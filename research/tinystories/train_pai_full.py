"""Full PerforatedAI integration (not demo mode): a dendrite on the output head.

Copy of train.py, changed only where PerforatedAI needs it (search "PAI"):
  * the dendrite target is the output head, set explicitly (".head"), not whatever
    demo mode would pick (it picks blocks.5.ple_proj). The head shares its weight
    with tok_emb, so GPA's experimental weight-tying support is switched on.
  * PerforatedAI decides when training ends, so the fixed-length cosine schedule
    cannot be used. Training is chunked into "epochs" of --epoch-steps steps, one
    validation per epoch, and the learning rate is warmup then constant. The
    optimizer is rebuilt (and warmup restarts) after each restructure.
  * --control runs the identical loop with no PerforatedAI, for --control-epochs.
    Compare the two at the same number of steps; the JSON has val per epoch.

Dendrites here are the open-source (gradient descent) kind: Perforated
Backpropagation (perforatedbp) is off unless --pb is given, and needs a license.

Run from the repo root with the PerforatedAI venv, e.g.
  PYTHONPATH=src .venv-pai/bin/python research/tinystories/train_pai_full.py \\
    --arm ple --capacity-test ...
Outputs: runs/pai-full-<name>.json (not checkpoints; PAI writes its own folder,
PAI_head*/). Dendrite checkpoints do not load into plain TinyLM.
"""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from perforatedai import globals_perforatedai as GPA  # PAI
from perforatedai import utils_perforatedai as UPA  # PAI
from perforatedai import demo_perforatedai as DPA  # PAI: only for set_conversion_lists

from model import Config, make_model

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tinystories"
RUNS = str(ROOT / "runs")


def variant_dir(vocab_size):
    return DATA / f"vocab-{vocab_size}"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Batcher:
    def __init__(self, split, batch_size, seq_len, device, dataset, seed=0):
        self.data = np.memmap(dataset / f"{split}.bin", dtype=np.uint16, mode="r")
        self.bs, self.sl, self.device = batch_size, seq_len, device
        self.rng = np.random.default_rng(1234 if split == "val" else seed)

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.sl - 1, self.bs)
        x = np.stack([self.data[i : i + self.sl] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.sl] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)


@torch.no_grad()
def evaluate(model, batcher, iters):
    model.eval()
    batcher.rng = np.random.default_rng(1234)  # same val batches every time
    losses = [model(*batcher())[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def build_optimizer(model, lr):
    """Same AdamW and weight-decay split as train.py. Called again after every PAI
    restructure, because the model's parameters change."""
    # No weight decay on 1-D params (norms) or lookup tables. Tied weights are
    # matched by identity so the split does not depend on which name is listed first.
    tok_ids = {id(p) for n, p in model.named_parameters(remove_duplicate=False)
               if "tok_emb" in n}
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in n or id(p) in tok_ids else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )


def warmup_lr(since_build, peak, warmup):  # PAI: replaces the cosine lr_at
    return peak * min(1.0, (since_build + 1) / max(1, warmup))


def n_params(model):
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "ple", "ple_notable", "fatembed", "bigcore"])
    ap.add_argument("--target-core", type=int, default=1_500_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--ple-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--fixed-ffn", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    # PAI options
    ap.add_argument("--epoch-steps", type=int, default=250,
                    help="training steps per PAI 'epoch' (one validation each)")
    ap.add_argument("--switch-epochs", type=int, default=8,
                    help="epochs between switches between normal and dendrite training")
    ap.add_argument("--max-dendrites", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=60, help="safety cap")
    ap.add_argument("--capacity-test", action="store_true",
                    help="debug: add a dendrite every epoch to check the path works")
    ap.add_argument("--pb", action="store_true",
                    help="Perforated Backpropagation (needs a perforatedbp license)")
    ap.add_argument("--control", action="store_true",
                    help="same loop, no PerforatedAI")
    ap.add_argument("--control-epochs", type=int, default=24)
    args = ap.parse_args()

    if not 0 < args.vocab <= 65536:
        raise SystemExit("--vocab must be 1..65536; the token bins are uint16")
    dataset = variant_dir(args.vocab)
    tok_path = dataset / "tokenizer.json"
    if not os.path.exists(tok_path):
        raise SystemExit(
            f"{tok_path} missing. Run: python -m research.tinystories.prepare "
            f"--vocab {args.vocab}")
    tok_sha = hashlib.sha256(open(tok_path, "rb").read()).hexdigest()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(RUNS, exist_ok=True)

    base = Config(seq_len=args.seq_len, ple_dim=args.ple_dim, vocab_size=args.vocab,
                  d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
    model = make_model(args.arm, args.target_core, base, fixed_ffn=args.fixed_ffn).to(device)
    budget = model.param_budget()
    cfg = model.cfg
    params_before = n_params(model)

    if not args.control:
        # PAI: configuration, then perforate_model, before the optimizer is built.
        GPA.pc.set_perforated_backpropagation(args.pb)
        GPA.pc.set_weight_tying_experimental(True)  # head.weight is tok_emb.weight
        GPA.pc.set_output_dimensions([-1, -1, 0])   # [batch, tokens, features]
        GPA.pc.set_configuration_confirmed(True)    # target is set below, skip the picker
        GPA.pc.set_unwrapped_modules_confirmed(True)
        # Perforate only the head; every other module with parameters is tracked.
        # (Reuses demo mode's helper for building those lists.)
        DPA.set_conversion_lists(model, ".head")
        if args.capacity_test:
            GPA.pc.set_testing_dendrite_capacity(True)
        else:
            GPA.pc.set_testing_dendrite_capacity(False)
            GPA.pc.set_max_dendrites(args.max_dendrites)
            GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
            GPA.pc.set_fixed_switch_num(args.switch_epochs)
            GPA.pc.set_first_fixed_switch_num(args.switch_epochs)
        save_name = "PAI_head" + (f"_{args.tag}" if args.tag else "")
        model = UPA.perforate_model(model, save_name=save_name, maximizing_score=False)
        model = model.to(device)

    opt = build_optimizer(model, args.lr)
    if not args.control:
        GPA.pai_tracker.set_optimizer_instance(opt)  # PAI

    train_b = Batcher("train", args.batch_size, args.seq_len, device, dataset,
                      seed=args.seed)
    val_b = Batcher("val", args.batch_size, args.seq_len, device, dataset)

    kind = "ctl" if args.control else ("cap" if args.capacity_test else "pai")
    name = f"{kind}-{args.arm}{'-' + args.tag if args.tag else ''}-s{args.seed}"
    history = []
    step = since_build = 0
    t0 = time.time()
    max_epochs = args.control_epochs if args.control else args.max_epochs

    for epoch in range(max_epochs):
        for _ in range(args.epoch_steps):
            for g in opt.param_groups:
                g["lr"] = warmup_lr(since_build, args.lr, args.warmup)
            x, y = train_b()
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            since_build += 1

        vl = evaluate(model, val_b, args.eval_iters)
        mode = ""
        if not args.control:
            try:
                mode = str(GPA.pai_tracker.member_vars.get("mode", ""))
            except Exception:
                pass
        rec = {"epoch": epoch, "step": step, "val": vl, "train": loss.item(),
               "params": n_params(model), "mode": mode}
        history.append(rec)
        print(f"{name} epoch {epoch:3d} step {step:6d} | train {rec['train']:.4f} "
              f"| val {vl:.4f} | ppl {math.exp(vl):7.2f} | params {rec['params']:,} "
              f"{mode} | {time.time() - t0:5.0f}s", flush=True)

        if args.control:
            continue
        model, restructured, complete = GPA.pai_tracker.add_validation_score(vl, model)  # PAI
        model = model.to(device)
        if complete:
            print(f"{name} PAI training complete after {epoch + 1} epochs ({step} steps)",
                  flush=True)
            break
        if restructured:
            opt = build_optimizer(model, args.lr)
            GPA.pai_tracker.set_optimizer_instance(opt)  # PAI
            since_build = 0  # warmup restarts with the new optimizer

    final_val = evaluate(model, val_b, args.eval_iters)  # for PAI: the best model PAI restored
    first_switch = args.switch_epochs - 1
    result = {
        "kind": kind,
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "training": {
            "batch_size": args.batch_size, "seq_len": args.seq_len, "lr": args.lr,
            "warmup": args.warmup, "epoch_steps": args.epoch_steps,
            "switch_epochs": args.switch_epochs, "max_dendrites": args.max_dendrites,
            "capacity_test": args.capacity_test, "perforated_backprop": args.pb,
            "target_core": args.target_core, "seed": args.seed,
            "schedule": "warmup then constant lr; warmup restarts after a restructure",
        },
        "tokenizer_sha256": tok_sha,
        "params_before": params_before,
        "params_after": n_params(model),
        "params_budget_before": budget,
        "val_at_first_switch": history[first_switch]["val"]
        if len(history) > first_switch else None,
        "final_val": final_val,
        "final_ppl": math.exp(final_val),
        "steps_run": step,
        "epochs_run": len(history),
        "wall_seconds": time.time() - t0,
        "history": history,
    }
    with open(os.path.join(RUNS, f"pai-full-{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"{name} DONE steps={step} params {params_before:,} -> {result['params_after']:,} "
          f"final_val={final_val:.4f} ppl={result['final_ppl']:.2f}")


if __name__ == "__main__":
    main()
