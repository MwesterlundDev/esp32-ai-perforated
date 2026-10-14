"""PerforatedAI pilot copy of train.py: one dendrite via perforate_model(demo_mode=True).

Differs from train.py only where demo mode needs it (search for "PAI PILOT"):
  * the model is wrapped to take one tensor and return one tensor (logits); the
    loss is computed outside, with the same cross-entropy as TinyLM.forward
  * the hand-set learning rate is replaced by a LambdaLR with the same lr_at curve
  * perforate_model is called before the optimizer is built
  * it trains 2 * --steps: phase 1 (dendrite frozen) is the baseline, phase 2 has
    the dendrite on and replays the schedule at 0.25x learning rate
  * outputs are written as pai-<name> (ctl-<name> with --control, which runs the same
    phases and learning rate restart with no dendrite), so they never overwrite
    train.py runs

Run from the repo root, with the PerforatedAI venv:
  PYTHONPATH=src .venv-pai/bin/python research/tinystories/train_pai.py --arm ple ...
Use --arm ple: demo mode targets the vocabulary head only while it is under 25% of
the parameters. Checkpoints contain the dendrite and do not load into TinyLM.
"""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from perforatedai import utils_perforatedai as UPA  # PAI PILOT

from model import Config, TinyLM, make_model

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# A vocabulary variant keeps its tokenizer and token bins together.
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
        # The bins are uint16. Reading a wider vocabulary through that dtype
        # yields plausible token ids rather than an error, so check before
        # opening.
        self.data = np.memmap(dataset / f"{split}.bin", dtype=np.uint16, mode="r")
        self.bs, self.sl, self.device = batch_size, seq_len, device
        # Batch order is part of the run. Without the seed here, torch.manual_seed
        # fixes initialisation only and two runs at the same --seed still see
        # different data order. Validation keeps a fixed stream so every arm is
        # scored on identical batches.
        self.rng = np.random.default_rng(1234 if split == "val" else seed)

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.sl - 1, self.bs)
        x = np.stack([self.data[i : i + self.sl] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.sl] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)


class LogitsOnly(nn.Module):  # PAI PILOT: demo mode needs tensor in, tensor out
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)[0]


def lm_loss(logits, y, out_vocab):  # PAI PILOT: same loss as TinyLM.forward
    return F.cross_entropy(logits.reshape(-1, out_vocab), y.reshape(-1), ignore_index=-1)


@torch.no_grad()
def evaluate(model, batcher, iters, out_vocab):
    model.eval()
    batcher.rng = np.random.default_rng(1234)  # same val batches for every arm
    losses = []
    for _ in range(iters):
        x, y = batcher()
        losses.append(lm_loss(model(x), y, out_vocab).item())
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["baseline", "ple", "ple_notable", "fatembed", "bigcore"],
    )
    ap.add_argument("--target-core", type=int, default=1_500_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--ple-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--fixed-ffn", type=int, default=None,
                    help="pin ffn_hidden and skip the core solver (table-scaling sweep)")
    # Published experiments always pass --vocab; the default is for ad-hoc runs.
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--control", action="store_true",
                    help="PAI PILOT: same phases and LR restart, no dendrite (doing_pai=False)")
    args = ap.parse_args()

    # Before anything expensive: the tokenizer that produced these bins. Its
    # hash is what lets the exporter and sampler refuse a mismatched tokenizer
    # later, and a checkpoint written without it cannot be tied to one. Failing
    # here costs nothing; failing after a 30-minute run costs the run.
    if not 0 < args.vocab <= 65536:
        raise SystemExit(f"--vocab must be 1..65536; the token bins are uint16 "
                         f"and are memmapped as uint16")

    dataset = variant_dir(args.vocab)
    tok_path = dataset / "tokenizer.json"
    if not os.path.exists(tok_path):
        raise SystemExit(
            f"{tok_path} missing. The bins this run trains on came from it, and "
            f"without its hash the checkpoint cannot be tied to a tokenizer. "
            f"Run: python -m research.tinystories.prepare --vocab {args.vocab}")
    tok_sha = hashlib.sha256(open(tok_path, "rb").read()).hexdigest()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(RUNS, exist_ok=True)

    base = Config(seq_len=args.seq_len, ple_dim=args.ple_dim, vocab_size=args.vocab,
                  d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
    core_model = make_model(args.arm, args.target_core, base, fixed_ffn=args.fixed_ffn).to(device)
    budget = core_model.param_budget()
    cfg = core_model.cfg
    out_vocab = cfg.resolved_out_vocab_size
    model = LogitsOnly(core_model)  # PAI PILOT

    train_b = Batcher("train", args.batch_size, args.seq_len, device, dataset,
                      seed=args.seed)
    val_b = Batcher("val", args.batch_size, args.seq_len, device, dataset)

    # PAI PILOT: the one PAI line. It sits before the optimizer is built and
    # before the parameter lists below. The sample batch comes from a throwaway
    # Batcher so train_b's data order is identical to train.py's. demo_phase_steps
    # is the length of the usual run (--steps optimizer steps).
    model = UPA.perforate_model(
        model, demo_mode=True, doing_pai=not args.control, demo_phase_steps=args.steps,
        demo_sample_input=Batcher("train", args.batch_size, args.seq_len, device,
                                  dataset, seed=args.seed)()[0])

    if args.control:
        # PAI PILOT: demo mode skips its output-dimensions hook when doing_pai=False,
        # but the wrapped Linear still sees (batch, tokens, features).
        for m in UPA.get_pai_modules(model, 0):
            m.set_this_output_dimensions([-1, -1, 0])

    # No weight decay on 1-D params (norms) or on lookup tables.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in n or "tok_emb" in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    # PAI PILOT: a real scheduler so demo mode can restart it. Same curve as lr_at,
    # built for --steps, not 2 * --steps; demo mode rewinds it at the switch.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_at(s, args.steps, args.lr, args.warmup) / args.lr)

    name = f"{'ctl' if args.control else 'pai'}-{args.arm}{'-' + args.tag if args.tag else ''}-s{args.seed}"
    history, best = [], float("inf")
    baseline_val = None
    t0 = time.time()

    total_steps = 2 * args.steps  # PAI PILOT: phase 1 (baseline) + phase 2 (dendrite)
    for step in range(total_steps):
        x, y = train_b()
        loss = lm_loss(model(x), y, out_vocab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        phase1_end = step == args.steps - 1
        if step % args.eval_every == 0 or phase1_end or step == total_steps - 1:
            vl = evaluate(model, val_b, args.eval_iters, out_vocab)
            best = min(best, vl)
            if phase1_end:
                baseline_val = vl  # the baseline: score at the end of phase 1
            tok = (step + 1) * args.batch_size * args.seq_len
            history.append({"step": step, "tokens": tok, "train": loss.item(), "val": vl})
            print(
                f"{name} step {step:5d}{' [END PHASE 1 = BASELINE]' if phase1_end else ''} | tok {tok / 1e6:6.1f}M | train {loss.item():.4f} "
                f"| val {vl:.4f} | ppl {math.exp(vl):7.2f} | {time.time() - t0:5.0f}s",
                flush=True,
            )

    result = {
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "training": {
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "warmup": args.warmup,
            "eval_every": args.eval_every,
            "eval_iters": args.eval_iters,
            "target_core": args.target_core,
            "fixed_ffn": args.fixed_ffn,
            "seed": args.seed,
        },
        "tokenizer_sha256": tok_sha,
        "params": budget,
        "pai_pilot": True,
        "control": args.control,
        "baseline_val": baseline_val,
        "baseline_ppl": math.exp(baseline_val),
        "final_val": history[-1]["val"],
        "best_val": best,
        "final_ppl": math.exp(history[-1]["val"]),
        "tokens_seen": total_steps * args.batch_size * args.seq_len,
        "steps": args.steps,
        "steps_run": total_steps,
        "wall_seconds": time.time() - t0,
        "history": history,
    }
    with open(os.path.join(RUNS, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    # Identity and schedule live only in the filename and the sidecar JSON
    # otherwise, so a checkpoint copied over another name, or trained on a
    # different schedule, would pass every content check.
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(),
                "tokenizer_sha256": tok_sha,
                "seed": args.seed, "tag": args.tag, "name": name,
                "training": result["training"]},
               os.path.join(RUNS, f"{name}.pt"))
    print(f"{name} baseline_val={baseline_val:.4f} (ppl {math.exp(baseline_val):.2f}) -> "
          f"{'control_no_dendrite' if args.control else 'with_dendrite'}_val={result['final_val']:.4f} (ppl {result['final_ppl']:.2f})")
    print(f"{name} DONE core={budget['core']:,} table={budget['table']:,} "
          f"val={result['final_val']:.4f} ppl={result['final_ppl']:.2f}")


if __name__ == "__main__":
    main()
