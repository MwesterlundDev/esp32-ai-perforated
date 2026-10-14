# PerforatedAI demo-mode pilot on the TinyStories PLE model

Date: 2026-10-14. One seed, one short run per condition. Read the
[Caveats](#caveats) before drawing conclusions.

Part 1 (this section) is the demo-mode pilot. Part 2, at the end, is the full
integration with a dendrite on the output head.

## Summary

- We ran the PerforatedAI **pilot** (`perforate_model(demo_mode=True)`) on the
  28.9M-parameter PLE TinyLM, with a no-dendrite **control** run of the same
  length.
- **Result: no measurable dendrite effect.** Final validation loss was 2.9161
  with the dendrite and 2.9170 for the control, a difference of 0.0009 nats.
- Both runs improved by about 0.144 nats between phase 1 and phase 2. That gain
  is the extra training in phase 2, not the dendrite. Without the control run
  it would have looked like a large win.
- **Demo mode did not put the dendrite on the output head.** It chose
  `blocks.5.ple_proj`, a 96x128 linear layer (12,288 parameters, about 0.04%
  of the model). A null result on a layer that small is expected. It says
  nothing about a dendrite on the 3.1M-parameter head.

## What was run

### Model and training setup

| | |
|---|---|
| Model | PLE TinyLM, arm `ple` (`src/model.py`) |
| Config | vocab 32768, d_model 96, 6 layers, ple_dim 128, `--target-core 560000` |
| Parameters | core 558,368; PLE table 25,165,824 (as reported by the script; excludes the dendrite) |
| Data | first 300 MB of TinyStories, 32768-vocab BPE tokenizer |
| Batch / sequence | 16 x 256 tokens |
| Steps | 1000 per phase (2000 total), seed 0 |
| Optimizer | AdamW, lr 1e-3, warmup 200, cosine decay to 10% (unchanged from `train.py`) |
| Hardware | Apple Silicon Mac, PyTorch MPS backend |
| Software | PerforatedAI 3.3.0 (local checkout, editable install), perforatedbp 3.2.9 (installed; demo mode turns Perforated Backpropagation off), torch 2.14.1 |

### How demo mode works

Phase 1 trains normally with the dendrite frozen. After `demo_phase_steps`
optimizer steps the dendrite unfreezes, and the optimizer and learning-rate
schedule restart at 0.25x their starting value and replay. The validation score
at the end of phase 1 is the baseline; the score at the end of phase 2 is
"baseline plus PerforatedAI".

Phase 2 is also 1000 extra training steps, which improves a model that is not yet
converged. The **control** run (`doing_pai=False`) runs the identical phases and
learning-rate restart with no dendrite, to separate that effect.

### Commands

From the repo root, using a separate venv with PerforatedAI installed:

```bash
# dendrite run
PYTHONPATH=src .venv-pai/bin/python research/tinystories/train_pai.py \
  --arm ple --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128 \
  --target-core 560000 --batch-size 16 --seq-len 256 --steps 1000 \
  --seed 0 --tag pilot

# control run (same, plus --control)
PYTHONPATH=src .venv-pai/bin/python research/tinystories/train_pai.py \
  --arm ple --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128 \
  --target-core 560000 --batch-size 16 --seq-len 256 --steps 1000 \
  --seed 0 --tag pilot --control
```

### Changes from `research/tinystories/train.py`

`train_pai.py` is a copy of `train.py`. The pilot skill says to leave the script
alone, but demo mode does not support two things in the original, so the copy
adapts them. Every edit is marked `PAI PILOT` in the file.

| Original `train.py` | Why demo mode can't use it | Change in the copy |
|---|---|---|
| Sets `param_group["lr"]` by hand each step | Demo mode restarts a real scheduler | `LambdaLR` using the same `lr_at` curve |
| `model(x, y)` returns `(logits, loss)` | Demo mode needs one tensor in and out | `LogitsOnly` wrapper returns logits; loss is computed outside with the same cross-entropy |

Other edits: `perforate_model` is called before the optimizer is built (using a
throwaway batcher for the sample batch, so the training data order is
unchanged); the loop runs 2x the steps; evaluation also runs at the end of
phase 1; outputs are named `pai-*` / `ctl-*` so they don't overwrite `train.py`
runs. Training hyperparameters were not changed.

One control-only fix: in `--control` mode (`doing_pai=False`) the library skips
its output-dimensions hook but still wraps the layer, which then fails on
3-D transformer activations (`this_output_dimensions` error, see
[Library issue](#library-issue)). The copy sets the dimensions by hand for the
control run only.

## Results

Validation loss is cross-entropy in nats over 40 fixed validation batches;
lower is better. Perplexity is exp(loss).

| | Val loss | Perplexity |
|---|---|---|
| Baseline, end of phase 1 (dendrite run, step 999) | 3.0591 | 21.31 |
| Baseline, end of phase 1 (control run, step 999) | 3.0612 | 21.35 |
| **With dendrite, end of training (step 1999)** | **2.9161** | **18.47** |
| **Control, no dendrite, end of training (step 1999)** | **2.9170** | **18.49** |

- Dendrite vs control: 0.0009 nats, a tie.
- Phase 1 -> phase 2 improvement: 0.1430 (dendrite) vs 0.1442 (control).
- The two runs differ by 0.002 or less at every evaluation after the phase
  switch (steps 1000-1999), which is the same-trajectory evidence.
- Phase 1 of the two runs differs by 0.0021 at step 999. The runs share a seed
  and data order, so this is likely GPU nondeterminism or the dendrite setup
  consuming a little random state. We did not isolate which.
- Runtime: about 280 seconds per run (2000 steps, about 7 steps/s).

### Full output: dendrite run

The `[PAI Config]` banner and the lines printed by `perforate_model` before
step 0 (target layer and parameter counts) were not captured. The target layer
appears in the phase-switch message below.

```text
----------------------------------------------------------------------
pai-ple-pilot-s0 step     0 | tok    0.0M | train 10.4081 | val 10.4100 | ppl 33188.82 |     2s
pai-ple-pilot-s0 step   250 | tok    1.0M | train 3.9746 | val 4.0214 | ppl   55.78 |    37s
pai-ple-pilot-s0 step   500 | tok    2.1M | train 3.3402 | val 3.3540 | ppl   28.62 |    71s
pai-ple-pilot-s0 step   750 | tok    3.1M | train 3.1721 | val 3.1343 | ppl   22.97 |   106s
pai-ple-pilot-s0 step   999 [END PHASE 1 = BASELINE] | tok    4.1M | train 3.1853 | val 3.0591 | ppl   21.31 |   140s
----------------------------------------------------------------------
PAI demo: phase 1 finished after 1000 optimizer steps.
  The dendrite on .inner.blocks.5.ple_proj is now training, and the learning rate restarted
  at 0.25x its starting value.
  Compare your validation score at the end of training with the score
  just before this message.
----------------------------------------------------------------------
pai-ple-pilot-s0 step  1000 | tok    4.1M | train 3.0685 | val 3.0591 | ppl   21.31 |   142s
pai-ple-pilot-s0 step  1250 | tok    5.1M | train 2.9213 | val 3.0319 | ppl   20.74 |   176s
pai-ple-pilot-s0 step  1500 | tok    6.1M | train 3.0341 | val 2.9761 | ppl   19.61 |   211s
pai-ple-pilot-s0 step  1750 | tok    7.2M | train 2.6630 | val 2.9352 | ppl   18.82 |   246s
pai-ple-pilot-s0 step  1999 | tok    8.2M | train 3.0007 | val 2.9161 | ppl   18.47 |   281s
pai-ple-pilot-s0 baseline_val=3.0591 (ppl 21.31) -> with_dendrite_val=2.9161 (ppl 18.47)
pai-ple-pilot-s0 DONE core=558,368 table=25,165,824 val=2.9161 ppl=18.47
```

### Full output: control run

The first control attempt failed before training with a library error, then
succeeded after the dimension fix (see [Library issue](#library-issue)).
Output of the successful run, with the local path in the config line removed:

```text
[PAI Config] Saved 103 variables -> PAI_demo/PAI_demo_config.json
Running Dendrite Experiment
----------------------------------------------------------------------
PAI demo mode, control run: same phases and restarts, no dendrite.
  Phase 1: your normal training for 1000 optimizer steps.
  Phase 2: the dendrite trains, and your optimizer and schedulers restart at
           0.25x their starting learning rate, replaying the same schedule.
  Run twice your usual number of epochs, and create the optimizer and
  scheduler after this call.
  Not supported: multiple optimizers, Muon, EMA or averaged model copies,
  DataParallel/DDP, or code that sets the learning rate by hand each step.
  Demo mode switches after a fixed number of steps, not on validation
  scores. If your model already overfits, the dendrite adds capacity and
  can make validation worse.
----------------------------------------------------------------------
ctl-ple-pilot-s0 step     0 | tok    0.0M | train 10.4081 | val 10.4100 | ppl 33188.82 |     2s
ctl-ple-pilot-s0 step   250 | tok    1.0M | train 3.9734 | val 4.0335 | ppl   56.46 |    36s
ctl-ple-pilot-s0 step   500 | tok    2.1M | train 3.3434 | val 3.3542 | ppl   28.62 |    71s
ctl-ple-pilot-s0 step   750 | tok    3.1M | train 3.1712 | val 3.1368 | ppl   23.03 |   105s
ctl-ple-pilot-s0 step   999 [END PHASE 1 = BASELINE] | tok    4.1M | train 3.1877 | val 3.0612 | ppl   21.35 |   139s
----------------------------------------------------------------------
PAI demo: phase 1 finished. Restarting the learning rate schedule at 0.25x
  its starting value (no new dendrite).
----------------------------------------------------------------------
ctl-ple-pilot-s0 step  1000 | tok    4.1M | train 3.0752 | val 3.0611 | ppl   21.35 |   141s
ctl-ple-pilot-s0 step  1250 | tok    5.1M | train 2.9228 | val 3.0339 | ppl   20.78 |   175s
ctl-ple-pilot-s0 step  1500 | tok    6.1M | train 3.0335 | val 2.9781 | ppl   19.65 |   210s
ctl-ple-pilot-s0 step  1750 | tok    7.2M | train 2.6666 | val 2.9361 | ppl   18.84 |   245s
ctl-ple-pilot-s0 step  1999 | tok    8.2M | train 3.0032 | val 2.9170 | ppl   18.49 |   279s
ctl-ple-pilot-s0 baseline_val=3.0612 (ppl 21.35) -> with_dendrite_val=2.9170 (ppl 18.49)
ctl-ple-pilot-s0 DONE core=558,368 table=25,165,824 val=2.9170 ppl=18.49
```

In this capture the summary line is labeled `with_dendrite_val`, but the control
has no dendrite. The label has since been fixed in `train_pai.py`
(`control_no_dendrite_val`).

## What the dendrite was attached to

Demo mode picks the layer itself: the last `nn.Linear`/`nn.Conv2d` in module
creation order, or the one before it if the last holds 25% or more of the
parameters. In `TinyLM` the head is created before the transformer blocks, so
the "last" linear is `blocks.5.ple_proj` (a 96x128 matrix, no bias; 12,288
parameters, computed from the shapes, not printed by the library).

The output head is `head`, tied to the token embedding, about 3.1M parameters
(about 11% of the model in the `ple` arm). It was **not** perforated in this
pilot. Demo mode has no option to choose the layer; targeting the head needs
the full PerforatedAI integration, where the target module is set explicitly.

## Caveats

- **One seed, one run per condition.** The repo's earlier multi-seed tests
  (`RESULTS.md`) put seed-to-seed noise at about 0.006 nats for the 5000-step
  deploy configuration. That figure was not measured for this 1000-step setup.
  The 0.0009 gap is well inside it either way.
- **Short, under-trained model.** 1000 steps of batch 16 reaches ppl 21, against
  ppl 11.94 for the shipped 5000-step model. A dendrite may behave differently
  on a converged model.
- **Tiny target layer.** The dendrite added about 0.04% parameters. The null
  result is what a dendrite that size would be expected to show.
- **The pilot measures quality only.** It says nothing about the cost of a
  dendrite on the ESP32: a head dendrite would add about as many parameters as
  the head (about 3.1M), which matters for the 15.6MB flash partition (about
  685 KB spare) and the head's roughly 59 ms of the 95 ms per-token compute.
- **MPS backend.** The PerforatedAI pilot skill states its GPU path was only
  checked on CPU. The runs completed on MPS without errors, but we did not
  compare against CPU.
- **Checkpoints are not usable by the exporter.** `pai-*.pt` files contain the
  dendrite and do not load into the plain `TinyLM`.

## Library issue

For `doing_pai=False` (control) with `demo_mode=True`, `start_demo_mode` in
`perforatedai/demo_perforatedai.py` registers the
`set_linear_output_dimensions` hook only inside `if doing_pai:`, but still
converts the layer. The wrapped `nn.Linear` then expects 2-D input and fails on
transformer activations:

```text
The following module has not properly set this_output_dimensions
.inner.blocks.5.ple_proj
it is expecting:
tensor([-1,  0])
but received
torch.Size([16, 256, 96])
```

Workaround used here: call `set_this_output_dimensions([-1, -1, 0])` on the
perforated module after `perforate_model`, for the control run only. A fix in
the library would move the hook registration out of the `if doing_pai:` block.

## Reproducing

```bash
uv sync
uv run python -m research.tinystories.prepare --vocab 32768      # tokenizer + bins
uv venv --python 3.12 .venv-pai
uv pip install --python .venv-pai/bin/python perforatedai perforatedbp
# or install perforatedai editable from a local checkout, as used here (3.3.0)
```

Then run the two commands above. Run artifacts go to `runs/` (gitignored) and
`PAI_demo/` (gitignored). Per-run JSON receipts:
`runs/pai-ple-pilot-s0.json`, `runs/ctl-ple-pilot-s0.json`.

# Part 2: full integration, dendrite on the output head

Open-source (gradient descent) dendrites, explicit target `.head`. Script:
`research/tinystories/train_pai_full.py`; driver:
`scripts/run_pai_full_experiment.sh`.

## Summary

- **No gain.** At 4000 steps the dendrite run finished at val loss 2.5010
  (ppl 12.20) against 2.4960 (ppl 12.13) for the no-dendrite control. The
  0.005 nat difference is inside seed noise, and the dendrite run was slightly
  behind at every evaluation after the dendrite was added.
- **It costs a lot.** About 3.18M extra parameters (the head's size), about 1.8x
  slower training epochs on the Mac, and, by estimate, no room on the ESP32
  (see [Fit on the ESP32](#fit-on-the-esp32)).
- **The mechanism works.** The capacity test added dendrites in sequence on the
  real 32K-vocab head, with the head/embedding weight tie restored
  (`Restored weight tying: tok_emb.main_module.weight -> head.main_module.weight`).
  There is no technical blocker; the dendrite just does not help here.

## Setup

Same model, data and hardware as Part 1 (arm `ple`, vocab 32768, d_model 96,
6 layers, ple_dim 128, batch 16 x 256, seed 0, MPS), with these differences:

| | Part 1 (demo mode) | Part 2 (full integration) |
|---|---|---|
| Target layer | chosen by demo mode (`blocks.5.ple_proj`) | the head, set with `DPA.set_conversion_lists(model, ".head")` |
| Weight tying | n/a | `GPA.pc.set_weight_tying_experimental(True)` (head shares `tok_emb`'s weight) |
| Run length | fixed, 2x normal | PerforatedAI decides; here 16 epochs = 4000 steps |
| Epoch | n/a | 250 steps, one validation per epoch |
| Switching | one switch at a set step | fixed: every 8 epochs (`DOING_FIXED_SWITCH`), `max_dendrites=1` |
| Learning rate | cosine, as `train.py` | warmup (200 steps) then constant; optimizer rebuilt, warmup restarted, after each restructure |
| Perforated Backpropagation | off | off (`--pb` not given); no license needed |

The cosine schedule cannot be used when the run length is decided at runtime,
so Part 2 numbers are **not comparable** to the shipped model's ppl 11.94 or to
Part 1. The control uses the identical loop and schedule, so compare within
Part 2.

Two things needed for a non-demo run in the local PerforatedAI 3.3.0 checkout:
`GPA.pc.set_configuration_confirmed(True)` (otherwise `perforate_model` opens an
interactive target picker) and `GPA.pc.set_unwrapped_modules_confirmed(True)`.

### Commands

```bash
caffeinate -i bash scripts/run_pai_full_experiment.sh
```

runs, in order: a capacity test (`--capacity-test`, adds a dendrite every epoch),
the real run (`--tag head`), and a control matched to the real run's epoch count
(`--control --control-epochs N`). Log: `runs/full-experiment.log`. Receipts:
`runs/pai-full-pai-ple-head-s0.json`, `runs/pai-full-ctl-ple-ctl-s0.json`,
`runs/pai-full-cap-ple-cap-s0.json`.

## Results

Validation loss, nats, lower is better. Dendrite and control at the same step.

| Step | Control | Dendrite | Gap (dendrite - control) |
|---|---|---|---|
| 250 | 4.0329 | 4.0347 | +0.002 |
| 500 | 3.3403 | 3.3400 | -0.000 |
| 750 | 3.0967 | 3.0987 | +0.002 |
| 1000 | 2.9483 | 2.9483 | 0.000 |
| 1250 | 2.8508 | 2.8507 | -0.000 |
| 1500 | 2.7792 | 2.7768 | -0.002 |
| 1750 | 2.7267 | 2.7256 | -0.001 |
| 2000 (switch) | 2.6854 | 2.6808 | -0.005 |
| 2250 | 2.6389 | 2.6492 | +0.010 |
| 2500 | 2.6117 | 2.6223 | +0.011 |
| 2750 | 2.5807 | 2.5964 | +0.016 |
| 3000 | 2.5565 | 2.5689 | +0.012 |
| 3250 | 2.5416 | 2.5497 | +0.008 |
| 3500 | 2.5156 | 2.5240 | +0.008 |
| 3750 | 2.5049 | 2.5103 | +0.005 |
| **4000** | **2.4960 (ppl 12.13)** | **2.5010 (ppl 12.20)** | **+0.005** |

- Before the dendrite is added (steps 250-2000) the two runs differ by 0.005 or
  less. That is the noise floor between two same-seed runs.
- After it is added (step 2250 on) the dendrite run is behind at every
  evaluation. The gap peaks at 0.016 and narrows to 0.005, so it never pulls
  ahead.
- PerforatedAI logged `Last Dendrites were good and this hit the max of 1`. Its
  test is whether the score kept improving, which it does with more training
  regardless; the control shows the same improvement without a dendrite.
- Reading of the log (not confirmed in the library code): at the switch the
  library enters and leaves its dendrite-training mode inside one call
  (`set mode p` ... `Switching back to N`), so epochs 8-15 were ordinary joint
  training with the dendrite present.

### Cost

| | Control | Dendrite |
|---|---|---|
| Parameters (script's count) | 28,869,920 | 35,194,144 |
| Wall time, 16 epochs | 557 s | 780 s |
| Seconds per epoch after the switch | about 35 | about 62 |

The count includes a frozen template copy of the head (+3,145,728, equal to
`32768 x 96`) that appears at `perforate_model`, before any training. The
dendrite itself looks like +3,178,496 (`32768 x 96` weights plus 32768
connection weights); the smoke test on a toy model showed the same pattern
(+16,896 = 512 x 32 + 512). We did not confirm in the library code that the
template is not exported.

## Fit on the ESP32

Estimates from repo numbers and the parameter counts above. Not tried on the
board.

- **Flash does not fit.** The deployed dendrite would be the 25,353 active rows
  x 96 = 2.43M weights: about 1.22MB at int4 plus about 0.05MB of scales and
  connection weights, around 1.3MB. The model partition has 685KB spare;
  shrinking the 1MB app partition (it uses 669KB) and dropping the 64KB coredump
  partition would free up to about 380KB more, still roughly 0.2MB short.
- **Speed.** The head is 59 ms of 95 ms per token and bandwidth-bound; the
  dendrite is another full matrix-vector product of that shape. Estimate: about
  154 ms per token (about 6.5 tok/s, down from 10.3).
- **PSRAM is fine.** Staging the dendrite as int8 needs about 2.4MB of the
  roughly 4.2MB free.
- **Software.** `runtime/llm.h` has no dendrite concept, and `export.py` reads a
  plain `TinyLM` state dict while PerforatedAI checkpoints use a different
  structure (`main_module`, `dendrite_module`). The host golden gate would need
  the dendrite's exact forward pass, including its activation, which we have not
  read.

## Caveats

- One seed, 4000 steps, constant learning rate with no final decay. A dendrite
  might help more on a converged model.
- Open-source dendrites only. The library prints that Perforated
  Backpropagation can give better results; that needs a license and was not
  part of this run.
- Single small model on one dataset; it says nothing about other models.
- The seed-noise figure (about 0.006 nats) is from the repo's 5000-step deploy
  ablations, not measured at this setup. The pre-dendrite differences here
  (up to 0.005) are consistent with it.

## Suggested next step

The head is the wrong target: no quality gain, about 1.3MB it cannot afford and
about a third of the speed. Remaining options, cheapest first:

1. Perforated Backpropagation on the same setup (needs the license).
2. A core layer (attention or FFN) as the target. It would cost kilobytes of
   flash and a few percent of compute, so it could fit on the chip.
