# Handoff: ESP32 TinyLM and PerforatedAI work

Written 2026-10-14. Everything below is the state at the end of this session.
Items marked **unverified** were estimated or assumed, not measured.

## Where things stand

- The board runs the 28.9M-parameter TinyStories model and reproduces the
  published numbers: perplexity 11.94 and about 10.3 tok/s.
- Five PerforatedAI dendrite experiments (three target sets, open-source and
  Perforated Backpropagation) found **no quality gain over a matched
  no-dendrite control**. Every gap is inside seed noise.
- A dendrite on the output head also would not fit on the chip (about 1.3MB
  needed against 685KB of spare flash; **unverified** estimate).
- One condition is still untested: dendrites on a **converged** baseline. All
  runs were 4000-6000 steps at a constant learning rate.
- The original goal, training your own model for the chip, has not started. The
  groundwork (benchmarks and a working training loop on the Mac GPU) is done.

## Git state

- Repo: your private repo `MwesterlundDev/esp32-ai` (`origin`). The original
  `slvDev/esp32-ai` is the `upstream` remote.
- Local `main` is **one commit ahead of `origin/main` and not pushed**:
  `24274ea` core-layer experiments and Part 4 of the team doc.
- Commits from this session, oldest first: `0491b2d` how-it-works doc and
  benchmark baselines, `1b6e1c1` display off in the TinyStories sketch,
  `a7837ef` demo-mode pilot script, `d465c30` full integration on the head,
  `52b22bc` Perforated Backpropagation run, `24274ea` core layers.
- **`runs/` is gitignored.** The JSON receipts and logs that the team doc cites
  exist only on this machine (`runs/`, 221MB). Copy the ones you want to keep
  somewhere shared. `data/` (450MB), `artifacts/` and the `PAI_*/` output
  folders are also local only.

## Environment

| | |
|---|---|
| Repo environment | `uv sync` creates `.venv` from `pyproject.toml` (Python 3.12, torch 2.13) |
| PerforatedAI environment | `.venv-pai` (Python 3.12, torch 2.14.1), gitignored |
| `perforatedai` | 3.3.0, editable install from `~/Projects/Perforatedai-cleanup/PerforatedAI` (a local checkout, not PyPI) |
| `perforatedbp` | 3.2.9 from PyPI (the licensed package) |
| Run the PAI scripts | from the repo root with `PYTHONPATH=src .venv-pai/bin/python ...` (the `model` module is not installed in `.venv-pai`) |
| GPU | Apple Silicon, PyTorch MPS. `train.py` and the PAI scripts pick it automatically |
| Tokenizer and data | `data/tinystories/vocab-32768/`, built with `prepare --vocab 32768`; the tokenizer hash matches `artifacts/tinystories/tokenizer.json` |

**Credentials.** Perforated Backpropagation needs your perforatedai.com license.
It is read from the environment variables `PAIEMAIL` and `PAITOKEN`. A
`perf-license.sh` file sits in the repo root; it is gitignored and was never
opened or committed from this session. Do not commit it or paste its contents
anywhere. The open-source dendrites need no license.

## Hardware state

- Board: ESP32-S3 N16R8. Use the **USB** port, not COM.
- Flashed with the TinyStories model and serial-only firmware (`USE_DISPLAY 0`
  in `firmware/esp32_tinystories/esp32_tinystories.ino`; set it to `1` once an
  OLED is wired to SDA 18 / SCL 46). Boot fingerprint `fp=a9bdd778`,
  `bytes=14912348`.
- Last seen on `/dev/cu.usbmodem1101`. The port name changes between plugs
  (`usbmodem1234561` earlier). **The board was not detected at the end of this
  session**, probably unplugged.
- If flashing fails to connect: hold BOOT, tap RST, release BOOT, retry.
- Only one program can hold the serial port. Close any `arduino-cli monitor`
  before running `scripts/deploy.sh` or `scripts/benchmark_device.py`.
- The bandwidth benchmark sketch rewrites the partition table. After running it,
  redeploy with `scripts/deploy.sh tinystories`.

## Baselines (recorded 2026-10-14)

| | Value |
|---|---|
| Val CE / perplexity (int8 activations, what the chip runs) | 2.4796 / 11.94 |
| Compute per token | 94.7 ms |
| End-to-end throughput, display off | 10.34 tok/s |
| Model size / flash partition | 14,912,348 B / 15,597,568 B |
| PSRAM sequential read | 60.7 MB/s |
| Flash random read, 512 B row | 20.3 us |
| Table cost per token | 122 us |

Commands and the full tables are in `docs/how-it-works.md`, section 9.

## What was run

| Experiment | Target | Mode | Gap to matched control |
|---|---|---|---|
| Demo-mode pilot | `blocks.5.ple_proj` (demo mode's pick) | open-source | -0.001 |
| Full integration | head | open-source | +0.005 |
| Full integration | head | Perforated Backpropagation | +0.003 |
| Full integration | `ffn.gate`, `ffn.down` (last block) | open-source | +0.002 |
| Full integration | `ffn.gate`, `ffn.down` (last block) | Perforated Backpropagation | 0.000 |

"Matched" means the same number of main-weight updates. Perforated
Backpropagation's dendrite phase trains only the dendrite, so comparing at equal
total steps would be unfair to the dendrite run. Details, per-epoch tables and
caveats: `docs/perforatedai-pilot-results.md` (Parts 1-4).

## Decisions and findings worth remembering

- **Demo mode did not target the head.** It picks the last linear layer in
  module creation order, and `TinyLM` creates the head before the blocks, so it
  chose `blocks.5.ple_proj`.
- **Weight tying.** `head.weight` is `tok_emb.weight`. PerforatedAI's checkpoint
  writer needs `GPA.pc.set_weight_tying_experimental(True)` even when the head is
  not a target. `train_pai_full.py` sets it.
- **A head dendrite is a poor trade even if it helped.** It adds about 3.18M
  parameters, trained about 1.8x slower per epoch on the Mac, and would cost
  about 59 ms more per token on the chip (**unverified**, about 6.5 tok/s).
- **Library bug (not fixed).** In `perforatedai/demo_perforatedai.py`,
  `start_demo_mode` registers the output-dimensions hook only when
  `doing_pai` is true, so a demo-mode control run on a Linear layer with 3-D
  input fails. `train_pai.py` works around it for `--control`. A fix is to move
  the hook registration out of the `if doing_pai:` block.
- **Why GPT-2 and nanoGPT do not fit.** GPT-2 small is 124M parameters (about
  62MB at int4) against 16MB of flash, and its parameters are dense. This
  repo's trick relies on a sparse per-token table, which GPT-2 lacks. A diagram
  comparing nanoGPT and TinyLM blocks is at
  https://claude.ai/artifact/EwQdsJ2WUuBvz6evqLGAab (private).

## What was added to the repo

| File | Purpose |
|---|---|
| `docs/how-it-works.md` | Architecture reference, commands and benchmark baselines |
| `docs/perforatedai-pilot-results.md` | Team report, Parts 1-4 |
| `docs/HANDOFF.md` | This file |
| `research/tinystories/train_pai.py` | Demo-mode copy of `train.py` (`--control` supported) |
| `research/tinystories/train_pai_full.py` | Full integration; `--targets`, `--pb`, `--control`, `--capacity-test` |
| `scripts/run_pai_full_experiment.sh` | Head: capacity test, open-source run, matched control |
| `scripts/run_pai_pb_experiment.sh` | Head with Perforated Backpropagation (reads `PAIEMAIL`, `PAITOKEN`) |
| `scripts/run_pai_core_experiment.sh` | Core layers, open-source then Perforated Backpropagation |

## Suggested next steps, in order

1. **Push** the unpushed commit (`git push origin main`) once you are happy with
   it, and copy any `runs/` receipts you want to keep.
2. **Converged-baseline dendrite test.** Rerun the core-layer experiment on the
   shipped schedule (5000 steps, cosine decay; settings in
   `research/tinystories/scripts/run_deploy_ablation.sh`). That needs
   `train_pai_full.py` to support a decaying schedule, which it does not today
   because PerforatedAI decides the run length. If it is also a tie, close the
   dendrite line for this model.
3. **Train your own model**, the original goal. The training loop works on the
   Mac GPU (about 7 steps/s at batch 16 and sequence 256, so about 9-12 minutes
   for 4000-5000 steps). The decision still to make is the dataset and domain;
   a narrow domain suits a model this small. Keep `TinyLM`'s architecture
   unless you are prepared to extend the C runtime, the exporter and the host
   golden check.
4. **Speed**, if tokens per second matters more than new capability: the
   head is PSRAM-bandwidth-bound, so the levers in `RESULTS.md` are an int4 head
   with SIMD unpack or a smaller output vocabulary.
5. **Optional:** a small budget calculator (flash, PSRAM, tokens per second) so
   model sizes can be tested before training. Offered, not built.

## Not verified

- ESP32 cost of any dendrite (flash, speed): arithmetic only.
- That the library's template copy of a perforated layer is not exported.
- Why the control and dendrite runs differ slightly before the dendrite is
  added (about 0.005 nats). Likely GPU nondeterminism; not isolated.
- The nanoGPT side of the comparison diagram, written from memory of its
  `model.py`, not checked against the source.
- Whether results hold on CPU, or for other seeds, models or datasets.
