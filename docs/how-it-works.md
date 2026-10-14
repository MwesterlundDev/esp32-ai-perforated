# How the ESP32 LLM works

Personal reference notes, drawn from `src/model.py`, `runtime/llm.h`,
`RESULTS.md` and `research/tinystories/README.md`.

## 1. The core problem: memory, not compute

The ESP32-S3 has three kinds of memory:

| Tier | Size | Speed | What lives there |
|---|---|---|---|
| SRAM | 512KB | ~240 MB/s | Activations, norm weights, scratch buffers (~29KB managed) |
| PSRAM | 8MB | ~61 MB/s | The dense transformer core and the output head (staged as int8) |
| Flash | 16MB | ~20µs per random read | The big lookup table (memory-mapped) |

A normal LLM touches every weight for every token. Here that would be a 15MB
model streamed through a 61 MB/s pipe. The design question is which weights must
be read every token and which can be read sparsely.

## 2. The trick: Per-Layer Embeddings (PLE)

The idea comes from Gemma 3n. The model has two kinds of parameters:

- **Core (559K params, dense):** attention, FFN and norms, 6 layers at
  d_model=96. Every token uses all of it.
- **PLE table (25M params, sparse):** a `32768 x (6 layers x 128)` embedding
  table. Each token ID selects one row, which is split into 6 slices, one per
  layer. A token reads only ~450 bytes.

So 25M of the 28.9M "parameters" sit in flash, and only ~6 small rows are read
per token. Measured table cost is ~0.12 ms per token, about 0.7% of the memory
time.

**How a layer uses it** (`Block.forward` in `src/model.py`):

```
x = x + attn(norm(x))
x = x + ffn(norm(x))
g = gelu(ple_gate(x))              # squeeze hidden state to 128 dims
x = x + norm(ple_proj(g * ple_l))  # gate it BY that layer's table slice, project back
```

The multiply is what makes the table act as conditioning rather than a bias.
Per-layer input is `(RMSNorm(proj(embed)) + table[token] * sqrt(128)) / sqrt(2)`,
so it mixes a context-aware part with the raw table row.

## 3. The output head

The head is a 32768x96 matrix, tied to the input embedding. It is dense but read
once per token as a sequential scan, so it costs bandwidth and not SRAM. It is
the biggest time sink at ~59 ms of the ~95 ms per token, and it is
PSRAM-bandwidth-bound. It is also why "bigger vocab makes the table cheap"
works: more vocab means more table rows, but each token still reads only a few.

## 4. Quantization

Everything big is **group-wise symmetric int4**: 4-bit codes with an fp16 scale
per 128 values. That is what shrinks the model to 14.9MB.

- It is **post-training quantization** (PTQ), with no quantization-aware
  training. The PLE gain survives it, and PLE degrades less than baseline.
- On-device, the head is unpacked to int8 once at boot, and activations are
  quantized to int8 per token. Each output row is then a plain int8 x int8 dot
  product.

## 5. The pipeline, end to end

```
TinyStories text
  -> prepare.py     trains a BPE tokenizer (32768 vocab), writes uint16 token bins
  -> train.py       PyTorch, src/model.py, arm=ple
  -> quantize_eval  checks the int4 version still performs
  -> export.py      model.bin (custom "PLE" format) + tokenizer.json + golden.txt
  -> deploy.sh      generates the vocab header, runs the C runtime on the laptop
                    and compares to the PyTorch golden (max diff 1e-5),
                    compiles, flashes the model partition (0x110000), then firmware
  -> ESP32          mmaps model.bin from flash, runs runtime/llm.h
```

`runtime/llm.h` is one portable C header that reimplements `model.py` op for op
(RoPE, RMSNorm, SwiGLU, GELU). The same code runs on the laptop for verification
and on the chip. The `verify.c` golden gate proves the C and PyTorch agree
before anything touches the board.

## 6. Why it is ~10 tok/s and not 58

The bandwidth ceiling is ~58 tok/s, but the first correct port ran at 0.57
tok/s. The speedups came from:

- staging the head in PSRAM instead of reading flash
- int8 head and activations
- splitting head rows across both cores
- `-O3`, caching attention scores, and similar cleanups

The S3's SIMD instructions are still unused. The next real wins are reading
fewer bytes (an int4 head) or a smaller vocabulary.

Current profile (ms/token): head 59.4, attention 20.5, PLE path 6.4, FFN 6.5,
input 2.2. Total ~94.9 ms, 9.88 tok/s attached.

## 7. Training your own

Where a new model has freedom:

- **Dataset and tokenizer.** The model is only as capable as its data.
  TinyStories gives simple stories and nothing else. A different dataset means a
  different `prepare.py` run.
- **Size knobs.** `--d-model`, `--n-layers`, `--ple-dim`, `--vocab`,
  `--target-core`. The headline config is d=96, 6 layers, ple_dim=128,
  vocab 32768.
- **The budget.** The flash model partition is 15.6MB, so table plus quantized
  core plus head must fit. The current model leaves ~685KB spare, so there is
  little room to grow. A bigger vocab or wider `ple_dim` needs a smaller core.
- **Expect limits.** The dense core (559K) caps reasoning, and the table does
  not fix that. A different dataset changes what it says, not how smart it is.

Things to be careful about:

- **Tokenizer hash check.** Exporting with the wrong tokenizer produces fluent
  text from the wrong token IDs rather than an error. The exporter and sampler
  check a recorded SHA-256 for this reason.
- **TinyStories-specific wiring.** The sketch has a fixed prompt (`PROMPT_IDS`
  in `esp32_tinystories.ino`). A new tokenizer needs matching prompt IDs, and
  `deploy.sh tinystories` generates the vocab header from the tokenizer being
  deployed.
- **Barista** shows separate input and output vocabularies, for question
  answering instead of free-form generation. Its training code is not published
  in this repo.

### Suggested path

1. Reproduce a small run to see the loop work:
   ```bash
   uv sync
   uv run python -m research.tinystories.prepare --vocab 4096
   uv run python -m research.tinystories.train --arm ple --vocab 4096 \
     --steps 3000 --target-core 1500000 --seed 0
   ```
2. Sample from the checkpoint:
   ```bash
   uv run python -m research.tinystories.sample \
     --run runs/<tag>.pt --tokenizer data/tinystories/vocab-4096/tokenizer.json
   ```
3. Change one thing (dataset, `ple_dim`, vocab) and compare.
4. Export and flash with `export.py`, then `scripts/deploy.sh tinystories`.

Open question: whether `train.py` supports a Mac GPU (MPS) or needs CUDA has not
been checked. That decides how realistic a full 32k-vocab run is locally.

## 8. Quick commands

```bash
# Flash a published model
scripts/fetch_model.sh tinystories
scripts/deploy.sh tinystories

# Watch serial output (port from deploy.sh)
arduino-cli monitor -p /dev/cu.usbmodemNNNN --config baudrate=115200
```

Board notes for this setup:

- Use the port labeled **USB** (native USB), not COM.
- If flashing fails to connect: hold BOOT, tap RST, release BOOT, then retry.
  The port name can change (`usbmodem1234561` to `usbmodem1101`).
- With no OLED attached, `USE_DISPLAY` is `0` in `esp32_tinystories.ino`. Set it
  back to `1` once a display is wired (SDA to GPIO18, SCL to GPIO46, SH1106 at
  `0x3C`).
- The sketch generates one story per boot from a fixed prompt. Tap RST for
  another.

## 9. Benchmarking

Baselines recorded 2026-10-14 on this board with the shipped TinyStories model
(`fp=a9bdd778`, 14,912,348 bytes, display off).

| | Value |
|---|---|
| Val CE / perplexity (int8 activations, what the chip runs) | **2.4796 / 11.94** |
| Val CE / perplexity (fp32 activations) | 2.4793 / 11.93 |
| Compute per token | **94.7 ms** |
| End-to-end throughput (200 tokens) | **10.34 tok/s** (96.75 ms/token) |
| Profile (ms/token) | head 59.1, attn 20.7, ffn 6.4, ple 6.3, input 2.1 |
| Flash model size | 14,912,348 bytes (partition 15,597,568) |

The published figure is 9.88 tok/s. The board runs a little faster end to end
with the display off; compute time matches.

Sample text is the usual "Once upon a time... Lily" story. It starts repeating
and restarts "Once upon a time" around token 130 (greedy decoding).

### Quality: validation perplexity

Runs the C runtime, the same code as on the chip, over held-out tokens.

```bash
# one-time: tokenizer plus val.bin (downloads ~300MB)
uv sync
uv run python -m research.tinystories.prepare --vocab 32768

cc -O3 -DLLM_INT8_ACT=1 -o /tmp/ppl_int8 runtime/host_verify/ppl.c -lm
/tmp/ppl_int8 artifacts/tinystories/model.bin \
  data/tinystories/vocab-32768/val.bin 128      # 128 windows = 32,768 predictions
```

Saved output: `runs/baseline-tinystories-ppl.txt`.

### Speed: on the board

Close any serial monitor first; the script resets the board over DTR and the
port can only be held by one program.

```bash
uv sync --extra device
uv run python scripts/benchmark_device.py tinystories --out runs/before.json
# after a change and a reflash:
uv run python scripts/benchmark_device.py tinystories --out runs/after.json \
  --baseline runs/before.json
```

`--baseline` fails if the generated text changed, so it suits speed-only
changes. For a new model the text differs by design; compare perplexity, speed
and samples by hand.

### Sampling from a checkpoint

```bash
uv run python -m research.tinystories.sample --run runs/<tag>.pt \
  --tokenizer data/tinystories/vocab-32768/tokenizer.json --seed 0
```

Needs a training checkpoint under `runs/`.

### Comparing fairly

- Same tokenizer and vocab, or per-token perplexity is not comparable. If the
  tokenizer or vocab changes, compare bits per byte of text instead.
- Same `val.bin` and the same window count every time.
- Compare the quantized model against the quantized model.
- Seed noise is about 0.006 nats; a smaller gap is not evidence. Run two seeds.
- Check speed and flash size alongside quality. A better model that overflows
  the 15.6MB partition or slows generation is not an improvement here.

### Memory bandwidth benchmark

`firmware/benchmarks/bandwidth/bandwidth.ino` measures what a token costs to
fetch on this chip, using the Xtensa cycle counter. It is the measurement the
PLE design rests on: it shows the flash table is nearly free and the output head
dominates memory traffic. It reports:

1. PSRAM sequential read bandwidth (where the head lives)
2. Internal SRAM sequential read bandwidth (upper bound)
3. Flash random-read latency for 512 B and 3072 B rows, and the cost of 6 table
   rows per token

It re-runs every 8 seconds, so a monitor that connects late still catches a full
run.

Measured on this board 2026-10-14 (output in `runs/baseline-bandwidth.txt`);
they match the published `RESULTS.md` figures.

| measurement | value |
|---|---|
| PSRAM sequential read | 60.7 MB/s |
| Internal SRAM sequential read | 240 MB/s |
| Flash random read, 512 B row | 20.3 us |
| Flash random read, 3072 B row | 117.1 us |
| Table cost per token (6 random rows) | 122 us (~0.12 ms) |
| 1MB PSRAM scan (the sketch scans 1MB) | 17.27 ms |
| Bandwidth-only ceiling | ~58 tok/s |

The ceiling is far above the measured ~10 tok/s because compute, not raw
bandwidth, dominates. The head is also PSRAM-bandwidth-bound at runtime
(int8 head: 2.43MB per token, ~40 ms floor at 60.7 MB/s).

To run it, flash this sketch in place of the TinyStories firmware. It uploads
with the default partition scheme, which rewrites the partition table, so the
model partition is no longer valid afterwards. Redeploy with
`scripts/deploy.sh tinystories` to restore the story demo; that rewrites both
the model and the firmware. This procedure worked on this board. The FQBN
options are copied from `deploy.sh` minus the custom partition scheme, since
this sketch ships no `partitions.csv`, so it probes the default `app0` partition
for flash latency. The sketch repeats every 8 s, so wait for a full `(done)`
block:

```bash
FQBN='esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,CPUFreq=240,FlashMode=qio,FlashSize=16M,PSRAM=opi'
arduino-cli compile --fqbn "$FQBN" firmware/benchmarks/bandwidth
arduino-cli upload  --fqbn "$FQBN" -p /dev/cu.usbmodemNNNN firmware/benchmarks/bandwidth
arduino-cli monitor -p /dev/cu.usbmodemNNNN --config baudrate=115200
```

Use it when a change affects where weights live (a bigger table, a different
head, int4 in PSRAM), to check the bandwidth floor before spending time on
compute.
