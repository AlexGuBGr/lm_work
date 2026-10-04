# Work log – lm_work

Chronological log of what was done and how.

## 2026-10-03 — Step 1: System check and environment

**Hardware (lscpu, free, df):**
- CPU: Intel Xeon E5-2690 v2 @ 3.00GHz (Ivy Bridge), 2 vCPUs (KVM), 1 thread/core
- Instruction sets: SSE4.2, AVX, F16C, AES — **no AVX2, FMA, AVX-512 or AMX/avx512_bf16**
  → no hardware support for bfloat16 → **float32** is used
- RAM: 7.8 GiB (+ 8 GiB swap). Disk: 41 GB free on `/`
- OS: Ubuntu 24.04, system Python 3.12.3. No gcc/g++, no passwordless sudo.

**Python environment:**
- `python3 -m venv` failed (package `python3.12-venv`/ensurepip is missing, and sudo requires a password).
- Therefore installed `uv` for the user: `curl -LsSf https://astral.sh/uv/install.sh | sh` → `~/.local/bin/uv`
- Python 3.12 chosen: fully supported by PyTorch including torch.compile, and already on the system.
- `uv venv --python 3.12 .venv`

## Step 2: PyTorch (CPU)
- `uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu`
  → torch 2.14.1+cpu (MKL, oneDNN, OpenMP)
- `uv pip install numpy tqdm` → numpy 2.5.3, tqdm 4.70.1
- `torch.backends.cpu.get_cpu_capability()` = `DEFAULT` (no AVX2 kernels can be used on this CPU)

## Step 3: Data (TinyStoriesV2-GPT4)
- Downloaded directly from HF (`roneneldan/TinyStories`, `resolve/main/…`) with `curl` into `data/raw/`:
  - `TinyStoriesV2-GPT4-train.txt` 2,227,753,162 B (sha256 6418d412…0443)
  - `TinyStoriesV2-GPT4-valid.txt`    22,502,601 B (sha256 6874bae9…585d)
- `prepare.py` (run: `.venv/bin/python prepare.py`, ~2.5 min, max ~650 MB RAM):
  - Reads the file in chunks of whole lines (~32 MB via `readlines(hint)`), so `<|endoftext|>` is never split.
  - Pass 1: counts characters with `np.bincount` over Unicode code points (UTF-32).
  - Vocabulary: `<unk>`=0, `<eot>`=1 (replaces `<|endoftext|>`), + every character with ≥200 occurrences
    in train. A first attempt with threshold 1000 dropped the digits 4, 6–9 → lowered to 200.
  - Result: **87 tokens**; 228 distinct characters in total, 1,271 occurrences (0.00006%) → `<unk>` (mostly CJK/noise).
  - Pass 2: lookup table code point → id (uint8) → `data/train.bin` (2,194,232,880 tokens),
    `data/val.bin` (22,161,827 tokens), `data/meta.json` (itos).
  - 2,717,700 stories in train.

## Step 4: Model and training script
- `model.py`: Llama-style decoder-only transformer – pre-norm `nn.RMSNorm`, RoPE (cos/sin cache),
  SwiGLU (hidden 352 ≈ 8/3·128 rounded to a multiple of 32), causal attention via `F.scaled_dot_product_attention`,
  tied embedding/lm_head weights, no biases. 5 layers, 4 heads, width 128, context 256
  → **1,016,064 parameters**.
- `train.py` (nanoGPT style): AdamW (weight decay only on 2D weights), warmup + cosine LR, grad clip 1.0,
  batches = random slices from `np.memmap` (new memmap per batch, no DataLoader workers),
  `torch.set_num_threads(os.cpu_count())`, dtype auto (bf16 only if `avx512_bf16`/`amx_bf16` is in
  /proc/cpuinfo → float32 here), optional `--compile=true`.
  Checkpoints: `out/ckpt.pt` (latest, written atomically via .tmp + rename) and `out/best.pt` (best val).
  Also saves on Ctrl+C/SIGTERM. Logs to `out/train_log.csv`, configuration per (re)start in `out/runs.jsonl`.
  Resumes automatically from `out/ckpt.pt` (`init_from=auto`). All settings can be set with `--name=value`.
- `sample.py`: generates text from a checkpoint.

## Step 5: Speed measurements
- Smoke test (100 iters, batch 32): ~1,700 tok/s – slower than expected → profiling:
  - `aten::mm` ≈ 47% of the time, SDPA flash attention ≈ 35%.
  - Raw matmul (1024³): 14 GFLOP/s on 1 thread, 23 GFLOP/s on 2 threads → the machine is simply slow
    (virtualised Ivy Bridge, AVX without FMA; PyTorch capability `DEFAULT`).
  - SDPA backends (B=16, per layer fwd+bwd): default 0.22 s, math 0.25 s, flash 0.23 s → default kept.
- torch.compile needs a C++ compiler + Python headers, which the system lacks (no sudo):
  - micromamba downloaded to `~/.local/bin/micromamba` (extracted with Python since `bzip2` is missing),
    gcc 13 from conda-forge: `micromamba create -p ~/.local/share/toolchain -c conda-forge gxx_linux-64=13`.
  - System Python has no `Python.h` → venv recreated with uv-managed CPython **3.12.15**
    (`uv python install 3.12.15; uv venv --python 3.12.15 --managed-python .venv`), torch reinstalled.
  - `train.py`/`bench.py` set `CXX` to the toolchain g++ themselves when compile is enabled.
- `bench.py` (fwd+bwd+AdamW, T=256):

  | batch | eager tok/s | compile tok/s |
  |---|---|---|
  | 8  | 1,164 | 1,371 |
  | 16 | 1,373 | 1,550 |
  | 32 | 1,448 | 1,532 |
  | 64 | 1,554 | 1,588 |

  Compilation takes ~70–85 s the first time. Measurements vary ±10% between runs (shared VM).

### Test run with train.py (batch 32, 80 iters, eager vs compile)
- Eager: 1,470–1,583 tok/s (≈5.2–5.6 s/iter). Compile: 1,511–1,558 tok/s after ~80 s of compilation.
  Identical loss → **no measurable gain from torch.compile in practice → disabled** (`compile=False`).
- Loss fell 4.4 → 2.54 (val 2.545) in 80 iters; stable with LR warmup.
- Resume tested: resumed from iter 80 → 90, CSV log continued correctly. `sample.py` tested.
- Eval (2×50 batches) takes ~3.3 min → ~3–4% overhead with eval every 1000 iters.
- **Batch size choice:** tok/s levels off from batch 16 (~1,400–1,550); batch 32 chosen (8,192 tokens/iter)
  for less noisy gradients with no loss of throughput.

**Estimate (~1,500 tok/s, ~125M tokens/day):**
10k iters ≈ 15 h · **30k iters ≈ 2 days (246M tokens, chosen default)** · 100k ≈ 6.5 days · 1 epoch (2.2B) ≈ 17 days.

## Step 6: Full training started (2026-10-03 20:41)
- tmux session `lm_train` (the name `lm` was already used by another tmux session):
  `tmux new-session -d -s lm_train "cd ~/analyses/lm_work && .venv/bin/python -u train.py >> out/train.out 2>&1"`
- Settings: batch 32, max_iters 30,000, LR 3e-3 → 3e-4 (cosine, warmup 500), wd 0.1, float32, 2 threads.
- Resources: ~180% CPU, ~1 GB RAM. Expected to finish around the evening of 2026-10-05.

## Step 7: sample.py and NOTES.md written
- See `NOTES.md` for monitoring/stopping/resuming/generating.

## 2026-10-04 — Architecture diagram
- `uv pip install matplotlib` (3.11.2) in the venv.
- `docs/make_architecture.py` draws `docs/architecture.png` (for download) and `docs/architecture.svg`
  (vector, sharp in a browser). Parameter counts are computed from `model.py`, so the diagram follows the model.

## 2026-10-04 — Git, tests and CI/CD
- Git identity set locally for the repo (same as the initial commit): `AlexGuBGr <AlexGuBGr@users.noreply.github.com>`.
- `requirements.txt`, `requirements-dev.txt` (pytest, ruff), `pyproject.toml` (ruff and pytest configuration).
- `tests/test_model.py`: parameter budget (~1M), tied weights, shapes/loss at init, causality,
  RoPE (preserves norm, depends only on relative position), generation.
- `tests/test_pipeline.py`: synthetic dataset → `prepare.main()` with small chunks (tests that
  `<|endoftext|>` is not split) → `train.py` 60 iters (loss must fall) → resume to 70 → `sample.py`.
- `scripts/check_repo.py`: rejects tracked files > 50 MB and `*.bin/*.pt/data/out/.venv`. Used in CI and
  as a local pre-commit hook.
- Lint fixes (ruff): unused f-strings, `open(..., "r")`, loop variable bound explicitly in `bench.py`.
- GitHub Actions: `ci.yml` (hygiene, lint, tests + report in job summary + artifacts) and
  `release.yml` (on a `v*` tag: run CI, publish a release).
- Locally: 10/10 tests pass (~45 s), ruff clean.
- Pushing requires a GitHub login on the server (not set up) → the user pushes.

## 2026-10-04 — English documentation
- All documentation, code comments, docstrings, script output, workflow step names and diagram labels
  translated from Danish to English. Commit messages are written in English from now on.
