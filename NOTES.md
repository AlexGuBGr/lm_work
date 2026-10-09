# NOTES – character-level GPT on TinyStories

All commands are run from `~/analyses/lm_work`.

## Monitor training
```bash
tail -f out/train.out            # live output (Ctrl+C only stops tail, not the training)
tmux attach -t lm_train          # view the session itself; detach again with Ctrl+B, then D
column -s, -t out/train_log.csv | tail -20   # loss, lr, tok/s per log line
grep eval out/train_log.csv      # validation loss (every 1000th iteration)
```
Training runs in the tmux session **`lm_train`** and keeps going after you log out.
Default: 30,000 iterations of 8,192 tokens (~246M tokens) ≈ 2 days at ~1,500 tok/s.

## Stop
```bash
tmux send-keys -t lm_train C-c   # or: pkill -INT -f "train.py"
```
The script saves `out/ckpt.pt` after the current iteration and exits (takes up to ~10 s).
Avoid `kill -9`: you would lose everything since the last checkpoint (saved every 1000 iterations).

## Resume
```bash
tmux new-session -d -s lm_train "cd ~/analyses/lm_work && .venv/bin/python -u train.py >> out/train.out 2>&1"
```
`train.py` resumes automatically from `out/ckpt.pt` (model, optimizer, iteration and RNG).
Train longer/shorter by adding e.g. `--max_iters=60000`. Note: the cosine LR schedule follows
`max_iters`, so changing it mid-run changes the learning rate from that point on.
Start over: `--init_from=scratch` (overwrites the checkpoints in `out/`) or use a different `--out_dir=...`.

## Generate text
```bash
.venv/bin/python sample.py                                   # from out/ckpt.pt (latest)
.venv/bin/python sample.py --ckpt=out/best.pt --prompt="Lily had a red ball" --num_samples=3
.venv/bin/python sample.py --temperature=0.6 --top_k=10 --max_new_tokens=1000
```
Can be run while training is running (briefly shares the 2 cores, though).

## Evaluate
```bash
.venv/bin/python evaluate.py                 # all stages (~2½ h on this server)
.venv/bin/python evaluate.py --stage=report  # rebuild docs/eval/EVAL.md from saved results
```
Stages: `val` (full validation loss + loss by position), `ngram` (n-gram baselines), `gen` (generated vs
real stories), `report`. Raw results in `out/eval/`, report and plot in `docs/eval/`.

## Files
| file | contents |
|---|---|
| `prepare.py` | TinyStories → `data/train.bin`, `data/val.bin` (uint8), `data/meta.json` (vocabulary, 87 tokens) |
| `model.py` | Llama-style transformer (RMSNorm, RoPE, SwiGLU, SDPA, tied weights), 1,016,064 parameters |
| `train.py` | training; settings at the top of the file or `--name=value` |
| `sample.py` | text generation |
| `analysis/figures.py` | README figures and `docs/figures/findings.json` (needs `evaluate.py` results) |
| `analysis/export_weights.py` | exports `out/best.pt` to `docs/demo/model.{json,bin}` for the browser demo |
| `docs/demo/` | browser demo (plain JavaScript, KV cache) + exported weights; served by GitHub Pages |
| `evaluate.py` | evaluation: validation loss, n-gram baselines, loss by position, generated text |
| `bench.py` | speed measurement: `.venv/bin/python bench.py [--compile] 16 32` |
| `out/` | `ckpt.pt`, `best.pt`, `train_log.csv`, `runs.jsonl`, `train.out` |
| `LOG.md` | detailed log of setup and decisions |
| `tests/`, `scripts/check_repo.py`, `.github/workflows/` | tests and CI/CD (see below) |
| `docs/architecture.png/.svg` | architecture diagram (regenerate with `.venv/bin/python docs/make_architecture.py`) |

## Environment
- venv: `.venv` (uv-managed CPython 3.12.15, torch 2.14.1+cpu). Optionally activate with `source .venv/bin/activate`.
- `uv` and `micromamba` live in `~/.local/bin`; gcc for torch.compile in `~/.local/share/toolchain`
  (only used with `--compile=true`, which gave no gain on this CPU).

## Tests and CI/CD
```bash
.venv/bin/python -m pytest -q        # unit tests + smoke test of prepare → train → resume → sample (~45 s)
.venv/bin/ruff check .               # lint
.venv/bin/python scripts/check_repo.py   # no files > 50 MB / data / checkpoints in git
```
- **CI** (`.github/workflows/ci.yml`) runs on every push and pull request: repo hygiene → lint →
  tests + MLOps smoke training. The model/training report is shown in the GitHub run's *Summary*, and
  the training log, junit and architecture diagram are stored as artifacts.
- **CD** (`.github/workflows/release.yml`): `git tag v0.1.0 && git push origin v0.1.0` runs CI and
  publishes a GitHub release with the diagram, NOTES.md and source code.
- A local pre-commit hook (`.git/hooks/pre-commit`) rejects commits with files > 50 MB or data/checkpoints.
