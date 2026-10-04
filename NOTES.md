# NOTES – tegn-niveau GPT på TinyStories

Alle kommandoer køres fra `~/analyses/lm_work`.

## Følg træningen
```bash
tail -f out/train.out            # live-output (Ctrl+C stopper kun tail, ikke træningen)
tmux attach -t lm_train          # se selve sessionen; forlad igen med Ctrl+B, derefter D
column -s, -t out/train_log.csv | tail -20   # loss, lr, tok/s pr. log-linje
grep eval out/train_log.csv      # validerings-loss (hver 1000. iteration)
```
Træningen kører i tmux-sessionen **`lm_train`** og fortsætter, selvom du logger ud.
Standard: 30.000 iterationer á 8.192 tokens (~246M tokens) ≈ 2 døgn ved ~1.500 tok/s.

## Stop
```bash
tmux send-keys -t lm_train C-c   # eller: pkill -INT -f "train.py"
```
Scriptet gemmer `out/ckpt.pt` efter den igangværende iteration og afslutter (tager op til ~10 s).
Undgå `kill -9`: så mister du det, der er sket siden sidste checkpoint (gemmes hver 1000. iteration).

## Genoptag
```bash
tmux new-session -d -s lm_train "cd ~/analyses/lm_work && .venv/bin/python -u train.py >> out/train.out 2>&1"
```
`train.py` genoptager automatisk fra `out/ckpt.pt` (model, optimizer, iteration og RNG).
Træn længere/kortere ved at tilføje fx `--max_iters=60000`. Bemærk: cosinus-lr-skemaet følger
`max_iters`, så en ændring midt i forløbet ændrer læringsraten fra det punkt.
Start helt forfra: `--init_from=scratch` (overskriver checkpoints i `out/`) eller brug et andet `--out_dir=...`.

## Generér tekst
```bash
.venv/bin/python sample.py                                   # fra out/ckpt.pt (seneste)
.venv/bin/python sample.py --ckpt=out/best.pt --prompt="Lily had a red ball" --num_samples=3
.venv/bin/python sample.py --temperature=0.6 --top_k=10 --max_new_tokens=1000
```
Kan køres mens træningen kører (deler dog de 2 kerner kortvarigt).

## Filer
| fil | indhold |
|---|---|
| `prepare.py` | TinyStories → `data/train.bin`, `data/val.bin` (uint8), `data/meta.json` (ordforråd, 87 tokens) |
| `model.py` | Llama-stil transformer (RMSNorm, RoPE, SwiGLU, SDPA, delte vægte), 1,016,064 parametre |
| `train.py` | træning; indstillinger øverst i filen eller `--navn=værdi` |
| `sample.py` | tekstgenerering |
| `bench.py` | hastighedsmåling: `.venv/bin/python bench.py [--compile] 16 32` |
| `out/` | `ckpt.pt`, `best.pt`, `train_log.csv`, `runs.jsonl`, `train.out` |
| `LOG.md` | detaljeret log over opsætning og beslutninger |
| `tests/`, `scripts/check_repo.py`, `.github/workflows/` | tests og CI/CD (se nedenfor) |
| `docs/architecture.png/.svg` | arkitekturdiagram (gendan med `.venv/bin/python docs/make_architecture.py`) |

## Miljø
- venv: `.venv` (uv-styret CPython 3.12.15, torch 2.14.1+cpu). Aktivér evt. med `source .venv/bin/activate`.
- `uv` og `micromamba` ligger i `~/.local/bin`; gcc til torch.compile i `~/.local/share/toolchain`
  (bruges kun med `--compile=true`, som ikke gav nogen gevinst på denne CPU).

## Tests og CI/CD
```bash
.venv/bin/python -m pytest -q        # unit-tests + røgtest af prepare → train → resume → sample (~45 s)
.venv/bin/ruff check .               # lint
.venv/bin/python scripts/check_repo.py   # ingen filer > 50 MB / data / checkpoints i git
```
- **CI** (`.github/workflows/ci.yml`) kører ved hvert push og pull request: repo-hygiejne → lint →
  tests + MLOps-røgtræning. Model-/træningsrapporten vises i GitHub-kørslens *Summary*, og
  træningslog, junit og arkitekturdiagram gemmes som artefakter.
- **CD** (`.github/workflows/release.yml`): `git tag v0.1.0 && git push origin v0.1.0` kører CI og
  udgiver en GitHub-release med diagram, NOTES.md og kildekode.
- Lokal pre-commit-hook (`.git/hooks/pre-commit`) afviser commits med filer > 50 MB eller data/checkpoints.
