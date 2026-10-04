# Arbejdslog – lm_work

Kronologisk log over hvad der er gjort og hvordan.

## 2026-10-03 — Trin 1: Systemtjek og miljø

**Hardware (lscpu, free, df):**
- CPU: Intel Xeon E5-2690 v2 @ 3.00GHz (Ivy Bridge), 2 vCPU (KVM), 1 tråd/kerne
- Instruktionssæt: SSE4.2, AVX, F16C, AES — **ingen AVX2, FMA, AVX-512 eller AMX/avx512_bf16**
  → bfloat16 har ingen hardwareunderstøttelse → **float32** bruges
- RAM: 7.8 GiB (+ 8 GiB swap). Disk: 41 GB ledig på `/`
- OS: Ubuntu 24.04, system-Python 3.12.3. Ingen gcc/g++, ingen sudo uden password.

**Python-miljø:**
- `python3 -m venv` fejlede (pakken `python3.12-venv`/ensurepip mangler, og sudo kræver password).
- Installerede derfor `uv` brugerlokalt: `curl -LsSf https://astral.sh/uv/install.sh | sh` → `~/.local/bin/uv`
- Python 3.12 valgt: fuldt understøttet af PyTorch inkl. torch.compile, og allerede på systemet.
- `uv venv --python 3.12 .venv`

## Trin 2: PyTorch (CPU)
- `uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu`
  → torch 2.14.1+cpu (MKL, oneDNN, OpenMP)
- `uv pip install numpy tqdm` → numpy 2.5.3, tqdm 4.70.1
- `torch.backends.cpu.get_cpu_capability()` = `DEFAULT` (ingen AVX2-kerner kan bruges på denne CPU)

## Trin 3: Data (TinyStoriesV2-GPT4)
- Hentet direkte fra HF (`roneneldan/TinyStories`, `resolve/main/…`) med `curl` til `data/raw/`:
  - `TinyStoriesV2-GPT4-train.txt` 2,227,753,162 B (sha256 6418d412…0443)
  - `TinyStoriesV2-GPT4-valid.txt`    22,502,601 B (sha256 6874bae9…585d)
- `prepare.py` (kør: `.venv/bin/python prepare.py`, ~2,5 min, maks. ~650 MB RAM):
  - Læser filen i bidder af hele linjer (~32 MB via `readlines(hint)`), så `<|endoftext|>` aldrig deles.
  - Gennemløb 1: tæller tegn med `np.bincount` over unicode-kodepunkter (utf-32).
  - Ordforråd: `<unk>`=0, `<eot>`=1 (erstatter `<|endoftext|>`), + alle tegn med ≥200 forekomster
    i train. Første forsøg med grænse 1000 tabte cifrene 4,6-9 → sænket til 200.
  - Resultat: **87 tokens**; 228 forskellige tegn i alt, 1,271 forekomster (0,00006 %) → `<unk>` (mest CJK/støj).
  - Gennemløb 2: opslagstabel kodepunkt→id (uint8) → `data/train.bin` (2,194,232,880 tokens),
    `data/val.bin` (22,161,827 tokens), `data/meta.json` (itos).
  - 2,717,700 historier i train.

## Trin 4: Model og træningsscript
- `model.py`: decoder-only transformer i Llama-stil – pre-norm `nn.RMSNorm`, RoPE (cos/sin-cache),
  SwiGLU (hidden 352 ≈ 8/3·128 rundet til 32), kausal attention via `F.scaled_dot_product_attention`,
  delte vægte embedding/lm_head, ingen bias. 5 lag, 4 hoveder, bredde 128, kontekst 256
  → **1,016,064 parametre**.
- `train.py` (nanoGPT-stil): AdamW (wd kun på 2D-vægte), warmup + cosinus-lr, grad clip 1.0,
  batches = tilfældige udsnit fra `np.memmap` (ny memmap pr. batch, ingen DataLoader-workers),
  `torch.set_num_threads(os.cpu_count())`, dtype auto (bf16 kun hvis `avx512_bf16`/`amx_bf16` i
  /proc/cpuinfo → her float32), valgfri `--compile=true`.
  Checkpoints: `out/ckpt.pt` (seneste, atomisk skrevet via .tmp + rename) og `out/best.pt` (bedste val).
  Gemmer også ved Ctrl+C/SIGTERM. Logger til `out/train_log.csv`, konfiguration pr. (gen)start i `out/runs.jsonl`.
  Genoptager automatisk fra `out/ckpt.pt` (`init_from=auto`). Alle indstillinger kan sættes med `--navn=værdi`.
- `sample.py`: genererer tekst fra checkpoint.

## Trin 5: Hastighedsmålinger
- Røgtest (100 iter, batch 32): ~1.700 tok/s – langsommere end ventet → profilering:
  - `aten::mm` ≈ 47 % af tiden, SDPA flash-attention ≈ 35 %.
  - Rå matmul (1024³): 14 GFLOP/s på 1 tråd, 23 GFLOP/s på 2 tråde → maskinen er simpelthen langsom
    (virtualiseret Ivy Bridge, AVX uden FMA; PyTorch-kapabilitet `DEFAULT`).
  - SDPA-backends (B=16, pr. lag fwd+bwd): default 0,22 s, math 0,25 s, flash 0,23 s → standard beholdes.
- torch.compile kræver C++-compiler + Python-headers, som systemet mangler (ingen sudo):
  - micromamba hentet til `~/.local/bin/micromamba` (udpakket med Python, da `bzip2` mangler),
    gcc 13 fra conda-forge: `micromamba create -p ~/.local/share/toolchain -c conda-forge gxx_linux-64=13`.
  - System-Python har ingen `Python.h` → venv genskabt med uv-styret CPython **3.12.15**
    (`uv python install 3.12.15; uv venv --python 3.12.15 --managed-python .venv`), torch geninstalleret.
  - `train.py`/`bench.py` sætter selv `CXX` til toolchain-g++ når compile er slået til.
- `bench.py` (fwd+bwd+AdamW, T=256):

  | batch | eager tok/s | compile tok/s |
  |---|---|---|
  | 8  | 1.164 | 1.371 |
  | 16 | 1.373 | 1.550 |
  | 32 | 1.448 | 1.532 |
  | 64 | 1.554 | 1.588 |

  Kompilering tager ~70-85 s første gang. Målingerne svinger ±10 % mellem kørsler (delt VM).

### Testkørsel med train.py (batch 32, 80 iter, eager vs compile)
- Eager: 1.470–1.583 tok/s (≈5,2–5,6 s/iter). Compile: 1.511–1.558 tok/s efter ~80 s kompilering.
  Identisk loss → **ingen målbar gevinst af torch.compile i praksis → slået fra** (`compile=False`).
- Loss faldt 4,4 → 2,54 (val 2,545) på 80 iter; stabil med lr-warmup.
- Resume testet: genoptog fra iter 80 → 90, CSV-log fortsat korrekt. `sample.py` testet.
- Eval (2×50 batches) tager ~3,3 min → ~3-4 % overhead ved eval hver 1000. iter.
- **Batch-valg:** tok/s flader ud fra batch 16 (~1.400-1.550); batch 32 valgt (8.192 tokens/iter)
  for mindre støjende gradienter uden tab af throughput.

**Estimat (~1.500 tok/s, ~125M tokens/døgn):**
10k iter ≈ 15 t · **30k iter ≈ 2 døgn (246M tokens, valgt standard)** · 100k ≈ 6,5 døgn · 1 epoke (2,2B) ≈ 17 døgn.

## Trin 6: Fuld træning startet (2026-10-03 20:41)
- tmux-session `lm_train` (navnet `lm` var allerede i brug af en anden tmux-session):
  `tmux new-session -d -s lm_train "cd ~/analyses/lm_work && .venv/bin/python -u train.py >> out/train.out 2>&1"`
- Indstillinger: batch 32, max_iters 30.000, lr 3e-3 → 3e-4 (cosinus, warmup 500), wd 0,1, float32, 2 tråde.
- Ressourcer: ~180 % CPU, ~1 GB RAM. Forventet færdig omkring 2026-10-05 aften.

## Trin 7: sample.py og NOTES.md skrevet
- Se `NOTES.md` for følg/stop/genoptag/generér.

## 2026-10-04 — Arkitekturdiagram
- `uv pip install matplotlib` (3.11.2) i venv.
- `docs/make_architecture.py` tegner `docs/architecture.png` (til download) og `docs/architecture.svg`
  (vektor, skarp i browser). Parametertal beregnes fra `model.py`, så diagrammet følger modellen.

## 2026-10-04 — Git, tests og CI/CD
- Git-identitet sat lokalt for repoet (samme som initial commit): `AlexGuBGr <AlexGuBGr@users.noreply.github.com>`.
- `requirements.txt`, `requirements-dev.txt` (pytest, ruff), `pyproject.toml` (ruff- og pytest-konfiguration).
- `tests/test_model.py`: parameterbudget (~1M), delte vægte, former/loss ved init, kausalitet,
  RoPE (bevarer norm, afhænger kun af relativ position), generering.
- `tests/test_pipeline.py`: syntetisk datasæt → `prepare.main()` med små bidder (tester at
  `<|endoftext|>` ikke deles) → `train.py` 60 iter (loss skal falde) → resume til 70 → `sample.py`.
- `scripts/check_repo.py`: afviser sporede filer > 50 MB og `*.bin/*.pt/data/out/.venv`. Bruges i CI og
  som lokal pre-commit-hook.
- Lint-rettelser (ruff): ubrugte f-strings, `open(..., "r")`, loop-variabel bundet eksplicit i `bench.py`.
- GitHub Actions: `ci.yml` (hygiejne, lint, tests + rapport i job-summary + artefakter) og
  `release.yml` (ved `v*`-tag: kør CI, udgiv release).
- Lokalt: 10/10 tests består (~45 s), ruff ren.
- Push kræver GitHub-login på serveren (ikke sat op) → commits ligger lokalt indtil `git push`.
