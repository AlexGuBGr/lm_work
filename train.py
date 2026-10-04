"""
Træning af en lille tegn-niveau GPT på TinyStories (kun CPU), i stil med nanoGPT.

Alle indstillinger nedenfor kan overskrives fra kommandolinjen, fx:
    .venv/bin/python train.py --max_iters=50000 --batch_size=32
    .venv/bin/python train.py --out_dir=out-test --max_iters=200 --init_from=scratch

init_from:
    "auto"    genoptag fra <out_dir>/ckpt.pt hvis den findes, ellers start forfra (standard)
    "resume"  genoptag (fejl hvis der ikke er noget checkpoint)
    "scratch" start altid forfra (overskriver checkpoints i out_dir)

Stop pænt med Ctrl+C eller `kill <pid>` (SIGTERM): der gemmes et checkpoint før afslutning.
"""
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, ModelConfig

# ---------------------------------------------------------------- konfiguration
out_dir = "out"
data_dir = "data"
init_from = "auto"
# model
n_layer = 5
n_head = 4
n_embd = 128
mlp_hidden = 352
block_size = 256
# træning
batch_size = 32
grad_accum = 1
max_iters = 30_000       # <- loft over antal iterationer (~2 døgn på denne server)
learning_rate = 3e-3
min_lr = 3e-4
warmup_iters = 500
lr_decay_iters = None    # None = max_iters (cosinus-henfald til min_lr)
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0
# evaluering, logging, checkpoints
eval_interval = 1000
eval_iters = 50
log_interval = 50
ckpt_interval = 1000     # gem ckpt.pt så ofte (desuden ved hver eval og ved stop)
# system
threads = os.cpu_count()
dtype = "auto"           # "auto" | "float32" | "bfloat16"
compile = False
seed = 1337

# simpel CLI-override: --navn=værdi
for arg in sys.argv[1:]:
    assert arg.startswith("--") and "=" in arg, f"ugyldigt argument: {arg}"
    key, val = arg[2:].split("=", 1)
    assert key in globals(), f"ukendt indstilling: {key}"
    old = globals()[key]
    if isinstance(old, bool):
        val = val.lower() in ("1", "true", "yes")
    elif isinstance(old, int):
        val = int(float(val))
    elif isinstance(old, float):
        val = float(val)
    elif old is None and val.lower() != "none":
        val = int(float(val))
    elif old is None:
        val = None
    globals()[key] = val
config = {k: globals()[k] for k in [
    "out_dir", "data_dir", "init_from", "n_layer", "n_head", "n_embd", "mlp_hidden", "block_size",
    "batch_size", "grad_accum", "max_iters", "learning_rate", "min_lr", "warmup_iters",
    "lr_decay_iters", "weight_decay", "beta1", "beta2", "grad_clip", "eval_interval", "eval_iters",
    "log_interval", "ckpt_interval", "threads", "dtype", "compile", "seed"]}
if lr_decay_iters is None:
    lr_decay_iters = max_iters


# ---------------------------------------------------------------- system
def cpu_supports_bf16():
    try:
        flags = open("/proc/cpuinfo").read()
    except OSError:
        return False
    return any(f in flags for f in ("avx512_bf16", "amx_bf16"))


if compile:
    # torch.compile (inductor) skal bruge en C++-compiler; systemet har ingen, så vi bruger
    # den brugerlokale gcc fra conda-forge (se LOG.md)
    cxx = os.path.expanduser("~/.local/share/toolchain/bin/x86_64-conda-linux-gnu-g++")
    if os.path.exists(cxx):
        os.environ.setdefault("CXX", cxx)
torch.set_num_threads(threads)
if dtype == "auto":
    dtype = "bfloat16" if cpu_supports_bf16() else "float32"
use_bf16 = dtype == "bfloat16"
torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------- data
with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
vocab_size = meta["vocab_size"]


def get_batch(split):
    # ny memmap hver gang (undgår at sider ophobes i RAM), tilfældige udsnit
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint8, mode="r")
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------- model / optimizer
ckpt_path = os.path.join(out_dir, "ckpt.pt")
best_path = os.path.join(out_dir, "best.pt")
iter_num, best_val_loss, tokens_seen = 0, float("inf"), 0

resume = init_from == "resume" or (init_from == "auto" and os.path.exists(ckpt_path))
if resume:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg)
    model.load_state_dict(ckpt["model"])
    iter_num, best_val_loss = ckpt["iter_num"], ckpt["best_val_loss"]
    tokens_seen = ckpt.get("tokens_seen", 0)
else:
    mcfg = ModelConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                       n_head=n_head, n_embd=n_embd, mlp_hidden=mlp_hidden)
    model = GPT(mcfg)

optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2))
if resume:
    optimizer.load_state_dict(ckpt["optimizer"])
    if "rng" in ckpt:
        torch.set_rng_state(ckpt["rng"]["torch"])
        np.random.set_state(ckpt["rng"]["numpy"])
    del ckpt

raw_model = model
if compile:
    print("kompilerer model med torch.compile ...")
    model = torch.compile(model)

print(f"{'genoptager fra iter ' + str(iter_num) if resume else 'starter forfra'} | "
      f"{raw_model.num_params():,} parametre | dtype={dtype} | threads={threads} | compile={compile}")
print(f"model: {asdict(mcfg)}")


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it >= lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


def autocast():
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with autocast():
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path):
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(mcfg),
        "train_config": config,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "tokens_seen": tokens_seen,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # atomisk: et afbrudt save ødelægger aldrig det gamle checkpoint


# logfiler: CSV til tal, og konfiguration ved hver (gen)start
log_path = os.path.join(out_dir, "train_log.csv")
new_log = not os.path.exists(log_path) or not resume
with open(log_path, "w" if new_log else "a") as f:
    if new_log:
        f.write("time,iter,kind,loss,val_loss,lr,tokens_per_sec,tokens_seen\n")
with open(os.path.join(out_dir, "runs.jsonl"), "a") as f:
    f.write(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "resume": resume,
                        "iter": iter_num, "config": config}) + "\n")


def log_row(kind, loss="", val_loss="", lr="", tps=""):
    with open(log_path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{iter_num},{kind},{loss},{val_loss},"
                f"{lr},{tps},{tokens_seen}\n")


# pænt stop ved Ctrl+C / SIGTERM
stop_requested = False


def _request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print(f"\nsignal {signum} modtaget: gemmer checkpoint og stopper efter denne iteration ...")


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)

# ---------------------------------------------------------------- træningsløkke
tokens_per_iter = batch_size * grad_accum * block_size
print(f"tokens pr. iteration: {tokens_per_iter:,} | max_iters: {max_iters:,}")
X, Y = get_batch("train")
t_log = time.time()
iters_since_log = 0
model.train()

while iter_num < max_iters and not stop_requested:
    lr = get_lr(iter_num)
    for g in optimizer.param_groups:
        g["lr"] = lr

    for _ in range(grad_accum):
        with autocast():
            _, loss = model(X, Y)
        X, Y = get_batch("train")
        (loss / grad_accum).backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    iter_num += 1
    tokens_seen += tokens_per_iter
    iters_since_log += 1

    if iter_num % log_interval == 0:
        dt = time.time() - t_log
        tps = iters_since_log * tokens_per_iter / dt
        lossf = loss.item()
        print(f"iter {iter_num:>7,} | loss {lossf:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s | "
              f"{1000 * dt / iters_since_log:.0f} ms/iter")
        log_row("train", loss=f"{lossf:.4f}", lr=f"{lr:.3e}", tps=f"{tps:.0f}")
        t_log, iters_since_log = time.time(), 0

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        losses = estimate_loss()
        print(f"--- eval iter {iter_num:,}: train {losses['train']:.4f} | val {losses['val']:.4f}")
        log_row("eval", loss=f"{losses['train']:.4f}", val_loss=f"{losses['val']:.4f}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(best_path)
        save_checkpoint(ckpt_path)
        t_log, iters_since_log = time.time(), 0  # eval-tid tæller ikke med i tok/s
    elif iter_num % ckpt_interval == 0:
        save_checkpoint(ckpt_path)

save_checkpoint(ckpt_path)
print(f"stoppet ved iter {iter_num:,} ({tokens_seen:,} tokens). checkpoint: {ckpt_path}")
