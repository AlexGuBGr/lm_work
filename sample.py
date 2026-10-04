"""
Generér tekst fra seneste checkpoint.

    .venv/bin/python sample.py
    .venv/bin/python sample.py --prompt="Once upon a time" --num_samples=3 --temperature=0.8
    .venv/bin/python sample.py --ckpt=out/best.pt --max_new_tokens=1000 --top_k=20

Generering stopper ved slutningen af en historie (<eot>), medmindre --until_eot=false.
"""
import argparse
import json
import os

import torch

from model import GPT, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="out/ckpt.pt")
p.add_argument("--data_dir", default="data")
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--num_samples", type=int, default=1)
p.add_argument("--max_new_tokens", type=int, default=600)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=None)
p.add_argument("--until_eot", default="true")
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

torch.set_num_threads(os.cpu_count())
if args.seed is not None:
    torch.manual_seed(args.seed)

with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
itos = meta["itos"]
stoi = {ch: i for i, ch in enumerate(itos) if i >= 2}
unk_id, eot_id = meta["unk_id"], meta["eot_id"]

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = GPT(ModelConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()
print(f"# {args.ckpt}: iter {ckpt['iter_num']:,}, {ckpt.get('tokens_seen', 0):,} tokens set, "
      f"bedste val loss {ckpt['best_val_loss']:.4f}\n")


def encode(s):
    return [stoi.get(ch, unk_id) for ch in s]


def decode(ids):
    return "".join("" if i == eot_id else ("�" if i == unk_id else itos[i]) for i in ids)


start = [eot_id] + encode(args.prompt)  # <eot> først = "starten af en ny historie"
stop = eot_id if args.until_eot.lower() in ("1", "true", "yes") else None
for n in range(args.num_samples):
    idx = torch.tensor([start], dtype=torch.long)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                         top_k=args.top_k, stop_id=stop)
    print(decode(out[0, 1:].tolist()).strip())
    if n < args.num_samples - 1:
        print("\n" + "-" * 60 + "\n")
