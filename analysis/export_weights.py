"""
Export the trained model for the browser demo and for a GitHub release.

    .venv/bin/python analysis/export_weights.py

Writes
  docs/demo/model.bin    all weights as little-endian float32, concatenated (≈4 MB)
  docs/demo/model.json   config, vocabulary and the offset/shape of every tensor in model.bin
  out/tinystories-char-1m.pt   weights + config + vocabulary only (no optimizer state), for sample.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, ModelConfig  # noqa: E402

ckpt = torch.load("out/best.pt", map_location="cpu", weights_only=False)
cfg = ckpt["model_config"]
model = GPT(ModelConfig(**cfg))
model.load_state_dict(ckpt["model"])
meta = json.load(open("data/meta.json", encoding="utf-8"))

os.makedirs("docs/demo", exist_ok=True)
tensors, offset, blobs = {}, 0, []
for name, t in model.state_dict().items():
    if name == "lm_head.weight":  # tied to tok_emb.weight
        continue
    a = t.detach().numpy().astype("<f4")
    tensors[name] = {"offset": offset, "shape": list(a.shape)}
    blobs.append(a.tobytes())
    offset += a.size
with open("docs/demo/model.bin", "wb") as f:
    for b in blobs:
        f.write(b)
with open("docs/demo/model.json", "w", encoding="utf-8") as f:
    json.dump({"config": cfg, "itos": meta["itos"], "eot_id": meta["eot_id"], "unk_id": meta["unk_id"],
               "iter": ckpt["iter_num"], "val_loss": ckpt["best_val_loss"], "tensors": tensors},
              f, ensure_ascii=False)

torch.save({"model": model.state_dict(), "model_config": cfg, "itos": meta["itos"],
            "iter_num": ckpt["iter_num"], "best_val_loss": ckpt["best_val_loss"],
            "tokens_seen": ckpt.get("tokens_seen")}, "out/tinystories-char-1m.pt")
print(f"docs/demo/model.bin: {offset * 4 / 1e6:.2f} MB, {len(tensors)} tensors; "
      f"out/tinystories-char-1m.pt: {os.path.getsize('out/tinystories-char-1m.pt') / 1e6:.2f} MB")
