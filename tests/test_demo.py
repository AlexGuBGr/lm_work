"""The browser demo's JavaScript model must give the same predictions as the PyTorch model."""
import json
import os
import shutil
import subprocess

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node") or os.path.expanduser("~/.local/share/nodejs/bin/node")
HAVE = os.path.exists(os.path.join(ROOT, "docs/demo/model.bin")) and os.path.exists(NODE or "")


@pytest.mark.skipif(not HAVE, reason="needs node and exported weights (analysis/export_weights.py)")
@pytest.mark.parametrize("prompt", ["Once upon a time, there was a little cat named Tom.",
                                    "x" * 300])  # 300 > 256: exercises the sliding context window
def test_js_matches_pytorch(prompt):
    meta = json.load(open(os.path.join(ROOT, "docs/demo/model.json"), encoding="utf-8"))
    out = json.loads(subprocess.run([NODE, os.path.join(ROOT, "tests/check_demo.mjs"), prompt],
                                    capture_output=True, text=True, check=True).stdout)
    js = torch.tensor(out["logits"])

    from model import load_exported

    model, _ = load_exported(os.path.join(ROOT, "docs/demo/model.json"))
    stoi = {c: i for i, c in enumerate(meta["itos"]) if i >= 2}
    ids = [meta["eot_id"]] + [stoi.get(c, meta["unk_id"]) for c in prompt]
    T = meta["config"]["block_size"]
    if len(ids) > T:  # same rule as the JS cache: when full, keep the last T/2 and continue
        keep = ids[:T]
        for t in ids[T:]:
            if len(keep) >= T:
                keep = keep[-T // 2:]
            keep.append(t)
        ids = keep
    with torch.no_grad():
        ref = model(torch.tensor([ids]))[0][0, -1]
    assert torch.allclose(js, ref, atol=2e-3), float((js - ref).abs().max())
