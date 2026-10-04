"""
End-to-end MLOps smoke test on a small synthetic dataset:
prepare -> train -> checkpoint -> resume -> sample.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import prepare

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORIES = [
    "Once upon a time, there was a little cat. The cat liked to play.",
    "Tom had a red ball. He played with the ball every day. He was happy!",
    "Lily saw a big dog. \"Hello, dog,\" she said. The dog was kind.",
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    raw = d / "raw"
    raw.mkdir()
    # repeat the stories so the model has something to learn; '米' is a rare char -> <unk>
    train = "\n<|endoftext|>\n".join(STORIES * 40) + "\n<|endoftext|>\n米\n"
    val = "\n<|endoftext|>\n".join(STORIES * 3) + "\n"
    (raw / "train.txt").write_text(train, encoding="utf-8")
    (raw / "val.txt").write_text(val, encoding="utf-8")

    old = (prepare.FILES, prepare.OUT_DIR, prepare.MIN_COUNT, prepare.CHUNK_BYTES)
    prepare.FILES = {"train": str(raw / "train.txt"), "val": str(raw / "val.txt")}
    prepare.OUT_DIR = str(d)
    prepare.MIN_COUNT = 2
    prepare.CHUNK_BYTES = 500  # many small chunks: tests that <|endoftext|> is not split
    try:
        prepare.main()
    finally:
        prepare.FILES, prepare.OUT_DIR, prepare.MIN_COUNT, prepare.CHUNK_BYTES = old
    return d


def test_prepare_vocab_and_roundtrip(data_dir):
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    itos = meta["itos"]
    assert itos[:2] == ["<unk>", "<eot>"]
    assert meta["vocab_size"] == len(itos) < 256
    assert "米" not in itos

    ids = np.memmap(data_dir / "train.bin", dtype=np.uint8, mode="r")
    assert (ids == 1).sum() == len(STORIES) * 40   # all <|endoftext|> found, even across chunk boundaries
    assert (ids == 0).sum() == 1                   # the rare character
    text = "".join("|" if i == 1 else ("?" if i == 0 else itos[i]) for i in ids[:200])
    assert text.startswith(STORIES[0] + "\n|\n" + STORIES[1][:20])


def run(*args):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                       env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


TINY = ["--n_layer=1", "--n_head=2", "--n_embd=32", "--mlp_hidden=64", "--block_size=32",
        "--batch_size=8", "--learning_rate=1e-2", "--min_lr=1e-3", "--warmup_iters=5",
        "--eval_interval=20", "--eval_iters=2", "--log_interval=5", "--threads=1"]


@pytest.fixture(scope="module")
def trained(data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    run("train.py", f"--data_dir={data_dir}", f"--out_dir={out}", "--init_from=scratch",
        "--max_iters=60", "--lr_decay_iters=100", *TINY)
    return out


def read_log(out):
    with open(out / "train_log.csv") as f:
        return list(csv.DictReader(f))


def test_training_learns(trained):
    rows = read_log(trained)
    train = [float(r["loss"]) for r in rows if r["kind"] == "train"]
    evals = [float(r["val_loss"]) for r in rows if r["kind"] == "eval"]
    assert len(train) >= 10 and len(evals) >= 2
    assert train[-1] < 0.75 * train[0], train
    assert evals[-1] < evals[0], evals
    assert all(float(r["tokens_per_sec"]) > 0 for r in rows if r["kind"] == "train")
    for name in ("ckpt.pt", "best.pt", "runs.jsonl"):
        assert (trained / name).exists()


def test_resume_continues(trained, data_dir):
    run("train.py", f"--data_dir={data_dir}", f"--out_dir={trained}", "--max_iters=70",
        "--lr_decay_iters=100", *TINY)
    ckpt = torch.load(trained / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["iter_num"] == 70
    assert ckpt["tokens_seen"] == 70 * 8 * 32
    runs = [json.loads(line) for line in (trained / "runs.jsonl").read_text().splitlines()]
    assert runs[-1]["resume"] is True and runs[-1]["iter"] == 60
    iters = [int(r["iter"]) for r in read_log(trained)]
    assert iters == sorted(iters)  # the log continues instead of starting over


def test_sample(trained, data_dir):
    out = run("sample.py", f"--ckpt={trained / 'ckpt.pt'}", f"--data_dir={data_dir}",
              "--prompt=Once", "--max_new_tokens=40", "--seed=0", "--num_samples=2")
    body = out.split("\n", 2)[2]
    assert body.strip().startswith("Once")
    assert "-" * 60 in body
