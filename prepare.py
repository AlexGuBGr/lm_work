"""
Prepare TinyStoriesV2-GPT4 for character-level training.

Reads the raw text files in chunks (never the whole file in RAM), builds a
character-level vocabulary from the training data, and writes data/train.bin and
data/val.bin as uint8 token ids that can be read with np.memmap.

Special tokens:
  0 = <unk>  (all rare characters are collected here)
  1 = <eot>  (replaces "<|endoftext|>" between stories)

Run:  .venv/bin/python prepare.py
"""
import json
import os
import time

import numpy as np

RAW_DIR = "data/raw"
OUT_DIR = "data"
FILES = {
    "train": os.path.join(RAW_DIR, "TinyStoriesV2-GPT4-train.txt"),
    "val": os.path.join(RAW_DIR, "TinyStoriesV2-GPT4-valid.txt"),
}
CHUNK_BYTES = 32 * 1024 * 1024  # ~32 MB of text at a time
MAX_VOCAB = 256                 # must fit in uint8
MIN_COUNT = 200                 # characters rarer than this -> <unk>

EOT_STR = "<|endoftext|>"
EOT_CHAR = "\ue000"             # private-use Unicode char, internal only
N_CODEPOINTS = 0x110000


def iter_chunks(path):
    """Read the file in chunks of whole lines, so <|endoftext|> is never split."""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            lines = f.readlines(CHUNK_BYTES)
            if not lines:
                break
            yield "".join(lines).replace(EOT_STR, EOT_CHAR)


def codepoints(text):
    return np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)


def count_chars(path):
    counts = np.zeros(N_CODEPOINTS, dtype=np.int64)
    for chunk in iter_chunks(path):
        counts += np.bincount(codepoints(chunk), minlength=N_CODEPOINTS)
    return counts


def build_vocab(counts):
    eot_cp = ord(EOT_CHAR)
    n_eot = int(counts[eot_cp])
    counts = counts.copy()
    counts[eot_cp] = 0
    order = np.argsort(-counts, kind="stable")
    keep = [int(cp) for cp in order[: MAX_VOCAB - 2] if counts[cp] >= MIN_COUNT]
    keep.sort()  # stable, readable order
    itos = ["<unk>", "<eot>"] + [chr(cp) for cp in keep]
    n_unk = int(counts.sum() - counts[keep].sum())
    return itos, n_eot, n_unk


def encode_file(path, out_path, lut):
    n = 0
    with open(out_path, "wb") as out:
        for chunk in iter_chunks(path):
            ids = lut[codepoints(chunk)]
            ids.tofile(out)
            n += len(ids)
    return n


def main():
    t0 = time.time()
    print("Pass 1: counting characters in training data ...")
    counts = count_chars(FILES["train"])
    itos, n_eot, n_unk = build_vocab(counts)
    total = int(counts.sum())
    print(f"  {total:,} chars, {int((counts > 0).sum())} distinct, {n_eot:,} stories")
    print(f"  vocabulary: {len(itos)} tokens; {n_unk:,} chars ({n_unk / total:.5%}) -> <unk>")

    lut = np.zeros(N_CODEPOINTS, dtype=np.uint8)  # default 0 = <unk>
    lut[ord(EOT_CHAR)] = 1
    for i, ch in enumerate(itos[2:], start=2):
        lut[ord(ch)] = i

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(itos), "itos": itos, "unk_id": 0, "eot_id": 1}, f,
                  ensure_ascii=False, indent=0)

    print("Pass 2: encoding to uint8 ...")
    for split, path in FILES.items():
        out_path = os.path.join(OUT_DIR, f"{split}.bin")
        n = encode_file(path, out_path, lut)
        print(f"  {split}: {n:,} tokens -> {out_path}")
    print(f"Done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
