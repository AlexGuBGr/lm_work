"""
Evaluate the trained character-level model.

    .venv/bin/python evaluate.py                    # all stages
    .venv/bin/python evaluate.py --stage=val        # one stage: val | ngram | gen | report

Stages
  val     loss on the full validation set (non-overlapping 256-char windows) + loss by position
  ngram   character n-gram baselines (Witten-Bell interpolation, n = 1..8) trained on train.bin
  gen     50 generated stories from 10 fixed prompts: real-word rate, repetition, story endings,
          compared with real validation stories
  report  write docs/eval/EVAL.md and docs/eval/loss_by_position.png from the saved metrics

Results are saved in out/eval/ (metrics JSON, per-window losses, samples) and docs/eval/ (report).
"""
import argparse
import json
import math
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, ModelConfig

p = argparse.ArgumentParser()
p.add_argument("--stage", default="all", choices=["all", "val", "ngram", "gen", "report"])
p.add_argument("--ckpt", default="out/best.pt")
p.add_argument("--data_dir", default="data")
p.add_argument("--out_dir", default="out/eval")
p.add_argument("--report_dir", default="docs/eval")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--ngram_train_chars", type=int, default=100_000_000)
p.add_argument("--ngram_val_chars", type=int, default=5_000_000)
p.add_argument("--max_order", type=int, default=8)
p.add_argument("--samples_per_prompt", type=int, default=5)
p.add_argument("--seed", type=int, default=1234)
args = p.parse_args()

torch.set_num_threads(os.cpu_count())
os.makedirs(args.out_dir, exist_ok=True)
LN2 = math.log(2)

with open(os.path.join(args.data_dir, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
ITOS, V, EOT, UNK = meta["itos"], meta["vocab_size"], meta["eot_id"], meta["unk_id"]
STOI = {ch: i for i, ch in enumerate(ITOS) if i >= 2}


def decode(ids):
    return "".join("\n<eot>\n" if i == EOT else ("�" if i == UNK else ITOS[i]) for i in ids)


def load_model():
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def save_json(name, obj):
    with open(os.path.join(args.out_dir, name), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    path = os.path.join(args.out_dir, name)
    return json.load(open(path)) if os.path.exists(path) else None


# ---------------------------------------------------------------------------- 1 + 3: validation loss
@torch.inference_mode()
def stage_val():
    model, ckpt = load_model()
    T = model.cfg.block_size
    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint8, mode="r")
    n_win = (len(val) - 1) // T
    pos_sum = np.zeros(T, dtype=np.float64)
    win_loss = np.zeros(n_win, dtype=np.float32)
    t0 = time.time()
    for b in range(0, n_win, args.batch_size):
        idx = np.arange(b, min(b + args.batch_size, n_win))
        x = np.stack([val[i * T:i * T + T] for i in idx]).astype(np.int64)
        y = np.stack([val[i * T + 1:i * T + T + 1] for i in idx]).astype(np.int64)
        logits, _ = model(torch.from_numpy(x))
        loss = F.cross_entropy(logits.reshape(-1, V), torch.from_numpy(y).reshape(-1),
                               reduction="none").view(len(idx), T).double().numpy()
        pos_sum += loss.sum(0)
        win_loss[idx] = loss.mean(1)
        if (b // args.batch_size) % 50 == 0:
            done = b + len(idx)
            rate = done * T / (time.time() - t0)
            print(f"  val {done:,}/{n_win:,} windows | {rate:,.0f} tok/s | "
                  f"~{(n_win - done) * T / rate / 60:.0f} min left", flush=True)
    np.save(os.path.join(args.out_dir, "val_window_loss.npy"), win_loss)
    by_pos = (pos_sum / n_win).tolist()
    mean = float(win_loss.mean())
    res = {"checkpoint": args.ckpt, "iter": ckpt["iter_num"], "tokens_seen": ckpt.get("tokens_seen"),
           "params": model.num_params(), "val_tokens": n_win * T, "windows": n_win,
           "loss_nats": mean, "bits_per_char": mean / LN2, "perplexity_per_char": math.exp(mean),
           "loss_by_position": by_pos, "seconds": round(time.time() - t0)}
    save_json("val.json", res)
    print(f"val loss {mean:.4f} nats = {mean / LN2:.4f} bits/char over {n_win * T:,} chars")


# ---------------------------------------------------------------------------- 2: n-gram baselines
def stage_ngram():
    """Interpolated Witten-Bell character n-grams:
       P_n(c|h) = (C(h,c) + T(h) * P_{n-1}(c|h')) / (C(h) + T(h)),  P_0 = uniform over V,
    with C = counts and T(h) = number of distinct characters seen after h."""
    train = np.fromfile(os.path.join(args.data_dir, "train.bin"), dtype=np.uint8,
                        count=args.ngram_train_chars).astype(np.int64)
    val = np.fromfile(os.path.join(args.data_dir, "val.bin"), dtype=np.uint8,
                      count=args.ngram_val_chars + 1).astype(np.int64)
    # evaluate the same characters the transformer is scored on: val[1..N] (each predicted from the left)
    tgt = np.arange(1, len(val))
    prob = np.full(len(tgt), 1.0 / V)
    results = []
    t0 = time.time()
    for n in range(1, args.max_order + 1):
        k = n - 1  # context length
        # n-gram codes over train: sum_j c[i+j] * V^(n-1-j)
        m = len(train) - n + 1
        codes = np.zeros(m, dtype=np.int64)
        for j in range(n):
            codes = codes * V + train[j:j + m]
        hc, hc_count = np.unique(codes, return_counts=True)
        del codes
        h_of = hc // V
        h, start = np.unique(h_of, return_index=True)
        h_total = np.add.reduceat(hc_count, start)
        h_types = np.diff(np.append(start, len(h_of)))
        # val n-gram codes for every target position with a full context (pos >= k)
        vcode = np.zeros(len(tgt), dtype=np.int64)
        ok = tgt >= k
        for j in range(n):
            vcode[ok] = vcode[ok] * V + val[tgt[ok] - k + j]
        vh = vcode // V
        # look up C(h,c), C(h), T(h)
        i_hc = np.searchsorted(hc, vcode)
        c_hc = np.where((i_hc < len(hc)) & (hc[np.minimum(i_hc, len(hc) - 1)] == vcode),
                        hc_count[np.minimum(i_hc, len(hc) - 1)], 0)
        i_h = np.searchsorted(h, vh)
        found = (i_h < len(h)) & (h[np.minimum(i_h, len(h) - 1)] == vh)
        c_h = np.where(found, h_total[np.minimum(i_h, len(h) - 1)], 0)
        t_h = np.where(found, h_types[np.minimum(i_h, len(h) - 1)], 0)
        upd = ok & (c_h > 0)  # unseen context: keep the lower-order estimate
        prob[upd] = (c_hc[upd] + t_h[upd] * prob[upd]) / (c_h[upd] + t_h[upd])
        loss = float(-np.log(prob).mean())
        results.append({"order": n, "loss_nats": loss, "bits_per_char": loss / LN2,
                        "distinct_ngrams": int(len(hc))})
        print(f"  {n}-gram: {loss:.4f} nats = {loss / LN2:.4f} bits/char "
              f"({len(hc):,} distinct n-grams, {time.time() - t0:.0f} s)", flush=True)
        del hc, hc_count, h_of, h, start, h_total, h_types
    save_json("ngram.json", {"train_chars": int(len(train)), "val_chars": int(len(tgt)),
                             "orders": results})


# ---------------------------------------------------------------------------- 4: generation quality
PROMPTS = [
    "Once upon a time", "One day, a little girl named Lily", "Tom and his dog",
    "There was a big tree in the park.", "Sam was sad because", "The sun was shining and",
    "Mom said, \"", "A little bird", "Once upon a time, in a small house,", "Ben wanted to",
]
WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def word_vocab(n_chars=50_000_000, min_count=3):
    text = decode(np.fromfile(os.path.join(args.data_dir, "train.bin"), dtype=np.uint8, count=n_chars))
    counts = Counter(w.lower() for w in WORD_RE.findall(text))
    return {w for w, c in counts.items() if c >= min_count}


def text_metrics(stories, vocab):
    words_total = words_known = 0
    rep4 = []
    lengths = []
    sentence_end = 0
    for s in stories:
        words = [w.lower() for w in WORD_RE.findall(s)]
        words_total += len(words)
        words_known += sum(w in vocab for w in words)
        grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        if grams:
            rep4.append(1 - len(set(grams)) / len(grams))
        lengths.append(len(s))
        sentence_end += s.rstrip().endswith((".", "!", "?", '"', "”"))
    return {"stories": len(stories), "known_word_rate": words_known / max(words_total, 1),
            "repeated_4gram_rate": float(np.mean(rep4)) if rep4 else 0.0,
            "mean_chars": float(np.mean(lengths)), "ends_with_sentence_end": sentence_end / len(stories)}


@torch.inference_mode()
def generate_batch(model, prompt, n, max_new=800, temperature=0.8):
    start = [EOT] + [STOI.get(c, UNK) for c in prompt]
    idx = torch.tensor([start] * n, dtype=torch.long)
    finished = torch.zeros(n, dtype=torch.bool)
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        finished |= nxt.squeeze(1) == EOT
        if finished.all():
            break
    out = []
    for row in idx[:, 1:].tolist():
        ended = EOT in row
        row = row[:row.index(EOT)] if ended else row
        out.append({"text": decode(row), "ended_with_eot": ended})
    return out


def stage_gen():
    torch.manual_seed(args.seed)
    model, _ = load_model()
    print("  building word vocabulary from 50M training characters ...", flush=True)
    vocab = word_vocab()
    samples, t0 = [], time.time()
    for i, prompt in enumerate(PROMPTS):
        for s in generate_batch(model, prompt, args.samples_per_prompt):
            samples.append({"prompt": prompt, **s})
        print(f"  prompt {i + 1}/{len(PROMPTS)} done ({time.time() - t0:.0f} s)", flush=True)
    gen = text_metrics([s["text"] for s in samples], vocab)
    gen["ended_with_eot"] = sum(s["ended_with_eot"] for s in samples) / len(samples)

    # reference: real validation stories, truncated to the generation limit for comparability
    val = np.fromfile(os.path.join(args.data_dir, "val.bin"), dtype=np.uint8, count=3_000_000)
    stories = [s.strip() for s in decode(val).split("<eot>")[1:-1] if s.strip()]
    rng = np.random.default_rng(args.seed)
    ref_stories = [stories[i][:800] for i in rng.choice(len(stories), size=500, replace=False)]
    ref = text_metrics(ref_stories, vocab)
    ref["ended_with_eot"] = float(np.mean([len(s) < 800 for s in ref_stories]))
    save_json("gen.json", {"generated": gen, "real_val_stories": ref, "vocab_size_words": len(vocab),
                           "temperature": 0.8, "max_new_chars": 800, "samples": samples})
    print(f"  generated: {gen}\n  real:      {ref}")


# ---------------------------------------------------------------------------- report
def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.report_dir, exist_ok=True)
    val, ng, gen = load_json("val.json"), load_json("ngram.json"), load_json("gen.json")
    lines = ["# Evaluation – character-level TinyStories model", ""]
    if val:
        lines += [f"Checkpoint `{val['checkpoint']}` (iteration {val['iter']:,}, "
                  f"{val['tokens_seen']:,} training tokens, {val['params']:,} parameters).", "",
                  "## 1. Validation loss (full validation set)", "",
                  "| metric | value |", "|---|---|",
                  f"| characters evaluated | {val['val_tokens']:,} ({val['windows']:,} windows of 256) |",
                  f"| cross-entropy | {val['loss_nats']:.4f} nats/char |",
                  f"| bits per character | {val['bits_per_char']:.4f} |",
                  f"| perplexity per character | {val['perplexity_per_char']:.3f} |", "",
                  "Non-overlapping windows: on average a character has 128 characters of context, so this is "
                  "slightly higher than a sliding-window evaluation would give.", ""]
    if ng:
        lines += ["## 2. Baselines: character n-grams", "",
                  f"Witten-Bell interpolated n-grams trained on the first {ng['train_chars']:,} training "
                  f"characters, scored on the first {ng['val_chars']:,} validation characters.", "",
                  "| model | context | bits/char |", "|---|---|---|"]
        lines += [f"| {o['order']}-gram | {o['order'] - 1} chars | {o['bits_per_char']:.4f} |"
                  for o in ng["orders"]]
        if val:
            wl = np.load(os.path.join(args.out_dir, "val_window_loss.npy"))
            n_sub = ng["val_chars"] // 256
            sub = float(wl[:n_sub].mean()) / LN2
            lines += [f"| **transformer (1M params)** | up to 255 chars | **{sub:.4f}** |", "",
                      f"(Transformer row: the same first {n_sub * 256:,} validation characters.)", ""]
            best = min(ng["orders"], key=lambda o: o["bits_per_char"])
            lines += [f"Best n-gram: {best['order']}-gram at {best['bits_per_char']:.4f} bits/char. "
                      f"The transformer is {100 * (1 - sub / best['bits_per_char']):.0f}% lower.", ""]
    if val:
        bp = np.array(val["loss_by_position"]) / LN2
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(bp) + 1), bp, color="#2f6db5", lw=1.6)
        ax.set_xscale("log")
        ax.set_xlabel("characters of context available (position in window)")
        ax.set_ylabel("bits per character")
        ax.set_title("Validation loss by context length")
        ax.grid(alpha=0.3, which="both")
        if ng:
            for o in ng["orders"]:
                if o["order"] in (3, 5, max(x["order"] for x in ng["orders"])):
                    ax.axhline(o["bits_per_char"], ls="--", lw=1, color="#999999")
                    ax.text(len(bp), o["bits_per_char"], f" {o['order']}-gram", va="center", fontsize=8,
                            color="#666666")
        fig.tight_layout()
        fig.savefig(os.path.join(args.report_dir, "loss_by_position.png"), dpi=120)
        pts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        lines += ["## 3. Loss by position in the context window", "",
                  "![loss by position](loss_by_position.png)", "",
                  "| context (chars) | " + " | ".join(str(x) for x in pts) + " |",
                  "|---|" + "---|" * len(pts),
                  "| bits/char | " + " | ".join(f"{bp[x - 1]:.3f}" for x in pts) + " |", ""]
    if gen:
        g, r = gen["generated"], gen["real_val_stories"]
        lines += ["## 4. Generated text", "",
                  f"{g['stories']} stories (10 prompts × {g['stories'] // 10}), temperature "
                  f"{gen['temperature']}, at most {gen['max_new_chars']} characters, compared with 500 real "
                  f"validation stories cut to the same length. Known words = words that occur ≥3 times in "
                  f"50M training characters ({gen['vocab_size_words']:,} words).", "",
                  "| metric | generated | real stories |", "|---|---|---|",
                  f"| known-word rate | {g['known_word_rate']:.2%} | {r['known_word_rate']:.2%} |",
                  f"| repeated word 4-grams within a story | {g['repeated_4gram_rate']:.2%} | "
                  f"{r['repeated_4gram_rate']:.2%} |",
                  f"| story finished (<eot> within 800 chars) | {g['ended_with_eot']:.0%} | "
                  f"{r['ended_with_eot']:.0%} |",
                  f"| ends with . ! ? or quote | {g['ends_with_sentence_end']:.0%} | "
                  f"{r['ends_with_sentence_end']:.0%} |",
                  f"| mean length (chars) | {g['mean_chars']:.0f} | {r['mean_chars']:.0f} |", "",
                  "### Examples (first sample of the first three prompts)", ""]
        seen = []
        for s in gen["samples"]:
            if s["prompt"] not in seen and len(seen) < 3:
                seen.append(s["prompt"])
                lines += ["```", s["text"].strip(), "```", ""]
    with open(os.path.join(args.report_dir, "EVAL.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.report_dir}/EVAL.md")


if __name__ == "__main__":
    stages = ["val", "ngram", "gen", "report"] if args.stage == "all" else [args.stage]
    for st in stages:
        print(f"== {st}", flush=True)
        globals()[f"stage_{st}"]()
