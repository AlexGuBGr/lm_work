"""
Figures and numbers for the README.

    .venv/bin/python analysis/figures.py          # writes docs/figures/*.png and docs/figures/findings.json

Uses out/best.pt, out/train_log.csv and the evaluation results in out/eval/ (run evaluate.py first).
"""
import csv
import json
import math
import os
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, ModelConfig  # noqa: E402

OUT = "docs/figures"
LN2 = math.log(2)
BLUE, GREY, ORANGE = "#2f6db5", "#8a94a3", "#d9822b"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})
torch.set_num_threads(os.cpu_count())
os.makedirs(OUT, exist_ok=True)

meta = json.load(open("data/meta.json", encoding="utf-8"))
ITOS, EOT, UNK = meta["itos"], meta["eot_id"], meta["unk_id"]
ckpt = torch.load("out/best.pt", map_location="cpu", weights_only=False)
model = GPT(ModelConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()
T = model.cfg.block_size
findings = {}


def show(i):
    return {EOT: "<eot>", UNK: "<unk>"}.get(i) or {"\n": "↵", " ": "␣"}.get(ITOS[i], ITOS[i])


# ------------------------------------------------------------------ headline numbers
rows = list(csv.DictReader(open("out/train_log.csv")))
t_start = datetime.strptime(rows[0]["time"], "%Y-%m-%d %H:%M:%S")
t_end = datetime.strptime(rows[-1]["time"], "%Y-%m-%d %H:%M:%S")
tokens = int(rows[-1]["tokens_seen"])
n_params = model.num_params()
train_flop = 6 * n_params * tokens  # standard estimate: forward + backward ≈ 6 FLOP per parameter per token
tps = [float(r["tokens_per_sec"]) for r in rows if r["kind"] == "train" and r["tokens_per_sec"]]
val = json.load(open("out/eval/val.json"))
ngram = json.load(open("out/eval/ngram.json"))
wl = np.load("out/eval/val_window_loss.npy")
sub_bits = float(wl[:ngram["val_chars"] // T].mean()) / LN2
best_ng = min(ngram["orders"], key=lambda o: o["bits_per_char"])
findings["headline"] = {
    "params": n_params, "train_chars": tokens,
    "wall_clock_hours": round((t_end - t_start).total_seconds() / 3600, 1),
    "median_tokens_per_sec": round(float(np.median(tps))), "train_petaflop": round(train_flop / 1e15, 2),
    "val_bits_per_char": round(val["bits_per_char"], 4), "val_bits_per_char_full_context":
        round(val["loss_by_position"][-1] / LN2, 3),
    "best_ngram_order": best_ng["order"], "best_ngram_bits": round(best_ng["bits_per_char"], 4),
    "transformer_bits_same_chars": round(sub_bits, 4),
    "improvement_over_best_ngram_pct": round(100 * (1 - sub_bits / best_ng["bits_per_char"]), 1),
}

# ------------------------------------------------------------------ 1. learning curve
ev = [r for r in rows if r["kind"] == "eval"]
x = np.array([int(r["tokens_seen"]) for r in ev]) / 1e6
y = np.array([float(r["val_loss"]) for r in ev]) / LN2
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=3, label="transformer, validation")
for o in ngram["orders"]:
    if o["order"] in (5, 8):
        ax.axhline(o["bits_per_char"], ls="--", lw=1, color=GREY)
        ax.text(x[-1], o["bits_per_char"] + 0.015, f"{o['order']}-gram baseline", ha="right", color="#555",
                fontsize=9)
hours = [(datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") - t_start).total_seconds() / 3600 for r in ev]
ax2 = ax.secondary_xaxis("top", functions=(lambda m: np.interp(m, x, hours),
                                           lambda h: np.interp(h, hours, x)))
ax2.set_xticks([0, 10, 20, 30, 40])
ax2.set_xlabel("hours of training on 2 vCPUs")
ax.set_xlabel("characters seen (millions)")
ax.set_ylabel("bits per character")
ax.set_ylim(0.5, 1.6)
ax.set_title("Learning curve", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/learning_curve.png")
plt.close(fig)

# ------------------------------------------------------------------ 2. n-gram comparison
labels = [f"{o['order']}-gram ({o['order'] - 1} chars)" for o in ngram["orders"]]
labels += ["transformer (1M params)"]
vals = [o["bits_per_char"] for o in ngram["orders"]] + [sub_bits]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = ax.barh(labels, vals, color=[GREY] * len(ngram["orders"]) + [BLUE])
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("bits per character (lower is better)")
ax.set_title("Transformer vs. character n-gram baselines", loc="left", fontweight="bold")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(f"{OUT}/ngram_comparison.png")
plt.close(fig)

# ------------------------------------------------------------------ 3. throughput vs batch size
# measured with bench.py on this server (see LOG.md, step 5); fwd+bwd+AdamW, 256-char sequences
bench = {"batch": [8, 16, 32, 64], "eager": [1164, 1373, 1448, 1554], "compile": [1371, 1550, 1532, 1588]}
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(bench["batch"], bench["eager"], marker="o", color=BLUE, lw=2, label="eager PyTorch")
ax.plot(bench["batch"], bench["compile"], marker="s", color=ORANGE, lw=2, label="torch.compile")
ax.set_xscale("log", base=2)
ax.set_xticks(bench["batch"], [str(b) for b in bench["batch"]])
ax.set_ylim(0, 1800)
ax.set_xlabel("batch size (sequences of 256 characters)")
ax.set_ylabel("training tokens / second")
ax.set_title("Throughput saturates: the two cores are already fully busy", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/throughput_vs_batch.png")
plt.close(fig)
findings["throughput"] = bench

# ------------------------------------------------------------------ 4. character embedding map
emb = model.tok_emb.weight.detach().numpy()
emb_c = emb - emb.mean(0)
_, s, vt = np.linalg.svd(emb_c, full_matrices=False)
xy = emb_c @ vt[:2].T
explained = (s[:2] ** 2 / (s ** 2).sum()).tolist()


def char_class(i):
    if i in (EOT, UNK):
        return "special"
    c = ITOS[i]
    if c in "aeiou":
        return "vowel (lower case)"
    if c.isalpha() and c.islower():
        return "consonant (lower case)"
    if c.isalpha():
        return "upper case"
    if c.isdigit():
        return "digit"
    if c.isspace():
        return "whitespace"
    return "punctuation"


classes = {"vowel (lower case)": "#d9822b", "consonant (lower case)": "#2f6db5", "upper case": "#7b52ab",
           "digit": "#2e8a4a", "punctuation": "#c0453a", "whitespace": "#333333", "special": "#999999"}
fig, ax = plt.subplots(figsize=(7.5, 6.2))
for name, col in classes.items():
    idx = [i for i in range(len(ITOS)) if char_class(i) == name]
    ax.scatter(xy[idx, 0], xy[idx, 1], s=26, color=col, label=name, alpha=0.85)
    for i in idx:
        ax.annotate(show(i), xy[i], xytext=(3, 2), textcoords="offset points", fontsize=8.5, color=col)
ax.set_xlabel(f"principal component 1 ({explained[0]:.0%} of variance)")
ax.set_ylabel(f"principal component 2 ({explained[1]:.0%} of variance)")
ax.set_title("The 87 learned character embeddings (PCA)", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="best")
fig.tight_layout()
fig.savefig(f"{OUT}/char_embeddings.png")
plt.close(fig)


# separation score: mean distance to own class centroid vs. to other centroids (in the full 128-d space)
def class_separation():
    labs = np.array([char_class(i) for i in range(len(ITOS))])
    keep = np.isin(labs, ["vowel (lower case)", "consonant (lower case)", "upper case", "digit",
                          "punctuation"])
    e, lab = emb_c[keep], labs[keep]
    cents = {c: e[lab == c].mean(0) for c in np.unique(lab)}
    correct = sum(min(cents, key=lambda c: np.linalg.norm(v - cents[c])) == lab_i for v, lab_i in zip(e, lab))
    return correct / len(e)


findings["embeddings"] = {"pca_explained": [round(v, 3) for v in explained],
                          "nearest_class_centroid_accuracy": round(class_separation(), 3)}


# ------------------------------------------------------------------ 5. surprisal
@torch.inference_mode()
def surprisal(ids, batch=64):
    """Bits needed for every character of ids[1:], each predicted with 128-255 characters of context
    (windows of 256 with stride 128; only the second half of each window is scored, except the first)."""
    ids = torch.tensor(ids)
    n = len(ids) - 1
    out = np.zeros(n)
    starts = list(range(0, n - T + 1, T // 2))
    for b0 in range(0, len(starts), batch):
        st = starts[b0:b0 + batch]
        x = torch.stack([ids[s:s + T] for s in st])
        y = torch.stack([ids[s + 1:s + T + 1] for s in st])
        logits, _ = model(x)
        lp = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                             reduction="none").view(len(st), T).numpy() / LN2
        for s, row in zip(st, lp):
            lo = 0 if s == 0 else T // 2
            out[s + lo:s + T] = row[lo:]
    return out[:starts[-1] + T]


val = np.fromfile("data/val.bin", dtype=np.uint8, count=200_001).astype(np.int64)
cache = "out/eval/surprisal_200k.npy"
if os.path.exists(cache):
    bits = np.load(cache)
else:
    bits = surprisal(val)
    np.save(cache, bits)
chars = val[1:len(bits) + 1]

# statistics by role of the character inside a word
roles = {"1st letter of a word": [], "2nd letter": [], "3rd letter": [], "4th+ letter": [],
         "space after a word": [], "punctuation": [], "line break": [], "end of story <eot>": []}
pos_in_word = 0
for c, b in zip(chars, bits):
    ch = ITOS[c] if c > 1 else ""
    if c == EOT:
        roles["end of story <eot>"].append(b)
        pos_in_word = 0
    elif ch.isalpha() or ch == "'":
        pos_in_word += 1
        key = ["1st letter of a word", "2nd letter", "3rd letter"][pos_in_word - 1] if pos_in_word <= 3 \
            else "4th+ letter"
        roles[key].append(b)
    else:
        if ch == " " and pos_in_word:
            roles["space after a word"].append(b)
        elif ch == "\n":
            roles["line break"].append(b)
        elif ch and not ch.isspace():
            roles["punctuation"].append(b)
        pos_in_word = 0
role_stats = {k: {"mean_bits": round(float(np.mean(v)), 3),
                  "share_of_chars": round(len(v) / len(chars), 3),
                  "share_of_total_bits": round(float(np.sum(v) / bits.sum()), 3)}
              for k, v in roles.items() if v}
findings["surprisal_by_role"] = role_stats

fig, ax = plt.subplots(figsize=(7.5, 3.6))
names = list(role_stats)
means = [role_stats[k]["mean_bits"] for k in names]
cols = [ORANGE if k == "1st letter of a word" else BLUE if "letter" in k else GREY for k in names]
bars = ax.bar(range(len(names)), means, color=cols)
ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
ax.set_xticks(range(len(names)), [n.replace(" of a word", "\nof a word").replace(" after", "\nafter")
                                 .replace("end of story", "end of\nstory") for n in names], fontsize=8.5)
ax.set_ylabel("mean surprisal (bits)")
ax.set_title("Where the model is surprised: mostly at the start of words", loc="left", fontweight="bold")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(f"{OUT}/surprisal_by_role.png")
plt.close(fig)

# heat map of one story
starts = np.where(val == EOT)[0]
s0, s1 = next((a + 1, b) for a, b in zip(starts, starts[1:]) if 450 < b - a < 750 and a > 2000)
story_bits, story = bits[s0 - 1:s1 - 1], val[s0:s1]
cols_per_line = 78
lines = []
line = []
for c, b in zip(story, story_bits):
    ch = ITOS[c]
    if ch == "\n" or len(line) >= cols_per_line:
        lines.append(line)
        line = []
        if ch == "\n":
            continue
    line.append((ch, b))
lines.append(line)
norm = Normalize(0, 6)
cmap = plt.get_cmap("YlOrRd")
fig_h = 0.27 * len(lines) + 1.1
fig, ax = plt.subplots(figsize=(10, fig_h))
ax.set_xlim(0, cols_per_line)
ax.set_ylim(len(lines), -0.6)
ax.axis("off")
for r, ln in enumerate(lines):
    for k, (ch, b) in enumerate(ln):
        ax.add_patch(plt.Rectangle((k, r - 0.5), 1, 1, color=cmap(norm(b)), lw=0))
        ax.text(k + 0.5, r, ch, ha="center", va="center", family="monospace", fontsize=9.5)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
cb.set_label("surprisal in bits (darker = harder to predict)")
ax.set_title("A validation story coloured by how surprised the model was by each character",
             loc="left", fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/surprisal_story.png")
plt.close(fig)
hard = sorted(zip(story_bits, range(len(story))), reverse=True)[:5]
findings["surprisal_story"] = {"chars": len(story), "mean_bits": round(float(story_bits.mean()), 3),
                               "hardest": [{"bits": round(float(b), 2),
                                            "context": "".join(ITOS[c] for c in story[max(0, i - 15):i]),
                                            "char": ITOS[story[i]]} for b, i in hard]}

with open(f"{OUT}/findings.json", "w") as f:
    json.dump(findings, f, indent=1, ensure_ascii=False)
print(json.dumps(findings, indent=1, ensure_ascii=False))
