"""
Tegner en detaljeret oversigt over modelarkitekturen.
Kør:  .venv/bin/python docs/make_architecture.py   ->  docs/architecture.png + docs/architecture.svg
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import GPT, ModelConfig  # noqa: E402

cfg = ModelConfig()
model = GPT(cfg)
C, H, L, T, V, F = cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block_size, cfg.vocab_size, cfg.mlp_hidden
hd = C // H
n_total = model.num_params()
n_emb = V * C
n_attn = C * 3 * C + C * C
n_mlp = 3 * C * F
n_block = n_attn + n_mlp + 2 * C

# farver pr. komponenttype
COL = {
    "io": ("#eef2f7", "#5b6b82"),
    "emb": ("#e3f0ff", "#2f6db5"),
    "norm": ("#fff4d6", "#b58a12"),
    "attn": ("#ffe3e0", "#c0453a"),
    "mlp": ("#e2f5e6", "#2e8a4a"),
    "head": ("#efe6ff", "#6a44b8"),
    "add": ("#ffffff", "#333333"),
    "note": ("#fafafa", "#999999"),
}

fig, ax = plt.subplots(figsize=(18, 22))
ax.set_xlim(0, 180)
ax.set_ylim(0, 220)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(cx, cy, w, h, title, sub=None, kind="io", fs=13, sfs=10.5, dashed=False, align="center"):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=fc, ec=ec, lw=2, ls="--" if dashed else "-", zorder=2))
    if sub:
        ax.text(cx, cy + h * 0.18, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color="#1d1d1d", zorder=3)
        ax.text(cx, cy - h * 0.2, sub, ha="center", va="center", fontsize=sfs,
                color="#333333", zorder=3, linespacing=1.35)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color="#1d1d1d", zorder=3)


def arrow(x1, y1, x2, y2, color="#444444", lw=1.8, style="-|>", conn="arc3", z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, connectionstyle=conn, zorder=z))


def plus(cx, cy, r=2.2):
    ax.add_patch(plt.Circle((cx, cy), r, fc="white", ec="#333333", lw=2, zorder=3))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=16, fontweight="bold", zorder=4)


def shape(x, y, s, ha="left"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=10, color="#555555", family="monospace",
            zorder=3)


def panel(x0, y0, w, h, title):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                fc="#fcfcfd", ec="#c9ced6", lw=1.5, zorder=0))
    ax.text(x0 + 2, y0 + h - 2.5, title, ha="left", va="center", fontsize=14,
            fontweight="bold", color="#222222")


# ------------------------------------------------------------------ titel
ax.text(90, 214, "Tegn-niveau GPT (Llama-stil) – TinyStories", ha="center", fontsize=22,
        fontweight="bold")
ax.text(90, 209.5,
        f"{n_total:,} parametre  ·  {L} lag  ·  {H} hoveder  ·  bredde {C}  ·  kontekst {T} tegn  ·  "
        f"ordforråd {V}  ·  float32 på CPU".replace(",", "."),
        ha="center", fontsize=13, color="#444444")

# ------------------------------------------------------------------ hovedsøjle
X = 44          # centrum for hovedsøjlen
W = 52          # bredde af bokse
RX = X - W / 2 - 5   # x for residual-strømmen

box(X, 199, W, 6, "Tekst", '"Once upon a time, there was a little…"', "io")
arrow(X, 196, X, 192.5)
box(X, 188.5, W, 7, "Tokenisering (tegn-niveau)",
    f"1 tegn = 1 token  ·  {V} tokens (<unk>, <eot> + 85 tegn)", "io", sfs=10)
shape(X + W / 2 + 2, 188.5, f"idx  (B, {T})")
arrow(X, 185, X, 181.5)
box(X, 177.5, W, 7, "Token-embedding",
    f"tabel {V}×{C}  ·  {n_emb:,} param".replace(",", ".") + "  ·  delt med LM-hoved", "emb", sfs=10)
shape(X + W / 2 + 2, 177.5, f"x  (B, {T}, {C})")
arrow(X, 174, X, 168.5)

# transformer-blok (×5)
BY0, BY1 = 76, 167
ax.add_patch(FancyBboxPatch((X - W / 2 - 9, BY0), W + 18, BY1 - BY0,
                            boxstyle="round,pad=0.4,rounding_size=2.5",
                            fc="#f7f9fc", ec="#7b8794", lw=2.2, ls="--", zorder=0))
ax.text(X + W / 2 + 7.5, BY1 - 3, f"× {L}", ha="right", va="center", fontsize=24,
        fontweight="bold", color="#5b6b82")
ax.text(X - W / 2 - 7, BY1 - 3, "Transformer-blok", ha="left", va="center", fontsize=14,
        fontweight="bold", color="#5b6b82")
ax.text(X - W / 2 - 7, BY1 - 6.5, f"{n_block:,} param pr. blok".replace(",", "."), ha="left",
        va="center", fontsize=10.5, color="#5b6b82")

# residual-strøm 1 (attention)
y_in = 165
ax.plot([X, X], [y_in, 154], color="#444444", lw=1.8, zorder=1)
ax.plot([RX, X], [156.5, 156.5], color="#888888", lw=1.8, zorder=1)
ax.plot([RX, RX], [156.5, 128], color="#888888", lw=1.8, zorder=1)
arrow(RX, 128, X - 2.2, 128, color="#888888")
ax.text(RX - 1, 143, "residual", rotation=90, ha="right", va="center", fontsize=10, color="#888888")

box(X, 151, W, 5, "RMSNorm", None, "norm", fs=12)
arrow(X, 148.5, X, 145.5)
box(X, 139, W, 12, "Kausal multi-head self-attention",
    f"{H} hoveder × {hd} dim  ·  RoPE på q, k\n"
    f"F.scaled_dot_product_attention(is_causal=True)\n"
    f"{n_attn:,} param".replace(",", "."), "attn")
arrow(X, 133, X, 130.2)
plus(X, 128)
shape(X + 4, 128, f"(B, {T}, {C})")

# residual-strøm 2 (MLP)
ax.plot([X, X], [125.8, 117], color="#444444", lw=1.8, zorder=1)
ax.plot([RX, X], [122, 122], color="#888888", lw=1.8, zorder=1)
ax.plot([RX, RX], [122, 84], color="#888888", lw=1.8, zorder=1)
arrow(RX, 84, X - 2.2, 84, color="#888888")
ax.text(RX - 1, 103, "residual", rotation=90, ha="right", va="center", fontsize=10, color="#888888")

box(X, 114.5, W, 5, "RMSNorm", None, "norm", fs=12)
arrow(X, 112, X, 108.5)
box(X, 100, W, 15, "SwiGLU feed-forward",
    f"down( SiLU(gate·x) ⊙ (up·x) )\n"
    f"{C} → {F} → {C}\n"
    f"{n_mlp:,} param".replace(",", "."), "mlp")
arrow(X, 92.5, X, 86.2)
plus(X, 84)
shape(X + 4, 84, f"(B, {T}, {C})")
arrow(X, 81.8, X, 70.5)

# afslutning
box(X, 67.5, W, 5, "RMSNorm (slut)", None, "norm", fs=12)
arrow(X, 65, X, 61.5)
box(X, 57, W, 8, "LM-hoved (lineær)",
    f"{C} → {V}  ·  vægte delt med embedding (0 ekstra param)", "head", sfs=10)
shape(X + W / 2 + 2, 57, f"logits  (B, {T}, {V})")
arrow(X, 53, X, 49.3)
box(X, 44.5, W, 9, "Softmax → næste tegn",
    "træning: cross-entropy mod teksten forskudt 1 tegn\n"
    "generering: temperatur / top-k sampling", "io", sfs=10)

# stiplet pil for delte vægte
arrow(X - W / 2 - 0.5, 177.5, X - W / 2 - 0.5, 57, color="#6a44b8", lw=1.6, style="<|-|>",
      conn="arc3,rad=0.2")
ax.text(2.2, 118, "delte vægte (weight tying)", rotation=90, ha="center", va="center",
        fontsize=10.5, color="#6a44b8", fontweight="bold")

# ------------------------------------------------------------------ panel: attention
PX, PW = 92, 85
panel(PX, 150, PW, 56, "Attention i detaljer (ét lag)")
ax_ = PX + 22
box(ax_, 196, 30, 4.5, f"x  (B, T, {C})", None, "io", fs=11)
arrow(ax_, 193.5, ax_, 190.7)
box(ax_, 188, 30, 5, f"Linear qkv  {C}→{3 * C}", None, "attn", fs=11)
arrow(ax_, 185.5, ax_, 182.7)
box(ax_, 180, 30, 5, "split → q, k, v", None, "attn", fs=11)
shape(ax_ + 17, 180, f"hver (B, {H}, T, {hd})")
arrow(ax_, 177.5, ax_, 174.7)
box(ax_, 172, 30, 5, "RoPE(q), RoPE(k)", None, "attn", fs=11)
arrow(ax_, 169.5, ax_, 166.7)
box(ax_, 164, 30, 5, "SDPA, kausal maske", None, "attn", fs=11)
arrow(ax_, 161.5, ax_, 158.7)
box(ax_, 156, 30, 5, f"concat → Linear {C}→{C}", None, "attn", fs=11)
ax.text(PX + 43, 172.5,
        r"$\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{" + str(hd) + r"}} + M\right)V$",
        fontsize=16, ha="left", va="center")
ax.text(PX + 43, 165.5, "M = −∞ over diagonalen:\net tegn ser kun sig selv\nog tidligere tegn",
        fontsize=10.5, ha="left", va="center", color="#444444")
ax.text(PX + 43, 157,
        f"q,k,v: {C}×{3 * C} = {C * 3 * C:,}\nproj:  {C}×{C} = {C * C:,}".replace(",", "."),
        fontsize=10.5, ha="left", va="center", family="monospace", color="#444444")
ax.text(PX + 43, 189, "Ingen bias-led i hele modellen.\nAttention-matrix pr. hoved:\n"
        f"(T × T) = ({T} × {T})", fontsize=10.5, ha="left", va="center", color="#444444")

# ------------------------------------------------------------------ panel: SwiGLU
panel(PX, 112, PW, 34, "SwiGLU feed-forward")
sx = PX + 22
box(sx, 136, 18, 4.5, f"x (B,T,{C})", None, "io", fs=11)
arrow(sx - 3, 133.7, sx - 11, 129.3)
arrow(sx + 3, 133.7, sx + 11, 129.3)
box(sx - 12, 127, 20, 4.5, f"gate {C}→{F}", None, "mlp", fs=11)
box(sx + 12, 127, 20, 4.5, f"up {C}→{F}", None, "mlp", fs=11)
arrow(sx - 12, 124.7, sx - 12, 122.3)
box(sx - 12, 120, 14, 4, "SiLU", None, "mlp", fs=11)
arrow(sx - 12, 118, sx - 2, 117.0)
arrow(sx + 12, 124.7, sx + 2, 117.0)
ax.add_patch(plt.Circle((sx, 116), 1.8, fc="white", ec="#333", lw=2, zorder=3))
ax.text(sx, 116, "⊙", ha="center", va="center", fontsize=14, zorder=4)
arrow(sx + 1.8, 116, sx + 10, 116)
box(sx + 21, 116, 20, 4.5, f"down {F}→{C}", None, "mlp", fs=11)
ax.text(PX + 52, 136, r"$\mathrm{SiLU}(z) = z\cdot\sigma(z)$", fontsize=14, va="center")
ax.text(PX + 52, 129.5,
        f"3 matricer á {C}×{F}\n= {n_mlp:,} param\n(hidden {F} ≈ 8/3·{C},\n rundet op til 32)".replace(",", "."),
        fontsize=10.5, va="center", color="#444444")

# ------------------------------------------------------------------ panel: RMSNorm + RoPE
panel(PX, 76, PW, 32, "RMSNorm og RoPE")
ax.text(PX + 3, 98.5, r"RMSNorm:  $y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g$",
        fontsize=14, va="center")
ax.text(PX + 3, 92, f"lærbar skala g ({C} param), ingen centrering/bias · pre-norm før hvert sub-lag",
        fontsize=10.5, va="center", color="#444444")
ax.text(PX + 3, 86,
        r"RoPE:  $(x'_{2i},\, x'_{2i+1}) = R(m\,\theta_i)\,(x_{2i},\, x_{2i+1})$,   "
        r"$\theta_i = 10000^{-2i/d}$",
        fontsize=14, va="center")
ax.text(PX + 3, 80.5, f"roterer par af dimensioner i q og k efter position m (0…{T - 1}) · "
        "0 param", fontsize=10.5, va="center", color="#444444")

# ------------------------------------------------------------------ panel: parametertabel
panel(PX, 22, PW, 50, "Parametre")
rows = [
    ("Komponent", "Form", "Antal", "Andel"),
    ("Token-embedding (= LM-hoved)", f"{V}×{C}", n_emb, None),
    ("Attention qkv  (pr. lag)", f"{C}×{3 * C}", C * 3 * C, None),
    ("Attention proj (pr. lag)", f"{C}×{C}", C * C, None),
    ("SwiGLU gate/up/down (pr. lag)", f"3×{C}×{F}", n_mlp, None),
    ("RMSNorm ×2 (pr. lag)", f"2×{C}", 2 * C, None),
    ("= én blok", "", n_block, None),
    (f"{L} blokke", "", L * n_block, None),
    ("Slut-RMSNorm", f"{C}", C, None),
    ("I alt", "", n_total, None),
]
y = 66
for i, (a, b, c, _) in enumerate(rows):
    bold = i == 0 or a.startswith("I alt") or a.startswith("=")
    fw = "bold" if bold else "normal"
    ax.text(PX + 3, y, a, fontsize=11, va="center", fontweight=fw)
    ax.text(PX + 50, y, b, fontsize=11, va="center", family="monospace", fontweight=fw)
    if i == 0:
        ax.text(PX + 72, y, c, fontsize=11, va="center", ha="right", fontweight=fw)
        ax.text(PX + 82, y, "Andel", fontsize=11, va="center", ha="right", fontweight=fw)
    else:
        ax.text(PX + 72, y, f"{c:,}".replace(",", "."), fontsize=11, va="center", ha="right",
                family="monospace", fontweight=fw)
        share = c / n_total
        if a.startswith(f"{L} blokke") or a.startswith("Token") or a.startswith("I alt"):
            ax.text(PX + 82, y, f"{share:.1%}".replace(".", ","), fontsize=11, va="center",
                    ha="right", family="monospace", fontweight=fw)
    if i in (0, 5, 8):
        ax.plot([PX + 2, PX + PW - 2], [y - 2.2, y - 2.2], color="#c9ced6", lw=1)
    y -= 4.6

# ------------------------------------------------------------------ bund: træning
ax.add_patch(FancyBboxPatch((3, 3), 174, 15.5, boxstyle="round,pad=0.4,rounding_size=2",
                            fc="#fcfcfd", ec="#c9ced6", lw=1.5, zorder=0))
ax.text(5, 15.5, "Træning", fontsize=14, fontweight="bold", va="center")
ax.text(5, 8.5,
        "Data: TinyStoriesV2-GPT4 · 2,19 mia. tegn (train) / 22 mio. (val) som uint8 i np.memmap, tilfældige udsnit på 257 tegn\n"
        "Batch 32 × 256 = 8.192 tokens/iter · AdamW (β=0,9/0,95, wd 0,1 på matricer) · lr 3e-3 → 3e-4 cosinus, warmup 500 · grad-clip 1,0\n"
        "30.000 iterationer ≈ 246 mio. tokens · ~1.500 tok/s på 2 vCPU (Xeon E5-2690 v2, float32) ≈ 2 døgn",
        fontsize=11, va="center", linespacing=1.6, color="#333333")

out = os.path.dirname(__file__)
fig.savefig(os.path.join(out, "architecture.png"), dpi=110, bbox_inches="tight")
fig.savefig(os.path.join(out, "architecture.svg"), bbox_inches="tight")
print("skrev docs/architecture.png og docs/architecture.svg")
