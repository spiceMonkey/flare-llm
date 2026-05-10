#!/usr/bin/env python3
"""Generate attention-architecture diagrams: MHA, GQA, MLA.

Three SVG figures for documentation/modeling/attention.md showing the per-token
data flow, projection matrices, and KV cache footprint of each variant. Box
widths and KV-cache labels use a common DSv3-class reference architecture so the
three figures can be visually compared.

Reference architecture: DSv3-class with H=7168, n_q=128, d_head=H/n_q=56.
GQA reference uses n_kv=8 (typical production grouping). MLA uses DSv3 latent
dimensions (d_c=512, d_q,c=1536, d_qk_nope=128, d_qk_rope=64, d_v=128).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Keep text editable in SVG output (no path conversion).
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"

# ── Reference DSv3-class architecture ─────────────────────────────────
H = 7168
N_Q = 128
D_HEAD = 56              # H / n_q
N_KV_GQA = 8             # typical GQA grouping
D_C = 512                # MLA latent dim
D_QK_NOPE = 128          # MLA non-positional Q/K head dim
D_QK_ROPE = 64           # MLA RoPE Q/K head dim
D_V = 128                # MLA value head dim
D_Q_C = 1536             # MLA query latent dim

# ── Colors ────────────────────────────────────────────────────────────
C_BG      = "#FFFFFF"
C_HIDDEN  = "#B3D9FF"    # hidden state h, output o
C_W       = "#ECEFF1"    # weight matrices
C_W_DOWN  = "#CFD8DC"    # down-projection weights (MLA)
C_W_UP    = "#E0E0E0"    # up-projection weights (MLA)
C_Q       = "#FFCDD2"    # Q-side intermediates
C_K       = "#C8E6C9"    # K-side intermediates
C_V       = "#FFF9C4"    # V-side intermediates
C_LATENT  = "#D1C4E9"    # MLA latent (compressed)
C_KV      = "#F8BBD0"    # KV cache box (highlighted)
C_ATTN    = "#FFE0B2"    # attention block
C_TEXT    = "#212121"
C_ARROW   = "#455A64"
C_DIM     = "#37474F"    # dimension labels
C_KV_EDGE = "#AD1457"    # KV cache border (highlight)


def rb(ax, x, y, w, h, color, label="", fontsize=9.5, fw="normal", lw=1.0, ec="#37474F"):
    """Rounded box."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                facecolor=color, edgecolor=ec, linewidth=lw, zorder=2))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=C_TEXT, fontweight=fw, zorder=3)


def stacked_box(ax, x, y, w, h, color, label="", n_stack=4, dx=0.07, dy=0.07,
                fontsize=9.5, fw="normal", lw=1.0, ec="#37474F"):
    """A stack of n_stack boxes (back-to-front, offset up-right) to convey
    'multiple heads/groups'. Back boxes are drawn under arrow z-order so
    arrows still appear on top; front box gets the label."""
    for i in range(n_stack - 1, 0, -1):
        ax.add_patch(FancyBboxPatch(
            (x + i * dx, y + i * dy), w, h, boxstyle="round,pad=0.03",
            facecolor=color, edgecolor=ec, linewidth=lw * 0.7,
            alpha=0.7, zorder=0.5))
    rb(ax, x, y, w, h, color, label=label, fontsize=fontsize, fw=fw, lw=lw, ec=ec)


def arrow(ax, x1, y1, x2, y2, label="", lpos=0.5, ldx=0.08, fontsize=8,
          color=None, lw=None, ls="-"):
    """Arrow with optional dimension label placed alongside."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=color or C_ARROW,
                                 linewidth=lw if lw is not None else 1.0,
                                 linestyle=ls, zorder=1))
    if label:
        lx = x1 + lpos * (x2 - x1) + ldx
        ly = y1 + lpos * (y2 - y1)
        ax.text(lx, ly, label, ha="left", va="center",
                fontsize=fontsize, color=C_DIM, fontstyle="italic", zorder=3)


# Distinct arrow styles for the three signal-flow roles.
C_SIGNAL = "#1565C0"   # token signal flow (h → Q → attention → W_O → o)
C_WRITE  = "#C62828"   # KV cache write (current K, V → cache)
C_READ   = "#2E7D32"   # KV cache read (cache → attention)


def signal_arrow(ax, x1, y1, x2, y2, label="", **kw):
    """Main token-signal path arrow: thicker, blue."""
    arrow(ax, x1, y1, x2, y2, label=label, color=C_SIGNAL, lw=1.6, **kw)


def write_arrow(ax, x1, y1, x2, y2, label="", **kw):
    """KV-cache write arrow: dashed, red — append current token's K or V."""
    arrow(ax, x1, y1, x2, y2, label=label, color=C_WRITE, lw=1.2, ls="--", **kw)


def read_arrow(ax, x1, y1, x2, y2, label="", **kw):
    """KV-cache read arrow: solid, green — read full S+1 history."""
    arrow(ax, x1, y1, x2, y2, label=label, color=C_READ, lw=1.4, **kw)


def add_legend(ax, y=0.15):
    """Horizontal arrow-convention legend, centered at the bottom of the figure."""
    items = [
        ("token signal flow",   C_SIGNAL, "-"),
        ("write to KV cache",   C_WRITE,  "--"),
        ("read from KV cache",  C_READ,   "-"),
    ]
    item_width = 3.0
    x_start = (10 - 3 * item_width) / 2
    for i, (label, color, ls) in enumerate(items):
        xi = x_start + i * item_width
        ax.plot([xi, xi + 0.4], [y, y], color=color, linewidth=1.8,
                linestyle=ls, solid_capstyle="round", zorder=3)
        ax.text(xi + 0.55, y, label, ha="left", va="center",
                fontsize=8.5, color="#37474F", zorder=3)


def setup_axes(figsize, title, subtitle, ymax=13):
    fig, ax = plt.subplots(figsize=figsize, facecolor=C_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(5, ymax - 0.5, title, ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_TEXT)
    ax.text(5, ymax - 0.95, subtitle, ha="center", va="center",
            fontsize=9.5, color="#546E7A", fontstyle="italic")
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# MHA — every head has its own K and V
# ──────────────────────────────────────────────────────────────────────
def gen_mha():
    fig, ax = setup_axes(
        figsize=(9, 11),
        title="Multi-Head Attention (MHA)",
        subtitle=f"Baseline: every head has its own K and V "
                 f"(reference: H={H}, n_q={N_Q}, d_head={D_HEAD})",
    )

    # Embedded token input (post-embedding + positional encoding)
    rb(ax, 2.5, 11.0, 5.0, 0.6, C_HIDDEN,
       label=f"Embedded token input:  h ∈ ℝ^H    (H = {H})", fw="bold", fontsize=10)

    # W_Q, W_K, W_V — single matrices
    y_w = 9.5
    rb(ax, 0.5, y_w, 2.5, 0.8, C_W,
       label=f"$W_Q$\n($H \\times n_q \\cdot d_{{head}}$)", fontsize=9.5)
    rb(ax, 3.75, y_w, 2.5, 0.8, C_W,
       label=f"$W_K$\n($H \\times n_q \\cdot d_{{head}}$)", fontsize=9.5)
    rb(ax, 7.0, y_w, 2.5, 0.8, C_W,
       label=f"$W_V$\n($H \\times n_q \\cdot d_{{head}}$)", fontsize=9.5)

    # h → W_* (signal)
    signal_arrow(ax, 4.0, 11.0, 1.75, 10.3)
    signal_arrow(ax, 5.0, 11.0, 5.0,  10.3)
    signal_arrow(ax, 6.0, 11.0, 8.25, 10.3)

    # Q, K, V (stacked = per-head)
    y_qkv = 8.0
    stacked_box(ax, 0.5, y_qkv, 2.5, 0.8, C_Q,
                label=f"$Q$\n$n_q \\times d_{{head}}$", n_stack=4, fontsize=9.5)
    stacked_box(ax, 3.75, y_qkv, 2.5, 0.8, C_K,
                label=f"$K$\n$n_q \\times d_{{head}}$", n_stack=4, fontsize=9.5)
    stacked_box(ax, 7.0, y_qkv, 2.5, 0.8, C_V,
                label=f"$V$\n$n_q \\times d_{{head}}$", n_stack=4, fontsize=9.5)

    # W_* → Q/K/V (signal)
    signal_arrow(ax, 1.75, 9.5, 1.75, 8.8)
    signal_arrow(ax, 5.0,  9.5, 5.0,  8.8)
    signal_arrow(ax, 8.25, 9.5, 8.25, 8.8)

    # KV cache box
    y_kv = 6.2
    rb(ax, 2.5, y_kv, 5.0, 1.0, C_KV,
       label="KV cache  (per token per layer)\n"
             f"$2 \\cdot n_q \\cdot d_{{head}} \\cdot b$",
       fontsize=11, fw="bold", lw=1.6, ec=C_KV_EDGE)

    # K, V → cache (write); Q bypasses cache to attention (signal)
    write_arrow(ax, 5.0,  8.0, 5.0, 7.2, label="  append", ldx=0.18, fontsize=8.5)
    write_arrow(ax, 8.25, 8.0, 6.5, 7.2)

    # Q → Attention (signal)
    signal_arrow(ax, 1.75, 8.0, 3.0, 5.4)

    # KV cache → Attention (read)
    read_arrow(ax, 5.0, 6.2, 5.0, 5.4, label="  read all $S{+}1$", ldx=0.18, fontsize=8.5)

    # Attention
    y_a = 4.4
    rb(ax, 2.0, y_a, 6.0, 1.0, C_ATTN,
       label="Attention(Q, K, V)\n"
             "per head:  $\\mathrm{softmax}(Q K^T / \\sqrt{d_{head}}) \\cdot V$",
       fontsize=10, fw="bold")

    # Attention → W_O → output (signal)
    signal_arrow(ax, 5.0, 4.4, 5.0, 3.6)
    y_o = 2.8
    rb(ax, 3.0, y_o, 4.0, 0.8, C_W,
       label="$W_O$   ($n_q \\cdot d_{head} \\times H$)", fontsize=10)
    signal_arrow(ax, 5.0, 2.8, 5.0, 2.0)
    rb(ax, 2.5, 1.4, 5.0, 0.6, C_HIDDEN,
       label="Embedded token output:  o ∈ ℝ^H", fw="bold", fontsize=10)

    # Convention footer
    ax.text(5, 0.75,
        "Single boxes = one tensor (matrix or latent vector).  "
        "Stacked boxes = per-head intermediates ($n_q$ heads).",
        ha="center", va="center", fontsize=8.5, color="#546E7A", fontstyle="italic")

    add_legend(ax, y=0.25)

    fig.savefig("assets/attention_mha.svg", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# GQA — n_kv ≪ n_q heads share K and V
# ──────────────────────────────────────────────────────────────────────
def gen_gqa():
    fig, ax = setup_axes(
        figsize=(9, 11),
        title="Grouped-Query Attention (GQA)",
        subtitle=f"$n_{{kv}}$ shared K/V groups serve $n_q$ query heads "
                 f"(reference: $n_{{kv}}={N_KV_GQA}$, $n_q/n_{{kv}}={N_Q // N_KV_GQA}$ queries per group)",
    )

    # Embedded token input (post-embedding + positional encoding)
    rb(ax, 2.5, 11.0, 5.0, 0.6, C_HIDDEN,
       label=f"Embedded token input:  h ∈ ℝ^H    (H = {H})", fw="bold", fontsize=10)

    # W_Q full-width; W_K, W_V narrower (single matrices; per-head structure in shape)
    y_w = 9.5
    rb(ax, 0.5, y_w, 2.5, 0.8, C_W,
       label=f"$W_Q$\n($H \\times n_q \\cdot d_{{head}}$)", fontsize=9.5)
    rb(ax, 4.0, y_w, 2.0, 0.8, C_W,
       label=f"$W_K$\n($H \\times n_{{kv}} \\cdot d_{{head}}$)", fontsize=8.5)
    rb(ax, 7.0, y_w, 2.0, 0.8, C_W,
       label=f"$W_V$\n($H \\times n_{{kv}} \\cdot d_{{head}}$)", fontsize=8.5)

    # h → W_* (signal)
    signal_arrow(ax, 4.0, 11.0, 1.75, 10.3)
    signal_arrow(ax, 5.0, 11.0, 5.0,  10.3)
    signal_arrow(ax, 6.0, 11.0, 8.0,  10.3)

    # Q (n_q heads, deep stack); K, V (n_kv groups, narrower + shallow stack)
    y_qkv = 8.0
    stacked_box(ax, 0.5, y_qkv, 2.5, 0.8, C_Q,
                label=f"$Q$\n$n_q \\times d_{{head}}$", n_stack=4, fontsize=9.5)
    stacked_box(ax, 4.0, y_qkv, 2.0, 0.8, C_K,
                label=f"$K$\n$n_{{kv}} \\times d_{{head}}$", n_stack=2, fontsize=8.5)
    stacked_box(ax, 7.0, y_qkv, 2.0, 0.8, C_V,
                label=f"$V$\n$n_{{kv}} \\times d_{{head}}$", n_stack=2, fontsize=8.5)

    # W_* → Q/K/V (signal)
    signal_arrow(ax, 1.75, 9.5, 1.75, 8.8)
    signal_arrow(ax, 5.0,  9.5, 5.0,  8.8)
    signal_arrow(ax, 8.0,  9.5, 8.0,  8.8)

    # KV cache — narrow box (only n_kv groups)
    y_kv = 6.2
    rb(ax, 3.0, y_kv, 4.0, 1.0, C_KV,
       label="KV cache  (per token per layer)\n"
             f"$2 \\cdot n_{{kv}} \\cdot d_{{head}} \\cdot b$",
       fontsize=11, fw="bold", lw=1.6, ec=C_KV_EDGE)

    # K, V → cache (write)
    write_arrow(ax, 5.0, 8.0, 5.0, 7.2, label="  append", ldx=0.18, fontsize=8.5)
    write_arrow(ax, 8.0, 8.0, 6.5, 7.2)

    # Q → Attention (signal); cache → Attention (read with broadcast note)
    y_a = 4.4
    rb(ax, 2.0, y_a, 6.0, 1.0, C_ATTN,
       label="Attention(Q, K, V)\n"
             f"each of $n_{{kv}}$ K/V groups serves $n_q / n_{{kv}}$ query heads",
       fontsize=10, fw="bold")
    signal_arrow(ax, 1.75, 8.0, 3.0, 5.4)
    read_arrow(ax, 5.0, 6.2, 5.0, 5.4,
               label="  read all $S{+}1$ (broadcast to $n_q$)",
               ldx=0.18, fontsize=8.5)

    # Attention → W_O → output (signal)
    signal_arrow(ax, 5.0, 4.4, 5.0, 3.6)
    y_o = 2.8
    rb(ax, 3.0, y_o, 4.0, 0.8, C_W,
       label="$W_O$   ($n_q \\cdot d_{head} \\times H$)", fontsize=10)
    signal_arrow(ax, 5.0, 2.8, 5.0, 2.0)
    rb(ax, 2.5, 1.4, 5.0, 0.6, C_HIDDEN,
       label="Embedded token output:  o ∈ ℝ^H", fw="bold", fontsize=10)

    # Convention footer
    ax.text(5, 0.75,
        "Single boxes = one tensor.  Stacked boxes = per-head intermediates "
        "($n_q$ for Q, $n_{kv}$ for K/V).",
        ha="center", va="center", fontsize=8.5, color="#546E7A", fontstyle="italic")

    add_legend(ax, y=0.25)

    fig.savefig("assets/attention_gqa.svg", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# MLA — joint KV latent + per-head reconstruction
# ──────────────────────────────────────────────────────────────────────
def gen_mla():
    fig, ax = setup_axes(
        figsize=(10, 14),
        title="Multi-head Latent Attention (MLA)",
        subtitle=f"Joint KV latent ($d_c={D_C}$) stored in cache; "
                 f"per-head K, V reconstructed on demand via up-projection",
        ymax=15.5,
    )

    # Embedded token input (post-embedding + positional encoding)
    rb(ax, 2.5, 13.5, 5.0, 0.6, C_HIDDEN,
       label=f"Embedded token input:  h ∈ ℝ^H    (H = {H})", fw="bold", fontsize=10)

    # Down-projections (single matrices)
    y_dw = 12.2
    rb(ax, 0.8, y_dw, 2.6, 0.8, C_W_DOWN,
       label=f"$W_{{DQ}}$\n($H \\times d_{{q,c}}$)", fontsize=9.5)
    rb(ax, 6.6, y_dw, 2.6, 0.8, C_W_DOWN,
       label=f"$W_{{DKV}}$\n($H \\times (d_c + d_{{qk,rope}})$)", fontsize=9)

    # h → W_D* (signal)
    signal_arrow(ax, 4.0, 13.5, 2.1, 13.0)
    signal_arrow(ax, 6.0, 13.5, 7.9, 13.0)

    # Latents (single boxes — head-shared)
    y_lat = 10.7
    rb(ax, 1.1, y_lat, 2.0, 0.7, C_LATENT,
       label=f"$c_Q$\n$d_{{q,c}} = {D_Q_C}$", fontsize=9, fw="bold")
    rb(ax, 6.9, y_lat, 2.0, 0.7, C_LATENT,
       label=f"$c_{{KV}}$\n$d_c + d_{{qk,rope}} = {D_C + D_QK_ROPE}$",
       fontsize=9, fw="bold")
    signal_arrow(ax, 2.1, 12.2, 2.1, 11.4, label=f"  H → {D_Q_C}", fontsize=8)
    signal_arrow(ax, 7.9, 12.2, 7.9, 11.4, label=f"  H → {D_C + D_QK_ROPE}", fontsize=8)

    # KV cache — TINY box, only c_KV stored
    y_kvc = 9.2
    rb(ax, 5.6, y_kvc, 2.6, 0.9, C_KV,
       label="KV cache  (only the latent!)\n"
             f"$(d_c + d_{{qk,rope}}) \\cdot b$",
       fontsize=10, fw="bold", lw=1.6, ec=C_KV_EDGE)

    # c_KV → cache (write); cache → up-projections (read)
    write_arrow(ax, 7.9, 10.7, 6.9, 10.1, label="  append", ldx=0.15, fontsize=8)

    # Up-projections (single matrices)
    y_uw = 7.6
    rb(ax, 0.6, y_uw, 2.4, 0.8, C_W_UP,
       label=f"$W_{{UQ}}$\n($d_{{q,c}} \\times n_q \\cdot d_{{qk}}$)", fontsize=9)
    rb(ax, 3.4, y_uw, 2.4, 0.8, C_W_UP,
       label=f"$W_{{UK}}$\n($d_c \\times n_q \\cdot d_{{qk,nope}}$)", fontsize=8.5)
    rb(ax, 6.2, y_uw, 2.4, 0.8, C_W_UP,
       label=f"$W_{{UV}}$\n($d_c \\times n_q \\cdot d_v$)", fontsize=9)

    # c_Q → W_UQ (signal); cache → W_UK and W_UV (read all S+1)
    signal_arrow(ax, 2.1, 10.7, 1.8, 8.4)
    read_arrow(ax, 6.9, 9.2, 4.6, 8.4, label="  read $c_{KV}$", ldx=0.18, fontsize=8)
    read_arrow(ax, 6.9, 9.2, 7.4, 8.4)

    # Reconstructed Q, K, V (stacked = per-head)
    y_qkv = 6.0
    stacked_box(ax, 0.6, y_qkv, 2.4, 0.7, C_Q,
                label=f"$Q$  ($n_q \\times d_{{qk}}$)", n_stack=4, fontsize=9.5)
    stacked_box(ax, 3.4, y_qkv, 2.4, 0.7, C_K,
                label=f"$K$  ($n_q \\times d_{{qk}}$)", n_stack=4, fontsize=9.5)
    stacked_box(ax, 6.2, y_qkv, 2.4, 0.7, C_V,
                label=f"$V$  ($n_q \\times d_v$)", n_stack=4, fontsize=9.5)
    signal_arrow(ax, 1.8, 7.6, 1.8, 6.7)
    signal_arrow(ax, 4.6, 7.6, 4.6, 6.7)
    signal_arrow(ax, 7.4, 7.6, 7.4, 6.7)

    # Attention
    y_a = 4.2
    rb(ax, 2.0, y_a, 6.0, 1.0, C_ATTN,
       label="Attention(Q, K, V)\n"
             "score uses $q^{nope}\\!,k^{nope}$ + shared $q^{rope}\\!,k^{rope}$;  value uses V",
       fontsize=9.5, fw="bold")
    signal_arrow(ax, 1.8, 6.0, 3.0, 5.2)
    signal_arrow(ax, 4.6, 6.0, 5.0, 5.2)
    signal_arrow(ax, 7.4, 6.0, 7.0, 5.2)

    # Attention → W_O → output (signal)
    signal_arrow(ax, 5.0, 4.2, 5.0, 3.3)
    y_o = 2.6
    rb(ax, 3.0, y_o, 4.0, 0.7, C_W,
       label="$W_O$   ($n_q \\cdot d_v \\times H$)", fontsize=10)
    signal_arrow(ax, 5.0, 2.6, 5.0, 1.9)
    rb(ax, 2.5, 1.3, 5.0, 0.6, C_HIDDEN,
       label="Embedded token output:  o ∈ ℝ^H", fw="bold", fontsize=10)

    # Convention + absorbed-mode footer (two lines)
    ax.text(5, 0.85,
        "Single boxes = one tensor.  Stacked boxes = per-head intermediates.\n"
        "Production decode (TensorRT-LLM, SGLang) absorbs $W_{UK}, W_{UV}$ into "
        "$W_{UQ}, W_O$ → attention runs in $d_c$ space.",
        ha="center", va="center", fontsize=8.2, color="#546E7A", fontstyle="italic")

    add_legend(ax, y=0.25)

    fig.savefig("assets/attention_mla.svg", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)


if __name__ == "__main__":
    gen_mha()
    gen_gqa()
    gen_mla()
    print("Generated 3 SVGs in assets/:")
    print("  attention_mha.svg")
    print("  attention_gqa.svg")
    print("  attention_mla.svg")
