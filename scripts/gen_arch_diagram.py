#!/usr/bin/env python3
"""Generate a high-level architecture diagram of the modeled inference cluster.

Side-by-side prefill/decode clusters with serving framework and distributed
KV cache layers.  Each block is annotated with the modeling doc and/or core
Python module that governs it.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 9.7))
ax.set_xlim(0, 16)
ax.set_ylim(2.3, 12)
ax.set_aspect("equal")
ax.axis("off")

# ── Colors ──────────────────────────────────────────────────────────
C_PREFILL = "#B3D9FF"
C_DECODE = "#FFE0B2"
C_SWITCH = "#E8F5E9"
C_DEVICE = "#FFF9C4"
C_HBM = "#E1BEE7"
C_SRAM = "#B2EBF2"
C_GPU = "#FFCDD2"
C_INTERCO = "#CFD8DC"
C_KV = "#F3E5F5"
C_FRAME = "#ECEFF1"
C_TEXT = "#212121"
C_REF = "#2E7D32"

def rounded_box(x, y, w, h, color, label="", fontsize=12, lw=1.5, ec="black",
                alpha=1.0, fontstyle="normal", fontweight="normal", zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=C_TEXT, fontweight=fontweight, fontstyle=fontstyle, zorder=zorder + 1)

def ref_text(x, y, text, fontsize=8.5, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=C_REF, family="monospace", fontstyle="italic", zorder=10)

# ═══════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════
ax.text(6.825, 11.65, "LLM Inference Cluster — Modeled Architecture", ha="center", va="center",
        fontsize=18, fontweight="bold", color=C_TEXT)

# ═══════════════════════════════════════════════════════════════════
# Serving Framework (top bar)
# ═══════════════════════════════════════════════════════════════════
rounded_box(0.5, 10.7, 12.65, 0.65, C_FRAME, lw=2.0, ec="#546E7A")
ax.text(6.825, 11.12, "Serving Framework", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#37474F")
ax.text(6.825, 10.83, "continuous batching · request scheduler · tokenizer · KV-aware router",
        ha="center", va="center", fontsize=10, color="#546E7A", fontstyle="italic")
ref_text(6.825, 10.52, "modeling/framework.md", fontsize=9)

# Arrow down
ax.annotate("", xy=(6.825, 10.2), xytext=(6.825, 10.47),
            arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=2.0))

# ═══════════════════════════════════════════════════════════════════
# Prefill Cluster (left)
# ═══════════════════════════════════════════════════════════════════
rounded_box(0.5, 5.4, 3.5, 4.75, C_PREFILL, lw=2.0, ec="#1976D2", alpha=0.35, zorder=1)
ax.text(2.25, 9.85, "Prefill Cluster", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#1565C0")
ax.text(2.25, 9.55, "(compute-heavy, large S)", ha="center", va="center",
        fontsize=10, color="#1565C0", fontstyle="italic")
ref_text(2.25, 9.28, "modeling/prefill.md · core/prefill_model.py", fontsize=7.5)

# Prefill devices — 2x2, tight
for dx, dy in [(0.8, 7.85), (2.35, 7.85), (0.8, 6.75), (2.35, 6.75)]:
    rounded_box(dx, dy, 1.35, 0.85, C_DEVICE, label="Device", fontsize=11, lw=1.0, ec="#F9A825")

# Prefill scale-up/out network
rounded_box(0.8, 5.7, 2.9, 0.42, C_SWITCH, label="Scale-up/out Network", fontsize=10, lw=1.2, ec="#2E7D32")

# ═══════════════════════════════════════════════════════════════════
# Decode Cluster (right)
# ═══════════════════════════════════════════════════════════════════
rounded_box(5.5, 5.4, 7.65, 4.75, C_DECODE, lw=2.0, ec="#E65100", alpha=0.35, zorder=1)
ax.text(9.325, 9.85, "Decode Cluster", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#BF360C")
ax.text(9.325, 9.55, "(memory-bound, autoregressive)", ha="center", va="center",
        fontsize=10, color="#BF360C", fontstyle="italic")
ref_text(9.325, 9.28, "modeling/decode.md · core/decode_model.py", fontsize=7.5)

# Decode devices — 4 top, 4 bottom; each device is a 3-tier memory stack:
# GPU compute (top) · SRAM fast tier (middle) · HBM/DRAM slow tier (bottom).
# The SRAM tier is what SRAM-augmented architectures (Groq, d-Matrix) expose
# at multi-TB/s; on a conventional GPU it represents on-die L1/L2.
dw, dh = 1.55, 0.85
top_y, bot_y = 7.85, 6.75
x_starts = [5.85, 7.65, 9.45, 11.25]
band_h = 0.245
gap = 0.025
for row_y in [top_y, bot_y]:
    for dx in x_starts:
        rounded_box(dx, row_y, dw, dh, C_DEVICE, lw=1.0, ec="#F9A825")
        gpu_y  = row_y + dh - gap - band_h
        sram_y = gpu_y - gap - band_h
        hbm_y  = sram_y - gap - band_h
        rounded_box(dx + 0.04, gpu_y,  dw - 0.08, band_h, C_GPU,  label="GPU",     fontsize=8.5, lw=0.8, ec="#C62828")
        rounded_box(dx + 0.04, sram_y, dw - 0.08, band_h, C_SRAM, label="SRAM",    fontsize=8.5, lw=0.8, ec="#00838F")
        rounded_box(dx + 0.04, hbm_y,  dw - 0.08, band_h, C_HBM,  label="HBM/DRAM", fontsize=8.5, lw=0.8, ec="#7B1FA2")

# Device-level references
ref_text(9.325, 9.0, "core/{memory_model, decode_model, memory_placement}.py · sram.md · primitives/", fontsize=7.5)

# Decode scale-up/out network — shown as the canonical hierarchical chain
# (innermost first). Single-tier deployments collapse to one element.
# All five parallelism dimensions (TP/EP/SP intra-domain, PP cross-stage,
# DP across replicas) ride this one fabric model.
rounded_box(5.85, 5.7, 6.95, 0.42, C_SWITCH, lw=1.5, ec="#2E7D32")
ax.text(9.325, 5.91, "Scale-up/out Network  (TP / EP / SP / PP / DP)", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#1B5E20")
ax.text(9.325, 5.55, "hierarchical α-β:  pair_mesh / NVLink → PCIe / fat-tree → ethernet  · INC short-circuit on sharp_class / hw_a2a tiers",
        ha="center", va="center", fontsize=8.5, color="#1B5E20", fontstyle="italic")
ref_text(9.325, 5.25, "modeling/collectives/00_summary.md · core/{decode_model, collective_algo_opt}.py · primitives/dispatch.py", fontsize=7.5)

# ═══════════════════════════════════════════════════════════════════
# KV Transfer interconnect (between clusters)
# ═══════════════════════════════════════════════════════════════════
ax.annotate("", xy=(5.35, 7.6), xytext=(4.15, 7.6),
            arrowprops=dict(arrowstyle="<->", color="#37474F", lw=2.5))
rounded_box(4.15, 7.0, 1.2, 0.45, C_INTERCO, lw=1.2, ec="#37474F")
ax.text(4.75, 7.22, "KV Transfer", ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="#263238")
ref_text(4.75, 6.72, "modeling/e2e.md", fontsize=7.5)

# ═══════════════════════════════════════════════════════════════════
# Distributed KV Cache (bottom bar)
# ═══════════════════════════════════════════════════════════════════
ax.annotate("", xy=(2.25, 4.55), xytext=(2.25, 5.15),
            arrowprops=dict(arrowstyle="-|>", color="#6A1B9A", lw=2.0))
ax.annotate("", xy=(9.325, 4.55), xytext=(9.325, 5.15),
            arrowprops=dict(arrowstyle="-|>", color="#6A1B9A", lw=2.0))

rounded_box(0.5, 3.85, 12.65, 0.65, C_KV, lw=2.0, ec="#6A1B9A")
ax.text(6.825, 4.28, "Distributed KV Cache", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#4A148C")
ax.text(6.825, 3.98, "HBM + host DRAM + SSD tiers · paged blocks · fragmentation model",
        ha="center", va="center", fontsize=10, color="#6A1B9A", fontstyle="italic")
ref_text(6.825, 3.65, "modeling/kv.md · core/kv_paging_model.py", fontsize=9)

# ═══════════════════════════════════════════════════════════════════
# Legend at bottom
# ═══════════════════════════════════════════════════════════════════
legend_y = 3.05
legend_items = [
    (C_GPU,    "#C62828", "GPU Compute"),
    (C_SRAM,   "#00838F", "SRAM (fast tier)"),
    (C_HBM,    "#7B1FA2", "HBM/DRAM (slow tier)"),
    (C_SWITCH, "#2E7D32", "Scale-up/out Network"),
    (C_INTERCO, "#37474F", "Disagg KV Transfer"),
    (C_KV,     "#6A1B9A", "Distributed KV Cache"),
]
for i, (fc, ec, label) in enumerate(legend_items):
    lx = 0.675 + i * 2.0
    rounded_box(lx, legend_y, 0.4, 0.25, fc, lw=0.8, ec=ec)
    ax.text(lx + 0.5, legend_y + 0.12, label, ha="left", va="center", fontsize=9, color=C_TEXT)

ax.text(6.825, 2.6, "Green italic labels reference modeling docs and core Python modules",
        ha="center", va="center", fontsize=10, color=C_REF, fontstyle="italic")

fig.tight_layout()
fig.savefig("assets/cluster_architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved → assets/cluster_architecture.png")
plt.close()
