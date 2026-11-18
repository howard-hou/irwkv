# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT_STEM = Path(__file__).with_name("rwkv7_hybrid_serving_comparison_2026-11-17")

lower_active = 33_695_827_968
lower_sessions = 2071.2
upper_active = 43_724_697_600
upper_sessions = 1321.6
target_sessions = 1924.0
fraction = (math.log(target_sessions) - math.log(lower_sessions)) / (
    math.log(upper_sessions) - math.log(lower_sessions)
)
irwkv_active = lower_active * math.exp(fraction * math.log(upper_active / lower_active))
irwkv_billions = irwkv_active / 1e9
assert math.isclose(irwkv_active, 35_167_595_824.517105, rel_tol=1e-12)

rows = [
    ("parent / 2021 reference", 3.7, "reference"),
    ("KDA, 3 of 4 layers", 13.3, "linear"),
    ("KDA, 1 global in 8", 24.5, "linear"),
    ("KDA, every other layer", 9.6, "linear"),
    ("Gated DeltaNet, 3 of 4", 15.2, "delta"),
    ("Gated DeltaNet-2, 3 of 4", 5.8, "delta"),
    ("RWKV-7, 3 of 4 — irwkv", irwkv_billions, "irwkv"),
    ("RWKV-7, 3 of 4 — bux", 7.97, "bux"),
    ("Mamba-2, 3 of 4", 14.4, "mamba"),
    ("SWA 512, 3 of 4", 13.6, "swa"),
    ("SWA 128, 3 of 4", 17.9, "swa"),
    ("SWA 2048, 3 of 4", 14.9, "swa"),
    ("SWA 512, every other layer", 9.5, "swa"),
    ("SWA 512, 1 global in 8", 26.8, "swa"),
]

colors = {
    "reference": "#98A2B3",
    "linear": "#5879B9",
    "delta": "#8274C8",
    "irwkv": "#07815D",
    "bux": "#D88A32",
    "mamba": "#A071B5",
    "swa": "#A8B6C7",
}

font_candidates = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
for candidate in font_candidates:
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=candidate
        ).get_name()
        break

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
    }
)

fig, ax = plt.subplots(figsize=(15.5, 11.5), facecolor="#F7F8FB")
ax.set_facecolor("#F7F8FB")
positions = list(range(len(rows)))
values = [row[1] for row in rows]
bar_colors = [colors[row[2]] for row in rows]

irwkv_index = next(i for i, row in enumerate(rows) if row[2] == "irwkv")
ax.axhspan(
    irwkv_index - 0.48,
    irwkv_index + 0.48,
    color="#DDF2E9",
    alpha=0.75,
    zorder=0,
)
ax.axvline(
    3.7,
    color="#7D8794",
    linewidth=1.25,
    linestyle=(0, (4, 4)),
    zorder=1,
)

bars = ax.barh(
    positions,
    values,
    height=0.61,
    color=bar_colors,
    edgecolor="none",
    zorder=2,
)
bars[irwkv_index].set_height(0.70)

for index, (bar, row) in enumerate(zip(bars, rows)):
    value = row[1]
    label = f"{value:.2f}" if row[2] == "irwkv" else f"{value:.1f}"
    ax.text(
        value + 0.45,
        bar.get_y() + bar.get_height() / 2,
        label,
        va="center",
        ha="left",
        fontsize=13 if row[2] == "irwkv" else 11.5,
        fontweight="bold" if row[2] == "irwkv" else "normal",
        color="#056649" if row[2] == "irwkv" else "#344054",
        zorder=3,
    )
    if row[2] == "irwkv":
        ax.text(
            value + 2.90,
            bar.get_y() + bar.get_height() / 2,
            "NEW",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#07815D",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#DDF2E9",
                "edgecolor": "none",
            },
        )

ax.set_yticks(positions)
ax.set_yticklabels([row[0] for row in rows], fontsize=11.5, color="#344054")
ax.get_yticklabels()[irwkv_index].set_fontweight("bold")
ax.get_yticklabels()[irwkv_index].set_color("#056649")
ax.invert_yaxis()
ax.set_xlim(0, 40)
ax.set_xticks(range(0, 41, 5))
ax.set_xlabel(
    "Active parameters at 1,924 sessions/h (billions)",
    fontsize=12.5,
    color="#475467",
    labelpad=13,
)
ax.xaxis.grid(True, color="#DDE2E9", linewidth=0.9, linestyle=(0, (2, 4)))
ax.set_axisbelow(True)
ax.tick_params(axis="x", colors="#667085", labelsize=11, length=0, pad=8)
ax.tick_params(axis="y", length=0, pad=12)
for spine in ax.spines.values():
    spine.set_visible(False)

fig.text(
    0.055,
    0.955,
    "Hybrids: SWA vs Linear Attention",
    fontsize=25,
    fontweight="bold",
    color="#172B3A",
)
fig.text(
    0.055,
    0.918,
    "FP8 serving on B300 — same workload and latency SLO across architectures",
    fontsize=13.5,
    color="#475467",
)
fig.text(
    0.055,
    0.888,
    "Each point is interpolated in log sessions/h × log active-parameter space; "
    "p50 TPOT ≤ 30 ms",
    fontsize=11.5,
    color="#667085",
)
fig.text(
    0.055,
    0.075,
    "irwkv: 35.17B active, interpolated between measured 33.70B and 43.72B points. "
    "bux RWKV-7 retained as a separate implementation reference.",
    fontsize=10.5,
    color="#475467",
)
fig.text(
    0.055,
    0.047,
    "Throughput only: irwkv measurements used dummy weights. "
    "Other architecture targets: bux-timeline / Every LLM Architecture Change.",
    fontsize=10,
    color="#7A8491",
)

fig.subplots_adjust(left=0.275, right=0.955, top=0.845, bottom=0.135)
fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=180, facecolor=fig.get_facecolor())
fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor=fig.get_facecolor())
print(f"irwkv_active={irwkv_active:.6f}")
print(f"irwkv_billions={irwkv_billions:.9f}")
print(OUTPUT_STEM.with_suffix(".png"))
print(OUTPUT_STEM.with_suffix(".svg"))
