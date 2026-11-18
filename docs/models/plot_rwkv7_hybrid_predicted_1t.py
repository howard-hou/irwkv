# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTPUT_STEM = Path(__file__).with_name("rwkv7_hybrid_predicted_1t_2026-11-17")

D = 1e12
E = 0.3204
B = 115.1
beta = 0.2999
parent = {"N": 3_696_034_851.7830844, "A": 10.93, "alpha": 0.2527}
rwkv = {"A": 7.338, "alpha": 0.2387}
old_target = 7_970_691_112.963164
new_target = 35_167_595_824.517136


def loss(n, a, alpha):
    return E + a * n ** (-alpha) + B * D ** (-beta)


parent_loss = loss(parent["N"], parent["A"], parent["alpha"])
old_rwkv_loss = loss(old_target, rwkv["A"], rwkv["alpha"])
new_rwkv_loss = loss(new_target, rwkv["A"], rwkv["alpha"])
rounded_old_prediction = (old_rwkv_loss / parent_loss - 1) * 100
rounded_new_prediction = (new_rwkv_loss / parent_loss - 1) * 100

# Full bux fit, not a translation of the old interval or a combination of
# independently sampled coefficient bounds. Both runs use 100 bootstrap refits,
# seed 0, and the original noise model. The old run reproduces the public report.
old_prediction = -2.565007497855687
old_low = -3.6217319564929027
old_high = -1.4591247738510849
new_prediction = -4.988535210292433
new_low = -6.142262245202043
new_high = -3.9306281961023193

rows = [
    ("KDA, 3 of 4 layers", -4.68, -5.84, -4.28, "linear"),
    ("KDA, 1 global in 8", -4.85, -5.39, -4.57, "linear"),
    ("KDA, every other layer", -3.36, -3.89, -3.11, "linear"),
    ("Gated DeltaNet, 3 of 4", -4.16, -5.36, -3.31, "delta"),
    ("Gated DeltaNet-2, 3 of 4", -2.93, -3.68, -2.64, "delta"),
    (
        "RWKV-7, 3 of 4 — irwkv · 35.17B",
        new_prediction,
        new_low,
        new_high,
        "irwkv",
    ),
    (
        "RWKV-7, 3 of 4 — bux · 7.97B",
        old_prediction,
        old_low,
        old_high,
        "bux",
    ),
    ("Mamba-2, 3 of 4", -3.64, -4.71, -2.67, "mamba"),
    ("SWA 512, 3 of 4", -2.78, -3.72, -1.97, "swa"),
    ("SWA 128, 3 of 4", -3.63, -4.08, -3.40, "swa"),
    ("SWA 2048, 3 of 4", -2.58, -3.07, -2.42, "swa"),
    ("SWA 512, every other layer", -2.40, -2.88, -2.28, "swa"),
    ("SWA 512, 1 global in 8", -3.87, -4.33, -3.64, "swa"),
]

colors = {
    "linear": "#5879B9",
    "delta": "#8274C8",
    "irwkv": "#07815D",
    "bux": "#D88A32",
    "mamba": "#A071B5",
    "swa": "#94A6BA",
}

for candidate in ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]:
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=candidate
        ).get_name()
        break
plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})

fig, ax = plt.subplots(figsize=(15.5, 11.2), facecolor="#F7F8FB")
ax.set_facecolor("#F7F8FB")
positions = list(range(len(rows)))
irwkv_index = next(i for i, row in enumerate(rows) if row[4] == "irwkv")
ax.axhspan(
    irwkv_index - 0.48,
    irwkv_index + 0.48,
    color="#DDF2E9",
    alpha=0.75,
    zorder=0,
)
ax.axvline(0, color="#7D8794", linewidth=1.2, zorder=1)

for index, (label, value, low, high, group) in enumerate(rows):
    color = colors[group]
    if low is not None:
        ax.hlines(index, low, high, color=color, linewidth=3.2, alpha=0.55, zorder=2)
        ax.vlines([low, high], index - 0.11, index + 0.11, color=color, linewidth=1.4)
    marker = "D" if group == "irwkv" else "o"
    size = 105 if group == "irwkv" else 62
    ax.scatter(
        value,
        index,
        s=size,
        marker=marker,
        color=color,
        edgecolor="#F7F8FB",
        linewidth=1.0,
        zorder=4,
    )
    ax.annotate(
        f"{value:.2f}%".replace("-", "−"),
        (value, index),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=12 if group == "irwkv" else 10.8,
        fontweight="bold" if group == "irwkv" else "normal",
        color="#056649" if group == "irwkv" else "#344054",
        bbox={"facecolor": "#F7F8FB", "edgecolor": "none", "pad": 0.7},
        zorder=5,
    )

ax.set_yticks(positions)
ax.set_yticklabels([row[0] for row in rows], fontsize=11.3, color="#344054")
ax.get_yticklabels()[irwkv_index].set_fontweight("bold")
ax.get_yticklabels()[irwkv_index].set_color("#056649")
ax.invert_yaxis()
ax.set_xlim(-6.6, 0.65)
ax.set_xticks(range(-6, 1))
ax.set_xlabel(
    "Predicted change in loss vs the 2021 reference at its 3.70B target (%) "
    "— left is better",
    fontsize=12.2,
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
    0.953,
    "Predicted: equal serving cost, 1T tokens",
    fontsize=25,
    fontweight="bold",
    color="#172B3A",
)
fig.text(
    0.055,
    0.915,
    "Hybrids: SWA vs Linear Attention — irwkv at 35.17B active",
    fontsize=14,
    color="#475467",
)
fig.text(
    0.055,
    0.884,
    "Same bux scaling law and parent target; only RWKV-7's measured serving "
    "target changes",
    fontsize=11.5,
    color="#667085",
)
fig.text(
    0.055,
    0.079,
    (
        f"irwkv: {new_prediction:.2f}% [{new_low:.2f}%, {new_high:.2f}%], "
        f"an additional {old_prediction - new_prediction:.2f} percentage points of "
        "predicted loss reduction vs bux RWKV-7."
    ).replace("-", "−"),
    fontsize=10.6,
    color="#475467",
)
fig.text(
    0.055,
    0.048,
    "100 bootstrap refits (seed 0, 5th–95th percentile). The old-target run "
    "reproduces the published fits; other rows are unchanged.",
    fontsize=10,
    color="#7A8491",
)
fig.text(
    0.055,
    0.022,
    "Scaling-law extrapolation, not measured model quality. irwkv serving "
    "measurements used dummy weights.",
    fontsize=10,
    color="#7A8491",
)

fig.subplots_adjust(left=0.29, right=0.955, top=0.845, bottom=0.16)
fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=180, facecolor=fig.get_facecolor())
fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor=fig.get_facecolor())

result = {
    "method": (
        "Full bux objective fit; replace only RWKV-7's FP8 serving target, then "
        "refit 100 bootstrap replicates with the original noise model and seed 0"
    ),
    "formula": "L(N,D) = E + A * N**(-alpha) + B * D**(-beta)",
    "D": D,
    "E": E,
    "B": B,
    "beta": beta,
    "parent": {**parent, "loss_at_1t_from_rounded_coefficients": parent_loss},
    "rwkv7": rwkv,
    "bootstrap": {
        "replicates": 100,
        "seed": 0,
        "percentiles": [5, 95],
        "converged": 100,
        "noise": "objective 0.33% own and 0.69% by seed; 415 dof",
    },
    "old": {
        "target_active": old_target,
        "loss_at_1t_from_rounded_coefficients": old_rwkv_loss,
        "predicted_change_percent": old_prediction,
        "bootstrap_range_percent": [old_low, old_high],
        "predicted_change_percent_from_rounded_coefficients": rounded_old_prediction,
        "published_predicted_change_percent": -2.57,
        "published_range_percent": [-3.62, -1.46],
    },
    "new": {
        "implementation": "irwkv",
        "target_active": new_target,
        "loss_at_1t_from_rounded_coefficients": new_rwkv_loss,
        "predicted_change_percent": new_prediction,
        "bootstrap_range_percent": [new_low, new_high],
        "predicted_change_percent_from_rounded_coefficients": rounded_new_prediction,
        "additional_reduction_percentage_points_vs_bux": old_prediction
        - new_prediction,
    },
    "validation": {
        "old_rwkv7_percent": [old_prediction, old_low, old_high],
        "old_kda_percent": [
            -4.684443215249701,
            -5.839171155490227,
            -4.2772685095013605,
        ],
        "old_mamba2_percent": [
            -3.643578981625961,
            -4.708880299730856,
            -2.6698393461968863,
        ],
        "other_architecture_rows_identical_between_old_and_new_runs": True,
    },
    "caveats": [
        "Scaling-law extrapolation, not measured quality.",
        "irwkv serving measurements used dummy weights.",
        (
            "The bootstrap range is the 5th–95th percentile of 100 refits, "
            "not a 95% confidence interval."
        ),
        "Published coefficients are rounded and are retained only as a cross-check.",
    ],
}
OUTPUT_STEM.with_suffix(".json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"parent_loss={parent_loss:.12f}")
print(f"old_prediction={old_prediction:.9f}% [{old_low:.9f}, {old_high:.9f}]")
print(f"new_prediction={new_prediction:.9f}% [{new_low:.9f}, {new_high:.9f}]")
print(f"rounded_old_prediction={rounded_old_prediction:.9f}%")
print(f"rounded_new_prediction={rounded_new_prediction:.9f}%")
print(OUTPUT_STEM.with_suffix(".png"))
print(OUTPUT_STEM.with_suffix(".svg"))
print(OUTPUT_STEM.with_suffix(".json"))
