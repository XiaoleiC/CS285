from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
WANDB_ROOT = ROOT / "wandb_export"
FIG_ROOT = ROOT / "report" / "figures"


def load_history(run_dir: str) -> list[dict]:
    path = WANDB_ROOT / run_dir / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_summary(run_dir: str) -> dict:
    return json.loads((WANDB_ROOT / run_dir / "summary.json").read_text())


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(row["_step"], row[key]) for row in rows if key in row and row[key] is not None]
    if not pts:
        return np.array([]), np.array([])
    xs, ys = zip(*pts)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) == 0:
        return y
    if window <= 1 or len(y) < window:
        return y
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fbfbfd",
            "grid.color": "#d9dce3",
            "grid.alpha": 0.6,
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def make_math_figure() -> None:
    grpo_rows = load_history("math_hard_grpo__zgbt75w1")
    rein_rows = load_history("math_hard_reinforce_rerun3__tuj2zpp2")
    grpo_summary = load_summary("math_hard_grpo__zgbt75w1")
    rein_summary = load_summary("math_hard_reinforce_rerun3__tuj2zpp2")

    blue = "#1565C0"
    red = "#C62828"
    gold = "#F9A825"

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, 4.4),
        gridspec_kw={"width_ratios": [1.05, 1.45, 1.0]},
        constrained_layout=True,
    )

    # Panel A: eval exact over the first 200 iterations.
    ax = axes[0]
    xg, yg = series(grpo_rows, "eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser")
    xr, yr = series(rein_rows, "eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser")
    xg = xg[xg <= 200]
    yg = yg[: len(xg)]
    xr = xr[xr <= 201]
    yr = yr[: len(xr)]
    ax.plot(xg, yg * 100, color=blue, marker="o", lw=2.4, ms=6, label="GRPO")
    ax.plot(xr, yr * 100, color=red, marker="o", lw=2.4, ms=6, label="GR-REINFORCE")
    ax.scatter([199], [33.3984375], color=blue, s=55, zorder=3)
    ax.scatter([199], [29.6875], color=red, s=55, zorder=3)
    ax.annotate("33.4%", (199, 33.3984375), xytext=(140, 35.1), color=blue, fontsize=9)
    ax.annotate("29.7%", (199, 29.6875), xytext=(126, 27.7), color=red, fontsize=9)
    ax.set_title("Held-out Boxed Exact Match")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-3, 205)
    ax.set_ylim(20, 39.5)
    add_panel_label(ax, "A")

    # Panel B: training reward, first 200 steps, raw + smoothed.
    ax = axes[1]
    xg, yg = series(grpo_rows, "rollout/mean_total_reward_across_all_completions_in_batch_and_groups")
    xr, yr = series(rein_rows, "rollout/mean_total_reward_across_all_completions_in_batch_and_groups")
    mask_g = xg <= 200
    mask_r = xr <= 200
    xg, yg = xg[mask_g], yg[mask_g]
    xr, yr = xr[mask_r], yr[mask_r]
    ax.plot(xg, yg, color=blue, alpha=0.18, lw=1.2)
    ax.plot(xr, yr, color=red, alpha=0.18, lw=1.2)
    ax.plot(xg, moving_average(yg, 11), color=blue, lw=2.6)
    ax.plot(xr, moving_average(yr, 11), color=red, lw=2.6)
    ax.set_title("Sampled Rollout Reward")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward")
    ax.set_xlim(0, 200)
    ax.set_ylim(0.0, 0.72)
    ax.text(0.02, 0.95, "opaque = smoothed\nfaint = raw", transform=ax.transAxes, va="top", fontsize=8, color="#555")
    add_panel_label(ax, "B")

    # Panel C: final greedy behavior.
    ax = axes[2]
    labels = ["Boxed exact", "Relaxed exact", "Contains $\\backslash$boxed"]
    grpo_vals = np.array(
        [
            grpo_summary["eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser"],
            grpo_summary["eval/math_hard_test_subset_split_fraction_exact_match_using_relaxed_last_number_parser"],
            grpo_summary["eval/math_hard_test_subset_split_fraction_completions_containing_boxed_answer_pattern"],
        ]
    )
    rein_vals = np.array(
        [
            rein_summary["eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser"],
            rein_summary["eval/math_hard_test_subset_split_fraction_exact_match_using_relaxed_last_number_parser"],
            rein_summary["eval/math_hard_test_subset_split_fraction_completions_containing_boxed_answer_pattern"],
        ]
    )
    pos = np.arange(len(labels))
    w = 0.35
    ax.bar(pos - w / 2, grpo_vals * 100, width=w, color=blue, alpha=0.92, label="GRPO")
    ax.bar(pos + w / 2, rein_vals * 100, width=w, color=red, alpha=0.92, label="GR-REINFORCE")
    for i, val in enumerate(grpo_vals):
        ax.text(i - w / 2, val * 100 + 1.3, f"{val*100:.1f}", ha="center", va="bottom", fontsize=8, color=blue)
    for i, val in enumerate(rein_vals):
        ax.text(i + w / 2, val * 100 + 1.3, f"{val*100:.1f}", ha="center", va="bottom", fontsize=8, color=red)
    ax.set_xticks(pos, labels)
    ax.set_title("Final Greedy Behavior")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    add_panel_label(ax, "C")

    handles = [
        plt.Line2D([0], [0], color=blue, lw=3, marker="o", label="GRPO"),
        plt.Line2D([0], [0], color=red, lw=3, marker="o", label="GR-REINFORCE"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Math-Hard Comparison: GRPO vs. GR-REINFORCE", y=1.08, fontweight="bold")
    fig.savefig(FIG_ROOT / "math_compare.pdf", bbox_inches="tight")
    plt.close(fig)


def make_ablation_figure() -> None:
    runs = {
        "default": ("format_copy_grpo__lfhm57r3", "#1f77b4"),
        "ppo=1": ("format_copy_grpo_ppo1__deco1vdm", "#9467bd"),
        "kl=0.01": ("format_copy_grpo_kl001__djxovs5j", "#2ca02c"),
        "kl=0.20": ("format_copy_grpo_kl020__xvxxhizh", "#d62728"),
        "clip=0.10": ("format_copy_grpo_clip010__6rohfdpt", "#ff7f0e"),
        "mb=24, ga=1": ("format_copy_grpo_mb24_ga1__x0r5c2wu", "#17becf"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6), constrained_layout=True)
    ax_reward, ax_kl, ax_clip, ax_grad = axes.flatten()

    max_grad = {}

    for label, (run_dir, color) in runs.items():
        rows = load_history(run_dir)
        x, reward = series(rows, "rollout/mean_total_reward_across_all_completions_in_batch_and_groups")
        _, kl = series(rows, "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches")
        _, clip = series(rows, "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches")
        _, grad = series(rows, "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps")
        max_grad[label] = float(np.max(grad)) if len(grad) else 0.0

        ax_reward.plot(x, moving_average(reward, 5), lw=2.2, color=color, label=label)
        ax_kl.plot(x[: len(kl)], moving_average(kl, 5), lw=2.0, color=color)
        ax_clip.plot(x[: len(clip)], moving_average(clip * 100, 5), lw=2.0, color=color)

    # Reward panel.
    ax_reward.set_title("Training Reward on format_copy")
    ax_reward.set_xlabel("Training step")
    ax_reward.set_ylabel("Mean reward")
    ax_reward.set_xlim(0, 50)
    ax_reward.set_ylim(0.0, 1.35)
    add_panel_label(ax_reward, "A")

    # KL panel.
    ax_kl.set_title("Approximate KL to Reference")
    ax_kl.set_xlabel("Training step")
    ax_kl.set_ylabel("Approximate KL")
    ax_kl.set_xlim(0, 50)
    ax_kl.set_ylim(0.0, 0.52)
    add_panel_label(ax_kl, "B")

    # Clipfrac panel.
    ax_clip.set_title("Fraction of Tokens Clipped")
    ax_clip.set_xlabel("Training step")
    ax_clip.set_ylabel("Clip fraction (%)")
    ax_clip.set_xlim(0, 50)
    ax_clip.set_ylim(0.0, 6.2)
    add_panel_label(ax_clip, "C")

    # Max gradient-norm bar chart.
    labels = list(runs.keys())
    values = [max_grad[label] for label in labels]
    colors = [runs[label][1] for label in labels]
    bars = ax_grad.bar(labels, values, color=colors, alpha=0.9)
    ax_grad.set_title("Peak Gradient Norm During Training")
    ax_grad.set_ylabel("Max clipped grad norm")
    ax_grad.set_ylim(0.0, 4.1)
    ax_grad.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax_grad.text(bar.get_x() + bar.get_width() / 2, value + 0.07, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    add_panel_label(ax_grad, "D")

    fig.legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handles=[plt.Line2D([0], [0], color=color, lw=3, label=label) for label, (_, color) in runs.items()],
    )
    fig.suptitle("GRPO Ablations on format_copy", y=1.05, fontweight="bold")
    fig.savefig(FIG_ROOT / "ablation_compare.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    style()
    make_math_figure()
    make_ablation_figure()


if __name__ == "__main__":
    main()
