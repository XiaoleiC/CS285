"""Generate training-curve PDFs for the HW5 report."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def read_eval(path: Path):
    steps, success = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            success.append(float(row["eval/success_rate"]) * 100.0)
    return np.array(steps), np.array(success)


def find_run(qdir: str, env_substr: str, alpha: float, agent_prefix: str):
    """Locate the longest matching run (largest final step in eval.csv).

    Some envs have multiple runs at the same alpha (e.g. an aborted partial run
    plus the full 1M run); we want the latter.
    """
    candidates = []
    for d in sorted((EXP / qdir).iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if env_substr not in name or agent_prefix not in name:
            continue
        if not name.endswith(f"_a{alpha}"):
            continue
        eval_csv = d / "eval.csv"
        if not eval_csv.exists():
            continue
        steps, _ = read_eval(eval_csv)
        candidates.append((steps.max() if len(steps) else -1, d))
    if not candidates:
        raise FileNotFoundError(f"no run match for {qdir}/{agent_prefix}/{env_substr}/a={alpha}")
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def fmt_steps(x, _pos):
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    if x >= 1e3:
        return f"{int(x/1e3)}K"
    return str(int(x))


def style_axes(ax, title, xmax):
    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate (%)")
    ax.set_title(title)
    ax.set_ylim(-3, 105)
    ax.set_xlim(0, xmax * 1.02)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_steps))
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.85")


# ---- Best-run plots: 2-panel (cube | antsoccer) ----

def best_two_panel(qdir, agent_prefix, cube_alpha, soccer_alpha, suptitle, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, env, alpha, label in [
        (axes[0], "cube-single", cube_alpha, "cube-single"),
        (axes[1], "antsoccer-arena", soccer_alpha, "antsoccer-arena"),
    ]:
        run = find_run(qdir, env, alpha, agent_prefix)
        x, y = read_eval(run / "eval.csv")
        ax.plot(x, y, marker="o", color="#1f77b4", label=fr"$\alpha={alpha:g}$")
        ax.fill_between(x, y, alpha=0.10, color="#1f77b4")
        style_axes(ax, label, x.max())
    fig.suptitle(suptitle, y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)


# ---- Alpha-sweep plots on cube-single ----

def alpha_sweep(qdir, agent_prefix, alphas, title, out_name):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(sorted(alphas)):
        run = find_run(qdir, "cube-single", a, agent_prefix)
        x, y = read_eval(run / "eval.csv")
        ax.plot(x, y, marker="o", color=cmap(i / max(1, len(alphas) - 1)),
                label=fr"$\alpha={a:g}$")
    style_axes(ax, title, x.max())
    fig.tight_layout()
    fig.savefig(OUT / out_name)
    plt.close(fig)


def main():
    # Q1 SAC+BC
    best_two_panel(
        "q1", "sacbc", cube_alpha=100.0, soccer_alpha=10.0,
        suptitle="SAC+BC: best-performing agents",
        out_name="q1_best.pdf",
    )
    alpha_sweep(
        "q1", "sacbc", alphas=[30.0, 100.0, 300.0, 1000.0],
        title=r"SAC+BC on cube-single: $\alpha$ sweep",
        out_name="q1_alpha_sweep.pdf",
    )

    # Q2 IQL
    best_two_panel(
        "q2", "iql", cube_alpha=1.0, soccer_alpha=10.0,
        suptitle=r"IQL ($\tau=0.9$): best-performing agents",
        out_name="q2_best.pdf",
    )
    alpha_sweep(
        "q2", "iql", alphas=[1.0, 3.0, 10.0],
        title=r"IQL on cube-single: $\alpha$ sweep ($\tau=0.9$)",
        out_name="q2_alpha_sweep.pdf",
    )

    # Q3 FQL
    best_two_panel(
        "q3", "fql", cube_alpha=300.0, soccer_alpha=10.0,
        suptitle="FQL: best-performing agents",
        out_name="q3_best.pdf",
    )

    print("wrote:", *(p.name for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    main()
