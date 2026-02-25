"""Generate all plots for hw2 report."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

EXP_DIR = "exp"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def load_csv(dirname):
    """Load log.csv from an experiment directory, handling nested dirs."""
    path = os.path.join(EXP_DIR, dirname, "log.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Try nested
    for root, dirs, files in os.walk(os.path.join(EXP_DIR, dirname)):
        if "log.csv" in files:
            return pd.read_csv(os.path.join(root, "log.csv"))
    raise FileNotFoundError(f"No log.csv in {dirname}")

def find_dir(prefix):
    """Find experiment directory matching prefix."""
    for d in sorted(os.listdir(EXP_DIR)):
        if d.startswith(prefix):
            return d
    raise FileNotFoundError(f"No dir matching {prefix}")

# ============================================================
# Figure 1: CartPole small batch (b=1000)
# ============================================================
plt.figure(figsize=(10, 6))
small_batch = [
    ("cartpole_sd", "No RTG, No NA"),
    ("cartpole_rtg_sd", "RTG, No NA"),
    ("cartpole_na_sd", "No RTG, NA"),
    ("cartpole_rtg_na_sd", "RTG, NA"),
]
for prefix, label in small_batch:
    d = find_dir(f"CartPole-v0_{prefix}")
    df = load_csv(d)
    plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=label)
plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
plt.ylabel("Eval Average Return")
plt.title("CartPole-v0: Small Batch (b=1000)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cartpole_small_batch.png"), dpi=150)
plt.close()

# ============================================================
# Figure 2: CartPole large batch (b=4000)
# ============================================================
plt.figure(figsize=(10, 6))
large_batch = [
    ("cartpole_lb_sd", "No RTG, No NA"),
    ("cartpole_lb_rtg_sd", "RTG, No NA"),
    ("cartpole_lb_na_sd", "No RTG, NA"),
    ("cartpole_lb_rtg_na_sd", "RTG, NA"),
]
for prefix, label in large_batch:
    d = find_dir(f"CartPole-v0_{prefix}")
    df = load_csv(d)
    plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=label)
plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
plt.ylabel("Eval Average Return")
plt.title("CartPole-v0: Large Batch (b=4000)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cartpole_large_batch.png"), dpi=150)
plt.close()

# ============================================================
# Figure 3: HalfCheetah Baseline Loss
# ============================================================
plt.figure(figsize=(10, 6))
cheetah_configs = [
    ("HalfCheetah-v4_cheetah_baseline_sd1", "Baseline (bgs=5, blr=0.01)"),
    ("HalfCheetah-v4_cheetah_baseline_decreased_bgs_sd1", "Decreased bgs (bgs=3)"),
    ("HalfCheetah-v4_cheetah_baseline_decreased_bgr_sd1", "Decreased blr (blr=0.001)"),
]
for prefix, label in cheetah_configs:
    d = find_dir(prefix)
    df = load_csv(d)
    plt.plot(df["Train_EnvstepsSoFar"], df["Baseline Loss"], label=label)
plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
plt.ylabel("Baseline Loss")
plt.title("HalfCheetah-v4: Baseline Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cheetah_baseline_loss.png"), dpi=150)
plt.close()

# ============================================================
# Figure 4: HalfCheetah Eval Return
# ============================================================
plt.figure(figsize=(10, 6))
for prefix, label in cheetah_configs:
    d = find_dir(prefix)
    df = load_csv(d)
    plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=label)
# Also plot no-baseline run
d = find_dir("HalfCheetah-v4_cheetah_sd1")
df = load_csv(d)
plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label="No Baseline", linestyle="--")
plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
plt.ylabel("Eval Average Return")
plt.title("HalfCheetah-v4: Eval Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cheetah_eval_return.png"), dpi=150)
plt.close()

# ============================================================
# Figure 5: LunarLander GAE lambda comparison
# ============================================================
plt.figure(figsize=(10, 6))
lambdas = ["0.00", "0.95", "0.98", "0.99", "1.00"]
for lam in lambdas:
    d = find_dir(f"LunarLander-v2_lunar_lander_lambda{lam}_sd1")
    df = load_csv(d)
    plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=f"$\\lambda={lam}$")
plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
plt.ylabel("Eval Average Return")
plt.title("LunarLander-v2: GAE $\\lambda$ Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lunarlander_gae.png"), dpi=150)
plt.close()

# ============================================================
# Figure 6: InvertedPendulum - all tuning runs
# ============================================================
plt.figure(figsize=(10, 6))

pendulum_runs = [
    ("InvertedPendulum-v4_pendulum_sd1_20260225_140616", "Default (lr=0.005, b=5000)", "C4", "--", 1.5),
    ("InvertedPendulum-v4_pendulum_sd1_20260224_132703", "Attempt 1 (lr=0.001, b=4000, blr=0.001)", "C2", "-", 1.2),
    ("InvertedPendulum-v4_pendulum_sd1_20260224_134328", "Attempt 2 (lr=0.001, b=1000, gae=0.99)", "C3", "-", 1.2),
    ("InvertedPendulum-v4_pendulum_sd1_20260224_134932", "Attempt 3 (lr=0.001, b=4000, blr=0.005)", "C1", "-", 1.2),
    ("InvertedPendulum-v4_pendulum_sd1_20260224_211743", "Best (lr=0.02, b=1000)", "C0", "-", 2.5),
]
for dirname, label, color, ls, lw in pendulum_runs:
    df = load_csv(dirname)
    plt.plot(df["Train_EnvstepsSoFar"], df["Eval_AverageReturn"], label=label, color=color, linestyle=ls, linewidth=lw)

plt.axhline(y=1000, color='r', linestyle='--', alpha=0.5, label="Target (1000)")
plt.xlabel("Environment Steps (Train_EnvstepsSoFar)")
plt.ylabel("Eval Average Return")
plt.title("InvertedPendulum-v4: Hyperparameter Tuning")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pendulum_tuning.png"), dpi=150)
plt.close()

print("All figures saved to figures/")
