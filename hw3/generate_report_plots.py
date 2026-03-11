import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

EXP_DIR = 'exp'

def load_log(dirname):
    path = os.path.join(EXP_DIR, dirname, 'log.csv')
    df = pd.read_csv(path)
    return df

def get_eval_data(df):
    """Extract rows that have eval data (non-NaN Eval_AverageReturn)."""
    mask = df['Eval_AverageReturn'].notna()
    return df.loc[mask, ['step', 'Eval_AverageReturn']].copy()

def get_train_data(df):
    """Extract rows that have train data."""
    mask = df['Train_EpisodeReturn'].notna()
    return df.loc[mask, ['step', 'Train_EpisodeReturn']].copy()

# ============================================================
# Figure 1: CartPole DQN (Section 2.4)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
df = load_log('CartPole-v1_dqn_sd1_20260309_145942')
ev = get_eval_data(df)
ax.plot(ev['step'], ev['Eval_AverageReturn'], linewidth=2)
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Eval Average Return')
ax.set_title('CartPole-v1 DQN')
ax.axhline(y=500, color='r', linestyle='--', alpha=0.5, label='Target (500)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_fig1_cartpole.png')
plt.close()

# ============================================================
# Figure 2: LunarLander DQN (Section 2.5)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
df = load_log('LunarLander-v2_dqn_sd1_20260309_151056')
ev = get_eval_data(df)
ax.plot(ev['step'], ev['Eval_AverageReturn'], linewidth=2)
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Eval Average Return')
ax.set_title('LunarLander-v2 Double DQN')
ax.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Target (200)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_fig2_lunarlander.png')
plt.close()

# ============================================================
# Figure 3: MsPacman DQN - train_return and eval_return (Section 2.5)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
df = load_log('MsPacman_dqn_sd1_20260309_171229')
ev = get_eval_data(df)
tr = get_train_data(df)
# For train, compute rolling average
tr_sorted = tr.sort_values('step')
window = max(1, len(tr_sorted) // 50)
tr_smooth = tr_sorted.rolling(window=window, min_periods=1).mean()
ax.plot(tr_smooth['step'], tr_smooth['Train_EpisodeReturn'], alpha=0.7, linewidth=1.5, label='Train Return (smoothed)')
ax.plot(ev['step'], ev['Eval_AverageReturn'], linewidth=2, label='Eval Return')
ax.axhline(y=1500, color='r', linestyle='--', alpha=0.5, label='Target (1500)')
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Return')
ax.set_title('MsPacman DQN: Train vs Eval Return')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_fig3_mspacman.png')
plt.close()

# ============================================================
# Figure 4: LunarLander Hyperparameter Experiment (Section 2.6)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
lr_exps = [
    ('LunarLander-v2_dqn_sd1_20260309_151056', 'lr=1e-3 (default)'),
    ('LunarLander-v2_dqn_lr1e-2_sd1_20260309_201403', 'lr=1e-2'),
    ('LunarLander-v2_dqn_lr5e-4_sd1_20260309_201409', 'lr=5e-4'),
    ('LunarLander-v2_dqn_lr1e-4_sd1_20260309_201403', 'lr=1e-4'),
]
for dirname, label in lr_exps:
    df = load_log(dirname)
    ev = get_eval_data(df)
    ax.plot(ev['step'], ev['Eval_AverageReturn'], linewidth=2, label=label)
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Eval Average Return')
ax.set_title('LunarLander-v2: Learning Rate Sensitivity')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_fig4_lunarlander_lr.png')
plt.close()

# ============================================================
# Figure 5: HalfCheetah SAC (Section 3.4)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
df = load_log('HalfCheetah-v4_sac_sd1_20260310_005613')
ev = get_eval_data(df)
ax.plot(ev['step'], ev['Eval_AverageReturn'], linewidth=2)
ax.set_xlabel('Environment Steps')
ax.set_ylabel('Eval Average Return')
ax.set_title('HalfCheetah-v4 SAC (Fixed Temperature)')
ax.axhline(y=6000, color='r', linestyle='--', alpha=0.5, label='Target (6000)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('report_fig5_halfcheetah_sac.png')
plt.close()

# ============================================================
# Figure 6: HalfCheetah SAC auto-tune vs fixed (Section 3.5)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Eval return comparison
df_fixed = load_log('HalfCheetah-v4_sac_sd1_20260310_005613')
df_auto = load_log('HalfCheetah-v4_sac_autotune_sd1_20260310_112518')
ev_fixed = get_eval_data(df_fixed)
ev_auto = get_eval_data(df_auto)
ax1.plot(ev_fixed['step'], ev_fixed['Eval_AverageReturn'], linewidth=2, label='Fixed Temperature')
ax1.plot(ev_auto['step'], ev_auto['Eval_AverageReturn'], linewidth=2, label='Auto-tuned Temperature')
ax1.set_xlabel('Environment Steps')
ax1.set_ylabel('Eval Average Return')
ax1.set_title('HalfCheetah-v4: Fixed vs Auto-tuned SAC')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Temperature (alpha) over training for auto-tuned
# alpha column exists in autotune log
mask_alpha = df_auto['alpha'].notna()
alpha_data = df_auto.loc[mask_alpha]
if len(alpha_data) > 0:
    ax2.plot(alpha_data['step'], alpha_data['alpha'], linewidth=2, color='orange')
else:
    # try temperature column
    mask_temp = df_auto['temperature'].notna()
    temp_data = df_auto.loc[mask_temp]
    ax2.plot(temp_data['step'], temp_data['temperature'], linewidth=2, color='orange')
ax2.set_xlabel('Environment Steps')
ax2.set_ylabel('Temperature (α)')
ax2.set_title('Auto-tuned Temperature over Training')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_fig6_halfcheetah_autotune.png')
plt.close()

# ============================================================
# Figure 7: Hopper single-Q vs clipped double-Q (Section 3.6)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

df_single = load_log('Hopper-v4_sac_singleq_sd1_20260310_124319')
df_clip = load_log('Hopper-v4_sac_clipq_sd1_20260310_151723')

# Subplot 1: Eval return
ev_single = get_eval_data(df_single)
ev_clip = get_eval_data(df_clip)
ax1.plot(ev_single['step'], ev_single['Eval_AverageReturn'], linewidth=2, label='Single-Q')
ax1.plot(ev_clip['step'], ev_clip['Eval_AverageReturn'], linewidth=2, label='Clipped Double-Q')
ax1.set_xlabel('Environment Steps')
ax1.set_ylabel('Eval Average Return')
ax1.set_title('Hopper-v4: Eval Return')
ax1.axhline(y=1500, color='r', linestyle='--', alpha=0.5, label='Target (1500)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Q-values
mask_qs = df_single['q_values'].notna()
qs_single = df_single.loc[mask_qs, ['step', 'q_values']]
mask_qc = df_clip['q_values'].notna()
qs_clip = df_clip.loc[mask_qc, ['step', 'q_values']]
ax2.plot(qs_single['step'], qs_single['q_values'], linewidth=2, label='Single-Q')
ax2.plot(qs_clip['step'], qs_clip['q_values'], linewidth=2, label='Clipped Double-Q')
ax2.set_xlabel('Environment Steps')
ax2.set_ylabel('Q-values')
ax2.set_title('Hopper-v4: Q-values')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('report_fig7_hopper.png')
plt.close()

print("All figures generated successfully!")
