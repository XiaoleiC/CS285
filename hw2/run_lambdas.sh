#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="LunarLander-v2"
LAMBDAS=(0.00 0.95 0.98 0.99 1.00)

for L in "${LAMBDAS[@]}"; do
  EXP_NAME="lunar_lander_lambda${L}"
  echo "===== Running lambda=${L} ====="

  uv run src/scripts/run.py \
    --env_name "$ENV_NAME" \
    --ep_len 1000 \
    --discount 0.99 \
    -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 \
    --use_reward_to_go --use_baseline \
    --gae_lambda "$L" \
    --exp_name "$EXP_NAME"
done