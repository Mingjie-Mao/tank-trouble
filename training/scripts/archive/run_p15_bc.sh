#!/bin/bash
# P15 混合路线: 行为克隆 Laika -> 克隆定级 -> 价值预热+RL微调 -> 定级
# 依据: Laika 镜像 40.2% > 冠军 33.4% (1000局@970000)
set -e
cd "$(dirname "$0")/.."
M=training/models

echo "===== [P15-BC] 采集 80 万样本 + 克隆训练 12 epochs ====="
python3 training/bc_laika.py --samples 800000 --epochs 12

echo "===== [定级] BC 克隆裸装上阵 (1000 局 @970000, 天花板=镜像 40.2%) ====="
python3 training/evaluate.py --policy model --model $M/p15_bc_clone.zip \
  --n 1000 --seed 970000

echo "===== [P15-FT] 价值预热 50 万步 + 联合微调 3M (lr 1e-4, ent 0.003) ====="
python3 training/train_ppo.py --steps 3000000 --envs 12 \
  --reward-version 5 --obs-traj --min-spawn-dist 4 --bad-shot -0.45 \
  --resume $M/p15_bc_clone.zip --value-warmup 500000 \
  --lr 1e-4 --ent-coef 0.003 --tag p15_ft
mv $M/best_model.zip $M/p15_bc_ft_best.zip
echo "[P15-FT] 已归档 p15_bc_ft_best.zip"
cp $M/p8_badshot_best.zip $M/best_model.zip

echo "===== [定级] P15 微调版 (1000 局 @970000, 冠军闸门 35.1%) ====="
python3 training/evaluate.py --policy model --model $M/p15_bc_ft_best.zip \
  --n 1000 --seed 970000
echo "===== P15 全部完成 ====="
