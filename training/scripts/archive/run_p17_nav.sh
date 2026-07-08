#!/bin/bash
# P17 导航观测实验 (排队在 P16 之后自动接力)
# 单变量: obs_nav (+4 维最短路方向), 其余与 P15 配方完全一致
# 对照: P15 (无导航) = 33.8% @1000局
set -e
cd "$(dirname "$0")/.."
M=training/models

echo "===== [P17] 等待 P16 过夜长训完成... ====="
while ! grep -q "P16 过夜长训全部完成" training/run_p16.log 2>/dev/null; do
  pgrep -f "run_p16_overnight.sh" >/dev/null || { echo "[P17] P16 进程已退出, 继续"; break; }
  sleep 60
done
echo "===== [P17] P16 已结束, 开始导航观测流水线 ====="

echo "===== [P17-BC] 采集 80 万样本 (125 维含导航) + 克隆训练 ====="
python3 training/bc_laika.py --samples 800000 --epochs 12 --obs-nav \
  --out $M/p17_bc_nav_clone

echo "===== [快检] 导航克隆裸装 (500 局 @970000, 对照 P15 克隆 8.1%) ====="
python3 training/evaluate.py --policy model --model $M/p17_bc_nav_clone.zip \
  --n 500 --seed 970000

echo "===== [P17-FT] 价值预热 50 万 + 微调 3M (与 P15 同配方) ====="
python3 training/train_ppo.py --steps 3000000 --envs 12 \
  --reward-version 5 --obs-traj --obs-nav --min-spawn-dist 4 --bad-shot -0.45 \
  --resume $M/p17_bc_nav_clone.zip --value-warmup 500000 \
  --lr 1e-4 --ent-coef 0.003 --tag p17_nav
mv $M/best_model.zip $M/p17_nav_best.zip
echo "[P17-FT] 已归档 p17_nav_best.zip"
cp $M/p8_badshot_best.zip $M/best_model.zip
chmod u+w $M/best_model.zip

echo "===== [定级] P17 (1000 局 @970000, 对照 P15=33.8% / 冠军=35.1%) ====="
python3 training/evaluate.py --policy model --model $M/p17_nav_best.zip \
  --n 1000 --seed 970000
echo "===== P17 全部完成 ====="
