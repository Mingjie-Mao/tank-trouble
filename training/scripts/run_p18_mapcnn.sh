#!/bin/bash
# P18 CNN 地图头实验 (排队在 P17 之后自动接力)
# 单变量: obs_map (迷宫栅格 + CNN 头), 其余与 P15 配方完全一致 (不含 P17 导航)
# 对照: P15 (无地图) = 33.8% | P17 (显式导航) 今晚出 | 冠军 35.1%
set -e
cd "$(dirname "$0")/.."
M=training/models

echo "===== [P18] 等待 P17 完成... ====="
while ! grep -q "P17 全部完成" training/run_p17.log 2>/dev/null; do
  pgrep -f "run_p17_nav.sh" >/dev/null || { echo "[P18] P17 进程已退出, 继续"; break; }
  sleep 60
done
echo "===== [P18] 开始 CNN 地图头流水线 ====="

echo "===== [P18-BC] 采集 80 万样本 (Dict观测: 121维+地图栅格) + 克隆训练 ====="
python3 training/bc_laika.py --samples 800000 --epochs 12 --obs-map \
  --out $M/p18_bc_map_clone

echo "===== [快检] 地图克隆裸装 (500 局 @970000, 对照 P15 克隆 8.1%) ====="
python3 training/evaluate.py --policy model --model $M/p18_bc_map_clone.zip \
  --n 500 --seed 970000

echo "===== [P18-FT] 价值预热 50 万 + 微调 3M (与 P15 同配方) ====="
python3 training/train_ppo.py --steps 3000000 --envs 12 \
  --reward-version 5 --obs-traj --obs-map --min-spawn-dist 4 --bad-shot -0.45 \
  --resume $M/p18_bc_map_clone.zip --value-warmup 500000 \
  --lr 1e-4 --ent-coef 0.003 --tag p18_map
mv $M/best_model.zip $M/p18_map_best.zip
echo "[P18-FT] 已归档 p18_map_best.zip"
cp $M/p8_badshot_best.zip $M/best_model.zip
chmod u+w $M/best_model.zip

echo "===== [定级] P18 (1000 局 @970000, 对照 P15=33.8% / 冠军=35.1%) ====="
python3 training/evaluate.py --policy model --model $M/p18_map_best.zip \
  --n 1000 --seed 970000
echo "===== P18 全部完成 ====="
