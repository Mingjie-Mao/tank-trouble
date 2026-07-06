#!/bin/bash
# P13 (空枪纪律) + P14 (闪避特训营→微调) 串联实验
# 每段结束后归档模型, 最后自动 1000 局定级, 恢复冠军 best_model
set -e
cd "$(dirname "$0")/.."
M=training/models

echo "===== [P13] 空枪纪律探针 3M (P8 配置 + waste -0.10 / near-miss 0.02) ====="
python3 training/train_ppo.py --steps 3000000 --envs 12 \
  --reward-version 5 --obs-traj --min-spawn-dist 4 --bad-shot -0.45 \
  --waste-shot -0.10 --near-miss 0.02 --tag p13_waste
mv $M/best_model.zip $M/p13_waste_best.zip
echo "[P13] 已归档 p13_waste_best.zip"

echo "===== [P14a] 闪避特训营 2M (缴械, 只练走位) ====="
python3 training/train_ppo.py --steps 2000000 --envs 12 \
  --reward-version 2 --obs-traj --dodge-drill --tag p14_drill
mv $M/latest.zip $M/p14_drill_final.zip
[ -f $M/best_model.zip ] && mv $M/best_model.zip $M/p14_drill_best.zip || true
echo "[P14a] 特训完成, 已归档 p14_drill_final.zip"

echo "===== [P14b] 特训模型回全规则微调 3M (P8 配置, lr 1.5e-4) ====="
python3 training/train_ppo.py --steps 3000000 --envs 12 \
  --reward-version 5 --obs-traj --min-spawn-dist 4 --bad-shot -0.45 \
  --resume $M/p14_drill_final.zip --lr 1.5e-4 --tag p14_ft
mv $M/best_model.zip $M/p14_dodge_ft_best.zip
echo "[P14b] 已归档 p14_dodge_ft_best.zip"

# 恢复冠军占位
cp $M/p8_badshot_best.zip $M/best_model.zip

echo "===== [定级] P13 (1000 局 @970000) ====="
python3 training/evaluate.py --policy model --model $M/p13_waste_best.zip \
  --n 1000 --seed 970000
echo "===== [定级] P14 (1000 局 @970000) ====="
python3 training/evaluate.py --policy model --model $M/p14_dodge_ft_best.zip \
  --n 1000 --seed 970000
echo "===== 全部完成 (冠军基准: p8 = 35.1% @1000局) ====="
