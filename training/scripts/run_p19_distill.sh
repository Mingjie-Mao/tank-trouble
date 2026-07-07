#!/bin/bash
# P19 蒸馏管线: MPC(96.0%) 老师 -> 神经网络学生
# 阶段: 并行采集3000局 -> BC训练 -> 学生裸装双基定级 -> RL精修 -> 终定级
set -e
cd "$(dirname "$0")/.."
M=training/models

echo "===== [P19] MPC 示范采集 (3000局, 10进程) + 蒸馏训练 ====="
python3 training/mpc_distill.py --rounds 3000 --workers 10 --epsilon 0.1 \
  --epochs 12 --out $M/p19_mpc_student

echo "===== [定级] 学生裸装 @970000 (1000局, 老师=96.0%, 保真率=学生/老师) ====="
python3 training/evaluate.py --policy model --model $M/p19_mpc_student.zip \
  --n 1000 --seed 970000
echo "===== [定级] 学生裸装 @990000 (500局, 双基复验) ====="
python3 training/evaluate.py --policy model --model $M/p19_mpc_student.zip \
  --n 500 --seed 990000

echo "===== [P19-FT] 学生 RL 精修: 价值预热50万 + 3M (P17环境配方) ====="
python3 training/train_ppo.py --steps 3000000 --envs 12 \
  --reward-version 5 --obs-traj --obs-nav --min-spawn-dist 4 --bad-shot -0.45 \
  --resume $M/p19_mpc_student.zip --value-warmup 500000 \
  --lr 1e-4 --ent-coef 0.003 --tag p19_ft
mv $M/best_model.zip $M/p19_student_ft_best.zip
echo "[P19-FT] 已归档 p19_student_ft_best.zip"
cp $M/p17_nav_best.zip $M/best_model.zip
chmod u+w $M/best_model.zip

echo "===== [定级] 精修版 @970000 (1000局) ====="
python3 training/evaluate.py --policy model --model $M/p19_student_ft_best.zip \
  --n 1000 --seed 970000
echo "===== P19 蒸馏管线全部完成 ====="
