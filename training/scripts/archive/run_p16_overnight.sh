#!/bin/bash
# P16 过夜长训: P15 配方续训 60M 步 + 卡墙惩罚
# 依据: P15 (BC热启动+微调3M) = 33.8% 统计平冠军, 学习3倍速,
#      截止时曲线仍在上行 (峰值37%@2.5M) => 长训是最高证据杠杆。
# 卡墙惩罚 -0.01/步: 录像观察 P15 顶墙不会脱困 (观测无速度/历史,
#      需显式负信号教它"正前方射线=0 且在推 => 松油门转向")。
set -e
cd "$(dirname "$0")/.."
M=training/models

echo "===== [P16] 过夜长训 60M 步 (从 p15_bc_ft_best 续训) ====="
python3 training/train_ppo.py --steps 60000000 --envs 12 \
  --reward-version 5 --obs-traj --min-spawn-dist 4 --bad-shot -0.45 \
  --resume $M/p15_bc_ft_best.zip \
  --lr 1.5e-4 --ent-coef 0.003 --stuck-penalty -0.01 \
  --eval-every 2000000 --tag p16_long
mv $M/best_model.zip $M/p16_long_best.zip
mv $M/latest.zip $M/p16_long_final.zip
echo "[P16] 已归档 p16_long_best.zip (回调最优) / p16_long_final.zip (终点)"
cp $M/p8_badshot_best.zip $M/best_model.zip
chmod u+w $M/best_model.zip

echo "===== [定级] P16 回调最优 (1000 局 @970000, 冠军闸门 35.1%) ====="
python3 training/evaluate.py --policy model --model $M/p16_long_best.zip \
  --n 1000 --seed 970000
echo "===== [定级] P16 训练终点 (1000 局 @970000) ====="
python3 training/evaluate.py --policy model --model $M/p16_long_final.zip \
  --n 1000 --seed 970000
echo "===== P16 过夜长训全部完成 ====="
