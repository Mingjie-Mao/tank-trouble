# Tank Trouble — 原版 1:1 复刻 + 击败 Laika 的 RL 训练

反编译 Flash 源码逐行移植的 Tank Trouble（vs Laika 模式），在其上训练 RL 智能体
击败原版 AI。游戏本体纯 Python 零依赖，无头 6000+ 帧/秒，种子完全可复现。

## 当前进度

**冠军：P17（BC 模仿热启动 + 导航观测 + RL 微调），真胜率 36.4% / 37.8%**
（双种子基各 1000 局，`training/models/best_model.zip`）

```
演进:  random 0.5% → 手写脚本 22.5% → 纯RL调参 35.1% → BC混合+导航 36.4-37.8%
参照:  Laika 镜像自打 = 40.2% (同实力线, 冠军距此 2-4 点)
```

18 轮单变量实验的完整台账见 **training/EXPERIMENTS.md**（接手必读）。

## 快速开始

```bash
# 游玩 (tkinter, 零依赖)          # 看 AI 打 Laika
python3 play_tank_trouble.py      python3 training/watch.py --policy model

# 评估 (双口径+行为指标; 判定必须 1000+ 局新种子)
python3 training/evaluate.py --policy model --model training/models/best_model.zip -n 1000 --seed 970000

# 训练 (依赖: pip install stable-baselines3 tensorboard gymnasium)
python3 training/bc_laika.py --samples 800000 --epochs 12 --obs-nav   # 1. BC 克隆
python3 training/train_ppo.py --steps 3000000 --envs 12 \
  --reward-version 5 --obs-traj --obs-nav --min-spawn-dist 4 --bad-shot -0.45 \
  --resume training/models/p15_bc_clone.zip --value-warmup 500000 \
  --lr 1e-4 --ent-coef 0.003 --tag my_probe                           # 2. RL 微调
# 后台跑加 caffeinate -i 防睡眠; 本机 ~4400 步/秒, 3M 探针约 25 分钟
```

## 两条铁律（血泪教训）

1. **口径**：先杀 ≠ 赢。原版规则先杀后死 = 双亡不得分。destroy 口径虚高
   15-29 点，对外数字一律用真胜率（evaluate.py 默认双口径输出）。
2. **测量**：训练回调的 100 局评估偏差 ±9 点，只可看趋势；**一切判定以
   1000+ 局全新种子为准，换冠军需第二种子基复验**。

## 核心设计

- **观测**（当前 125 维）：自车/敌车状态 + 24 射线 + 6 子弹 + **弹道预演**
  （来袭弹未来轨迹/射击扇模拟，+8 点）+ **导航**（最短路方向，+3 点）。
  CNN 地图头（Dict 观测）已验证劣于显式导航（P18=22.5%，勿再投入）。
- **奖励 v5**：原版计分终局 ±1 + 击杀事件 ±0.5 + 开火质量塑形（模拟命中
  +0.3 / 自杀弹 -0.45）+ 密集闪避压力 + 势能塑形。
- **训练配方（P15+）**：BC 克隆 Laika（80 万样本）→ 价值预热 50 万步
  （冻结策略只训价值头，防随机价值头毁先验）→ PPO 微调 3M（lr 1e-4）。
  比从零训练快 3 倍。

## 已知问题与风险

| 问题 | 现状 |
|---|---|
| **双亡 28-30%**（杀完被余弹带走） | 头号短板；胜局自伤 0% vs 败局 22% |
| 无记忆策略（观测无速度/历史帧） | frame stack / RNN 未试；BC 克隆有状态的 Laika 只到 8% 即此因 |
| **微调有效窗口 ≈2M 步** | 恒定 lr 续训必崩（P16 用 60M 步实证）；KL 锚定/lr 衰减未实现 |
| 奖励微调已饱和 | P8 后八连 NO-GO；行为矫正会破坏自平衡（P13/P14），改观测才涨分 |
| ~~逼近镜像线后的天花板未知~~ | **已破解：MPC 搜索 96.0%**（无未来函数沙盒，18 动作×48 帧前推，零训练）——镜像线 40.2% 作废，游戏真实上限远高于此 |

## 下一步（蒸馏 v1 已试，argmax 标签路线失败）

- P19 蒸馏 v1（argmax 标签）失败：学生裸装 8.9%，精修 30.8%。
  教训：老师棋力 ≠ 可教性（临界打平随机选边 → 标签矛盾）。
- **P20 混合体成功**：P17 网络剪枝 + MPC 深推。**K=12 = 96.0%（追平全量，
  快 4.3 倍，36ms/决策实时可用）**；K=8 = 95.0%@24ms；K=5 = 81%@15ms。
  `python3 training/watch.py --policy hybrid --k 12` 可实时观看。

**主线 P21 — 专家迭代（把 96% 内化成纯网络权重）**：
循环 [混合体(K=8)打几千局 → 记录 18 候选的推演**评分**（非 argmax，治
标签矛盾）→ 网络回归价值地形 → 更强的网络给下一轮剪枝]。
终点验收只认**拆掉搜索的纯网络裸装成绩**；搜索永久降级为训练工具。

## 文件结构

```
tank_trouble_original/   游戏本体 (1:1 移植, 勿改逻辑; 详见 PORT_NOTES.md)
play_tank_trouble.py     本地游玩 | test_original_port.py 25 项验证
swf_decompiled/          反编译源码 (移植依据, 行号引用见 PORT_NOTES)
training/                # 核心 .py 固定在包根 (import 链与已存模型引用, 勿移)
├── tt_gym_env.py        环境: 奖励 v1-v5, 观测 76/121/125/Dict(+地图)
├── train_ppo.py         PPO (resume/价值预热/lr覆盖/各惩罚参数)
├── bc_laika.py          行为克隆 | map_extractor.py CNN头 | mpc_distill.py 蒸馏采集
├── mpc_agent.py         B线: 无未来函数沙盒 + MPC 搜索 (96.0%)
├── hybrid_agent.py      B线: 网络剪枝混合体 (K=12: 96.0% @36ms 实时)
├── evaluate.py          双口径评估 | baselines.py 基线 | watch.py 回放(含hybrid/mpc)
├── EXPERIMENTS.md       ★ 全部实验台账
├── analysis/ scripts/ logs/   挖掘工具 / 实验运行脚本 / 全部历史日志
└── models/              模型 (gitignore, 线下传; best_model.zip=冠军P17)
```

## 游戏机制速览

25 FPS ｜ 迷宫每局随机 (4-12×4-10 格) ｜ 子弹每帧 7 子步、反弹、250 帧寿命、
**可打死自己** ｜ 弹匣 5 发 ｜ 死亡后 125 帧才计分（双亡窗口）｜ Laika 为
目标优先级脚本 AI（决策周期 10 帧，会躲一切子弹）。忠实度细节见 PORT_NOTES.md。
