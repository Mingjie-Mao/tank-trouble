# Tank Trouble — 原版 1:1 复刻 + 击败 Laika 的 AI 全技术路线

反编译 Flash 源码逐行移植的 Tank Trouble（vs Laika 模式），在其上完整走通
「RL → 搜索 → 蒸馏」三条技术路线。游戏本体纯 Python 零依赖，无头 6000+ 帧/秒，
种子完全可复现。项目叙事与教训见 **docs/REPORT.md**，全部实验见 **training/EXPERIMENTS.md**。

## 当前进度（已验证成绩单）

| 路线 | 代表 | 真胜率 | 决策成本 | 性质 |
|---|---|---:|---:|---|
| RL 训练 | P17 (BC+导航+PPO) | 36.4% | 0.3ms | 权重, 净胜 Laika |
| 旧决策时搜索 | MPC / 混合体 K=12 | 96.0% | 91/36ms | 现场推演, 早期上限测量仪 |
| **快速网络** | **P27b（P26 底座 + 风险/价值头）** | **88.2 / 89.6%**\* | 零在线搜索 | **当前部署冠军** |
| **精确状态搜索** | **Exact-State Safety-Shielded MPC** | **99.0%**\*\* | 离线慢搜索 | **当前搜索老师冠军** |

\* P27b 双基正式定级为 1000 局@970000 / 500 局@990000，合计 1330/1500，
真胜率 88.67%。\*\* 精确老师在三组未见种子共 300 局中 297 胜、3 负、0 双亡；
固定三组基准另为 120/120。它读取本地引擎的完整状态和 RNG，只用于标注和回归，
不是快速纯网络，也不是任意未知分布上的 100% 证明。
参照：random 0.5% / 手写脚本 22.5% / Laika 镜像 40.2%。
观测脚注：含物理预演特征（事实非价值，无候选比较）。

```
演进主线: 5%(random) → 35%(奖励工程到顶) → 36.4%(观测补全) → 96%(旧MPC)
          → 62-64%(P21b) → ~68%(P22) → 77%(P25) → 87%(P26)
          → 88.2/89.6%(P27b快速网络) → 99%(精确状态搜索老师)
```

## 网络架构演进

当前快速冠军 P27b 由 `p26_amortized_mpc_iter05.pt` 动作评分底座和
`p27b_risk_value_iter00.pt` 风险/价值头组成。每帧只做网络前向和确定性动作重排，
不运行在线 rollout。完整正式结果和后续负结果见
`training/analysis/P27_POLICY_RESULTS.md`。

下面是奠定 P22 路线的历史 P21b 评分网络架构；P26/P27b 延续了“对全部候选动作
预测结果，再进行风险校准”的核心思路。

![P21b 评分网络架构](docs/scorenet_arch.svg)

**核心思想：网络学的不是"模仿哪个动作"，而是"每个动作会导向什么结局"。**
输出 18 个评分对应全部动作组合（油门 3 × 转向 3 × 开火 2），训练标签是 MPC
老师对每个候选沙盘推演 48 帧的结局评分（价值地形）；部署时一次前向取
argmax，零搜索，8.6ms/决策。这也是它区别于 BC 行为克隆、能治"并列陷阱"
（多个动作打平时 argmax 标签自相矛盾）的原因。

**408 维输入 = 当前帧的全部物理事实**（无未来函数，只报事实不做判断）：

| 块 | 维数 | 内容 |
|---|---:|---|
| 基础观测 | 125 | 自车 6 · 敌车 8（含路径距离/LOS）· 射线 24 · 弹槽 6×6 · 弹道预演 6×4 · 射击扇 7×3 · 导航 4 · 计时/扳机 2 |
| ★ 动作条件预演 | 18 | 9 种走法各推 24 帧 × [会中弹?, 几帧后] —— 蒸馏成败的关键一块：标签问"往哪走能活"，观测里必须有动作条件信息 |
| 弹槽补全 | 24 | 第 7-10 颗子弹 × 6 |
| 全迷宫位图 | 240 | 12×10 格 × [下墙, 左墙] |
| 卡墙标志 | 1 | 上一帧是否撞墙 |

**尺寸哲学**：1024×3 共约 2.5M 参数——只在实验证明容量是瓶颈时才加大。
三次对照实验（观测两针 +11 点 / 奖励八连败 / P21a 大网仅 +1%）证明瓶颈
从来是信息和数据分布，不是参数量；不用 RNN/时序结构是因为公平性规则下
老师标签是当前态的纯函数。历代网络：RL 线 = SB3 PPO 双塔 [256,256]
（~0.2M）；P18 CNN 头惨败于 4 维手工导航特征后废弃。

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
| 搜索老师仍非严格 100% | 精确状态安全 MPC 未见种子 297/300；剩余 3 局需要逐局修复，固定基准 120/120 不能外推为全分布 100% |

## 当前战线（详见 training/EXPERIMENTS.md 台账）

- **P27b 快速网络冠军**：正式双基 88.2% / 89.6%，零在线搜索。
- **Exact-State Safety-Shielded MPC 搜索老师**：固定基准 120/120；未见种子
  297/300、0 双亡，通过 297 胜门槛，但仍有 3 个明确失败局。
- **Sparse Exact Safety 混合部署未晋升**：固定基准 119/120，未见种子在
  152 局出现第 4 个非胜后提前停止，说明减少搜索会重新暴露反弹弹道风险。
- **精确老师蒸馏先导未晋升**：直接全网络微调破坏原冠军校准；下一步保留
  P27b 默认动作，只学习有高置信优势的残差修正。

**后备方向**：算力放大（租多核 CPU）｜特征自举（感知网络替换手写前端）｜
在线平台视觉栈 V1-V3

## 文件结构

```
README.md                本文件 (总览)
play_tank_trouble.py     本地游玩 | test_original_port.py 25 项验证
tank_trouble_original/   游戏本体 (1:1 移植, 勿改逻辑; 详见 docs/PORT_NOTES.md)
swf_decompiled/          反编译源码 (移植依据, 行号引用见 docs/PORT_NOTES.md)
docs/                    文档归档: REPORT(项目叙事) PAPER(论文骨架) PORT_NOTES(移植)
                         GAME_MECHANICS_ANALYSIS(机制解读) HANDOFF/scorenet_arch.svg
training/                # 核心 .py 固定在包根 (import 链与已存模型引用, 勿移)
├── tt_gym_env.py        环境: 奖励 v1-v5, 观测 76/121/125/408/Dict(+地图)
├── mpc_agent.py         B线: 无未来函数沙盒 + MPC 搜索 (96.0%)
├── hybrid_agent.py      B线: 网络剪枝混合体 (K=12: 96.0% @36ms 实时)
├── score_distill.py     C线: 评分蒸馏 (P21b) | expert_iter.py 专家迭代 (P22)
├── p26_amortized_mpc.py 快速动作评分底座 | p27_risk_value.py 风险/价值部署头
├── exact_state.py       精确引擎状态克隆 | exact_state_mpc_teacher.py 搜索老师
├── sparse_exact_safety_policy.py 低频搜索混合策略（实验性，未替代快速冠军）
├── survival_mode.py     ★ P24 生存课程 (进行中) | value_leaf.py P23 (负结果存档)
├── train_ppo.py         A线: PPO | bc_laika.py BC | map_extractor.py CNN头
│                        | mpc_distill.py P19 —— A线已终结, 仅存档
├── evaluate.py          双口径评估 | baselines.py 基线 | watch.py 回放(hybrid/mpc/网络)
├── EXPERIMENTS.md       ★ 全部实验台账
├── analysis/            挖掘工具
├── scripts/             run_expert_iter.sh (现役) | archive/ 历史实验脚本
├── logs/                全部历史日志 (实验证据, 入库)
└── models/              模型（gitignore；当前部署需 P26 iter05 + P27b risk/value）
```

## 游戏机制速览

25 FPS ｜ 迷宫每局随机 (4-12×4-10 格) ｜ 子弹每帧 7 子步、反弹、250 帧寿命、
**可打死自己** ｜ 弹匣 5 发 ｜ 死亡后 125 帧才计分（双亡窗口）｜ Laika 为
目标优先级脚本 AI（决策周期 10 帧，会躲一切子弹）。忠实度细节见 docs/PORT_NOTES.md。
