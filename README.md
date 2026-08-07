# Tank Trouble：原版复刻与击败 Laika 的 AI

本项目从 Flash 反编译源码出发，用 Python 复刻 Tank Trouble 的迷宫、坦克、
反弹弹道、回合计分和 Laika，并在原版规则下训练可部署策略。

最终指标始终是**原版真胜率**：先击杀 Laika、随后被余弹击中属于双亡，不算胜利。
训练课程可以改变规则，但不能替代原版验收。

## 当前结果

| 路线 | 代表 | 原版真胜率 | 状态 |
| --- | --- | ---: | --- |
| Laika 镜像 | Laika vs Laika | 40.2% | 行为基线 |
| RL | P17 | 36.4% / 37.8% | RL 路线最佳 |
| 评分蒸馏 | P21b | 62.2% / 63.8% | 1000/500 局双基 |
| 专家迭代 | P22 iter02 | 约 68% | 前纯网络冠军 |
| 生存课程学生 | P24v2.1 | 58.7% | 课程老师成功，旧蒸馏失败 |
| **生存课程重蒸馏** | **P24v4** | **70.0%（80 局门测）** | **三轮 DAgger 进行中** |
| **机会课程学生** | **P25v2 iter02** | **76.6%** | **当前最佳，1000 局新种子基线** |
| 在线搜索老师 | P25v2 360° MPC | 98.0%（100 局） | 老师上限探针，不是部署模型 |

P25v2 当前最佳的正式独立基线：

- 1000 局，种子 `980000–980999`
- 真胜率 **76.6%**
- 负 16.8%，双亡 5.6%，平局 1.0%
- 场均开火 2.7 发，命中率 14.3%，平均局长 22.2 秒

模型晋升曲线为 66.2% → 70.0% → 80.0%（固定 80 局评测门），随后在独立
200 局中得到 77.0%，在全新 1000 局中得到 76.6%。第二种子基复验仍建议补做。

## 当前最佳模型

```text
training/models/best_model.pt
```

它是 `p25v2_opportunity_best.pt` 的当前冠军快照。P22 仍单独保存在
`training/models/scorenet_best.pt`，因为后续实验需要将它作为引导策略，不能覆盖。

P25v2 学生不是在线 MPC：部署时只有物理观测前端和一次评分网络前向，不比较
候选动作的未来价值。但其 440 维观测包含 9 种移动的短物理预演，因此前端比 P22/P24
学生更重；严格意义上属于“网络策略 + 确定性物理感知”，不是原始像素端到端模型。

## 观看回放

所有命令在仓库根目录运行。回放窗口按 `R` 更换地图。

### 当前最佳 P25v2 学生：原版规则

```bash
python3 training/watch.py --policy best
```

固定随机种子：

```bash
python3 training/watch.py --policy best --seed 980000
```

指定其他 P25v2 权重：

```bash
python3 training/watch.py --policy best \
  --net training/models/p25v2_opportunity_iter01.pt
```

### P25v3 反演击杀场 MPC 老师：原版规则

```bash
python3 training/watch.py --policy p25v3-teacher --seed 984100
```

这是慢速在线老师：每次决策进行 MPC、反演射击格和弹道验证，不是最终学生。
20 局机制探针为 90% 真胜、10% 双亡，95 次物理确认机会全部尝试开火。

### P24v2.1 生存课程学生：回到原版规则

```bash
python3 training/watch.py --policy p24-student-original --seed 970000
```

### P24v2 风格生存老师：回到原版规则

```bash
python3 training/watch.py --policy p24-teacher-original --seed 970000
```

该老师仍用生存课程账本选择动作，但战斗规则为原版：Laika 可以被击杀，没有受击免疫。

### P24v2 风格生存老师：生存课程规则

```bash
python3 training/watch.py --policy p24-teacher-survival --seed 970000
```

此模式中 Laika 无敌，我方一发即死，命中增加分数池；使用表现最好的 P24v2 规则，
不包含后来失败的 2 秒受击免疫和空弹夹税。

### P24v4 生存学生：原版规则

```bash
/opt/anaconda3/bin/python3 training/watch.py \
  --policy p24v4-student-original --seed 970000
```

### P24v4 生存学生：生存课程规则

```bash
/opt/anaconda3/bin/python3 training/watch.py \
  --policy p24v4-student-survival --seed 970000
```

### P24-P22 Replica-530 学生

原版规则：

```bash
/opt/anaconda3/bin/python3 training/watch.py \
  --policy p24r530-student-original --seed 970000
```

生存课程规则：

```bash
/opt/anaconda3/bin/python3 training/watch.py \
  --policy p24r530-student-survival --seed 970000
```

该模型使用 P22 同型的 18 路联合动作评分网络；正式训练完成前，权重文件
`training/models/p24r530_best.pt` 可能尚不存在。

### 本地游玩

```bash
python3 play_tank_trouble.py
```

## 重新评测当前最佳

Mac 长任务统一加 `caffeinate -i`，允许屏幕关闭但阻止系统因空闲休眠。

```bash
caffeinate -i python3 training/opportunity_distill_v2.py eval \
  --net training/models/best_model.pt \
  --eval-n 1000 \
  --eval-seed 990000 \
  --workers 8
```

## P25v2 架构

### 观测

P25v2 使用 440 维输入：

| 部分 | 维数 | 内容 |
| --- | ---: | --- |
| P21b 物理观测 | 408 | 坦克、子弹、迷宫、导航、弹道和动作条件物理事实 |
| 当前机会状态 | 5 | 炮线质量、射击位进度、来弹风险、目标方向 |
| 9 种移动预演 | 27 | 每种移动后的炮线变化、射击位变化和风险 |

评分网络为三层 1024 宽 MLP，输出 18 个动作分数。部署时按 9 对“不开火/开火”
分数选择移动；只有网络偏好开火且对准后的可信炮线不低于 0.58 时才允许开火。

### 老师与数据

P25v2 老师扫描坦克全部 32 个物理朝向，预测 75 帧、最多两次反弹。数据完全独立：

1. 128 局老师分布；
2. 128 局 P22 分布，由 P25v2 老师重标；
3. 两轮各 128 局学生 on-policy DAgger；
4. 每轮从全部聚合数据重新训练并通过固定评测门晋升。

## P24v4 生存老师重蒸馏

旧 P24 学生没有完整看到老师的覆盖账本，并把移动与开火压在同一个 18 动作回归头中。
新管线位于 `training/survival_distill_v2.py`：

- 530 维观测：408 物理状态 + 分数池/剩余时间 + 12×10 覆盖冷却图；
- 移动与火控使用两个独立编码器，避免稀少开火事件破坏移动表征；
- 9 路移动头用相对优势软排序，不再回归被公共生存基线支配的绝对分数；
- 9 路条件开火头按学生最终选择的移动分支触发；
- 开火正样本加权，并额外进行机会状态平衡重放；
- 128 局老师 + 128 局 P22 引导 + 三轮各 128 局学生 DAgger；
- 生存课程评测和原版真胜率评测同时保留。

当前门测冠军为 `training/models/p24v4_survival_best.pt`：生存课程 40 局达到
5.0 秒/中、卡墙 8.7%、风格 +0.07/秒；原版 80 局真胜率 70.0%、双亡 1.2%。
这些仍是迭代门指标，第三轮 DAgger 和更大独立种子基线尚未完成。

启动命令：

```bash
caffeinate -i python3 training/survival_distill_v2.py pipeline --fresh \
  --teacher-rounds 128 \
  --bootstrap-rounds 128 \
  --dagger-rounds 3 \
  --rounds-per-dagger 128 \
  --workers 8 \
  --epochs 12 \
  --gate-n 80 \
  --survival-gate-n 40 \
  --eval-n 200 \
  --final-survival-n 80
```

新数据写入 `training/survival_data_v2/`，不会与旧 P24 数据混合。

## 生存课程规则

当前重蒸馏使用表现最好的 P24v2.1 规则：

- Laika 无敌，我方一发即死；
- 初始分数 100，每秒衰减 10；
- 命中 Laika +50；
- 净靠近一格 +3，远离会扣回，绕圈净额为零；
- 首次或冷却后进入格子 +2，冷却 4 秒；
- 卡墙额外 −5/秒；
- 分数耗尽或死亡立即结束，死亡结算清零；
- 最长 30 秒。

失败的 P24v3“受击后 2 秒免疫 + 子弹穿透 + 空弹夹税”没有用于新训练。

## 当前路线：exploit 搜索 → 对手建模 → 飞轮（2026-08-05）

### 已判死的两条腿

AlphaZero 式飞轮的两个主要杠杆，在这个游戏上都被实验否决：

| 杠杆 | 结论 | 证据 |
| --- | --- | --- |
| V(s) 价值叶子 | ❌ | 六个受控消融，验证 R² 全在 0 附近或为负 |
| 更深的搜索 | ❌ | horizon 72 vs 36 头对头 41.1% ± 12.9 |

价值那条的消融链：自博弈对称结局（击杀 223:243、超时 226:226）方差 0.343
仍 R² +0.017；网络 463k→6k 参数 R² 不动（排除过拟合）；按局内位置分桶，
连最后 5% 帧 R² 也只有 0.109（排除"未来不确定"）；加入击杀场特征后
R² 从 −0.013 掉到 −0.091（排除缺任务特征）。

两个结果互相印证同一个游戏性质：**局面几乎不决定胜负**。一局由一次
短促的反射性交锋决定，开火瞬间的几何一秒内就完全变了。这与围棋象棋那类
"局面强烈决定结局、看得越远越强"的游戏不是一类。

### 活下来的路线

搜索的 rollout 里对手模型写死 `opp_model="L2"`（把所有人当 Laika），
这是群友能稳定打赢它的根因。修这个不需要价值函数：

```
TAS 式存档回溯搜索  →  枚举出大量能杀死当前模型的时间线
        ↓
网络建模对手空间（对手 = Laika + 偏差，z 从实战推断）
        ↓
搜索的推演变准 → 搜索变强 → 旧 exploit 失效 → 再搜
```

每轮的新信息来自**新发现的打法**，不来自任何估计。技术前提是这个游戏
确定性：(game, teacher) 一起 pickle 后回溯能逐位复现决策（54KB / 0.4ms）。

以 Laika 为参照系而不是从零建模，因为对手模型跑在 rollout **里面**
（一次决策约 90 次对手推理），必须是微秒级；且 `z=0` 时严格等于现状，
遇到没见过的对手最坏退回今天的水平。

### 规则

30 秒（750 帧）结算；击杀并存活 = 1.0；超时按追猎链分裁决（领先 0.4 /
落后 0.2）；双亡 0.1；自己死 0.0。超时对搜索算失败，要回溯。

### 命令

```bash
# 地毯式 exploit 搜索（断点续跑，重启自动跳过已完成的图）
caffeinate -i python3 training/exploit_search.py run --maps 64 --workers 8

# 回放搜索出来的击杀路径
python3 training/watch.py --policy exploit-replay

# 新规则擂台
python3 training/watch.py --policy arena --ranked --seed 40000000
```

## 项目结构

```text
README.md                              当前总览与回放命令
play_tank_trouble.py                   本地游玩
tank_trouble_original/                 原版 Python 复刻
training/
├── evaluate.py                        原版双口径评测
├── mpc_agent.py                       48 帧 MPC 基础老师
├── score_distill.py                   P21b 评分网络
├── expert_iter.py                     P22 专家迭代
├── survival_mode.py                   P24 生存课程
├── survival_distill.py                P24v2.1 旧蒸馏
├── survival_distill_v2.py             P24v4 双头 + 完整账本 + DAgger
├── opportunity_distill.py             P25v1 机会课程
├── opportunity_teacher_v2.py          P25v2 360°老师
├── opportunity_distill_v2.py          P25v2 独立蒸馏与 DAgger
├── opportunity_teacher_v3.py          P25v3 反演击杀场老师
├── watch.py                           统一回放入口
├── EXPERIMENTS.md                     完整实验台账
└── models/                            本地模型产物，不进入 Git
docs/
├── HANDOFF_COMPLETE_CONTEXT.md        当前结论与跨会话恢复
├── CLAUDE_CODE_COMPLETE_CONTEXT.md    两份 Claude Code 可见对话全文
├── REPORT.md                          项目叙事
└── PORT_NOTES.md                      Flash 到 Python 的移植依据
tools/export_claude_dialogue.py        重新导出 Claude 会话上下文
```

训练数据、`.pt`/`.zip` 模型和本机会话原始文件均不进入 Git。

## 测量纪律

1. 原版真胜率是最终标准，双亡不算胜利。
2. 80/100/200 局只能作为评测门或机制探针。
3. 正式结果至少使用 1000 局全新种子，换冠军建议补第二种子基。
4. 老师表现好不代表学生已经蒸馏成功；必须分别报告老师、课程学生和原版学生。
5. 同维度但语义不同的数据禁止混合。

更完整的实验数字、失败路线和当前训练状态见 `training/EXPERIMENTS.md`；跨会话恢复先读
`docs/HANDOFF_COMPLETE_CONTEXT.md`。
