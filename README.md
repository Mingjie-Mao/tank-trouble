# Tank Trouble — 原版 1:1 复刻 + 击败 Laika 的 RL 训练

基于反编译源码逐行移植的 Tank Trouble vs Laika 模式（逻辑/常量/帧序与原版
Flash 一致），并在其上训练强化学习智能体击败原版 AI「Laika」。

**当前最好成绩：原版真规则胜率 33.4% ± 2.1%**（2000 局全新种子终审，
冠军模型 `training/models/p8_badshot_best.zip`）。
旧口径（destroy 即判胜）下为 52.1%，与早期 README 的 50.5% 同口径可比。
口径差异见下方「胜率口径」——**这是接手前必读的一节**。

---

## 一、快速开始（游戏本体）

```bash
# 本地游玩 (vs Laika, 25 FPS, 无第三方依赖, 仅需 Python3 自带 tkinter)
python3 play_tank_trouble.py
python3 play_tank_trouble.py --seed 42          # 指定种子
python3 play_tank_trouble.py --two-players       # 本地双人

python3 test_original_port.py                    # 运行验证 (25 项检查)
```

**操作**: E/W/↑ 前进 · D/S/↓ 倒车 · A/← 左转 · F/→ 右转 · Q/Space/M 开火 · R 重开 · Esc 退出

---

## 二、胜率口径（重要！两套数字不能混）

- **先杀率（旧口径）**：训练环境在首个 destroy 事件即判胜——先杀掉 Laika 就算赢，
  哪怕 3 秒内被余弹/自己的反弹弹打死。早期"50.5%"即此口径。
- **真胜率（原版规则）**：原版在击杀后 125 帧、endCount==50 计分点看谁存活；
  先杀后死 = 双亡 = 不得分。**对外宣称"击败 Laika"必须用这个口径。**

实测差异（同种子对比）：早期 v2 模型先杀率 51.2%，真胜率仅 22.1%——差的
29 个点全是"神风换子"回合（双亡率 40.8%）。`training/evaluate.py` 现默认输出
双口径 + 行为指标，旧协议保留在 `--legacy`。

---

## 三、训练成果（模型演进，全部同协议可比）

| 模型 | 关键改动 | 真胜率 | 双亡率 | 备注 |
|---|---|---:|---:|---|
| random 基线 | — | 0.5% | 8.5% | 91% 死于自己反弹弹 |
| v1 `v1_best_19pct.zip` | 纯胜负奖励 | 12.4% | 14.4% | 苟活不开火 |
| v2 `v2_teammate_50pct.zip` | + 开火质量塑形 | 22.1% | 40.8% | 先杀率 51.2%, 神风换子流 |
| Hunter 手写脚本 | 利用已知弱点 | 22.5% | 14.5% | RL 及格线 |
| v4 | 终局改原版计分 + 击杀/死亡事件奖励 | 23.8% | 38.2% | 只改激励几乎无效 |
| P2 | + **弹道预演观测** (76→121 维) | 30.4% | 29.6% | **最大单项 +8** |
| P6 `p6_sharpen_best.zip` | + 去偏/闪避压力/熵精修 | 31.5% | 28.9% | |
| **P8 `p8_badshot_best.zip`** | + **自伤弹惩罚 -0.45** | **33.4%** | 27.3% | **← 交付冠军** |

冠军行为画像：击杀 57% 经反弹、45% 隔墙发射（高级弹道成型）；自杀率
28.6%→19.6%；场均移动 4.1→5.7 格。完整 12 轮探针记录（含全部失败实验）
见 **training/EXPERIMENTS.md**。

### 关键决策复盘（接手必读）

1. **v1→v2**（前任）：开火质量塑形治"苟活不开火"，胜负奖励卡 19% 的破局点。
2. **口径修正**（本轮）：训练目标必须定义在原版计分点上，否则 PPO 会把
   "杀完同归于尽"当成赢（reward v4 起修正）。
3. **缺信息不缺激励**：只改奖励 +1.7，给观测加弹道几何 +8——模型看不见
   反弹弹道时，任何闪避奖励都学不动（v3 失败的根因）。
4. **胜负头号变量 = 自己的反弹弹**：胜局自伤率 0%、败局 22%、双亡 40%
   （1500 局对照）。自伤弹惩罚从 -0.15 提到 -0.45 得 +3.6，
   剂量曲线 -0.15/-0.45/-0.75 = 31.5/35.1/31.9，峰值在 -0.45。
5. **测量纪律**：训练回调的 100 局评估噪声 ±9 个点、择优保存会过拟合运气；
   **一切判定以全新种子 1000+ 局离线评估为准**（本轮用 970000 种子基）。

### 已证伪的方向（别再走）

- frame_skip 2→1（细瞄准粒度）：时序翻倍学习难度 > 收益，-3 个点；
- 递增时间惩罚、近距图混训、纯续训整合、低 lr 单独用：均无增益；
- **ExploitBot 漏洞课程**：弹尽/卡墙/角落等单一窗口漏洞经 2000 局分析 +
  三版脚本验证，利用上限 ≈ Hunter 水平（<40% 闸门）。"Laika 卡墙"局胜率
  49% 是真的，但那是模型用弹幕压迫诱导的隐式技能，脚本造不出来；
- 恒定 lr 3e-4 长训：3M 探针后只在最优附近震荡（P2/P4 长训两次验证）。

---

## 四、下一步思路（按证据强度排序）

1. **命中率专项**：12% 命中率全程横盘，是最顽固短板。候选：收紧
   R_WASTE_SHOT、射击扇加宽（±30°→±50°）给更多预瞄选项；
2. **近距图双亡 43%**：训练去偏(min-spawn 4)使近图成为分布盲区，
   朴素混训已证伪，需距离加权的双亡惩罚等更精细方案；
3. **网络加宽 512**：121 维特征可能受 256×256 瓶颈（需从零训，成本高）；
4. **算力升级**：本机 3M 步探针尺度已挖尽，租多核 CPU 跑 5000 万步
   （瓶颈在 CPU 采样，不需要 GPU）。

---

## 五、训练与评估指令

```bash
pip install stable-baselines3 tensorboard gymnasium

# 训练冠军配置 (reward v5 + 弹道预演观测 + 去偏 + 自伤纪律)
python3 -u training/train_ppo.py --steps 3000000 --envs 12 \
    --reward-version 5 --obs-traj --min-spawn-dist 4 \
    --lr 0.0001 --ent-coef 0.003 --bad-shot -0.45 \
    --resume training/models/p8_badshot_best.zip --tag my_probe
# 注意: 后台运行必须加 python3 -u (stdout 块缓冲会让日志看似卡死)

# 评估 (默认双口径+行为指标; 判定用 1000+ 局全新种子)
python3 training/evaluate.py --policy model --model training/models/p8_badshot_best.zip --n 1000 --seed 970000
python3 training/evaluate.py --policy hunter --n 500        # 脚本基线
python3 training/evaluate.py --policy model --legacy ...    # 旧单口径

# 分析工具
python3 training/exploit_analysis.py --model <zip> --n 2000       # Laika 状态窗口 lift
python3 training/kill_pattern_analysis.py --model <zip> --n 1500  # 击杀画像/胜负对照/组合搜索

# 看录像
python3 training/watch.py --policy model --model training/models/p8_badshot_best.zip
```

---

## 六、文件与产物

```
training/
├── tt_gym_env.py          # 训练环境: 奖励 v1-v5, 观测 76/121 维(--obs-traj),
│                          #   terminal_mode destroy/score, 去偏/自伤/时间惩罚参数
├── train_ppo.py           # PPO 训练 (resume 可覆盖 lr/熵; 回调口径随奖励版本)
├── evaluate.py            # 双口径评估协议 + 行为指标电池 (--legacy 兼容旧口径)
├── baselines.py           # idle/random/hunter + ExploitBot(漏洞验证, 负结果留档)
├── exploit_analysis.py    # Laika 漏洞挖掘 (状态窗口 lift)
├── kill_pattern_analysis.py # 组合模式挖掘 (击杀画像/胜负对照)
├── EXPERIMENTS.md         # 12 轮探针完整记录 ★接手必读
└── models/
    ├── p8_badshot_best.zip    # ★ 冠军 (真胜率 33.4%)
    ├── p6_sharpen_best.zip    # 次优 (31.5%)
    ├── v2_teammate_50pct.zip  # 早期 v2 原件 (真胜率 22.1%)
    └── ...                    # 各探针存档 (p2-p12, 见 EXPERIMENTS.md)

tank_trouble_original/     # 游戏本体 (纯 Python 零依赖, 1:1 移植)
play_tank_trouble.py       # 本地游玩入口
test_original_port.py      # 25 项验证
PORT_NOTES.md              # 移植忠实度说明 ★
swf_decompiled/            # 反编译源码 (81 个 .as 文件)
GAME_MECHANICS_ANALYSIS.md # 游戏机制解读
```

## 七、AI 训练接口（自建训练时用）

```python
from tank_trouble_original import TankTroubleEnv

env = TankTroubleEnv(seed=42)      # tank0 = 智能体, tank1 = Laika
state = env.reset()
state, events = env.step({"forward": True, "turn_left": False, "fire": True})
# 无头 ~6000+ 帧/秒, 种子完全可复现
```

## 八、游戏核心机制速览（全部来自反编译源码）

| 机制 | 原版行为 |
|---|---|
| 帧率 | 25 FPS 固定步进 |
| 迷宫 | random(4) 模板法, W∈[4,12] H∈[4,10], SCALE 每回合变化 |
| 坦克 | 每帧 5 子步移动, 21 碰撞点, 贴墙滑动, 10° 转向吸附 |
| 子弹 | 每帧 7 子步, X/Y 独立反弹, 250 帧寿命, 最多 5 发 |
| 命中 | 出膛即致命, **可打死自己** (贴墙开火会反弹自杀) |
| 回合 | 死亡→125帧继续→**50帧冻结计分**→5帧后新迷宫 (计分点=真胜负判定点) |
| Laika | 目标优先级 AI, 每回合重建, 会躲避一切子弹(含自己的) |

详细忠实度说明（含全部已知近似）见 **PORT_NOTES.md**。
