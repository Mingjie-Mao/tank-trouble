# Tank Trouble — 原版 1:1 复刻 (Python)

基于反编译源码逐行移植的 Tank Trouble vs Laika 模式，逻辑、常量、
帧序与原版 Flash 一致，内置面向 AI 训练的数据接口。

## 快速开始

```bash
# 本地游玩 (vs Laika, 25 FPS, 无第三方依赖, 仅需 Python3 自带 tkinter)
python3 play_tank_trouble.py

# 指定种子 / 本地双人
python3 play_tank_trouble.py --seed 42
python3 play_tank_trouble.py --two-players

# 运行验证 (25 项检查)
python3 test_original_port.py
```

**操作**: E/W/↑ 前进 · D/S/↓ 倒车 · A/← 左转 · F/→ 右转 · Q/Space/M 开火 · R 重开 · Esc 退出

## AI 训练接口

```python
from tank_trouble_original import TankTroubleEnv

env = TankTroubleEnv(seed=42)      # tank0 = 智能体, tank1 = Laika
state = env.reset()
state, events = env.step({"forward": True, "turn_left": False, "fire": True})

# state: 完整真值 (坦克位姿/子弹/迷宫墙体/回合状态/比分/Laika当前目标)
# events: fire / bounce / hit / destroy / round_end / new_round
# 无头速度 ~6000+ 帧/秒 (实时的 240 倍), 种子完全可复现
```

可选 gymnasium 包装: `from tank_trouble_original import TankTroubleGymEnv`
(动作 MultiDiscrete([3,3,2]), 击杀 ±1 奖励, 需 `pip install gymnasium`)。

## 项目结构

```
tank_trouble_original/        # 游戏本体 (纯 Python, 零依赖)
├── constants.py              # 原版全部常量 (frame_53:2152+)
├── maze.py                   # 迷宫生成/寻路 (frame_53:1-586)
├── game.py                   # 坦克/子弹/主循环 (tank+bullet+root)
├── laika.py                  # Laika AI 全移植 (1386 行 AI 脚本)
└── env.py                    # 训练数据接口 + gym 包装
play_tank_trouble.py          # 本地游玩入口 (tkinter)
test_original_port.py         # 验证脚本 (25 项)
PORT_NOTES.md                 # 移植对照与忠实度说明 ★
swf_decompiled/               # 反编译源码 (81 个 .as 文件)
GAME_MECHANICS_ANALYSIS.md    # 游戏机制解读
```

## 核心机制速览 (全部来自反编译源码)

| 机制 | 原版行为 |
|---|---|
| 帧率 | 25 FPS 固定步进 |
| 迷宫 | random(4) 模板法, W∈[4,12] H∈[4,10], SCALE 每回合变化 |
| 坦克 | 每帧 5 子步移动, 21 碰撞点, 贴墙滑动, 10° 转向吸附 |
| 子弹 | 每帧 7 子步, X/Y 独立反弹, 250 帧寿命, 最多 5 发 |
| 命中 | 出膛即致命, **可打死自己** (贴墙开火会反弹自杀) |
| 回合 | 死亡→125帧继续→50帧冻结计分→5帧后新迷宫, 比分保留 |
| Laika | 目标优先级 AI, 每回合重建, 会躲避一切子弹(含自己的) |

详细忠实度说明 (含全部已知近似) 见 **PORT_NOTES.md**。
