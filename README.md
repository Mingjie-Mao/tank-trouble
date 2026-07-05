# Tank Trouble — 原版 1:1 复刻 + 击败 Laika 的 RL 训练

基于反编译源码逐行移植的 Tank Trouble vs Laika 模式（逻辑/常量/帧序与原版
Flash 一致），并在其上训练强化学习智能体击败原版 AI「Laika」。

**当前最好成绩：稳定 ~50% 胜率**（1500 局评估均值 50.5%，自杀 0，平局 ≈0）。
即从对手手里拿到半数胜场。详见下方「训练成果」。

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

## 二、训练成果与结论

### 做了什么，到了什么程度

| 阶段 | 真实胜率 | 说明 |
|---|---|---|
| 随机策略 | 5% | 87% 死于自己的反弹弹 |
| 手写猎杀脚本 | 30% | RL 的「及格线」基准 |
| **v1**（纯胜负奖励） | 19% | ❌ 卡死——学会「苟活躲弹」的局部最优，不敢开火 |
| **v2**（+ 开火塑形）★ | **50–53%** | ✅ 突破。奖励「瞄准后果断开火」，根治苟活+自杀 |
| v3（+ 闪避奖励） | 37% | ❌ 失败。闪避奖励太弱被模型忽略，已放弃 |

**最终交付模型 = v2**（`training/models/best_model.zip`）。

### 关键决策复盘（给接手队友）

1. **v1→v2 的转折**：用 200 局对局数据定位到「每局只打 1.1 发弹」的苟活陷阱
   （躲到平局比开火冒险更划算）。重构奖励激励果断开火后，胜率翻 2.5 倍。
2. **顶住「假平台」**：训练中确定性评估多次走平，但通过看**训练胜率+熵**能
   区分「真平台」和「收敛滞后」，避免了过早干预。→ 训练回调的 100 局评估
   噪声极大（系统性偏低 8–11 个点），**一律以 500 局离线评估为准**。
3. **v3 及时止损**：行为数据（危险步 6.9% vs 6.2%、闪避 0.37% vs 0.3%）证明
   闪避奖励**根本没改变模型行为**——稀疏(+0.15)信号被 PPO 忽略。避免浪费算力。

---

## 三、观察到的 AI 行为（看录像总结，重要）

当前 v2 模型胜率 ~50%，但**打法是「投机」而非「真本事」**，天花板由此而来：

1. **不会跑图**。它不主动穿越迷宫找 Laika，而是**原地等 Laika 过来**。
2. **以量取胜，不靠瞄准**。策略是**朝 Laika 方向一次性泼很多发子弹**（哪怕
   中间隔着墙也照准方向打），本质是「以逸待劳 + 乱拳打死老师傅」。
3. **没学会躲避**。根因疑为**地图种子机制**：原版大量存在「双方出生就很近」
   的布局，很多胜局其实是**刷到近距离图 → 谁瞄准快谁赢**的对射。所以调闪避
   它无动于衷——与其学躲，不如赌运气刷个近图直接对射。
4. **在摆烂**。相当多回合它消极拖时间。

> 一句话：v2 学到的是「站桩泼弹 + 赌近距离图」，不是「机动 + 精确打击」。
> 要突破 50%，必须逼它学会主动进攻和真正的走位躲避。

---

## 四、接下来的思路（队友可直接接手）

按优先级排列，都值得各跑一轮 300 万步验证（约 20 分钟/轮）：

### 思路 A：加大时间惩罚，逼它主动出击（★ 首选，对应观察 4）
当前 `R_TIME_PENALTY = -0.002` 太温和，纵容摆烂。改为**随时间递增的惩罚**
（局面拖得越久扣得越狠），逼它尽快解决战斗而非站桩等运气。
- 改 `training/tt_gym_env.py` 的 `R_TIME_PENALTY`，或在 `step()` 里改成
  `reward += R_TIME_PENALTY * (self._frames / TRUNCATE_FRAMES)` 让惩罚随帧数放大。
- 也可同时**缩短 `TRUNCATE_FRAMES`**（当前 2500），压缩苟活空间。

### 思路 B：改进闪避奖励（对应观察 3，v3 失败的正确修法）
v3 的方向没错，错在信号**稀疏又微弱**。改成**密集连续信号**：每帧按「距最近
来袭弹多远」给梯度奖励（越靠近来袭弹扣分越多），而非只在「化解锁定」的稀疏
瞬间给 +0.15。这样 PPO 才学得到。相关代码：`_incoming_will_hit()` 已能判定
威胁，把它改成返回「最近来袭弹的距离」即可做梯度。

### 思路 C：地图课程/去偏（对应观察 3 根因）
如果对射胜局是「近距离图运气」，可在训练时**过滤或降采样近距离出生的种子**，
逼它在「必须跑图接近」的图上学习机动。`Game` 的出生点由种子决定，可在 env
的 `reset()` 里检测双方初始格距离、太近就换种子。

### 思路 D：先手奖励（对应观察 2）
奖励「比 Laika 更早开出有效弹」，鼓励主动压制而非对灌。

---

## 五、训练相关指令

```bash
# 安装依赖 (torch 已有的话只需后两个)
pip install stable-baselines3 tensorboard gymnasium

# 训练 (reward-version: 1=纯胜负 2=开火塑形(最优) 3=闪避实验)
python3 training/train_ppo.py --steps 10000000 --envs 12 --reward-version 2 --tag v2
#   后台+防睡眠: 前面加 caffeinate -i, 末尾加 > training/train_v2.log 2>&1 &

# 评估 (务必用 500+ 局, 回调的 100 局评估噪声太大不可信)
python3 training/evaluate.py --policy model --model training/models/best_model.zip --n 500 --seed 950000
python3 training/evaluate.py --policy hunter --n 200      # 手写脚本基线 (~30%)
python3 training/evaluate.py --policy random --n 200      # 随机基线 (~5%)

# 看录像 (tkinter 里回放模型 vs Laika, 可指定种子复现某一局)
python3 training/watch.py --policy model --model training/models/best_model.zip
python3 training/watch.py --policy hunter --seed 910007

# 训练曲线
tensorboard --logdir training/tb_logs
```

**算力说明**：瓶颈在 CPU 采样（游戏引擎+Laika 决策），策略网络是小 MLP，
本机 ~4400 步/秒，1000 万步约 40 分钟。**不需要 GPU**；若要大扩采样，租
**多核 CPU** 机器（非 GPU）性价比最高。

---

## 六、文件与产物

```
training/
├── tt_gym_env.py         # 训练环境 (v1/v2/v3 三档奖励, 76维观测: 自车+敌车+24射线+子弹)
├── baselines.py          # 基线策略 (idle/random/hunter 手写猎杀脚本)
├── evaluate.py           # 标准评估协议 (固定种子集, 胜/负/平/自杀/局长)
├── train_ppo.py          # PPO 训练 (12并行环境, 自动保存最优)
├── watch.py              # tkinter 录像回放
├── models/
│   ├── best_model.zip        # 最终模型 = v2 (~50%) ← 用这个
│   ├── v2_best_53pct.zip     # v2 备份
│   └── v1_best_19pct.zip     # v1 对照
├── train_run1.log        # v1 训练日志
├── train_v2.log          # v2 训练日志 (胜率从 5%→52% 全过程)
└── train_v3.log          # v3 训练日志 (闪避实验, 失败)

tank_trouble_original/     # 游戏本体 (纯 Python 零依赖, 1:1 移植)
├── constants.py  maze.py  game.py  laika.py  env.py
play_tank_trouble.py       # 本地游玩入口
test_original_port.py      # 25 项验证
PORT_NOTES.md              # 移植忠实度说明 ★
swf_decompiled/            # 反编译源码 (81 个 .as 文件)
GAME_MECHANICS_ANALYSIS.md # 游戏机制解读
```

---

## 七、AI 训练接口（自建训练时用）

```python
from tank_trouble_original import TankTroubleEnv

env = TankTroubleEnv(seed=42)      # tank0 = 智能体, tank1 = Laika
state = env.reset()
state, events = env.step({"forward": True, "turn_left": False, "fire": True})

# state:  完整真值 (坦克位姿/子弹/迷宫墙体/回合状态/比分/Laika当前目标)
# events: fire / bounce / hit / destroy / round_end / new_round
# 无头 ~6000+ 帧/秒 (实时的 240 倍), 种子完全可复现
```

## 八、游戏核心机制速览（全部来自反编译源码）

| 机制 | 原版行为 |
|---|---|
| 帧率 | 25 FPS 固定步进 |
| 迷宫 | random(4) 模板法, W∈[4,12] H∈[4,10], SCALE 每回合变化 |
| 坦克 | 每帧 5 子步移动, 21 碰撞点, 贴墙滑动, 10° 转向吸附 |
| 子弹 | 每帧 7 子步, X/Y 独立反弹, 250 帧寿命, 最多 5 发 |
| 命中 | 出膛即致命, **可打死自己** (贴墙开火会反弹自杀) |
| 回合 | 死亡→125帧继续→50帧冻结计分→5帧后新迷宫, 比分保留 |
| Laika | 目标优先级 AI, 每回合重建, 会躲避一切子弹(含自己的) |

详细忠实度说明（含全部已知近似）见 **PORT_NOTES.md**。
