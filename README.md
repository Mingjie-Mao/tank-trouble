# 坦克对战游戏本体与老师 AI

这个目录现在只保留两类内容：

- 游戏本体
- 一个符合 `plan.md` 思路的规则型老师 AI

这意味着这里不再保留旧的课程环境、历史模型、训练脚本和策略运行时。当前版本的定位不是“已经开始完整训练”，而是：

> 先稳定一个可玩、可观察、可扩展、可接训练的 Tank Trouble 风格环境。

如果你要先审整体路线，请先看根目录的 [plan.md](/Users/cichlidfish/deeprl/plan.md)。

## 当前版本在整个计划中的位置

按照 `plan.md`，完整路线是：

```text
TeacherBot
→ 行为克隆 Behavior Cloning
→ DAgger
→ 强化学习微调
→ 自博弈
→ 域随机化
→ 与外部黑盒 AI 对测
```

当前这个目录只完成了第一阶段所需的基础设施：

- 一个类 Tank Trouble 的对战环境
- 一个规则型老师 `TeacherTankBot`
- 一个可视化手动游玩入口

也就是说，我们现在还处在：

```text
环境搭建 + TeacherBot 基线阶段
```

后续训练都会建立在这套环境上，而不是重新换一套游戏逻辑。

## 当前保留的文件

### `tank_env.py`

核心游戏环境。

它负责：

- 地图生成与固定地图配置
- 迷宫线墙几何
- 坦克移动、转向、碰撞
- 子弹发射、飞行、反弹、自伤
- 回合重置
- 观测空间与动作空间
- 敌方老师 AI 的接入
- 画面渲染

如果你之后要继续改游戏规则、观测空间、动作空间、奖励接口，优先看这个文件。

### `bots.py`

规则型老师 AI。

当前保留的主角是：

- `TeacherTankBot`

它的行为优先级是：

1. 卡住时先尝试脱困
2. 发现明显来弹时先躲避
3. 有直线射界时瞄准并开火
4. 否则沿导航点追击玩家

这不是最终要战胜的神经网络模型，而是一个“AI 老师 / 陪练 / 基线对手”。

### `play_tank.py`

本地可视化游玩入口。

它会打开一个 Tkinter 窗口，让你直接手动操作自己的坦克，对战 `TeacherTankBot`，实时观察：

- 双方位置
- 出生点
- 视线情况
- 场上子弹数量

### `README.md`

就是你现在正在看的这份说明。

## 当前游戏能做什么

现在这套环境已经具备：

- 俯视角 2D 坦克对战
- 迷宫型地图
- 随机地图与小型固定地图
- 细线墙体风格的迷宫边界
- 坦克连续移动和转向
- 子弹反弹
- 自己发出的子弹也可能打到自己
- 一个会追击、躲弹、瞄准、开火的老师 AI
- 本地可视化对战
- Gymnasium 风格的 `reset / step / render` 接口

## 如何运行游戏

推荐两种启动方式。

### 方式一：在项目根目录直接运行

```bash
python3 /Users/cichlidfish/deeprl/tank_trouble/play_tank.py
```

### 方式二：进入游戏目录再运行

```bash
cd /Users/cichlidfish/deeprl/tank_trouble
python3 play_tank.py
```

这两种方式都可以。

## 常用启动参数

当前入口脚本支持：

- `--seed`：固定随机种子
- `--fps`：控制刷新帧率
- `--arena-preset`：切换地图预设

例如：

```bash
python3 /Users/cichlidfish/deeprl/tank_trouble/play_tank.py --arena-preset small_fixed --fps 20 --seed 0
```

目前支持的地图预设有：

- `micro_fixed`
- `small_fixed`
- `cover_fixed`
- `maze_random`

建议你先这样试玩：

- `small_fixed`：最容易看清双方行为
- `maze_random`：最接近后续泛化训练目标

## 键盘操作

- `W / S`：前进 / 后退
- `A / D`：左转 / 右转
- `Space`：开火
- `R`：重新生成并重开地图
- `Esc`：退出

## 运行前需要什么环境

当前代码依赖：

- `numpy`
- `gymnasium`
- `Pillow`
- `tkinter`

如果你本机的 `python3` 已经装好这些依赖，直接运行即可。

## 这个版本为什么没有训练脚本

因为你现在要先审 `plan.md`，并重新整理项目方向，所以我把历史训练部分先清掉了，只保留：

- 稳定的游戏本体
- 一个符合计划方向的老师 AI

这样你后面再定训练路线时，不会被旧实验文件干扰。

## 按当前 plan，接下来要补什么

如果严格按照 `plan.md` 继续推进，下一批最应该补的是：

1. TeacherBot 轨迹采样脚本
2. 行为克隆训练脚本
3. DAgger 数据回流脚本
4. 强化学习微调脚本
5. 自博弈训练与评估脚本

也就是说，下一步不是先乱改游戏，而是先把：

```text
TeacherBot -> 数据采样 -> 模仿学习
```

这一段接完整。

## 之后如果要训练，学习数据怎么来

虽然当前目录里已经不保留训练脚本，但这套环境本身已经具备训练接口雏形。

环境是 Gymnasium 风格的：

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

所以未来做强化学习时，学习数据不是手工标注文件，而是通过智能体和环境交互在线生成。每一步都会自然产出：

- `obs`
- `action`
- `reward`
- `terminated`
- `truncated`
- `info`

如果后续你要按 `plan.md` 开始实现：

- 行为克隆
- DAgger
- PPO / SAC 微调
- 自博弈

那我们可以直接在这套环境上继续往外搭训练骨架，而不需要重写游戏本体。

## 目前最适合你的阅读顺序

建议你先按这个顺序审：

1. 根目录的 `/Users/cichlidfish/deeprl/plan.md`
2. `/Users/cichlidfish/deeprl/tank_trouble/tank_env.py`
3. `/Users/cichlidfish/deeprl/tank_trouble/bots.py`
4. `/Users/cichlidfish/deeprl/tank_trouble/play_tank.py`

这样你会先看清楚目标，再看环境，再看老师 AI，最后看可视化入口。
