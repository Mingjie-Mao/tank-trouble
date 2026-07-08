# Tank Trouble 1:1 复刻 — 移植对照与忠实度说明

**移植方式**: 逐行对照反编译源码 (`swf_decompiled/scripts/`) 重写为 Python。
**范围**: vs Laika 单人模式的完整逻辑 (原版该模式 `settingsActiveWeapons=[]`，
无武器箱、纯子弹对决，故其他 8 种武器不会出现)。

---

## 1. 文件对照表

| Python 文件 | 反编译源码 | 内容 |
|---|---|---|
| `tank_trouble_original/constants.py` | frame_53:2152-2221 + SWF header | 全部游戏常量 |
| `tank_trouble_original/maze.py` | frame_53:1-586 | createMaze / calcReachable / calcDistances / findDeadEnds / getShortestPathWithDistances / followGradientPath* |
| `tank_trouble_original/game.py` Tank | DefineSprite_135_tank | 坦克移动/碰撞/开火 (onEnterFrame 全逻辑) |
| `tank_trouble_original/game.py` Bullet | DefineSprite_175_bullet | 子弹 7 子步移动 + 反弹 + 命中 |
| `tank_trouble_original/game.py` Game | frame_53 (\_root.onEnterFrame, setupBattle, setupStandardMaze, deployTank, destroyTank, fireBullet, weaponReady, lockedControl) | 主循环/回合/生成 |
| `tank_trouble_original/laika.py` | DefineSprite_186_tankTroubleAI (全部 1386 行) | Laika AI 完整移植 |
| `tank_trouble_original/env.py` | — | AI 训练数据接口 (新增) |
| `play_tank_trouble.py` | — | 本地渲染入口 (tkinter, 25 FPS) |

## 2. 关键机制的精确复刻点

### 帧率与执行顺序
- 25 FPS 固定帧步进 (SWF header)。
- 每帧顺序: `_root` 逻辑 → tank0 → tank1(Laika) → 子弹(按创建序)。
- **当帧发射的子弹下一帧才开始移动** (Flash attachMovie 语义)。

### 坦克移动 (tank:322-410)
- 每帧分 **5 子步** (STEPS=5)；先无碰撞走完 5 步，若终点碰墙则回滚逐步重走。
- 前进 4×(SCALE/50)/帧，倒车 2.5×(SCALE/50)/帧，转向 10°/帧。
- 21 个碰撞探测点 (前 6 / 后 5 / 左 5 / 右 5)，本地坐标经缩放+旋转变换。
- 前进只挡前点、倒车只挡后点 → 原版特有的"贴墙滑动"。
- 转向后吸附到 10° 倍数 (`(360+rot)%10` 取整逻辑)。
- `_rotation` 写入即归一化到 (-180,180] (Flash 语义, AI 依赖)。
- 坦克之间**不碰撞** (原版如此)。

### 弹道判定 (bullet:10-167)
- 每帧 **7 个子步**，每子步 0.643×(SCALE/50) px。
- 反弹: 撞墙后分别测试"只反X"与"只反Y"，都撞则双反 (逐行复刻)。
- `deadly=0`: 出膛即可命中**包括发射者自己** (可反弹自杀，原版特性)。
- 命中判定在 7 子步完成后进行一次 (帧末采样)。
- 寿命 250 帧，到期归还弹数。最多同时 5 发 (settingsMaxBullets)。

### 墙壁碰撞
- 墙 = 有厚度线条: 厚 2×floor(SCALE/16)，方头端帽 (两端外延半厚)。
- 坐标按原版 `Math.floor(格线×SCALE)` 取整。
- 命中 = 点是否落入线条笔画内 (`hitTest(x,y,true)` 等价)。
- `WallGrid` 空间索引仅加速，结果与逐段判定**完全一致** (轴对齐线段+方头帽的笔画恰为外扩矩形)。

### 迷宫生成 (frame_53:1-42, 1838-1873)
- `tempmaze[x][y]=random(4)` 随机模板法 (不是递归回溯!):
  下墙 ⟺ `t[x][y+1]==2 || t[x+1][y+1]==0`；左墙 ⟺ `t[x][y]==1 || t[x][y+1]==3`。
- 尺寸: W=random(9)+4 (4..12)，H=random(7)+4 (4..10)。
- SCALE = min(400/(H+0.125), 692/(W+0.125)) — 每回合随迷宫尺寸变化。
- 连通区 < 2×TANKS 时整体重掷。出生格从连通区随机不重复抽取。
- 出生朝向 = random(32)×11.25°。
- AS2 越界语义 (`undefined==0` 为 false) 和 NaN 传播 (未达格距离参与运算) 均已复刻。

### 回合循环 (frame_53:2456-2535)
- 死亡 → `endCount=125` 且继续运行 (幸存者仍可被反弹弹打死，再触发重置 125)。
- `endCount==50` → 全场冻结 + 计分 (存活者 +1)。
- `endCount==0` → 清场；5 帧后新迷宫新回合，比分保留。
- Laika AI **每回合重建** (常量依赖当回合 SCALE)。

### Laika AI (1386 行全移植)
- 目标优先级系统: dodgeBullet / dodgeFragbomb / dodgeLaser / shootAfter /
  goForCrate / sprayBullets / runAway / backAway / driveTo / idle。
- `dodgeTrajectories`: 最近逼近点 + 反弹预判 (含 checkBounce 分支)。
- `checkBulletPath`: 以 1 子步/帧粗粒度模拟 83.3 帧弹道 (原版就是粗的)，
  含 SUICIDE 判定与"最近距离"追踪 (曼哈顿距离 + 格距 ≤2 限制)。
- `shootAfter`: 先测直线弹道，再 ±turnSpeed×k² 随机探测 3 角。
- 性格常量: AGGRESIVENESS=0.5, COWARDNESS=0.7000000000000001, GREEDY=1
  (浮点字面量与反编译输出一致)。
- runAway 的求和距离图保留了原版 (W-1)×(H-1) 的 off-by-one。
- 动作栈 (LIFO) 与 setInputToDoActions 的弹出/复推/窥视逻辑逐行复刻。
- **Laika 会躲避自己的子弹** (dodgeTrajectories 扫描所有子弹, 原版行为)。

## 3. 已知近似 (与原版的全部差异)

1. **坦克命中形状**: 原版用矢量图形做 `hitTest(点, true)`；本移植用
   base 矩形 (61×81) ∪ 炮管矩形 (17 宽, 尖端 y=-55) 近似，
   尺寸取自 SWF 矢量精确边界 (shape78/shape82)，误差 < 2px (圆顶均在 base 矩形内)。
2. **hitTest(点, false) 包围盒**: 用 frame1 联合边界 (-30.5,-55,30.5,40.5)
   旋转后的 AABB, 与 Flash 语义一致。
3. **计分服务器**: 原版排位模式等待服务器响应；本地版走"非排位"分支
   (endCount==50 时立即计分)，去除网络依赖。
4. **随机数**: Python `random.Random` 替代 Flash 的 RNG (分布相同:
   `random(n)=floor(rand*n)`)。序列不同但统计特性一致，支持 seed 复现。
5. **视觉特效** (烟雾/碎片/Laika 狗动画): 逻辑无关，未移植。

## 4. 精确提取的图形数据 (来自 SWF 矢量边界)

| 元素 | 本地边界 | 说明 |
|---|---|---|
| base (shape78) | [-30.5,-40.5 .. 30.5,40.5] | 61×81, 注册点居中 |
| turret frame1 (shape82) | [-22.5,-55.0 .. 22.5,22.5] | 45×77.5 |
| tank 整体 | [-30.5,-55.0 .. 30.5,40.5] | bbox 用 |
| bullet (shape167) | ±3.5 | 视觉半径 |
| 显示缩放 | 0.55×SCALE % | deployTank |

推导: hitPointsFront[4/5] = (±45/6, -77.5/16×11) = (±7.5, -53.28)。
出膛点 SCALE×4.5/16 (=0.281×SCALE) 在炮管尖端 (0.303×SCALE) 之内 1px，
但子弹首次命中判定发生在移动一帧后 → **直射不自杀** (与原版一致)；
贴墙开火反弹弹可打死自己 (与原版一致)。

## 5. 验证结果 (test_original_port.py)

- 25/25 项通过: 常量、弹速、反弹、回合循环、种子确定性等。
- 行为统计: Laika vs 随机乱动玩家 362:2 (平均 4.2 秒/杀)；
  vs 静止玩家 62:5 (双亡 5)。
- 无头性能: ~6000-12000 帧/秒 (实时 25fps 的 240-470 倍)，适合 RL 训练。

## 6. 使用

```bash
# 本地游玩 (vs Laika)
python3 play_tank_trouble.py [--seed N]
python3 play_tank_trouble.py --two-players   # 本地双人

# 验证
python3 test_original_port.py

# AI 训练
python3 -c "
from tank_trouble_original import TankTroubleEnv
env = TankTroubleEnv(seed=42)
state = env.reset()
state, events = env.step({'forward': True, 'fire': True})
print(state['tanks'][0], events)
"
```
