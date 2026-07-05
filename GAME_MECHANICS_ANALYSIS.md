# Tank Trouble Flash Game - 完整机制分析

## 概览
Tank Trouble 是一个多人坦克对战游戏，已成功反编译。包含以下核心系统：

---

## 1. 游戏参数 (Global Settings)

### 地图和视窗
- **SCALE**: 每个格子的像素大小 (基础单位)
- **MOVIEWIDTH/HEIGHT**: 712x490 像素
- **帧率**: 25 FPS

### 坦克物理
```
forwardSpeed  = 4 * (SCALE / 50)      // 前进速度
backUpSpeed   = 2.5 * (SCALE / 50)    // 倒退速度
turnSpeed     = 10 degrees/frame       // 转向速度
```

### 武器参数
```
BULLETSPEED   = (SCALE / 50) * speed_multiplier
BULLETLIFETIME = frames until bullet disappears
BULLETDEADLY  = frames until bullet can hit tanks (延迟伤害判定)
BULLETHITCHECKINTERVALS = collision detection frequency

FRAGSPEED     = speed of frag bomb
FRAGBOMBDETONATEDIST = explosion radius
FRAGBOMBSAFETYDIST = safe distance to detonate

GATLINGSPEED  = rapid fire bullet speed
LASER: instant hit (aimed beam)
```

---

## 2. 坦克系统

### 坦克结构
```actionscript
Tank {
  x, y           // 世界坐标
  _rotation      // 0-360°, 相对炮塔方向
  alive: bool
  
  // 速度
  forwardSpeed
  backUpSpeed
  turnSpeed
  
  // 控制状态
  forward: bool
  backup: bool
  turnLeft: bool
  turnRight: bool
  fire: bool
  
  // 武器装备
  currentWeapon: "bullet" | "laser" | "frag" | "gatling"
  currentEquipment: "aimer" | "shield" | "mine" | ...
  
  // 武器计数
  bulletsFired: int           // 当前已发射子弹数
  laserReady: bool            // 激光就绪
  fragFired: bool             // 已发射碎片炸弹
  lastFrag: Frag object
  gatlingReady: bool
  homingReady: bool
  electricReady: bool
  deathRayReady: bool
  minesLayed: int
}
```

### 碰撞检测点
坦克周围有多个碰撞检测点：
- **hitPointsFront**: 前方 (6个点)
- **hitPointsRear**: 后方 (5个点)
- **hitPointsLeft**: 左侧 (5个点)
- **hitPointsRight**: 右侧 (5个点)

这允许更细致的碰撞检测，而不仅仅是中心点。

### 主游戏循环 (onEnterFrame)
```
每帧 (25fps = 40ms):
1. 更新颜色和UI
2. 读取控制输入 (鼠标/AI)
3. 处理坦克移动
4. 更新位置和旋转
5. 碰撞检测
6. 武器发射
7. 条件检查 (存活等)
```

---

## 3. AI 系统 (DefineSprite_186_tankTroubleAI)

### AI 决策树

AI 不是直接操作坦克，而是产生 **目标 (Goals)** 和 **行动 (Actions)**：

```
Goal {
  goal: string        // 目标类型
  priority: float     // 优先级 (0-1)
  period: int         // 决策周期
  id: int             // 唯一ID
  updateContinuously: bool
}

Action {
  action: string      // 行动类型
  delay?: int         // 延迟帧数
  ... (其他参数取决于行动类型)
}
```

### 目标类型 (按优先级顺序评估)

1. **dodgeBullet** (优先级: 1)
   - 躲避敌方子弹
   - 使用距离梯度跟踪

2. **dodgeFragbomb** (优先级: 1)
   - 远离碎片炸弹

3. **dodgeFrag Fragment** (优先级: 1)
   - 躲避碎片炸弹爆炸碎片

4. **dodgeGatlingBullet** (优先级: 1)
   - 躲避盖特林枪

5. **dodgeLaser** (优先级: 1)
   - 躲避激光瞄准器

6. **shootAfter** (优先级: varies)
   ```
   priority = distance < LONGESTPATHTONOTHESITATETOSHOOT 
              ? 1 
              : (LONGESTPATHTOSHOOT - distance) / LONGESTPATHTOSHOOT * currentAggresiveness
   ```
   - 追踪敌方坦克并射击
   - 如果距离太远 (> LONGESTPATHTOSHOOT = 7), 优先级降低

7. **goForCrate** (优先级: varies)
   - 前往获取武器箱 (仅当使用普通子弹时)
   ```
   priority = (MAXCELLDISTTOGOFORCRATE - distance) / MAXCELLDISTTOGOFORCRATE 
              * GREEDY (1.0) 
              * (MAX_BULLETS - bulletsFired) / MAX_BULLETS
   ```

8. **sprayBullets** (优先级: varies)
   - 使用盖特林枪快速射击

9. **runAway** (优先级: varies)
   ```
   priority = (LONGESTPATHTORUN - distance) / LONGESTPATHTORUN 
              * COWARDNESS (0.7) 
              * (bulletsFired / MAX_BULLETS)
   ```
   - 在弹药耗尽时逃离

10. **backAway** (优先级: stuckTime / MAXSTUCKTIME)
    - 卡住时后退 (MAXSTUCKTIME = 1 frame!)

11. **driveTo** (优先级: 0.1)
    - 向随机敌人移动 (空闲时)

### 每帧决策流程

```python
makeDecisionsAndUpdateGoal():
  1. 收集所有子弹、碎片、激光信息
  2. 评估每个目标，选择最高优先级的
  3. 目标优先级随时间衰减 (* 0.9 每帧)
  4. 调用 decideActionsToAchieveGoal() 转换为具体行动

decideActionsToAchieveGoal():
  // 根据当前目标生成行动列表
  // 例如 "shootAfter" -> ["turnTo angle X", "fireWeapon delay=5"]

setInputToDoActions():
  // 逐个执行行动
  // 转换为坦克的控制信号 (forward, backup, turnLeft, turnRight, fire)
```

### AI 性格参数

```
AGGRESIVENESS = 0.5        // 主动性 (越高越好战)
COWARDNESS = 0.7           // 逃跑倾向
GREEDY = 1.0               // 对武器箱的渴望度

IDLEDRIVETOWARDENEMYPRIORITY = 0.1  // 空闲时探索优先级

// 躲避相关
MAXTIMETODODGEBULLET = 75 frames      // 躲避预测窗口
MAXDISTTODODGEBULLET = 4 * SCALE      // 最大躲避距离
MAXCELLDISTTODODGEBULLET = SCALE * 75 / 50 (BulletSpeed)

// 其他躲避类似...

// 射击相关
LONGESTPATHTOSHOOT = 7         // 最远可以射击的距离 (格子)
LONGESTPATHTONOTHESITATETOSHOOT = 2  // 不犹豫的距离 (优先级 = 1)
MAXCLOSESTDISTANCE = SCALE * 2  // 考虑射击的最近距离

// 逃离相关
LONGESTPATHTORUN = 10           // 开始逃离的距离
```

### 躲避算法 (dodgeTrajectories)

关键函数：计算子弹轨迹，预测碰撞点

```
输入: 子弹列表, 当前位置, 躲避参数
输出: 躲避目标位置

算法:
  对每个子弹:
    1. 预测子弹轨迹 (直线运动)
    2. 找到子弹与坦克的最近接近点
    3. 如果距离 < maxDistToDodge 且时间 < maxTimeToDodge:
       计算垂直于子弹轨迹的逃脱点
       使用梯度下降找到最安全的位置
    4. 考虑墙壁反弹
```

---

## 4. 地图系统

### 地图数据结构
```
maze: 2D array
  - 0: 空地 (可通行)
  - 1: 墙壁 (不可通行)
  - 其他值可能表示特殊瓷砖

distancesForMaze: 4D array
  distancesForMaze[fieldX][fieldY][cellX][cellY]
  = A* 或 BFS 距离从 (fieldX, fieldY) 到 (cellX, cellY)
  (预计算的路径查找结果缓存)
```

### 路径查找函数
```
getShortestPathWithDistances(maze, distanceMap, startX, startY, endX, endY)
  → 返回路径数组: [{x, y}, {x, y}, ...]

followGradientPathWithDistances(maze, distanceMap, startX, startY, steps)
  → 沿着距离梯度移动 `steps` 步

followGradientPathWithDistancesAndDeadEnds(maze, distanceMap, deadEnds, x, y, steps)
  → 避免死胡同的梯度跟踪
```

---

## 5. 碰撞检测系统

### 子弹碰撞 (checkPathForCollision)

```python
checkPathForCollision(x, y, xSpeed, ySpeed, hitCheckInterval, maxTime, lifetime):
  """
  模拟子弹运动，检测墙壁碰撞和反弹
  
  参数:
    x, y: 起始位置
    xSpeed, ySpeed: 每帧移动量
    hitCheckInterval: 每隔N次循环检查一次
    maxTime: 最大模拟时间 (帧)
    lifetime: 子弹寿命 (帧)
  
  返回:
    { x, y, xSpeed, ySpeed, t } - 碰撞点和反弹速度
    undefined - 无碰撞
  """
  
  循环直到lifetime < 0:
    x += xSpeed
    y += ySpeed
    
    if mazemc.hitTest(x, y):  // 与墙壁碰撞
      处理反弹 (xSpeed 或 ySpeed 反向或两者)
      return 碰撞信息
```

### 坦克与墙壁碰撞
使用 expandedHitCheck() 检查坦克周围的多个点

---

## 6. 武器系统

### 普通子弹
- **Speed**: BULLETSPEED
- **Lifetime**: BULLETLIFETIME
- **Deadly Delay**: BULLETDEADLY (击中前延迟)
- **Max Bullets**: settingsMaxBullets
- **Hit Detection**: 每 BULLETHITCHECKINTERVALS 帧检查一次

### 激光
- **Type**: 瞄准光束 (Aimer)
- **Hit**: 即时
- **Target**: 来自另一个坦克的瞄准器 (DefineSprite_177_aimer)
- **Dodge**: 检查激光线段与坦克是否相交

### 碎片炸弹 (Frag Bomb)
- **Behavior**: 抛物线弧线
- **Detonation**: 靠近敌人或按下按钮
- **Blast Radius**: FRAGBOMBDETONATEDIST
- **Fragment**: 爆炸产生多个碎片 (DefineSprite_166_fragbombfragment)

### 盖特林枪
- **Type**: 连射
- **Speed**: GATLINGSPEED
- **Duration**: GATLINGLIFETIME

### 特殊武器
- **Homing Bullet**: 追踪目标
- **Electric Bullet**: 电击效果
- **Remote Bullet**: 远程操控
- **Mine**: 地雷 (DefineSprite_217_mine)
- **Shield**: 护盾 (DefineSprite_218_shield)
- **Death Ray**: 死亡射线

---

## 7. 射击算法 (checkBulletPath)

```python
checkBulletPath(angle):
  """
  沿着给定角度预测子弹轨迹
  
  返回:
    { result: "HIT" | "NOTHING" | "SUICIDE", time, closest }
  """
  
  xSpeed = cos(angle - 90°) * BULLETSPEED
  ySpeed = sin(angle - 90°) * BULLETSPEED
  
  模拟BULLETLIFETIME帧:
    更新位置
    检查墙壁碰撞 (反弹)
    检查坦克碰撞:
      如果击中自己 → "SUICIDE"
      如果击中敌人 → "HIT"
    计算最近距离到其他坦克
  
  返回结果
```

### AI 瞄准 (shootAfter 目标)

```python
// 直接射击
if 敌人在直线上:
  targetAngle = atan2(enemy.y - myTank.y, enemy.x - myTank.x)
  
// 间接射击 (反弹)
else:
  尝试±1, ±2, ±3 * turnSpeed 角度
  选择最近或有击中的角度
```

---

## 8. 行动类型

### 移动行动
```
"driveToField"   - 驾驶到格子中心
"driveToPos"     - 驾驶到绝对坐标
"forward"        - 向前移动 N 帧
"backup"         - 后退 N 帧
"forwardAndTurn" - 向前并转向
"backupAndTurn"  - 后退并转向
```

### 转向行动
```
"turnTo" - 转到特定角度 (自动计算最短路径)
```

### 攻击行动
```
"fireWeapon" - 发射当前武器
"detonate"   - 引爆碎片炸弹
"sprayBullets" - 连续射击
```

### 其他
```
"idle" - 什么都不做
```

---

## 9. 游戏常量总结

| 常量 | 值 | 用途 |
|------|-----|------|
| SCALE | grid_size (像素) | 基础度量单位 |
| BULLETSPEED | SCALE / 50 * multiplier | 子弹速度 |
| BULLETLIFETIME | ~100-200 | 子弹最大生存时间 |
| TANKS | 游戏中坦克数 | 用于循环 |
| MOVIEWIDTH | 712 | 游戏宽度 |
| MOVIEHEIGHT | 490 | 游戏高度 |
| FPS | 25 | 帧率 |
| settingsMaxBullets | 用户设置 | 最大子弹数 |

---

## 10. 实现提示

### 重写为 Python 时需要注意

1. **坐标系**: Flash 使用左上角为原点，Y向下。Python/Pygame 通常也是这样。

2. **角度**: Flash 使用 0-360°，其中 0° 向右，90° 向下。请确保转换正确。

3. **碰撞检测**: 使用 `pygame.mask.Mask` 或射线投射实现精确碰撞。

4. **距离缓存**: `distancesForMaze` 是预计算的，在初始化时执行 BFS。

5. **AI 循环**: 每帧执行决策 → 行动 → 控制转换。

6. **物理**: 使用简单的离散时间步进 (每帧更新一次位置)。

7. **武器生命周期**: 每个子弹都是独立对象，需要跟踪其寿命和致命延迟。

---

## 11. 关键源文件

已反编译到 `/Users/cichlidfish/tank_trouble/swf_decompiled/scripts/`：

- `DefineSprite_186_tankTroubleAI/frame_1/DoAction.as` - **核心 AI 代码** ⭐
- `DefineSprite_135_tank/frame_1/DoAction.as` - 坦克物理和控制
- `DefineSprite_175_bullet/frame_1/DoAction.as` - 子弹逻辑
- `DefineSprite_176_laser/frame_1/DoAction.as` - 激光系统
- `DefineSprite_185_gatling/frame_1/DoAction.as` - 盖特林枪
- 以及其他武器/设备脚本
- `frame_1/DoAction.as` - 主游戏循环

---

## 下一步：重写为 Python

建议使用以下技术栈：
- **Pygame**: 图形和事件处理
- **OpenAI Gym**: AI 环境接口
- **NumPy**: 数学运算
- **NetworkX/SciPy**: 路径查找和距离计算

要开始实现，需要：
1. 设计游戏循环和物理引擎
2. 实现地图和碰撞检测
3. 创建坦克和武器系统
4. 移植 AI 决策逻辑
5. 包装成 Gym 环境供 RL 算法训练

