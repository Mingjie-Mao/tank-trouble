"""
Tank Trouble 原版游戏常量 — 1:1 提取自反编译源码
来源: swf_decompiled/scripts/frame_53/DoAction.as (第 2152-2221 行)
     以及 DefineSprite_135_tank / frame_43 / SWF header

所有数值与原版 Flash 完全一致，请勿随意修改。
"""

# ---- 帧率 (SWF header: Frame rate: 25.0) ----
FPS = 25

# ---- 画面布局 (frame_53:2152-2154) ----
MOVIEWIDTH = 692
MOVIEHEIGHT = 480
HEIGHTTOBOTTOM = 80

# ---- 武器 (frame_53:2155-2212) ----
STARTWEAPON = "bullet"
BULLETSPEED = 4.5            # 像素/帧 (以 SCALE=50 为基准, 实际乘 SCALE/50)
BULLETLIFETIME = 250         # 帧 (25fps 下 = 10 秒)
BULLETHITCHECKINTERVALS = 7  # 每帧子步数 (每子步移动 speed/7)
BULLETDEADLY = 0             # 出膛立即可命中 (包括发射者自己!)

LASERSPEED = 70
LASERLIFETIME = 8
LASERHITCHECKINTERVALS = 70
LASERDEADLY = 7
AIMERLENGTH = 200
AIMERHITCHECKINTERVALS = 100
AIMERACTIVE = 7

FRAGSPEED = 4.5
FRAGLIFETIME = 250
FRAGHITCHECKINTERVALS = 7
FRAGDEADLY = 0
FRAGFRAGMENTS = 40
FRAGLEVELS = 1

GATLINGSPEED = 5.5
GATLINGLIFETIME = 125
GATLINGBULLETS = 20
GATLINGHITCHECKINTERVALS = 7
GATLINGDEADLY = 2
GATLINGSPINSPEED = 10

HOMINGSPEED = 4.5
HOMINGLIFETIME = 250
HOMINGHITCHECKINTERVALS = 5
HOMINGDEADLY = 0
HOMINGSTARTUPTIME = 50

MINESPEED = 5.5
MINEHITCHECKINTERVALS = 4
MINEDEADLY = 20
MINEHIDETIME = 0
MINENUMBER = 3
MINEDETONATETIME = 6

REMOTESPEED = 4.5
REMOTETURNSPEED = 15
REMOTELIFETIME = 250
REMOTEHITCHECKINTERVALS = 5
REMOTEDEADLY = 0

DEATHRAYLIFETIME = 30
DEATHRAYWARMUPTIME = 9
DEATHRAYDEADLY = 0

SHIELDSIZE = 70

# ---- 武器箱 (frame_53:2173-2175) ----
CRATESPAWNTIMEBASE = 350
CRATESPAWNTIMERANDOM = 200
CRATESPAWNMAZESIZESCALE = 2000

# ---- 回合流程 (frame_53:2197-2199) ----
NUMBEROFFRAMESBEFOREEND = 125   # 击杀后继续运行的帧数
NUMBEROFFRAMESFROZEN = 50       # 冻结帧数 (endCount 减到 50 时冻结并计分)
NUMBEROFFRAMESBEFORERESET = 5   # 清场后到新回合的帧数

# ---- 视觉特效数量 (逻辑无关, 渲染参考) ----
NUMBEROFFRAGMENTS = 8
NUMBEROFSMOKECLOUDS = 10
MAXSHAKE = 8

# ---- 寻路 (frame_53:2220) ----
MAXDEADENDPENALTY = 5

# ---- 默认设置 (frame_57 设置面板默认值) ----
SETTINGS_MAX_BULLETS = 5          # settingsMaxBullets = 5
SETTINGS_MAX_CRATES = 3           # settingsMaxCrates = 3
SETTINGS_CRATE_SPAWN_MODIFIER = 1 # settingsCrateSpawnModifier = 1

# ---- 坦克物理 (DefineSprite_135_tank:42-44, 322) ----
TANK_FORWARD_SPEED_BASE = 4.0    # * (SCALE/50) 像素/帧
TANK_BACKUP_SPEED_BASE = 2.5     # * (SCALE/50) 像素/帧
TANK_TURN_SPEED = 10             # 度/帧
TANK_MOVE_STEPS = 5              # 每帧移动子步数 (STEPS = 5)

# ---- 坦克图形尺寸 (SWF 矢量边界精确提取, 单位: 本地像素) ----
# base = sprite79 (shape78 描边边界), turret = sprite134 frame1 (shape82)
# tank._xscale = 0.55 * SCALE (百分比), 即显示缩放 = 0.55*SCALE/100
TANK_BASE_WIDTH = 61.0     # base._width : shape78 [-30.5,-40.5 .. 30.5,40.5]
TANK_BASE_HEIGHT = 81.0    # base._height
TANK_TURRET_WIDTH = 45.0   # turret._width : frame1 [-22.5,-55.0 .. 22.5,22.5]
TANK_TURRET_HEIGHT = 77.5  # turret._height
TANK_DISPLAY_SCALE_FACTOR = 0.55 / 100.0   # 乘以 SCALE 得到实际缩放

# 整车 frame1 联合边界 (sprite135 精确值): hitTest(x,y,false) 用
TANK_BOUNDS_LOCAL = (-30.5, -55.0, 30.5, 40.5)  # (xmin, ymin, xmax, ymax)

# 墙壁碰撞探测点用 (hitPointsFront[4/5] = ±turret._width/6, -turret._height/16*11):
TANK_BARREL_HALF_WIDTH = TANK_TURRET_WIDTH / 6.0        # 7.5
TANK_BARREL_TIP_Y = -TANK_TURRET_HEIGHT / 16.0 * 11.0   # -53.28125

# 命中判定形状用 (点 vs 坦克图形, hitTest(x,y,true)):
# 炮管矢量图形: 尖端 y=-55, 宽约 17 (像素实测), 圆顶半径约 23.5 (含在 base 矩形内)
TANK_SHAPE_BARREL_HALF_WIDTH = 8.5
TANK_SHAPE_BARREL_TIP_Y = -55.0

# 子弹图形半径 (sprite175 内 shape167 边界 ±3.5, 渲染用)
BULLET_VISUAL_RADIUS = 3.5

# ---- 玩家/Laika 颜色 (frame_43:88-89; 玩家色为账号自选, 此处用经典红) ----
LAIKA_BASE_COLOR = 0x262626    # 2500134
LAIKA_TURRET_COLOR = 0x666666  # 6710886
PLAYER_BASE_COLOR = 0xB02A22
PLAYER_TURRET_COLOR = 0xD23B32
