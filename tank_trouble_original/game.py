"""
游戏引擎核心 — 1:1 移植自反编译源码

移植对照:
  Tank.update()      <- DefineSprite_135_tank/frame_1/DoAction.as (onEnterFrame)
  Bullet.update()    <- DefineSprite_175_bullet/frame_1/DoAction.as (onEnterFrame)
  Game.step()        <- frame_53/DoAction.as (_root.onEnterFrame)
  Game.setup_battle()<- frame_53 setupBattle()/setupStandardMaze()/deployTank()
  Game.destroy_tank()<- frame_53 destroyTank()

帧执行顺序 (对应 Flash enterFrame 派发, 按创建顺序):
  _root 逻辑 -> tank0 -> tank1 -> 子弹(按创建顺序)
  当帧新建的子弹下一帧才开始移动 (Flash 中 attachMovie 的剪辑
  在下一次 enterFrame 才收到事件)。
"""

import math
import random as _random

from . import constants as C
from .maze import (
    create_maze, calc_reachable, calc_distances, find_dead_ends,
    build_wall_segments, WallGrid,
)
from .laika import LaikaAI

DEG = math.pi / 180.0


def norm_rot(deg):
    """Flash _rotation 语义: 写入后归一化到 (-180, 180]"""
    deg = math.fmod(deg, 360.0)
    if deg > 180.0:
        deg -= 360.0
    elif deg <= -180.0:
        deg += 360.0
    return deg


# ============================================================== Tank

class Tank:
    """坦克 — 状态与逐帧逻辑与 DefineSprite_135_tank 一致"""

    def __init__(self, game, number, cell, scale, rng):
        self.game = game
        self.number = number
        # deployTank: 位置 = (格子+0.5)*SCALE, 朝向 = random(32)*11.25
        self.x = (cell["x"] + 0.5) * scale
        self.y = (cell["y"] + 0.5) * scale
        self.rotation = norm_rot(math.floor(rng.random() * 32) * 11.25)
        # 物理参数 (tank:42-44)
        self.forward_speed = C.TANK_FORWARD_SPEED_BASE * (scale / 50.0)
        self.backup_speed = C.TANK_BACKUP_SPEED_BASE * (scale / 50.0)
        self.turn_speed = C.TANK_TURN_SPEED
        # 显示缩放 (deployTank: _xscale = 0.55*SCALE)
        self.display_scale = C.TANK_DISPLAY_SCALE_FACTOR * scale
        # 状态 (tank:45-55)
        self.trigger_released = True
        self.bullets_fired = 0
        self.laser_ready = True
        self.frag_fired = False
        self.alive = True
        self.gatling_ready = True
        self.homing_ready = True
        self.mines_layed = 0
        self.death_ray_ready = True
        self.remote_controlling = False
        self.electric_ready = True
        self.current_weapon = C.STARTWEAPON
        self.current_equipment = "none"
        self.last_frag = None
        self.hit_something = False
        # 控制输入
        self.forward = False
        self.backup = False
        self.turn_left = False
        self.turn_right = False
        self.fire = False
        # AI (deployTank: AIEnabled && number==1)
        self.ai = None
        # 碰撞点 (tank:58-82, 本地坐标; bullet 武器)
        bw, bh = C.TANK_BASE_WIDTH, C.TANK_BASE_HEIGHT
        tw, th = C.TANK_TURRET_WIDTH, C.TANK_TURRET_HEIGHT
        self.hit_points_front = [
            (-bw / 2, -bh / 2), (-bw / 4, -bh / 2),
            (bw / 4, -bh / 2), (bw / 2, -bh / 2),
            (-tw / 6, -th / 16 * 11), (tw / 6, -th / 16 * 11),
        ]
        self.hit_points_rear = [
            (-bw / 2, bh / 2), (-bw / 4, bh / 2), (0, bh / 2),
            (bw / 4, bh / 2), (bw / 2, bh / 2),
        ]
        self.hit_points_right = [
            (bw / 2, -bh / 6 * 2), (bw / 2, -bh / 6), (bw / 2, 0),
            (bw / 2, bh / 6), (bw / 2, bh / 6 * 2),
        ]
        self.hit_points_left = [
            (-bw / 2, -bh / 6 * 2), (-bw / 2, -bh / 6), (-bw / 2, 0),
            (-bw / 2, bh / 6), (-bw / 2, bh / 6 * 2),
        ]

    # ---- 坐标变换 (localToGlobal: 缩放 -> 旋转 -> 平移) ----

    def local_to_global(self, lx, ly):
        s = self.display_scale
        th = self.rotation * DEG
        c, sn = math.cos(th), math.sin(th)
        gx = s * (lx * c - ly * sn)
        gy = s * (lx * sn + ly * c)
        return self.x + gx, self.y + gy

    def hit_check(self, points):
        """tank:12-26 — 各点是否命中迷宫墙壁"""
        g = self.game
        for (lx, ly) in points:
            px, py = self.local_to_global(lx, ly)
            if g.wall_hit(px, py):
                return True
        return False

    def expanded_hit_check(self, points, factor):
        """tank:27-41 — 本地坐标先放大 factor 倍再检测 (AI backAway 用)"""
        g = self.game
        for (lx, ly) in points:
            px, py = self.local_to_global(lx * factor, ly * factor)
            if g.wall_hit(px, py):
                return True
        return False

    def _any_side_hit(self):
        return (self.hit_check(self.hit_points_front)
                or self.hit_check(self.hit_points_rear)
                or self.hit_check(self.hit_points_left)
                or self.hit_check(self.hit_points_right))

    # ---- 命中判定 (子弹点 vs 坦克图形) ----

    def point_in_shape(self, px, py):
        """hitTest(x, y, true) 的等价: 点是否在坦克图形内。

        近似为 base 矩形 ∪ 炮管矩形 (原版为矢量图形, 差异 <2px)。
        """
        s = self.display_scale
        th = self.rotation * DEG
        c, sn = math.cos(th), math.sin(th)
        dx, dy = px - self.x, py - self.y
        # 逆旋转 + 逆缩放 -> 本地坐标
        lx = (dx * c + dy * sn) / s
        ly = (-dx * sn + dy * c) / s
        bw2, bh2 = C.TANK_BASE_WIDTH / 2, C.TANK_BASE_HEIGHT / 2
        if -bw2 <= lx <= bw2 and -bh2 <= ly <= bh2:
            return True
        if (abs(lx) <= C.TANK_SHAPE_BARREL_HALF_WIDTH
                and C.TANK_SHAPE_BARREL_TIP_Y <= ly <= 0):
            return True
        return False

    def point_in_bbox(self, px, py):
        """hitTest(x, y, false) 的等价: 点是否在旋转后包围盒的 AABB 内"""
        s = self.display_scale
        th = self.rotation * DEG
        c, sn = math.cos(th), math.sin(th)
        xmin, ymin, xmax, ymax = C.TANK_BOUNDS_LOCAL
        xs, ys = [], []
        for lx, ly in ((xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)):
            xs.append(self.x + s * (lx * c - ly * sn))
            ys.append(self.y + s * (lx * sn + ly * c))
        return min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys)

    # ---- 逐帧更新 (tank onEnterFrame 138-433) ----

    def update(self):
        g = self.game
        if g.frozen:
            return
        if not (self.alive and not g.locked_control(self)):
            return

        old_x, old_y, old_rot = self.x, self.y, self.rotation

        # AI 决策 (tank:314-321, 在移动之前设置输入)
        if self.ai is not None:
            if self.ai.make_decisions_and_update_goal():
                self.ai.decide_actions_to_achieve_goal()
            self.ai.set_input_to_do_actions()

        # 分步移动 (tank:322-351)
        STEPS = C.TANK_MOVE_STEPS
        move_size = 0.0
        turn_size = 0.0
        if self.forward:
            move_size = self.forward_speed / STEPS
        if self.backup:
            move_size -= self.backup_speed / STEPS
        if self.turn_left:
            turn_size = -self.turn_speed / STEPS
        if self.turn_right:
            turn_size += self.turn_speed / STEPS

        self.hit_something = False
        # 第一遍: 不检测碰撞直接走完 5 步
        for _ in range(STEPS):
            self.rotation = norm_rot(self.rotation + turn_size)
            rad = (self.rotation - 90) * DEG
            self.x += math.cos(rad) * move_size
            self.y += math.sin(rad) * move_size

        # 终点有碰撞 -> 回滚, 逐步重走 (tank:354-398)
        if self._any_side_hit():
            self.x, self.y, self.rotation = old_x, old_y, old_rot
            for _ in range(STEPS):
                # 键盘/AI 坦克 (非 mouseTank): 逐步转向, 撞墙回退转角
                step_old_rot = self.rotation
                self.rotation = norm_rot(self.rotation + turn_size)
                if self._any_side_hit():
                    self.rotation = step_old_rot
                    self.hit_something = True
                step_old_x, step_old_y = self.x, self.y
                rad = (self.rotation - 90) * DEG
                self.x += math.cos(rad) * move_size
                self.y += math.sin(rad) * move_size
                if move_size > 0 and self.hit_check(self.hit_points_front):
                    self.x, self.y = step_old_x, step_old_y
                    self.hit_something = True
                elif move_size < 0 and self.hit_check(self.hit_points_rear):
                    self.x, self.y = step_old_x, step_old_y
                    self.hit_something = True

        # 旋转吸附 (tank:399-410)
        offset = (360 + self.rotation) % self.turn_speed
        if not self.hit_something and turn_size != 0 and offset != 0:
            if offset < self.turn_speed / 2:
                self.rotation = norm_rot(self.rotation - offset)
            else:
                self.rotation = norm_rot(self.rotation + (self.turn_speed - offset))

        # 开火 (tank:411-419, 扳机边沿触发)
        if self.fire and self.trigger_released and g.weapon_ready(self):
            self.trigger_released = False
            g.fire_weapon(self)
        elif not self.fire:
            self.trigger_released = True


# ============================================================== Bullet

class Bullet:
    """子弹 — 与 DefineSprite_175_bullet 一致

    每帧 BULLETHITCHECKINTERVALS(7) 个子步, 每子步含反弹判定;
    子步完成后判定命中坦克 (deadly==0 立即生效, 含发射者自己);
    lifetime 每帧减 1, 到 0 消失并归还弹数。
    """

    def __init__(self, game, name, owner, scale):
        self.game = game
        self.name = name
        self.owner = owner
        rad = (owner.rotation - 90) * DEG
        # fireBullet (frame_53:1259-1282)
        self.x = owner.x + math.cos(rad) * scale * 4.5 / 16
        self.y = owner.y + math.sin(rad) * scale * 4.5 / 16
        self.x_speed = (math.cos(rad) * C.BULLETSPEED
                        / C.BULLETHITCHECKINTERVALS * (scale / 50.0))
        self.y_speed = (math.sin(rad) * C.BULLETSPEED
                        / C.BULLETHITCHECKINTERVALS * (scale / 50.0))
        self.lifetime = C.BULLETLIFETIME
        self.deadly = C.BULLETDEADLY
        self.removed = False

    def update(self):
        g = self.game
        if g.frozen:
            return
        # 7 个子步 (bullet:16-87)
        for _ in range(C.BULLETHITCHECKINTERVALS):
            prev_x, prev_y = self.x, self.y
            self.x += self.x_speed
            self.y += self.y_speed
            if g.wall_hit(self.x, self.y):
                g.events.append(("bounce", self.name))
                # X 反转测试 (bullet:31-44)
                tx = prev_x - self.x_speed
                ty = prev_y + self.y_speed
                hit_on_x_invert = g.wall_hit(tx, ty)
                # Y 反转测试 (bullet:45-58)
                tx = prev_x + self.x_speed
                ty = prev_y - self.y_speed
                hit_on_y_invert = g.wall_hit(tx, ty)
                # 决定反弹方向 (bullet:59-71)
                if hit_on_x_invert and not hit_on_y_invert:
                    self.y_speed = -self.y_speed
                elif hit_on_y_invert and not hit_on_x_invert:
                    self.x_speed = -self.x_speed
                else:
                    self.x_speed = -self.x_speed
                    self.y_speed = -self.y_speed
                self.x = prev_x + self.x_speed
                self.y = prev_y + self.y_speed
            # (护盾反弹检查: bullet:77-85, vs Laika 模式无护盾)

        # 命中坦克 (bullet:90-104) — 原版无发射者豁免, 可命中自己!
        # 规则开关: self_harm_immune 里的坦克对自己的子弹免疫 (不自伤)。
        if self.deadly == 0:
            for i in range(g.tanks_count):
                tank = g.tanks[i]
                if tank.alive and tank.point_in_shape(self.x, self.y):
                    if (tank is self.owner
                            and self.owner.number in g.self_harm_immune):
                        continue          # 自伤免疫: 跳过自己的子弹
                    if g.hit_immunity_remaining[i] > 0:
                        continue          # 受击免疫: 子弹穿透且不消失
                    g.register_hit(self.owner, tank)
                    self.owner.bullets_fired -= 1
                    if tank.number not in g.invincible:
                        g.destroy_tank(i)  # 无敌坦克: 触发命中事件但不死
                    self.removed = True
        if self.deadly > 0:
            self.deadly -= 1

        # 寿命 (bullet:109-167)
        self.lifetime -= 1
        if self.lifetime <= 0 and not self.removed:
            self.owner.bullets_fired -= 1
            self.removed = True


# ============================================================== Game

class Game:
    """游戏主体 — 对应 frame_53 的 _root 逻辑与回合循环。

    vs Laika 模式 (frame_43:81-90):
      TANKS=2, tank0=玩家, tank1=Laika(AI);
      settingsActiveWeapons=[] -> 永不生成武器箱, 纯子弹对决。
    """

    def __init__(self, seed=None, ai_enabled=True, tanks=2,
                 self_harm_immune=None, invincible=None,
                 hit_immunity_frames=None):
        self.rng = _random.Random(seed)
        self.ai_enabled = ai_enabled
        self.tanks_count = tanks
        # 规则开关: 这些坦克编号对**自己发射的**子弹免疫 (不自伤)。
        # 默认空 = 原版行为 (可自杀)。用于把爱神风的对手改造成真对手。
        self.self_harm_immune = set(self_harm_immune or ())
        # 规则开关: 这些坦克编号无敌 (对**所有**子弹免疫), 但仍触发 hit 事件。
        # 用于生存模式: 对手打不死, 我方命中它只为"加时", 非击杀。
        self.invincible = set(invincible or ())
        # 规则开关: 指定坦克被命中后获得若干帧免疫。免疫期间子弹穿透，
        # 不触发 hit 事件也不消失。默认全 0，原版行为不变。
        immunity = dict(hit_immunity_frames or {})
        self.hit_immunity_duration = tuple(
            max(0, int(immunity.get(i, 0))) for i in range(tanks))
        self.hit_immunity_remaining = [0] * tanks
        # 设置项 (默认值)
        self.settings_max_bullets = C.SETTINGS_MAX_BULLETS
        self.settings_max_crates = C.SETTINGS_MAX_CRATES
        self.settings_crate_spawn_modifier = C.SETTINGS_CRATE_SPAWN_MODIFIER
        # AI 模式下 settingsActiveWeapons = [] (frame_53:2228-2231)
        self.settings_active_weapons = [] if ai_enabled else [
            "laser", "frag", "gatling", "homing", "deathRay"]
        # 回合状态 (frame_53:2214-2219)
        self.alive_count = 0
        self.end_count = -1
        self.reset_count = -1
        self.frozen = False
        self.shake = 0.0
        self.number_of_crates = 0
        self.crate_spawn_points = []
        self.crate_timer = ((C.CRATESPAWNTIMEBASE
                             + self.rng.randrange(C.CRATESPAWNTIMERANDOM))
                            * self.settings_crate_spawn_modifier)
        # 计分 (scoreboard, 跨回合保留)
        self.scores = [0] * tanks
        self.round_number = 0
        self.frame = 0
        self.events = []
        # 战场数据
        self.maze = None
        self.scale = 50.0
        self.walls = []
        self.wall_half_t = 3
        self.reachable = []
        self.reachable_index = None
        self.distances_for_maze = None
        self.dead_ends = None
        self.tank_fields = []
        self.tanks = []
        self.bullets = []
        self._bullet_depth = 0
        self.setup_battle()

    # ---------------------------------------------- setupBattle

    def setup_battle(self):
        """frame_53 setupBattle + setupStandardMaze + deployTank"""
        self.round_number += 1
        self.hit_immunity_remaining = [0] * self.tanks_count
        rng = self.rng
        TANKS = self.tanks_count

        # setupStandardMaze (frame_53:1838-1873)
        spawn_cells = [None] * TANKS
        self.reachable = []
        while len(self.reachable) < 2 * TANKS:
            width = rng.randrange(9) + 4    # 4..12
            height = rng.randrange(7) + 4   # 4..10
            self.scale = min(
                (C.MOVIEHEIGHT - C.HEIGHTTOBOTTOM) / (height + 0.125),
                C.MOVIEWIDTH / (width + 0.125))
            self.maze = create_maze(width, height, rng)
            spawn_cells[0] = {"x": math.floor(rng.random() * width),
                              "y": math.floor(rng.random() * height)}
            self.reachable, self.reachable_index = calc_reachable(
                self.maze, spawn_cells[0]["x"], spawn_cells[0]["y"])
        self.reachable[0]["used"] = True
        i = 1
        while i < TANKS:
            k = math.floor(rng.random() * len(self.reachable))
            if not self.reachable[k]["used"]:
                spawn_cells[i] = {"x": self.reachable[k]["x"],
                                  "y": self.reachable[k]["y"]}
                self.reachable[k]["used"] = True
            else:
                i -= 1
            i += 1
        for cell in self.reachable:
            cell["used"] = False
        self.number_of_crates = 0

        # drawMaze -> 碰撞墙体 (+ 等价加速索引)
        self.walls = build_wall_segments(self.maze, self.scale)
        self.wall_half_t = math.floor(self.scale / 16)
        self._wall_grid = WallGrid(self.walls, self.wall_half_t,
                                   bucket_size=self.scale)

        # deployTank (frame_53:791-899) — 每回合全新坦克对象
        self.tanks = []
        self.bullets = []
        self._bullet_depth = 0
        self.crates = {}
        for n in range(TANKS):
            tank = Tank(self, n, spawn_cells[n], self.scale, rng)
            self.tanks.append(tank)
        # AI 附着 (deployTank: AIEnabled && number == 1); AI 每回合重建
        if self.ai_enabled:
            self.tanks[1].ai = LaikaAI(self, self.tanks[1])

        self.alive_count = TANKS

        # distancesForMaze: 每个可达格到全图的距离 (frame_53:1816-1828)
        w, h = len(self.maze), len(self.maze[0])
        self.distances_for_maze = [[None] * h for _ in range(w)]
        for cell in self.reachable:
            self.distances_for_maze[cell["x"]][cell["y"]] = calc_distances(
                self.maze, cell["x"], cell["y"])

        # tankFields (frame_53:1829-1835)
        self.tank_fields = [{"x": spawn_cells[n]["x"], "y": spawn_cells[n]["y"]}
                            for n in range(TANKS)]

        # deadEnds (frame_53:1836)
        self.dead_ends = find_dead_ends(self.maze, self.reachable,
                                        C.MAXDEADENDPENALTY)
        self.events.append(("new_round", self.round_number))

    def wall_hit(self, px, py):
        """点是否命中墙壁 (mazemc.hitTest 等价, 经空间索引加速)"""
        return self._wall_grid.hit(px, py)

    def dist_map(self, fx, fy):
        """distancesForMaze[fx][fy] 的受保护读取 (不可达格 -> None)"""
        if (self.distances_for_maze is not None
                and 0 <= fx < len(self.distances_for_maze)
                and 0 <= fy < len(self.distances_for_maze[fx])):
            return self.distances_for_maze[fx][fy]
        return None

    # ---------------------------------------------- 武器

    def locked_control(self, tank):
        """frame_53:1077-1090 (bullet 恒为 False)"""
        w = tank.current_weapon
        if w == "laser":
            return not tank.laser_ready
        if w == "deathRay":
            return not tank.death_ray_ready
        if w == "remote":
            return tank.remote_controlling
        return False

    def weapon_ready(self, tank):
        """frame_53:1091-1115"""
        w = tank.current_weapon
        if w == "bullet":
            return tank.bullets_fired < self.settings_max_bullets
        if w == "frag" or w == "mine":
            return True
        if w == "laser":
            return tank.laser_ready
        if w == "gatling":
            return tank.gatling_ready
        if w == "homing":
            return tank.homing_ready
        if w == "remote":
            return not tank.remote_controlling
        if w == "electric":
            return tank.electric_ready
        if w == "deathRay":
            return tank.death_ray_ready
        return False

    def fire_weapon(self, tank):
        """frame_53:1227-1258 (vs Laika 模式仅 bullet)"""
        if tank.current_weapon == "bullet":
            self._bullet_depth += 1
            name = "bullet%d" % self._bullet_depth
            b = Bullet(self, name, tank, self.scale)
            b.just_created = True   # 当帧不更新 (Flash: 下一帧才收到 enterFrame)
            self.bullets.append(b)
            tank.bullets_fired += 1
            self.events.append(("fire", tank.number))
        # 其他武器在 vs Laika 模式中不存在 (settingsActiveWeapons=[])

    # ---------------------------------------------- 命中/摧毁

    def register_hit(self, owner, victim):
        """frame_53:1950-2013 — 本地版只发事件 (Laika growl/scoff)"""
        self.events.append(("hit", owner.number, victim.number))
        duration = self.hit_immunity_duration[victim.number]
        if duration > 0:
            self.hit_immunity_remaining[victim.number] = duration

    def destroy_tank(self, number):
        """frame_53:900-911"""
        tank = self.tanks[number]
        tank.alive = False
        self.alive_count -= 1
        self.end_count = C.NUMBEROFFRAMESBEFOREEND
        self.shake = max(C.MAXSHAKE, self.shake + 7)
        self.events.append(("destroy", number))

    def assign_points(self):
        """frame_53:2015-2050 (本地简化: 存活者 +1 分)"""
        winner = None
        for i in range(self.tanks_count):
            if self.tanks[i].alive:
                self.scores[i] += 1
                winner = i
        self.events.append(("round_end", winner))

    def clean_up_battle(self):
        """frame_53:1942-1949 — 移除场上所有对象"""
        self.bullets = []
        self.crates = {}

    # ---------------------------------------------- 主循环

    def step(self):
        """推进一帧 (1/25 秒), 对应一次 enterFrame 派发。

        返回本帧事件列表。
        """
        self.frame += 1
        self.events = []
        rng = self.rng

        for i, remaining in enumerate(self.hit_immunity_remaining):
            if remaining > 0:
                self.hit_immunity_remaining[i] = remaining - 1

        # ---- _root.onEnterFrame (frame_53:2366-2536) ----
        # tankFields 更新 (2390-2397)
        for i in range(self.tanks_count):
            self.tank_fields[i] = {
                "x": math.floor(self.tanks[i].x / self.scale),
                "y": math.floor(self.tanks[i].y / self.scale)}

        # 武器箱计时 (2398-2448); AI 模式 settingsActiveWeapons=[] -> 不生成
        if not self.frozen:
            self.crate_timer -= 1
        if not self.frozen and self.crate_timer <= 0:
            self.crate_timer = ((C.CRATESPAWNTIMEBASE
                                 + rng.randrange(C.CRATESPAWNTIMERANDOM)
                                 + C.CRATESPAWNMAZESIZESCALE / len(self.reachable))
                                * self.settings_crate_spawn_modifier)
            # placeCrate: settingsActiveWeapons 为空时直接返回 (frame_53:1568-1573)

        # 屏幕震动衰减 (2450-2455, 仅视觉)
        if self.shake >= 0:
            self.shake -= 0.5

        # 回合结束状态机 (2456-2525)
        if self.alive_count <= 1:
            if self.end_count >= 0:
                self.end_count -= 1
            if self.end_count == C.NUMBEROFFRAMESFROZEN:
                self.frozen = True
                self.assign_points()   # 本地(非排位)分支: 立即计分
            if self.end_count == 0:
                self.clean_up_battle()
                self.reset_count = C.NUMBEROFFRAMESBEFORERESET
        if self.reset_count >= 0:
            self.reset_count -= 1
        if self.reset_count == 0:
            self.end_count = C.NUMBEROFFRAMESBEFOREEND + C.NUMBEROFFRAMESFROZEN
            self.frozen = False
            self.setup_battle()

        # ---- 坦克 enterFrame (创建顺序) ----
        for tank in self.tanks:
            tank.update()

        # ---- 子弹 enterFrame (创建顺序; 当帧新建的跳过) ----
        for b in list(self.bullets):
            if getattr(b, "just_created", False):
                b.just_created = False
                continue
            if not b.removed:
                b.update()
        self.bullets = [b for b in self.bullets if not b.removed]

        return self.events

    # ---------------------------------------------- 对象注册表 (供 AI 遍历)

    def iter_named_objects(self):
        """等价于 AS2 `for(name in _root.game)` — AI 用于扫描子弹/道具"""
        for b in self.bullets:
            yield b.name, b
        for name, crate in self.crates.items():
            yield name, crate
