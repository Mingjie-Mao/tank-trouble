"""
Laika AI — 1:1 移植自 DefineSprite_186_tankTroubleAI/frame_1/DoAction.as

结构 (与原版一致):
  make_decisions_and_update_goal()  每帧调用; goal.period 计数, 到期后重新
                                    评估所有候选目标, 取最高 priority
  decide_actions_to_achieve_goal()  目标 -> 动作栈 (myActionsForGoal)
  set_input_to_do_actions()         动作栈 -> 坦克输入 (forward/turn/fire)

AI 每回合随坦克重建 (deployTank 中 attachMovie), 状态不跨回合。
常量数值(含浮点字面量)与反编译输出完全一致。
"""

import math

from . import constants as C
from .maze import (
    get_shortest_path_with_distances,
    follow_gradient_path_with_distances,
    follow_gradient_path_with_distances_and_dead_ends,
)

DEG = math.pi / 180.0
PI = 3.141592653589793


class LaikaAI:

    def __init__(self, game, my_tank):
        self.game = game
        self.my_tank = my_tank
        scale = game.scale
        # ---- 常量 (AI 源码 1355-1385, 依赖当回合 SCALE) ----
        self.AGGRESIVENESS = 0.5
        self.COWARDNESS = 0.7000000000000001
        self.GREEDY = 1
        self.LONGESTPATHTOSHOOT = 7
        self.LONGESTPATHTONOTHESITATETOSHOOT = 2
        self.FRAGBOMBSAFETYDIST = 3 * scale
        self.FRAGBOMBDETONATEDIST = 3 * scale
        self.LONGESTPATHTORUN = 10
        self.MAXSTUCKTIME = 1
        self.stuck_time = 0
        self.current_aggresiveness = self.AGGRESIVENESS
        self.IDLEDRIVETOWARDENEMYPRIORITY = 0.1
        self.IDLEDRIVEPRIORITY = 0.1
        self.MAXCLOSESTCELLDISTANCE = 2
        self.MAXCLOSESTDISTANCE = scale * self.MAXCLOSESTCELLDISTANCE
        self.MAXTIMETODODGEBULLET = 75
        self.MAXDISTTODODGEBULLET = 4 * scale
        self.MAXCELLDISTTODODGEBULLET = (self.MAXTIMETODODGEBULLET
                                         * C.BULLETSPEED / 50)
        self.MAXCELLDISTTODODGEFRAGBOMB = 5
        self.MAXTIMETODODGEFRAGBOMBFRAGMENT = 50
        self.MAXDISTTODODGEFRAGBOMBFRAGMENT = 3 * scale
        self.MAXCELLDISTTODODGEFRAGBOMBFRAGMENT = (
            self.MAXTIMETODODGEFRAGBOMBFRAGMENT * (C.FRAGSPEED + 4) / 50)
        self.MAXTIMETODODGEGATLINGBULLET = 75
        self.MAXDISTTODODGEGATLINGBULLET = 3 * scale
        self.MAXCELLDISTTODODGEGATLINGBULLET = (
            self.MAXTIMETODODGEGATLINGBULLET * C.GATLINGSPEED / 50)
        self.MAXCELLDISTTODODGELASER = 2
        self.MAXCELLDISTTOGOFORCRATE = 10
        self.goal_id = 1
        self.my_goal = {"goal": "idle", "priority": 0, "period": 15,
                        "id": 0, "updateContinuously": True}
        self.my_actions = []

    # ------------------------------------------------ 工具

    def _rand(self, n):
        """AS2 random(n)"""
        return math.floor(self.game.rng.random() * n)

    def _cell_dist(self, fx, fy, cx, cy):
        """_root.distancesForMaze[fx][fy][cx][cy], 缺失 -> NaN (比较恒 False)"""
        dm = self.game.dist_map(fx, fy)
        if dm is None:
            return float("nan")
        if 0 <= cx < len(dm) and 0 <= cy < len(dm[cx]):
            v = dm[cx][cy]
            return float("nan") if v is None else v
        return float("nan")

    def update_goal(self, temp):
        """AI:1-7"""
        if self.my_goal["priority"] < temp["priority"]:
            self.my_goal = temp

    # ------------------------------------------------ 弹道模拟

    def check_path_for_collision(self, x, y, x_speed, y_speed,
                                 hit_check_interval, maxtime, lifetime):
        """AI:165-229 — 模拟直线运动, 遇墙反弹并立即返回反弹信息。

        返回 {'x','y','xSpeed','ySpeed','t'} 或 None。
        """
        g = self.game
        lifetime = min(maxtime, lifetime)
        t = 0
        while lifetime > 0:
            for _ in range(hit_check_interval):
                prev_x, prev_y = x, y
                x += x_speed
                y += y_speed
                if g.wall_hit(x, y):
                    # X 反转测试
                    tx = prev_x - x_speed
                    ty = prev_y + y_speed
                    hit_x_inv = g.wall_hit(tx, ty)
                    # Y 反转测试
                    tx = prev_x + x_speed
                    ty = prev_y - y_speed
                    hit_y_inv = g.wall_hit(tx, ty)
                    if hit_x_inv and not hit_y_inv:
                        y_speed = -y_speed
                    elif hit_y_inv and not hit_x_inv:
                        x_speed = -x_speed
                    else:
                        x_speed = -x_speed
                        y_speed = -y_speed
                    x = prev_x + x_speed
                    y = prev_y + y_speed
                    return {"x": x, "y": y, "xSpeed": x_speed,
                            "ySpeed": y_speed, "t": t}
            lifetime -= 1
            t += 1
        return None

    def check_bullet_path(self, angle):
        """AI:230-336 — 沿角度模拟一颗子弹, 返回 HIT/SUICIDE/NOTHING。

        速度 = BULLETSPEED*(SCALE/50)/步, 模拟 BULLETLIFETIME/3 步。
        """
        g = self.game
        scale = g.scale
        my = self.my_tank
        rad = (angle - 90) * PI / 180
        x = my.x + math.cos(rad) * scale * 4.5 / 16
        y = my.y + math.sin(rad) * scale * 4.5 / 16
        xs = math.cos(rad) * C.BULLETSPEED * (scale / 50)
        ys = math.sin(rad) * C.BULLETSPEED * (scale / 50)
        life = C.BULLETLIFETIME / 3          # 浮点, 与原版一致
        deadly = C.BULLETDEADLY
        closest = C.MOVIEWIDTH + C.MOVIEHEIGHT
        while life > 0:
            # 单子步 + 反弹 (不返回)
            prev_x, prev_y = x, y
            x += xs
            y += ys
            if g.wall_hit(x, y):
                tx = prev_x - xs
                ty = prev_y + ys
                hit_x_inv = g.wall_hit(tx, ty)
                tx = prev_x + xs
                ty = prev_y - ys
                hit_y_inv = g.wall_hit(tx, ty)
                if hit_x_inv and not hit_y_inv:
                    ys = -ys
                elif hit_y_inv and not hit_x_inv:
                    xs = -xs
                else:
                    xs = -xs
                    ys = -ys
                x = prev_x + xs
                y = prev_y + ys
            if deadly == 0:
                for i in range(g.tanks_count):
                    tank = g.tanks[i]
                    if tank.alive and tank.point_in_bbox(x, y):
                        if tank.point_in_shape(x, y):
                            if tank is my:
                                return {"result": "SUICIDE",
                                        "time": C.BULLETLIFETIME / 3 - life}
                            return {"result": "HIT",
                                    "time": C.BULLETLIFETIME / 3 - life}
                    elif tank.alive and tank is not my:
                        d = abs(tank.x - x) + abs(tank.y - y)  # 曼哈顿距离
                        if d < self.MAXCLOSESTDISTANCE:
                            cx = math.floor(x / scale)
                            cy = math.floor(y / scale)
                            tf = g.tank_fields[i]
                            if (self._cell_dist(tf["x"], tf["y"], cx, cy)
                                    <= self.MAXCLOSESTCELLDISTANCE):
                                if d < closest:
                                    closest = d
            if deadly > 0:
                deadly -= 1
            life -= 1
        return {"result": "NOTHING", "time": C.BULLETLIFETIME / 3,
                "closest": closest}

    # ------------------------------------------------ 躲避轨迹

    def dodge_trajectories(self, fieldx, fieldy, bullets, max_time_to_dodge,
                           max_dist_to_dodge, max_cell_dist_to_dodge,
                           hit_check_interval, check_bounce):
        """AI:8-88 — 对每颗子弹求最近逼近点, 生成 dodgeBullet 目标。"""
        g = self.game
        scale = g.scale
        my = self.my_tank
        best_dist = max_dist_to_dodge
        result = {"priority": 0}
        for b in bullets:
            bx, by = b.x, b.y
            cell_x = math.floor(bx / scale)
            cell_y = math.floor(by / scale)
            if not (self._cell_dist(fieldx, fieldy, cell_x, cell_y)
                    <= max_cell_dist_to_dodge):
                continue
            # 直线段: 当前位置 -> hit_check_interval 帧后
            x2 = b.x + b.x_speed * hit_check_interval
            y2 = b.y + b.y_speed * hit_check_interval
            tx, ty = my.x, my.y
            seg_sq = (x2 - bx) * (x2 - bx) + (y2 - by) * (y2 - by)
            t = (((tx - bx) * (x2 - bx) + (ty - by) * (y2 - by)) / seg_sq
                 if seg_sq else 0.0)
            if -1 < t < max_time_to_dodge:
                cx = bx + t * (x2 - bx)
                cy = by + t * (y2 - by)
                dx, dy = tx - cx, ty - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    col = self.check_path_for_collision(
                        cx, cy, dx / dist, dy / dist, 1,
                        math.ceil(dist), math.ceil(dist))
                else:
                    col = None
                if col is None and dist < best_dist:
                    dx2, dy2 = x2 - cx, y2 - cy
                    d2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
                    if d2 > 0:
                        col = self.check_path_for_collision(
                            cx, cy, dx2 / d2, dy2 / d2, 1,
                            math.ceil(d2), math.ceil(d2))
                    else:
                        col = None
                    if col is None:
                        best_dist = min(best_dist, dist)
                        result = {
                            "goal": "dodgeBullet", "x": b.x, "y": b.y,
                            "closest": {"x": cx, "y": cy}, "dist": dist,
                            "t": t, "dir": {"x": x2 - bx, "y": y2 - by},
                            "maxTime": max_time_to_dodge,
                            "maxDist": max_dist_to_dodge,
                            "period": 10, "priority": 1,
                            "updateContinuously": False,
                            "id": self.goal_id}
                        self.goal_id += 1
            # 反弹后的威胁 (AI:50-83)
            if best_dist > scale / 4 and check_bounce:
                col5 = self.check_path_for_collision(
                    bx, by, b.x_speed, b.y_speed,
                    hit_check_interval, 12, b.lifetime)
                if col5 is not None:
                    bx2, by2 = col5["x"], col5["y"]
                    x2 = col5["x"] + col5["xSpeed"] * hit_check_interval
                    y2 = col5["y"] + col5["ySpeed"] * hit_check_interval
                    seg_sq = (x2 - bx2) ** 2 + (y2 - by2) ** 2
                    t = ((((tx - bx2) * (x2 - bx2) + (ty - by2) * (y2 - by2))
                          / seg_sq) if seg_sq else 0.0)
                    if 0 < t < max_time_to_dodge - col5["t"]:
                        cx = bx2 + t * (x2 - bx2)
                        cy = by2 + t * (y2 - by2)
                        dx, dy = tx - cx, ty - cy
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist > 0:
                            col = self.check_path_for_collision(
                                cx, cy, dx / dist, dy / dist, 1,
                                math.ceil(dist), math.ceil(dist))
                        else:
                            col = None
                        if col is None and dist < best_dist:
                            dx2, dy2 = cx - bx2, cy - by2
                            d2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
                            if d2 > 0:
                                col = self.check_path_for_collision(
                                    bx2, by2, dx2 / d2, dy2 / d2, 1,
                                    math.ceil(d2), math.ceil(d2))
                            else:
                                col = None
                            if col is None:
                                best_dist = min(best_dist, dist)
                                result = {
                                    "goal": "dodgeBullet",
                                    "x": b.x, "y": b.y,
                                    "closest": {"x": cx, "y": cy},
                                    "dist": dist, "t": t + col5["t"],
                                    "dir": {"x": x2 - bx2, "y": y2 - by2},
                                    "maxTime": max_time_to_dodge,
                                    "maxDist": max_dist_to_dodge,
                                    "period": 10, "priority": 1,
                                    "updateContinuously": False,
                                    "id": self.goal_id}
                                self.goal_id += 1
        return result

    # ------------------------------------------------ 反击

    def try_to_retaliate(self):
        """AI:89-164"""
        g = self.game
        my = self.my_tank
        if self.current_aggresiveness < self.AGGRESIVENESS / 2:
            return
        w = my.current_weapon
        if w in ("bullet", "laser"):
            if my.bullets_fired < g.settings_max_bullets:
                found = False
                closest = C.MOVIEWIDTH + C.MOVIEHEIGHT
                res = self.check_bullet_path(my.rotation)
                if res["result"] == "HIT":
                    found = True
                elif res["result"] == "NOTHING" and not found:
                    if res["closest"] < closest:
                        closest = res["closest"]
                if found or closest < self.MAXCLOSESTDISTANCE / 2:
                    self.my_actions.append({"action": "fireWeapon", "delay": 1})
                    self.current_aggresiveness = max(
                        0, self.current_aggresiveness - 0.2)
        elif w == "frag":
            # (vs Laika 模式无 frag; 移植保留)
            if my.frag_fired and my.last_frag is not None:
                frag = my.last_frag
                dx = my.x - frag.x
                dy = my.y - frag.y
                d = math.sqrt(dx * dx + dy * dy)
                col = self.check_path_for_collision(
                    frag.x, frag.y, dx / d, dy / d, 1,
                    math.ceil(d), math.ceil(d)) if d > 0 else None
                if col is not None or d >= self.FRAGBOMBSAFETYDIST:
                    for i in range(g.tanks_count):
                        t = g.tanks[i]
                        if t.alive and t is not my:
                            dx = t.x - frag.x
                            dy = t.y - frag.y
                            d = math.sqrt(dx * dx + dy * dy)
                            if d <= self.FRAGBOMBDETONATEDIST and d > 0:
                                col = self.check_path_for_collision(
                                    frag.x, frag.y, dx / d, dy / d, 1,
                                    math.ceil(d), math.ceil(d))
                                if col is None:
                                    self.my_actions.append(
                                        {"action": "fireWeapon", "delay": 1})

    # ------------------------------------------------ 路径 -> 动作

    def push_actions_to_follow_path(self, path):
        """AI:337-375 — 注意动作栈后进先出:
        先压 driveToField(远端..path[1]), 最后压 driveToPos(path[0]) 先执行。
        """
        scale = self.game.scale
        for i in range(len(path) - 1, 0, -1):
            self.my_actions.append({"action": "driveToField",
                                    "x": path[i]["x"], "y": path[i]["y"]})
        if path:
            self.my_actions.append({
                "action": "driveToPos",
                "x": (path[0]["x"] + 0.5) * scale,
                "y": (path[0]["y"] + 0.5) * scale,
                "canReverse": len(path) <= 2})

    # ------------------------------------------------ 目标评估 (主决策)

    def make_decisions_and_update_goal(self):
        """AI:376-721 — 返回 True 表示需要重新生成动作栈。"""
        g = self.game
        scale = g.scale
        my = self.my_tank

        if self.my_goal["period"] > 0:
            self.my_goal["period"] -= 1
            return self.my_goal["updateContinuously"]

        self.my_goal["priority"] *= 0.9000000000000002
        old_goal = self.my_goal
        fx = math.floor(my.x / scale)
        fy = math.floor(my.y / scale)

        # ---- goForCrate (AI:387-414; vs Laika 无箱, 循环为空) ----
        if g.alive_count > 1 and my.current_weapon == "bullet":
            crates = [obj for name, obj in g.iter_named_objects()
                      if name[:5] == "crate"]
            best_cd = self.MAXCELLDISTTOGOFORCRATE
            goal = {"priority": 0}
            for crate in crates:
                cx = math.floor(crate.x / scale)
                cy = math.floor(crate.y / scale)
                cd = self._cell_dist(fx, fy, cx, cy)
                if cd <= best_cd:
                    best_cd = cd
                    goal = {"goal": "goForCrate", "x": cx, "y": cy,
                            "period": 10,
                            "priority": ((self.MAXCELLDISTTOGOFORCRATE - cd)
                                         / self.MAXCELLDISTTOGOFORCRATE
                                         * self.GREEDY
                                         * (g.settings_max_bullets
                                            - my.bullets_fired)
                                         / g.settings_max_bullets),
                            "updateContinuously": False, "id": self.goal_id}
                    self.goal_id += 1
            self.update_goal(goal)

        # ---- dodgeBullet (AI:415-424) ----
        bullets = [obj for name, obj in g.iter_named_objects()
                   if name[:6] == "bullet"]
        goal = self.dodge_trajectories(
            fx, fy, bullets, self.MAXTIMETODODGEBULLET,
            self.MAXDISTTODODGEBULLET, self.MAXCELLDISTTODODGEBULLET,
            C.BULLETHITCHECKINTERVALS, True)
        self.update_goal(goal)

        # ---- dodgeFragbomb / fragment (AI:425-461; vs Laika 为空) ----
        # ---- dodgeGatlingBullet (AI:462-471; vs Laika 为空) ----
        # ---- dodgeLaser (AI:472-525; 无 aimer 装备, 循环无效果) ----

        # ---- shootAfter (AI:526-616) ----
        w = my.current_weapon
        if w in ("bullet", "laser"):
            if my.bullets_fired < g.settings_max_bullets or w == "laser":
                for i in range(g.tanks_count):
                    t = g.tanks[i]
                    if t.alive and t is not my:
                        dm = g.dist_map(fx, fy)
                        if dm is None:
                            continue
                        path = get_shortest_path_with_distances(
                            g.maze, dm, fx, fy,
                            g.tank_fields[i]["x"], g.tank_fields[i]["y"])
                        if len(path) < self.LONGESTPATHTOSHOOT:
                            pr = (1 if len(path)
                                  <= self.LONGESTPATHTONOTHESITATETOSHOOT
                                  else (self.LONGESTPATHTOSHOOT - len(path))
                                  / self.LONGESTPATHTOSHOOT
                                  * self.current_aggresiveness)
                            goal = {"goal": "shootAfter", "target": t,
                                    "period": 10, "priority": pr,
                                    "updateContinuously": False,
                                    "id": self.goal_id}
                            self.goal_id += 1
                            self.update_goal(goal)

        # ---- runAway (AI:617-662) ----
        if (g.alive_count > 1 and my.current_weapon == "bullet"
                and my.bullets_fired == g.settings_max_bullets):
            w_, h_ = len(g.maze), len(g.maze[0])
            # 原版 off-by-one: 数组尺寸为 (W-1) x (H-1), 缺行列按 NaN 处理
            summed = [[0.0] * (h_ - 1) for _ in range(w_ - 1)]
            for i in range(g.tanks_count):
                t = g.tanks[i]
                if (t.alive and t is not my
                        and t.bullets_fired != g.settings_max_bullets):
                    dm = g.dist_map(g.tank_fields[i]["x"],
                                    g.tank_fields[i]["y"])
                    for xx in range(w_ - 1):
                        for yy in range(h_ - 1):
                            if dm is None or dm[xx][yy] is None:
                                summed[xx][yy] = float("nan")
                            else:
                                summed[xx][yy] += dm[xx][yy]
            here = (summed[fx][fy]
                    if fx < w_ - 1 and fy < h_ - 1 else float("nan"))
            if here < self.LONGESTPATHTORUN:
                goal = {"goal": "runAway", "dist": summed, "period": 10,
                        "priority": ((self.LONGESTPATHTORUN - here)
                                     / self.LONGESTPATHTORUN
                                     * self.COWARDNESS
                                     * (my.bullets_fired
                                        / g.settings_max_bullets)),
                        "updateContinuously": False, "id": self.goal_id}
                self.goal_id += 1
                self.update_goal(goal)

        # ---- backAway (AI:663-672) ----
        if my.hit_something:
            self.stuck_time = min(self.stuck_time + 1, self.MAXSTUCKTIME)
        else:
            self.stuck_time = 0
        goal = {"goal": "backAway", "period": 5,
                "priority": self.stuck_time / (self.MAXSTUCKTIME - 0.1),
                "updateContinuously": False, "id": self.goal_id}
        self.goal_id += 1
        self.update_goal(goal)

        # ---- 空闲追击 driveTo (AI:673-685) ----
        if g.alive_count > 1:
            k = self._rand(g.tanks_count)
            guard = 0
            while (g.tanks[k] is my or not g.tanks[k].alive) and guard < 1000:
                k = self._rand(g.tanks_count)
                guard += 1
            if g.tanks[k] is not my:
                goal = {"goal": "driveTo", "period": 10,
                        "priority": self.IDLEDRIVETOWARDENEMYPRIORITY,
                        "x": g.tank_fields[k]["x"],
                        "y": g.tank_fields[k]["y"],
                        "updateContinuously": False, "id": self.goal_id}
                self.goal_id += 1
                self.update_goal(goal)

        # ---- 目标切换 (AI:686-720) ----
        if old_goal["id"] != self.my_goal["id"]:
            gname = self.my_goal.get("goal")
            if gname == "shootAfter":
                self.current_aggresiveness = max(
                    0, self.current_aggresiveness - 0.2)
            elif gname == "sprayBullets":
                self.current_aggresiveness = max(
                    0, self.current_aggresiveness - 0.1)
            elif gname == "detonate":
                self.current_aggresiveness = max(
                    0, self.current_aggresiveness - 0.1)
            return True
        self.current_aggresiveness = min(
            self.AGGRESIVENESS,
            self.current_aggresiveness + self.AGGRESIVENESS / 50)
        return self.my_goal["updateContinuously"]

    # ------------------------------------------------ 目标 -> 动作栈

    def decide_actions_to_achieve_goal(self):
        """AI:722-1051"""
        g = self.game
        scale = g.scale
        my = self.my_tank
        self.my_actions = []
        fx = math.floor(my.x / scale)
        fy = math.floor(my.y / scale)
        goal = self.my_goal
        gname = goal.get("goal")

        if gname == "shootAfter":
            best_angle = my.rotation
            found = False
            best_time = C.BULLETLIFETIME
            closest = C.MOVIEWIDTH + C.MOVIEHEIGHT
            angle = my.rotation
            # 直线可达判定 (AI:735-767)
            dx = goal["target"].x - my.x
            dy = goal["target"].y - my.y
            d = math.sqrt(dx * dx + dy * dy)
            col = (self.check_path_for_collision(
                my.x, my.y, dx / d, dy / d, 1, math.ceil(d), math.ceil(d))
                if d > 0 else None)
            if col is None:
                found = True
                closest = 0
                if dx != 0:
                    if dx > 0:
                        best_angle = 90 + math.atan(dy / dx) * 180 / PI
                    else:
                        best_angle = -90 + math.atan(dy / dx) * 180 / PI
                elif dy > 0:
                    best_angle = 180
                elif dy < 0:
                    best_angle = 0
                else:
                    best_angle = angle
            if not found:
                # 角度探测 ±turnSpeed*k^2 (AI:768-810)
                for k in range(1, 4):
                    res = self.check_bullet_path(angle)
                    if res["result"] == "HIT":
                        found = True
                        if res["time"] < best_time:
                            best_time = res["time"]
                            closest = 0
                            best_angle = angle
                    elif res["result"] == "NOTHING" and not found:
                        if res["closest"] < closest:
                            closest = res["closest"]
                            best_angle = angle
                    if self.game.rng.random() < 0.5:
                        angle += my.turn_speed * k * k
                    else:
                        angle -= my.turn_speed * k * k
                    if angle < -180:
                        angle = 360 + angle
                    if angle > 180:
                        angle -= 360
            # 生成动作 (AI:811-829)
            div = 1 if my.current_weapon != "laser" else 2
            if found or closest < self.MAXCLOSESTDISTANCE / div:
                self.my_actions.append({"action": "fireWeapon", "delay": 5})
                self.my_actions.append({"action": "turnTo",
                                        "angle": best_angle})
            elif best_angle != my.rotation:
                self.my_actions.append({"action": "turnTo",
                                        "angle": best_angle})
            else:
                a = my.rotation + 180
                if a > 180:
                    a -= 360
                self.my_actions.append({"action": "turnTo", "angle": a})

        elif gname == "sprayBullets":
            # (AI:831-895; vs Laika 无 gatling, 移植保留)
            best_angle = my.rotation
            found = False
            best_time = C.GATLINGLIFETIME
            closest = C.MOVIEWIDTH + C.MOVIEHEIGHT
            angle = my.rotation
            for k in range(1, 4):
                res = self.check_bullet_path(angle)
                if res["result"] == "HIT":
                    found = True
                    if res["time"] < best_time:
                        best_time = res["time"]
                        closest = 0
                        best_angle = angle
                elif res["result"] == "NOTHING" and not found:
                    if res["closest"] < closest:
                        closest = res["closest"]
                        best_angle = angle
                if self.game.rng.random() < 0.5:
                    angle += my.turn_speed * k * k
                else:
                    angle -= my.turn_speed * k * k
                if angle < -180:
                    angle = 360 + angle
                if angle > 180:
                    angle -= 360
            if found or closest < self.MAXCLOSESTDISTANCE:
                self.my_actions.append({"action": "fireWeapon", "delay": 75})
                self.my_actions.append({"action": "turnTo",
                                        "angle": best_angle})
            elif best_angle != my.rotation:
                self.my_actions.append({"action": "turnTo",
                                        "angle": best_angle})
            else:
                a = my.rotation + 180
                if a > 180:
                    a -= 360
                self.my_actions.append({"action": "turnTo", "angle": a})

        elif gname == "detonate":
            self.my_actions.append({"action": "fireWeapon", "delay": 1})

        elif gname == "driveTo":
            dm = g.dist_map(fx, fy)
            if dm is not None:
                path = get_shortest_path_with_distances(
                    g.maze, dm, fx, fy, goal["x"], goal["y"])
                self.push_actions_to_follow_path(path)

        elif gname == "runAway":
            path = follow_gradient_path_with_distances_and_dead_ends(
                g.maze, goal["dist"], g.dead_ends, fx, fy, 5)
            self.push_actions_to_follow_path(path)

        elif gname == "backAway":
            # (AI:909-951)
            self.my_actions.append({
                "action": "driveToPos",
                "x": (fx + 0.5) * scale, "y": (fy + 0.5) * scale,
                "canReverse": False})
            if my.expanded_hit_check(my.hit_points_front, 1.1):
                if my.expanded_hit_check(my.hit_points_rear, 1.1):
                    if my.expanded_hit_check(my.hit_points_left,
                                             1.3000000000000005):
                        self.my_actions.append({"action": "backupAndTurn",
                                                "dist": 5, "dir": "left"})
                    else:
                        self.my_actions.append({"action": "backupAndTurn",
                                                "dist": 5, "dir": "right"})
                else:
                    self.my_actions.append({"action": "backup", "dist": 3})
            elif my.expanded_hit_check(my.hit_points_rear, 1.1):
                if my.expanded_hit_check(my.hit_points_front, 1.1):
                    if my.expanded_hit_check(my.hit_points_left,
                                             1.3000000000000005):
                        self.my_actions.append({"action": "backupAndTurn",
                                                "dist": 5, "dir": "left"})
                    else:
                        self.my_actions.append({"action": "backupAndTurn",
                                                "dist": 5, "dir": "right"})
                else:
                    self.my_actions.append({"action": "forward", "dist": 3})
            else:
                self.my_actions.append({"action": "backup", "dist": 3})

        elif gname == "dodgeBullet":
            # (AI:952-1019)
            bx = math.floor(goal["x"] / scale)
            by = math.floor(goal["y"] / scale)
            dm = g.dist_map(bx, by)
            if dm is not None:
                path = follow_gradient_path_with_distances_and_dead_ends(
                    g.maze, dm, g.dead_ends, fx, fy, 5)
            else:
                path = []
            close_call = (goal["t"] < goal["maxTime"] / 3
                          and goal["dist"] < goal["maxDist"] / 5)
            if close_call or len(path) <= 1:
                # 被逼入角落 / 来不及跑: 转向与弹道平行 (AI:962-995)
                cur = my.rotation
                gd = goal["dir"]
                if gd["x"] != 0:
                    if gd["x"] > 0:
                        a = 90 + math.atan(gd["y"] / gd["x"]) * 180 / PI
                    else:
                        a = -90 + math.atan(gd["y"] / gd["x"]) * 180 / PI
                elif gd["y"] > 0:
                    a = 180
                elif gd["y"] < 0:
                    a = 0
                else:
                    a = cur
                if 90 < abs(a - cur) < 270:
                    a += 180
                    if a > 180:
                        a -= 360
                a = round(a / my.turn_speed) * my.turn_speed
                self.my_actions.append({"action": "turnTo", "angle": a})
                if goal["dist"] < scale / 4:
                    # 垂直闪避 (AI:996-1012)
                    dl = math.sqrt(gd["x"] ** 2 + gd["y"] ** 2)
                    if dl > 0:
                        perp = {"x": -gd["y"] / dl, "y": gd["x"] / dl}
                        p1 = {"x": goal["closest"]["x"] + perp["x"] * scale / 2,
                              "y": goal["closest"]["y"] + perp["y"] * scale / 2}
                        p2 = {"x": goal["closest"]["x"] - perp["x"] * scale / 2,
                              "y": goal["closest"]["y"] - perp["y"] * scale / 2}
                        d1 = math.hypot(my.x - p1["x"], my.y - p1["y"])
                        d2 = math.hypot(my.x - p2["x"], my.y - p2["y"])
                        if d1 < d2:
                            self.my_actions.append(
                                {"action": "driveToPos", "x": p1["x"],
                                 "y": p1["y"], "canReverse": True})
                        else:
                            self.my_actions.append(
                                {"action": "driveToPos", "x": p2["x"],
                                 "y": p2["y"], "canReverse": True})
            else:
                self.push_actions_to_follow_path(path)
            self.try_to_retaliate()

        elif gname == "dodgeFragbomb":
            # (AI:1020-1034; vs Laika 不触发, 移植保留)
            frag = goal["frag"]
            bx = math.floor(frag.x / scale)
            by = math.floor(frag.y / scale)
            dm = g.dist_map(bx, by)
            if dm is not None:
                path = follow_gradient_path_with_distances_and_dead_ends(
                    g.maze, dm, g.dead_ends, fx, fy, 5)
                if len(path) > 1:
                    self.push_actions_to_follow_path(path)
                else:
                    path = follow_gradient_path_with_distances(
                        g.maze, dm, fx, fy, 5)
                    self.push_actions_to_follow_path(path)
            self.try_to_retaliate()

        elif gname == "dodgeLaser":
            # (AI:1035-1041; vs Laika 不触发, 移植保留)
            owner = goal["owner"]
            bx = math.floor(owner.x / scale)
            by = math.floor(owner.y / scale)
            dm = g.dist_map(bx, by)
            if dm is not None:
                path = follow_gradient_path_with_distances_and_dead_ends(
                    g.maze, dm, g.dead_ends, fx, fy, 2)
                self.push_actions_to_follow_path(path)
            self.try_to_retaliate()

        elif gname == "goForCrate":
            dm = g.dist_map(fx, fy)
            if dm is not None:
                path = get_shortest_path_with_distances(
                    g.maze, dm, fx, fy, goal["x"], goal["y"])
                if path:
                    self.my_actions.append({
                        "action": "driveToPos",
                        "x": (path[-1]["x"] + 0.5) * scale,
                        "y": (path[-1]["y"] + 0.5) * scale,
                        "canReverse": True})
                    self.push_actions_to_follow_path(path)

        elif gname == "idle":
            self.my_actions.append({"action": "idle"})

    # ------------------------------------------------ 动作栈 -> 输入

    def set_input_to_do_actions(self):
        """AI:1052-1354 — 弹出/检查动作完成度, 再按栈顶动作设置坦克输入。"""
        g = self.game
        scale = g.scale
        my = self.my_tank
        fx = math.floor(my.x / scale)
        fy = math.floor(my.y / scale)

        # 第一段: 弹出栈顶, 未完成则压回 (AI:1056-1113)
        action = self.my_actions.pop() if self.my_actions else None
        if action is not None:
            a = action["action"]
            if a == "driveToField":
                if (abs(my.x - (action["x"] + 0.5) * scale) > scale / 3
                        or abs(my.y - (action["y"] + 0.5) * scale) > scale / 3):
                    self.my_actions.append(action)
            elif a == "turnTo":
                if abs(my.rotation - action["angle"]) >= my.turn_speed:
                    self.my_actions.append(action)
            elif a == "fireWeapon":
                if action["delay"] != 0:
                    action["delay"] -= 1
                    self.my_actions.append(action)
            elif a == "driveToPos":
                if (abs(my.x - action["x"]) > scale / 4
                        or abs(my.y - action["y"]) > scale / 4):
                    self.my_actions.append(action)
            elif a == "forward":
                if action["dist"] != 0:
                    action["dist"] -= 1
                    self.my_actions.append(action)
            elif a == "forwardAndTurn":
                if action["dist"] != 0:
                    action["dist"] -= 1
                    self.my_actions.append(action)
                # 注: 原版此 case 无 break, 会落入 backup 分支再判一次;
                # 由于 dist 已减 1, 若仍非 0 会再压一次 -> 复刻该 fallthrough
                if action["dist"] != 0:
                    action["dist"] -= 1
                    self.my_actions.append(action)
            elif a == "backup":
                if action["dist"] != 0:
                    action["dist"] -= 1
                    self.my_actions.append(action)
            elif a == "backupAndTurn":
                if action["dist"] != 0:
                    action["dist"] -= 1
                    self.my_actions.append(action)
            elif a == "idle":
                self.my_actions.append(action)

        # 第二段: 按新的栈顶设置输入 (AI:1114-1353)
        action = self.my_actions[-1] if self.my_actions else None
        if action is None:
            my.turn_left = my.turn_right = False
            my.forward = my.backup = my.fire = False
            self.my_goal["period"] = 0
            return

        a = action["action"]
        if a == "driveToField":
            cur = my.rotation
            if fx > action["x"]:
                target = -90
            elif fx < action["x"]:
                target = 90
            elif fy > action["y"]:
                target = 0
            elif fy < action["y"]:
                target = 180
            else:
                target = cur
            self._turn_toward(target, cur)
            if 90 < abs(target - cur) < 270:
                my.forward = False
                my.backup = False
            else:
                my.forward = True
                my.backup = False
            my.fire = False

        elif a == "turnTo":
            self._turn_toward(action["angle"], my.rotation)
            my.forward = False
            my.backup = False
            my.fire = False

        elif a == "fireWeapon":
            my.turn_left = my.turn_right = False
            my.forward = my.backup = False
            my.fire = True

        elif a == "driveToPos":
            cur = my.rotation
            reverse = False
            dx = action["x"] - my.x
            dy = action["y"] - my.y
            if dx != 0:
                if dx > 0:
                    target = 90 + math.atan(dy / dx) * 180 / PI
                else:
                    target = -90 + math.atan(dy / dx) * 180 / PI
            elif dy > 0:
                target = 180
            elif dy < 0:
                target = 0
            else:
                target = cur
            target = my.turn_speed * round(target / my.turn_speed)
            if action["canReverse"]:
                if 90 < abs(target - cur) < 270:
                    reverse = True
                    target += 180
                    if target > 180:
                        target -= 360
            # 带死区的转向 (AI:1268-1298)
            if target > cur:
                if abs(target - cur) > 180:
                    my.turn_left = abs(target - cur) < 360 - my.turn_speed
                    my.turn_right = False
                else:
                    my.turn_left = False
                    my.turn_right = abs(target - cur) > my.turn_speed
            elif target < cur:
                if abs(target - cur) > 180:
                    my.turn_left = False
                    my.turn_right = abs(target - cur) < 360 - my.turn_speed
                else:
                    my.turn_left = abs(target - cur) > my.turn_speed
                    my.turn_right = False
            else:
                my.turn_left = False
                my.turn_right = False
            if 45 < abs(target - cur) < 315:
                my.forward = False
                my.backup = False
            else:
                my.forward = not reverse
                my.backup = reverse
            my.fire = False

        elif a == "forward":
            my.turn_left = my.turn_right = False
            my.forward = True
            my.backup = False
            my.fire = False

        elif a == "forwardAndTurn":
            my.turn_left = action["dir"] == "left"
            my.turn_right = action["dir"] == "right"
            my.forward = True
            my.backup = False
            my.fire = False

        elif a == "backup":
            my.turn_left = my.turn_right = False
            my.forward = False
            my.backup = True
            my.fire = False

        elif a == "backupAndTurn":
            my.turn_left = action["dir"] == "left"
            my.turn_right = action["dir"] == "right"
            my.forward = False
            my.backup = True
            my.fire = False

        elif a == "idle":
            my.turn_left = my.turn_right = False
            my.forward = my.backup = my.fire = False

        else:
            my.turn_left = my.turn_right = False
            my.forward = my.backup = my.fire = False
            self.my_goal["period"] = 0

    def _turn_toward(self, target, cur):
        """AI:1140-1170 / 1186-1216 — 选择较近方向转向 (无死区版本)"""
        my = self.my_tank
        if target > cur:
            if abs(target - cur) > 180:
                my.turn_left = True
                my.turn_right = False
            else:
                my.turn_left = False
                my.turn_right = True
        elif target < cur:
            if abs(target - cur) > 180:
                my.turn_left = False
                my.turn_right = True
            else:
                my.turn_left = True
                my.turn_right = False
        else:
            my.turn_left = False
            my.turn_right = False
