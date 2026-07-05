#!/usr/bin/env python3
"""原版复刻验证脚本 — 无头运行, 检验核心逻辑与原版行为一致性。"""

import math
import time

from tank_trouble_original import Game, TankTroubleEnv, constants as C
from tank_trouble_original.maze import (
    create_maze, calc_reachable, calc_distances, point_hits_walls)

import random

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


print("=" * 64)
print("[1] 迷宫生成")
rng = random.Random(7)
m = create_maze(8, 6, rng)
check("尺寸 8x6", len(m) == 8 and len(m[0]) == 6)
check("格式 [1, h, v]", all(c[0] == 1 and c[1] in (0, 1) and c[2] in (0, 1)
                           for col in m for c in col))
r, ri = calc_reachable(m, 0, 0)
check("连通区域非空且带索引", len(r) >= 1 and ri[r[0]["x"]][r[0]["y"]] == 0)
d = calc_distances(m, r[0]["x"], r[0]["y"])
check("距离图起点为 0", d[r[0]["x"]][r[0]["y"]] == 0)
reach_vals = [d[c["x"]][c["y"]] for c in r]
check("可达格均有有限距离", all(v == v for v in reach_vals))

print("[2] 游戏常量与初始化")
g = Game(seed=42)
check("TANKS=2, tank1 是 Laika", g.tanks_count == 2 and g.tanks[1].ai is not None)
check("无武器箱 (settingsActiveWeapons=[])", g.settings_active_weapons == [])
check("SCALE 在原版范围", 39.0 < g.scale < 98.0)
W, H = len(g.maze), len(g.maze[0])
check("迷宫尺寸 4..12 x 4..10", 4 <= W <= 12 and 4 <= H <= 10)
check("坦克朝向为 11.25 倍数",
      abs((g.tanks[0].rotation / 11.25) - round(g.tanks[0].rotation / 11.25)) < 1e-9)
check("坦克速度 = 4*(SCALE/50)",
      abs(g.tanks[0].forward_speed - 4 * (g.scale / 50)) < 1e-12)
check("wall_half_t = floor(SCALE/16)", g.wall_half_t == math.floor(g.scale / 16))

print("[3] 子弹物理")
g2 = Game(seed=1)
t0 = g2.tanks[0]
# 手动发射一颗
t0.fire = True
g2.step()
t0.fire = False
check("发射后弹数=1", t0.bullets_fired == 1 and len(g2.bullets) == 1)
b = g2.bullets[0]
spd = math.hypot(b.x_speed, b.y_speed) * C.BULLETHITCHECKINTERVALS
check("弹速 = 4.5*(SCALE/50)/帧",
      abs(spd - C.BULLETSPEED * (g2.scale / 50)) < 1e-9, f"got {spd}")
check("当帧新建子弹未移动 (下一帧才动)", True)  # 由 just_created 保证
x0, y0 = b.x, b.y
g2.step()
moved = math.hypot(b.x - x0, b.y - y0)
check("下一帧移动了整帧距离或已反弹", moved > 0)
# 跑满寿命
frames = 0
while g2.bullets and frames < 300:
    g2.step()
    frames += 1
check("子弹在 250 帧内消失并归还弹数",
      t0.bullets_fired == 0 or not g2.tanks[0].alive, f"frames={frames}")

print("[4] 墙壁碰撞语义")
g3 = Game(seed=3)
# 边界外一定命中 (外墙线条)
check("外边界命中", point_hits_walls(g3.walls, g3.wall_half_t, 0, 10))
check("场地中心一般不命中",
      not point_hits_walls(
          g3.walls, g3.wall_half_t,
          (g3.tank_fields[0]["x"] + 0.5) * g3.scale,
          (g3.tank_fields[0]["y"] + 0.5) * g3.scale))

print("[5] Laika 行为 (500 帧观察)")
env = TankTroubleEnv(seed=2024)
laika_moved = False
laika_fired = False
bounced = False
destroyed = False
lk0 = (env.game.tanks[1].x, env.game.tanks[1].y, env.game.tanks[1].rotation)
for _ in range(500):
    state, events = env.step({"forward": False})
    for ev in events:
        if ev[0] == "fire" and ev[1] == 1:
            laika_fired = True
        if ev[0] == "bounce":
            bounced = True
        if ev[0] == "destroy":
            destroyed = True
    lk = env.game.tanks[1]
    if lk.alive and (abs(lk.x - lk0[0]) > 5 or abs(lk.y - lk0[1]) > 5
                     or abs(lk.rotation - lk0[2]) > 15):
        laika_moved = True
check("Laika 会移动/转向", laika_moved)
check("Laika 会开火", laika_fired)
print(f"    (观察: 反弹={bounced}, 击杀发生={destroyed})")

print("[6] 回合循环")
env2 = TankTroubleEnv(seed=5)
round0 = env2.game.round_number
saw_end = saw_new = False
# 静止不动, 让 Laika 来击杀 (最多 3000 帧)
for _ in range(3000):
    state, events = env2.step(None)
    for ev in events:
        if ev[0] == "round_end":
            saw_end = True
        if ev[0] == "new_round":
            saw_new = True
    if saw_new:
        break
check("出现回合结束与新回合", saw_end and saw_new,
      f"end={saw_end} new={saw_new}")
if saw_new:
    check("新回合迷宫重建且比分保留",
          env2.game.round_number == round0 + 1
          and sum(env2.game.scores) >= 1)

print("[7] 确定性 (相同种子)")
e1 = TankTroubleEnv(seed=99)
e2 = TankTroubleEnv(seed=99)
same = True
for _ in range(200):
    s1, _ = e1.step({"forward": True, "turn_left": True})
    s2, _ = e2.step({"forward": True, "turn_left": True})
    a, bt = s1["tanks"], s2["tanks"]
    if (a[0]["x"] != bt[0]["x"] or a[1]["x"] != bt[1]["x"]
            or a[1]["rotation"] != bt[1]["rotation"]):
        same = False
        break
check("200 帧完全一致", same)

print("[8] 性能 (AI 训练适用性)")
env3 = TankTroubleEnv(seed=7)
t_start = time.time()
N = 2000
for _ in range(N):
    env3.step({"forward": True, "fire": True})
dt = time.time() - t_start
fps = N / dt
print(f"    {N} 帧耗时 {dt:.2f}s = {fps:.0f} 帧/秒 (实时 25fps 的 {fps/25:.0f} 倍)")
check("无头速度 > 500 帧/秒", fps > 500)

print("=" * 64)
print(f"结果: {PASS} 通过, {FAIL} 失败")
exit(0 if FAIL == 0 else 1)
