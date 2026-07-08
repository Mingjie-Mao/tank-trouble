#!/usr/bin/env python3
"""
Tank Trouble 本地运行入口 — vs Laika

严格按原版 25 FPS 运行, 游戏逻辑全部在 tank_trouble_original 包内
(渲染层只读状态, 不参与逻辑)。

操作 (对应原版控制方案 1, 同时提供常用替代键):
    E / W / ↑      前进
    D / S / ↓      倒车
    S(原版)=A / ←  左转     (A 和左方向键)
    F / D / →      右转     (F 和右方向键)
    Q / Space / M  开火
    R              重新开始 (新 Game)
    Esc            退出

用法:
    python play_tank_trouble.py [--seed N] [--two-players]
"""

import argparse
import math
import random
import tkinter as tk

from tank_trouble_original import Game, constants as C

DEG = math.pi / 180.0

BG_COLOR = "#FFFFFF"
GROUND_COLOR = "#E6E6E6"   # 15132390
WALL_COLOR = "#4D4D4D"     # 5066061
LAIKA_BASE = "#262626"
LAIKA_TURRET = "#666666"
PLAYER_BASE = "#B02A22"
PLAYER_TURRET = "#D23B32"


class App:
    def __init__(self, seed=None, two_players=False, self_harm_immune=None,
                 invincible=None):
        self.seed = seed
        self.two_players = two_players
        self.self_harm_immune = self_harm_immune
        self.invincible = invincible
        self.game = Game(seed=seed, ai_enabled=not two_players,
                         self_harm_immune=self_harm_immune,
                         invincible=invincible)
        self.rng = random.Random()

        self.root = tk.Tk()
        self.root.title("Tank Trouble (原版 1:1 复刻)")
        self.root.configure(bg=BG_COLOR)
        self.canvas = tk.Canvas(
            self.root, width=C.MOVIEWIDTH + 20, height=C.MOVIEHEIGHT + 20,
            bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack()

        # 键位: 集合中的任一按键按下即视为该输入
        self.p1_keys = {
            "forward": {"e", "w", "Up"},
            "backup": {"d", "s", "Down"},
            "turn_left": {"a", "Left"},
            "turn_right": {"f", "Right"},
            "fire": {"q", "space", "m"},
        }
        # 双人模式: P1 用 ESDF+Q (原版), P2 用方向键+M (原版)
        if two_players:
            self.p1_keys = {
                "forward": {"e"}, "backup": {"d"},
                "turn_left": {"s"}, "turn_right": {"f"}, "fire": {"q"},
            }
            self.p2_keys = {
                "forward": {"Up"}, "backup": {"Down"},
                "turn_left": {"Left"}, "turn_right": {"Right"}, "fire": {"m"},
            }
        self.pressed = set()
        self.root.bind("<KeyPress>", self._on_key_down)
        self.root.bind("<KeyRelease>", self._on_key_up)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self._tick()

    # ------------------------------------------------ 输入

    def _key_name(self, event):
        k = event.keysym
        return k if len(k) > 1 else k.lower()

    def _on_key_down(self, event):
        k = self._key_name(event)
        if k == "r":
            self.game = Game(seed=None, ai_enabled=not self.two_players,
                             self_harm_immune=self.self_harm_immune,
                             invincible=self.invincible)
            return
        self.pressed.add(k)

    def _on_key_up(self, event):
        self.pressed.discard(self._key_name(event))

    def _apply_input(self, tank, keymap):
        tank.forward = bool(self.pressed & keymap["forward"])
        tank.backup = bool(self.pressed & keymap["backup"])
        tank.turn_left = bool(self.pressed & keymap["turn_left"])
        tank.turn_right = bool(self.pressed & keymap["turn_right"])
        tank.fire = bool(self.pressed & keymap["fire"])

    # ------------------------------------------------ 主循环 (25 FPS)

    def _tick(self):
        g = self.game
        self._apply_input(g.tanks[0], self.p1_keys)
        if self.two_players:
            self._apply_input(g.tanks[1], self.p2_keys)
        g.step()
        self._draw()
        self.root.after(1000 // C.FPS, self._tick)  # 40ms

    # ------------------------------------------------ 渲染

    def _draw(self):
        g = self.game
        cv = self.canvas
        cv.delete("all")

        # 屏幕震动 (原版: random(shake) - shake/2)
        ox, oy = 10.0, 10.0
        if g.shake > 1:
            s = int(g.shake)
            ox += self.rng.randrange(max(1, s)) - g.shake / 2
            oy += self.rng.randrange(max(1, s)) - g.shake / 2
        # 居中
        world_w = len(g.maze) * g.scale
        ox += max(0, (C.MOVIEWIDTH - world_w) / 2)

        # 地面
        w, h = len(g.maze), len(g.maze[0])
        cv.create_rectangle(
            ox, oy, ox + math.floor(w * g.scale), oy + math.floor(h * g.scale),
            fill=GROUND_COLOR, outline="")

        # 墙壁 (厚度与碰撞一致: 2*half_t, 方头端帽)
        t2 = g.wall_half_t * 2
        for (x1, y1, x2, y2) in g.walls:
            cv.create_line(ox + x1, oy + y1, ox + x2, oy + y2,
                           width=t2, fill=WALL_COLOR, capstyle="projecting")

        # 子弹 (视觉半径随 SCALE 缩放)
        br = max(2.0, 2.5 * (g.scale / 50.0))
        for b in g.bullets:
            cv.create_oval(ox + b.x - br, oy + b.y - br,
                           ox + b.x + br, oy + b.y + br,
                           fill="#222222", outline="")

        # 坦克
        colors = [(PLAYER_BASE, PLAYER_TURRET), (LAIKA_BASE, LAIKA_TURRET)]
        for t in g.tanks:
            if not t.alive:
                continue
            base_c, turret_c = colors[t.number % len(colors)]
            self._draw_tank(cv, t, ox, oy, base_c, turret_c)

        # 记分板
        cv.create_text(
            ox + 10, oy + math.floor(h * g.scale) + 24, anchor="w",
            text=("你  %d   :   %d  Laika    (回合 %d)"
                  % (g.scores[0], g.scores[1], g.round_number)),
            font=("Helvetica", 16, "bold"), fill="#333333")
        if g.frozen:
            cv.create_text(ox + world_w - 10,
                           oy + math.floor(h * g.scale) + 24, anchor="e",
                           text="回合结束…", font=("Helvetica", 13),
                           fill="#888888")

    def _draw_tank(self, cv, t, ox, oy, base_c, turret_c):
        s = t.display_scale
        th = t.rotation * DEG
        c, sn = math.cos(th), math.sin(th)

        def pt(lx, ly):
            return (ox + t.x + s * (lx * c - ly * sn),
                    oy + t.y + s * (lx * sn + ly * c))

        bw2 = C.TANK_BASE_WIDTH / 2
        bh2 = C.TANK_BASE_HEIGHT / 2
        # 履带底盘
        body = [pt(-bw2, -bh2), pt(bw2, -bh2), pt(bw2, bh2), pt(-bw2, bh2)]
        cv.create_polygon(body, fill=base_c, outline="#1A1A1A", width=1.5)
        # 两侧履带条
        for side in (-1, 1):
            track = [pt(side * bw2, -bh2), pt(side * bw2 * 0.62, -bh2),
                     pt(side * bw2 * 0.62, bh2), pt(side * bw2, bh2)]
            cv.create_polygon(track, fill="#1A1A1A", outline="")
        # 炮管 (矢量图形: 宽 17, 尖端 y=-55)
        hw = C.TANK_SHAPE_BARREL_HALF_WIDTH
        tip = C.TANK_SHAPE_BARREL_TIP_Y
        barrel = [pt(-hw, 0), pt(hw, 0), pt(hw, tip), pt(-hw, tip)]
        cv.create_polygon(barrel, fill=turret_c, outline="#1A1A1A", width=1)
        # 炮塔圆顶 (半径约 23.5)
        dome_r = s * 23.5
        cx, cy = pt(0, 0)
        cv.create_oval(cx - dome_r, cy - dome_r, cx + dome_r, cy + dome_r,
                       fill=turret_c, outline="#1A1A1A", width=1.5)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Tank Trouble 原版复刻")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--two-players", action="store_true",
                    help="本地双人 (关闭 Laika AI): P1=ESDF+Q, P2=方向键+M")
    args = ap.parse_args()
    App(seed=args.seed, two_players=args.two_players).run()


if __name__ == "__main__":
    main()
