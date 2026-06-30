from __future__ import annotations

import argparse
import tkinter as tk
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageTk

from tank_env import make_tank_duel_env


@dataclass
class ControlState:
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    fire: bool = False
    restart_requested: bool = False

    def to_action(self) -> np.ndarray:
        throttle = 1
        turn = 1

        if self.up and not self.down:
            throttle = 2
        elif self.down and not self.up:
            throttle = 0

        if self.left and not self.right:
            turn = 0
        elif self.right and not self.left:
            turn = 2

        fire = 1 if self.fire else 0
        return np.array([throttle, turn, fire], dtype=np.int64)


@dataclass
class GameApp:
    env_seed: int | None = None
    fps: int = 20
    arena_preset: str = "maze_random"
    control: ControlState = field(default_factory=ControlState)

    def __post_init__(self) -> None:
        self.env = make_tank_duel_env(render_mode="rgb_array", arena_preset=self.arena_preset)
        self.root = tk.Tk()
        self.root.title("Tank Trouble Arena")
        self.root.configure(bg="#f2eee5")

        self.photo: ImageTk.PhotoImage | None = None
        self.last_info: dict[str, object] = {}
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.seed_offset = 0

        self.status_var = tk.StringVar(value="Loading arena...")
        self.map_var = tk.StringVar(value="Map: -")
        self.player_var = tk.StringVar(value="Player: -")
        self.enemy_var = tk.StringVar(value="Teacher: -")
        self.los_var = tk.StringVar(value="Line of sight: -")
        self.bullet_var = tk.StringVar(value="Bullets: -")

        shell = tk.Frame(self.root, bg="#f2eee5", padx=16, pady=16)
        shell.pack(fill="both", expand=True)

        left = tk.Frame(shell, bg="#f2eee5")
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(shell, bg="#e6decd", width=260, padx=16, pady=16)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        title = tk.Label(
            left,
            text="Tank Trouble AI Playground",
            font=("Helvetica", 20, "bold"),
            bg="#f2eee5",
            fg="#1b1b1b",
        )
        title.pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Label(left, bd=0, bg="#f2eee5")
        self.canvas.pack(anchor="w")

        self.status = tk.Label(
            left,
            textvariable=self.status_var,
            font=("Menlo", 12),
            bg="#f2eee5",
            fg="#2b2b2b",
            justify="left",
        )
        self.status.pack(anchor="w", pady=(10, 0))

        sidebar_title = tk.Label(
            right,
            text="Arena Readout",
            font=("Helvetica", 16, "bold"),
            bg="#e6decd",
            fg="#191919",
        )
        sidebar_title.pack(anchor="w", pady=(0, 12))

        for variable in [self.map_var, self.player_var, self.enemy_var, self.los_var, self.bullet_var]:
            tk.Label(
                right,
                textvariable=variable,
                font=("Menlo", 11),
                bg="#e6decd",
                fg="#2b2b2b",
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=4)

        tk.Button(
            right,
            text="New Maze (R)",
            command=self._request_restart,
            font=("Helvetica", 12, "bold"),
            bg="#2f5dff",
            fg="white",
            activebackground="#2146cc",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=8,
        ).pack(fill="x", pady=(18, 10))

        tk.Button(
            right,
            text="Quit (Esc)",
            command=self.root.destroy,
            font=("Helvetica", 12),
            bg="#f2eee5",
            fg="#202020",
            activebackground="#ddd5c6",
            relief="flat",
            padx=12,
            pady=8,
        ).pack(fill="x")

        help_text = (
            "Controls\n"
            "W / S  move\n"
            "A / D  turn\n"
            "Space  fire\n"
            "R      new maze\n"
            "Esc    quit"
        )
        tk.Label(
            right,
            text=help_text,
            font=("Menlo", 11),
            bg="#e6decd",
            fg="#3a3a3a",
            justify="left",
            anchor="nw",
        ).pack(fill="x", pady=(18, 0))

        self._bind_keys()
        self._reset()

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress-w>", lambda _e: self._set("up", True))
        self.root.bind("<KeyRelease-w>", lambda _e: self._set("up", False))
        self.root.bind("<KeyPress-s>", lambda _e: self._set("down", True))
        self.root.bind("<KeyRelease-s>", lambda _e: self._set("down", False))
        self.root.bind("<KeyPress-a>", lambda _e: self._set("left", True))
        self.root.bind("<KeyRelease-a>", lambda _e: self._set("left", False))
        self.root.bind("<KeyPress-d>", lambda _e: self._set("right", True))
        self.root.bind("<KeyRelease-d>", lambda _e: self._set("right", False))
        self.root.bind("<KeyPress-space>", lambda _e: self._set("fire", True))
        self.root.bind("<KeyRelease-space>", lambda _e: self._set("fire", False))
        self.root.bind("<KeyPress-r>", lambda _e: self._request_restart())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def _set(self, field_name: str, value: bool) -> None:
        setattr(self.control, field_name, value)

    def _request_restart(self) -> None:
        self.control.restart_requested = True

    def _current_seed(self) -> int | None:
        if self.env_seed is None:
            return None
        return self.env_seed + self.seed_offset

    def _reset(self) -> None:
        _, self.last_info = self.env.reset(seed=self._current_seed())
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.control.restart_requested = False
        self._render()
        if self.env_seed is not None:
            self.seed_offset += 1

    def _render(self) -> None:
        frame = self.env.render()
        if frame is None:
            return

        image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.configure(image=self.photo)

        player = self.last_info.get("player_tank", {})
        enemy = self.last_info.get("enemy_tank", {})
        bullets = self.last_info.get("bullet_count", 0)
        player_spawn = self.last_info.get("player_spawn_cell", ("?", "?"))
        enemy_spawn = self.last_info.get("enemy_spawn_cell", ("?", "?"))
        player_los = self.last_info.get("player_has_line_of_sight", False)
        enemy_los = self.last_info.get("enemy_has_line_of_sight", False)

        status = (
            f"reward={self.last_reward:+.3f}   "
            f"player=({player.get('x', 0):.0f}, {player.get('y', 0):.0f})   "
            f"teacher=({enemy.get('x', 0):.0f}, {enemy.get('y', 0):.0f})"
        )
        if self.terminated:
            status += "   GAME OVER"
        elif self.truncated:
            status += "   TIME UP"
        self.status_var.set(status)

        self.map_var.set(
            f"Map #{self.last_info.get('episode_index', 0)}   preset={self.arena_preset}   "
            f"seed={self._current_seed() if self.env_seed is not None else 'random'}"
        )
        self.player_var.set(f"Player spawn cell: {player_spawn}")
        self.enemy_var.set(f"Teacher spawn:    {enemy_spawn}")
        self.los_var.set(f"Line of sight: player={player_los}  teacher={enemy_los}")
        self.bullet_var.set(f"Bullets in arena: {bullets}")

    def tick(self) -> None:
        if self.control.restart_requested:
            self._reset()
        elif not self.terminated and not self.truncated:
            action = self.control.to_action()
            _, reward, self.terminated, self.truncated, self.last_info = self.env.step(action)
            self.last_reward = reward
            self._render()

        delay_ms = max(1, int(1000 / self.fps))
        self.root.after(delay_ms, self.tick)

    def run(self) -> None:
        self.root.after(0, self.tick)
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play the Tank Duel environment locally.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--arena-preset",
        type=str,
        choices=["maze_random", "small_fixed", "micro_fixed", "cover_fixed"],
        default="maze_random",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = GameApp(
        env_seed=args.seed,
        fps=args.fps,
        arena_preset=args.arena_preset,
    )
    app.run()
