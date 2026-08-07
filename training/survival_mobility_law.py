"""Hard mobility curriculum: wall contact or two seconds in one cell loses."""

import numpy as np

from training.survival_mode import ECON, FPS, Ledger


MAX_CELL_FRAMES = 2 * FPS


def tank_cell(game):
    tank = game.tanks[0]
    return int(tank.x // game.scale), int(tank.y // game.scale)


class MobilityLawLedger(Ledger):
    def __init__(self, game, econ=ECON):
        super().__init__(game, econ)
        self.mobility_cell = tank_cell(game)
        self.cell_frames = 0
        self.mobility_death = None

    def mobility_features(self, game):
        elapsed = min(self.cell_frames / MAX_CELL_FRAMES, 1.0)
        return np.asarray([
            elapsed,
            1.0 - elapsed,
            float(game.tanks[0].hit_something),
        ], dtype=np.float32)

    def on_frame(self, game, events):
        end = super().on_frame(game, events)
        if end == "death":
            return end
        current = tank_cell(game)
        if current == self.mobility_cell:
            self.cell_frames += 1
        else:
            self.mobility_cell = current
            self.cell_frames = 0
        if game.tanks[0].hit_something:
            self.mobility_death = "wall"
            return "wall_death"
        if self.cell_frames >= MAX_CELL_FRAMES:
            self.mobility_death = "cell_timeout"
            return "cell_death"
        return end
