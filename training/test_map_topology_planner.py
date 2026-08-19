import unittest

import numpy as np

from tank_trouble_original import Game
from training.map_topology_planner import (
    TOPOLOGY_FEATURE_DIM,
    MapTopologyPlanner,
    MovementHysteresis,
    cardinal_neighbors,
    dead_end_depth,
    drive_action_to_cell,
    shortest_cardinal_path,
)
from training.opportunity_teacher_v2 import OpportunityAnalyzer360


class MapTopologyPlannerTest(unittest.TestCase):
    def setUp(self):
        self.game = Game(seed=970000, ai_enabled=True)
        self.planner = MapTopologyPlanner()
        self.analyzer = OpportunityAnalyzer360(self.game)

    def test_graph_edges_are_symmetric(self):
        for item in self.game.reachable:
            cell = (item["x"], item["y"])
            for neighbor in cardinal_neighbors(self.game.maze, cell):
                self.assertIn(cell, cardinal_neighbors(
                    self.game.maze, neighbor))

    def test_shortest_path_reaches_goal_through_open_edges(self):
        start = (self.game.reachable[0]["x"], self.game.reachable[0]["y"])
        goal = (self.game.reachable[-1]["x"], self.game.reachable[-1]["y"])
        path = shortest_cardinal_path(self.game.maze, start, goal)
        self.assertTrue(path or start == goal)
        previous = start
        for cell in path:
            self.assertIn(cell, cardinal_neighbors(self.game.maze, previous))
            previous = cell
        self.assertEqual(previous, goal)

    def test_dead_end_goal_routes_toward_zero_depth(self):
        dead_cells = [
            (item["x"], item["y"])
            for item in self.game.reachable
            if dead_end_depth(self.game, (item["x"], item["y"])) > 0
        ]
        self.assertTrue(dead_cells)
        cell = max(dead_cells, key=lambda item: dead_end_depth(self.game, item))
        tank = self.game.tanks[0]
        tank.x = (cell[0] + 0.5) * self.game.scale
        tank.y = (cell[1] + 0.5) * self.game.scale
        goal = self.planner.choose_goal(self.game, self.analyzer)
        self.assertEqual(goal.kind, "escape_dead_end")
        self.assertTrue(goal.path)
        self.assertEqual(dead_end_depth(self.game, goal.target), 0.0)

    def test_features_are_bounded_and_stable(self):
        goal = self.planner.choose_goal(self.game, self.analyzer)
        features = self.planner.features(self.game, goal)
        self.assertEqual(features.shape, (TOPOLOGY_FEATURE_DIM,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertLessEqual(np.abs(features).max(), 1.0)

    def test_route_control_matches_tank_heading(self):
        tank = self.game.tanks[0]
        cell = (2, 2)
        tank.x = (cell[0] + 0.5) * self.game.scale
        tank.y = (cell[1] + 0.5) * self.game.scale
        tank.rotation = 0.0
        self.assertEqual(
            drive_action_to_cell(self.game, tank, (2, 1)),
            (2, 1, 0),
        )
        self.assertEqual(
            drive_action_to_cell(self.game, tank, (3, 2)),
            (1, 2, 0),
        )
        self.assertEqual(
            drive_action_to_cell(self.game, tank, (2, 3)),
            (0, 1, 0),
        )

    def test_movement_hysteresis_suppresses_jitter_but_not_fire(self):
        smoother = MovementHysteresis(hold_frames=4)
        self.assertEqual(smoother.choose((2, 0, 0)), (2, 0, 0))
        self.assertEqual(smoother.choose((0, 2, 0)), (2, 0, 0))
        self.assertEqual(smoother.choose((1, 1, 1)), (1, 1, 1))
        self.assertEqual(smoother.choose((0, 2, 0)), (0, 2, 0))
        self.assertEqual(smoother.suppressions, 1)

    def test_movement_hysteresis_interrupts_on_danger(self):
        smoother = MovementHysteresis(hold_frames=4)
        smoother.choose((2, 0, 0))
        self.assertEqual(
            smoother.choose((0, 2, 0), interrupt=True),
            (0, 2, 0),
        )


if __name__ == "__main__":
    unittest.main()
