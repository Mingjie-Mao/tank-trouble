import random
import unittest

from tank_trouble_original.game import Game
from training.exact_state import (
    apply_controls,
    clone_exact_game,
    state_fingerprint,
    verify_clone_trajectory,
)
from training.exact_state_mpc_teacher import (
    exact_root_search,
    exact_successor_viability,
    _parse_seeds,
    prefer_nonfire_secured_kill,
    prefer_nonfire_low_gain,
)
from training.opportunity_teacher_v2 import OpportunityAnalyzer360


def _schedule(seed):
    rng = random.Random(seed)
    actions = []
    for _ in range(512):
        throttle = rng.randrange(3)
        turn = rng.randrange(3)
        actions.append({
            "forward": throttle == 2,
            "backup": throttle == 0,
            "turn_left": turn == 0,
            "turn_right": turn == 2,
            "fire": rng.random() < 0.1,
        })
    return lambda frame: actions[frame % len(actions)]


class ExactStateCloneTest(unittest.TestCase):
    def test_clone_preserves_complete_object_graph(self):
        game = Game(seed=970000, ai_enabled=True)
        schedule = _schedule(1)
        for frame in range(73):
            apply_controls(game, schedule(frame))
            game.step()

        cloned = clone_exact_game(game, verify=True)

        self.assertEqual(state_fingerprint(game), state_fingerprint(cloned))
        self.assertIsNot(game, cloned)
        self.assertIs(cloned.tanks[1].ai.game, cloned)
        self.assertIs(cloned.tanks[1].ai.my_tank, cloned.tanks[1])
        self.assertEqual(game.rng.getstate(), cloned.rng.getstate())

    def test_clone_is_independent(self):
        game = Game(seed=990000, ai_enabled=True)
        cloned = clone_exact_game(game)
        cloned.tanks[0].x += 1.0
        cloned.rng.random()

        self.assertNotEqual(game.tanks[0].x, cloned.tanks[0].x)
        self.assertNotEqual(game.rng.getstate(), cloned.rng.getstate())

    def test_trajectory_matches_across_hidden_ai_and_rng_state(self):
        for seed in (970000, 990000, 973000):
            with self.subTest(seed=seed):
                game = Game(seed=seed, ai_enabled=True)
                warmup = _schedule(seed)
                for frame in range(91):
                    apply_controls(game, warmup(frame))
                    game.step()
                result = verify_clone_trajectory(
                    game,
                    _schedule(seed + 17),
                    350,
                )
                self.assertTrue(result.matched, result)

    def test_exact_root_search_does_not_mutate_live_game(self):
        game = Game(seed=973034, ai_enabled=True)
        analyzer = OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        before = state_fingerprint(game)

        best, rows = exact_root_search(
            game,
            analyzer,
            metrics,
            (0, 1, 8, 9),
            horizon=12,
        )

        self.assertIn(best, (0, 1, 8, 9))
        self.assertEqual(len(rows), 4)
        self.assertEqual(before, state_fingerprint(game))

    def test_successor_viability_does_not_mutate_live_game(self):
        game = Game(seed=973002, ai_enabled=True)
        analyzer = OpportunityAnalyzer360(game)
        metrics = analyzer.metrics(game)
        before = state_fingerprint(game)
        _, rows = exact_root_search(
            game,
            analyzer,
            metrics,
            (0, 1, 8, 9),
            horizon=12,
            max_death=0.0,
            max_dd=0.0,
        )

        viable, evaluations, details = exact_successor_viability(
            game,
            analyzer,
            rows,
            continuation_indices=(0, 1, 8, 9),
            horizon=12,
        )

        allowed = {row["index"] for row in rows if row["allowed"]}
        self.assertTrue(set(viable).issubset(allowed))
        self.assertEqual(evaluations, len(viable) * 4)
        self.assertLessEqual(len(viable), len(allowed))
        self.assertEqual(set(details), set(viable))
        self.assertEqual(before, state_fingerprint(game))

    def test_secured_kill_tie_prefers_nonfire(self):
        rows = [
            {"index": 3, "allowed": True, "kill": 1.0, "value": 100.0},
            {"index": 2, "allowed": True, "kill": 1.0, "value": 100.0},
            {"index": 14, "allowed": True, "kill": 0.0, "value": 101.0},
        ]

        self.assertEqual(prefer_nonfire_secured_kill(rows, 3), 2)

    def test_unsecured_fire_requires_minimum_gain(self):
        rows = [
            {"index": 3, "allowed": True, "kill": 0.0, "value": 10.9},
            {"index": 2, "allowed": True, "kill": 0.0, "value": 10.0},
        ]

        self.assertEqual(prefer_nonfire_low_gain(rows, 3, 2.0), 2)
        self.assertEqual(prefer_nonfire_low_gain(rows, 3, 0.5), 3)

    def test_seed_ranges_expand(self):
        self.assertEqual(
            _parse_seeds("970000:3,990001,973000:2"),
            (970000, 970001, 970002, 990001, 973000, 973001),
        )


if __name__ == "__main__":
    unittest.main()
