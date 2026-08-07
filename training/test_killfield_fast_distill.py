import os
import tempfile
import unittest

import numpy as np
import torch

from tank_trouble_original.game import Game
from training.killfield_distill import DistillLedger
from training.killfield_fast_distill import (
    FAST_MAP_CHANNELS,
    FAST_VECTOR_DIM,
    KillFieldFastPolicy,
    RuntimeDecisionState,
    build_fast_network,
    fast_spatial_observation,
    fast_vector_observation,
    post_kill_survival_scores,
)
from training.survival_frontier_rl import MAP_H, MAP_W, FrontierState
from training.tt_gym_env import TankTroubleGym


class KillFieldFastDistillTests(unittest.TestCase):
    def test_fast_observation_shapes_and_phase(self):
        game = Game(seed=38_500_001, ai_enabled=True)
        encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        ledger = DistillLedger(game, 500)
        frontier = FrontierState(game, dense=True)
        decision = RuntimeDecisionState(game)
        vector = fast_vector_observation(
            encoder, game, ledger, frontier, decision)
        spatial = fast_spatial_observation(game, frontier)
        self.assertEqual(vector.shape, (FAST_VECTOR_DIM,))
        self.assertEqual(
            spatial.shape, (FAST_MAP_CHANNELS, MAP_H, MAP_W))
        self.assertEqual(vector[-4:].tolist(), [1.0, 0.0, 0.0, 0.0])

        game.destroy_tank(1)
        vector = fast_vector_observation(
            encoder, game, ledger, frontier, decision)
        self.assertEqual(vector[-4:].tolist(), [0.0, 1.0, 1.0, 0.0])

    def test_post_kill_teacher_disables_fire(self):
        game = Game(seed=38_500_002, ai_enabled=True)
        game.destroy_tank(1)
        scores = post_kill_survival_scores(game, horizon=8)
        self.assertEqual(scores.shape, (18,))
        self.assertTrue(np.all(scores[::2] > scores[1::2]))

    def test_runtime_post_kill_does_not_call_sandbox_or_fire(self):
        import training.killfield_fast_distill as module

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pt")
            torch.save({
                "state_dict": build_fast_network().state_dict(),
                "version": "test",
                "vector_dim": FAST_VECTOR_DIM,
                "map_channels": FAST_MAP_CHANNELS,
            }, path)
            policy = KillFieldFastPolicy(path)
            game = Game(seed=38_500_003, ai_enabled=True)
            game.destroy_tank(1)
            original = module.make_sandbox

            def forbidden(*_args, **_kwargs):
                raise AssertionError("deployed P38 called make_sandbox")

            module.make_sandbox = forbidden
            try:
                action = policy.act(game)
            finally:
                module.make_sandbox = original
            self.assertFalse(action["fire"])
            self.assertTrue(policy.last_planned)
            self.assertEqual(policy.telemetry()["field_builds"], 0)


if __name__ == "__main__":
    unittest.main()
