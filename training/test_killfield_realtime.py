import time
import math
import unittest
from unittest import mock

import numpy as np

from tank_trouble_original import constants as C
from tank_trouble_original.game import Bullet, Game
from training.killfield_realtime import (
    RealtimeKillFieldTeacher,
    _human_hypotheses,
    _plan_worker,
    _robust_action_score,
)
from training.killfield_teacher import (
    CANDIDATES,
    HuntChainState,
    InverseDensityFieldBuilder,
    density_rollout,
    mask_moving_fire_scores,
)
from training.survival_expert_iter_530 import apply_action


class RealtimeKillFieldTests(unittest.TestCase):
    def test_realtime_execution_never_combines_motion_and_fire(self):
        game = Game(seed=38_700_006, ai_enabled=True)
        policy = RealtimeKillFieldTeacher(
            seed=3, ray_count=16, max_bounces=1,
            max_flight_frames=12, horizon=2, hold=1)
        try:
            moving_fire = CANDIDATES.index((2, 0, 1))
            scores = np.zeros(len(CANDIDATES), dtype=np.float32)
            scores[moving_fire] = 1_000_000.0
            policy._ready = {
                "phase": "combat",
                "round": game.round_number,
                "frame": game.frame,
                "target": (int(game.tanks[1].x // game.scale),
                           int(game.tanks[1].y // game.scale)),
                "field": InverseDensityFieldBuilder(
                    game, 16, 1, 12).build(
                        (int(game.tanks[1].x // game.scale),
                         int(game.tanks[1].y // game.scale))),
                "scores": scores,
                "field_built": True,
                "elapsed": 0.0,
            }
            action = policy._take_ready(game, "combat")
            emitted = policy._emit_action(game, action, "test")
            self.assertFalse(
                emitted["fire"] and (
                    emitted["forward"] or emitted["backup"]
                    or emitted["turn_left"] or emitted["turn_right"]))
            self.assertEqual(policy.last_action, (1, 1, 1))
        finally:
            policy.close()

    def test_post_kill_survival_moves_off_a_lethal_bullet_line(self):
        game = Game(seed=38_700_400, ai_enabled=True)
        me, enemy = game.tanks
        for angle in (index * math.pi / 4.0 for index in range(8)):
            x = me.x + math.cos(angle) * game.scale * 0.75
            y = me.y + math.sin(angle) * game.scale * 0.75
            if game.wall_hit(x, y):
                continue
            bullet = Bullet(game, "test_threat", enemy, game.scale)
            bullet.x, bullet.y = x, y
            speed = (C.BULLETSPEED / C.BULLETHITCHECKINTERVALS
                     * (game.scale / 50.0))
            bullet.x_speed = -math.cos(angle) * speed
            bullet.y_speed = -math.sin(angle) * speed
            bullet.just_created = False
            game.bullets.append(bullet)
            enemy.bullets_fired += 1
            break
        game.destroy_tank(1)
        from training.killfield_fast_distill import post_kill_survival_scores
        scores = post_kill_survival_scores(game, 75)
        idle = CANDIDATES.index((1, 1, 0))
        self.assertLess(scores[idle], 0.0)
        self.assertGreater(float(scores.max()), 0.0)
        self.assertNotEqual(CANDIDATES[int(scores.argmax())], (1, 1, 0))

    def test_human_hypotheses_ignore_private_current_buttons(self):
        game = Game(seed=38_700_000, ai_enabled=True)
        enemy = game.tanks[1]
        before = _human_hypotheses(game, 5, 4, include_current=False)
        enemy.forward = enemy.turn_left = enemy.fire = True
        after = _human_hypotheses(game, 5, 4, include_current=False)
        self.assertEqual(before, after)

    def test_human_profile_never_requests_laika_rollout(self):
        game = Game(seed=38_700_001, ai_enabled=True)
        field = InverseDensityFieldBuilder(
            game, ray_count=16, max_bounces=1, max_frames=12).build(
                (int(game.tanks[1].x // game.scale),
                 int(game.tanks[1].y // game.scale)))
        settings = {
            "horizon": 2,
            "hold": 1,
            "human_samples": 3,
            "human_mean_weight": 0.65,
            "laika_weight": 0.70,
        }
        models = []

        def fake(*_args, **kwargs):
            models.append(kwargs["opp_model"])
            return 1.0

        with mock.patch(
                "training.killfield_realtime.density_rollout", fake):
            score = _robust_action_score(
                game, (1, 1, 0), field, 7,
                HuntChainState(), settings, "human")
        self.assertEqual(score, 1.0)
        self.assertTrue(models)
        self.assertEqual(set(models), {"L1"})

    def test_laika_worker_matches_original_p37_scores(self):
        game = Game(seed=38_700_004, ai_enabled=True)
        settings = {
            "rays": 32, "bounces": 1, "flight_frames": 20,
            "horizon": 4, "hold": 2, "post_kill_horizon": 20,
            "opponent_profile": "laika", "human_samples": 3,
            "laika_weight": 0.70, "human_mean_weight": 0.65,
        }
        target = (int(game.tanks[1].x // game.scale),
                  int(game.tanks[1].y // game.scale))
        field = InverseDensityFieldBuilder(
            game, 32, 1, 20).build(target)
        seed = 91
        expected = np.asarray([
            density_rollout(
                game, action, field, seed, HuntChainState(), 4, 2)
            for action in CANDIDATES
        ], dtype=np.float32)
        mask_moving_fire_scores(expected)
        result = _plan_worker(
            game, settings, seed, HuntChainState(), None, "combat")
        np.testing.assert_allclose(result["scores"], expected)

    def test_async_act_returns_while_worker_plans(self):
        game = Game(seed=38_700_002, ai_enabled=True)
        policy = RealtimeKillFieldTeacher(
            seed=3, ray_count=32, max_bounces=1,
            max_flight_frames=20, horizon=4, hold=2,
            max_plan_seconds=3.0, max_stale_frames=100,
            worker_count=1)
        try:
            elapsed = []
            for _ in range(120):
                started = time.perf_counter()
                action = policy.act(game)
                elapsed.append(time.perf_counter() - started)
                apply_action(game, policy.last_action)
                game.step()
                if policy.async_results:
                    break
                time.sleep(0.005)
            self.assertGreater(policy.async_results, 0)
            self.assertLess(float(np.percentile(elapsed, 95)), 0.050)
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
