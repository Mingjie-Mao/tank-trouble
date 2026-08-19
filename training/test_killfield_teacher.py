import math
import unittest

import numpy as np

from tank_trouble_original import constants as C
from tank_trouble_original.game import Bullet, Game
from training.killfield_teacher import (
    DensityField,
    HuntChainState,
    InverseDensityFieldBuilder,
    KillFieldTeacher,
    action_self_hits,
)
from training.killfield_student import (
    KILLFIELD_EXTRA_DIM,
    KILLFIELD_MAP_DIM,
    P37_OBS_DIM,
    KillFieldFeatureState,
)
from training.killfield_full_distill import (
    ACTION_PREVIEW_DIM,
    P37_FULL_OBS_DIM,
    action_preview_features,
)
from training.coin_path_rules import neighbors
from training.survival_expert_iter_530 import apply_action


class InverseDensityFieldTests(unittest.TestCase):
    def setUp(self):
        self.game = Game(seed=37_500_001, ai_enabled=True)
        self.target = (
            int(self.game.tanks[1].x // self.game.scale),
            int(self.game.tanks[1].y // self.game.scale),
        )

    def _build(self):
        return InverseDensityFieldBuilder(
            self.game, ray_count=128, max_bounces=2).build(self.target)

    def test_votes_are_deterministic_and_unique_per_ray(self):
        first = self._build()
        second = self._build()
        np.testing.assert_array_equal(first.counts, second.counts)
        self.assertGreater(first.max_count, 0)
        self.assertLessEqual(first.max_count, first.ray_count)
        self.assertTrue(np.all(first.counts >= 0))

    def test_public_value_is_discrete_exponential(self):
        field = self._build()
        values = set(float(value) for value in np.unique(field.values))
        allowed = {0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0}
        self.assertTrue(values <= allowed)
        self.assertGreater(len(values - {0.0}), 1)

    def test_best_aim_and_success_rate_are_well_formed(self):
        field = self._build()
        index = np.unravel_index(np.argmax(field.counts), field.counts.shape)
        angle, concentration = field.best_aim_at(index, current_heading=0.0)
        self.assertIsNotNone(angle)
        self.assertGreaterEqual(angle, 0.0)
        self.assertLess(angle, 2.0 * np.pi)
        self.assertGreater(concentration, 0.0)
        self.assertLessEqual(field.success_rate_at(index), 1.0)

    def test_field_excludes_paths_after_flight_deadline(self):
        field = self._build()
        finite = field.min_frames[np.isfinite(field.min_frames)]
        self.assertEqual(field.max_flight_frames, 75)
        self.assertTrue(np.all(finite <= field.max_flight_frames + 1e-5))

    def test_guidance_covers_map_and_only_stops_at_firing_cells(self):
        field = self._build()
        for item in self.game.reachable:
            cell = (int(item["x"]), int(item["y"]))
            self.assertGreater(field.guidance_at(cell), 0.0)
            self.assertLessEqual(field.guidance_at(cell), 1.0 + 1e-6)
            higher = any(
                field.guidance_at(adjacent) > field.guidance_at(cell) + 1e-7
                for adjacent in neighbors(self.game, cell))
            if not higher:
                self.assertGreater(field.count_at(cell), 0)


class HuntChainTests(unittest.TestCase):
    class FakeField:
        target_cell = (9, 9)

        @staticmethod
        def guidance_at(cell):
            return float(cell[0])

    def test_chain_is_exponential_one_shot_and_uphill_only(self):
        state = HuntChainState()
        field = self.FakeField()
        self.assertEqual(state.collect_ascent(field, (0, 0), (1, 0)), 1.0)
        self.assertEqual(state.collect_ascent(field, (1, 0), (2, 0)), 2.0)
        self.assertEqual(state.collect_ascent(field, (2, 0), (3, 0)), 4.0)
        self.assertEqual(state.collect_ascent(field, (3, 0), (2, 0)), 0.0)
        self.assertEqual(state.collect_ascent(field, (2, 0), (3, 0)), 0.0)

    def test_timeout_resets_multiplier_but_not_collected_cells(self):
        state = HuntChainState()
        field = self.FakeField()
        self.assertEqual(state.collect_ascent(field, (0, 0), (1, 0)), 1.0)
        state.advance(75)
        self.assertEqual(state.count, 0)
        self.assertEqual(state.collect_ascent(field, (1, 0), (2, 0)), 1.0)
        self.assertEqual(state.collect_ascent(field, (0, 0), (1, 0)), 0.0)

    def test_target_change_cannot_create_a_reward(self):
        state = HuntChainState()
        reward = state.collect_ascent(
            self.FakeField(), (0, 0), (1, 0), target_stable=False)
        self.assertEqual(reward, 0.0)


class KillFieldTeacherTests(unittest.TestCase):
    def test_moving_fire_is_sanitized_to_stationary_fire(self):
        game = Game(seed=37_500_011, ai_enabled=True)
        teacher = KillFieldTeacher(
            seed=31, ray_count=32, max_bounces=1, horizon=4, hold=2)
        action = teacher._emit_action(game, (2, 0, 1), "test")
        self.assertFalse(action["forward"])
        self.assertFalse(action["turn_left"])
        self.assertTrue(action["fire"])
        self.assertEqual(teacher.last_action, (1, 1, 1))

    def test_post_kill_replans_survival_without_firing(self):
        # 击杀后不再有独立的状态机, 同一个目标函数覆盖回合两半, 所以这里
        # 不断言 decision_kind 这种实现标签, 只断言可观察行为。
        game = Game(seed=37_500_009, ai_enabled=True)
        teacher = KillFieldTeacher(
            seed=29, ray_count=32, max_bounces=1, horizon=4, hold=2)
        teacher.last_motion_action = (2, 0, 0)
        game.destroy_tank(1)
        action = teacher.act(game)
        self.assertFalse(action["fire"])
        self.assertIsNotNone(teacher.last_scores)

    def test_post_kill_objective_is_not_flat(self):
        """P41 回归守卫: 击杀后目标函数曾塌成一张平面。

        当时 10 个候选里 9 个精确等于 OPPONENT_SELF_SCORE——incoming_risk 是
        个 0/1 阶跃, 场上没有飞行弹时对每个候选恒为 0, argmax 在平面上乱选,
        坦克 75 帧一动不动。见 docs/P41_POSTKILL_OBJECTIVE_FLATNESS.md。
        """
        for seed in (37_500_009, 37_500_002, 37_500_011):
            with self.subTest(seed=seed):
                game = Game(seed=seed, ai_enabled=True)
                teacher = KillFieldTeacher(
                    seed=29, ray_count=32, max_bounces=1, horizon=4, hold=2)
                teacher.last_motion_action = (2, 0, 0)
                game.destroy_tank(1)
                scores = teacher.scores(game)
                live = scores[np.isfinite(scores) & (scores > -1e8)]
                self.assertEqual(len(live), 10)
                ties = np.max(np.unique(
                    np.round(live, 2), return_counts=True)[1])
                # 塌陷时这里是 9。静止的三个转向候选位移与净空都相同,
                # 并列是物理事实, 所以允许并列但不允许压倒性并列。
                self.assertLessEqual(ties, 3)

    def test_post_kill_window_keeps_the_tank_moving(self):
        """塌陷的可观察后果是坦克变雕像, 这里直接量它走了多远。"""
        for seed in (37_500_009, 37_500_002, 37_500_011):
            with self.subTest(seed=seed):
                game = Game(seed=seed, ai_enabled=True)
                teacher = KillFieldTeacher(
                    seed=29, ray_count=32, max_bounces=1, horizon=4, hold=2)
                teacher.last_motion_action = (2, 0, 0)
                game.destroy_tank(1)
                me = game.tanks[0]
                previous = (me.x, me.y)
                travelled = 0.0
                for _ in range(75):
                    teacher.act(game)
                    apply_action(game, teacher.last_action)
                    game.step()
                    travelled += math.hypot(
                        me.x - previous[0], me.y - previous[1])
                    previous = (me.x, me.y)
                    if not me.alive:
                        break
                # 塌陷时三个种子分别只走了 0.34 / 0.45 / 0.97 格。
                self.assertGreater(travelled / game.scale, 2.0)

    def test_original_round_end_timing_is_preserved(self):
        game = Game(seed=37_500_010, ai_enabled=True)
        starting_round = game.round_number
        game.destroy_tank(1)
        for _ in range(74):
            events = game.step()
            self.assertFalse(any(event[0] == "round_end" for event in events))
        events = game.step()
        self.assertTrue(any(event[0] == "round_end" for event in events))
        self.assertEqual(game.scores, [1, 0])
        self.assertEqual(game.round_number, starting_round)
        frames = 75
        while game.round_number == starting_round:
            game.step()
            frames += 1
        self.assertEqual(frames, 129)

    def test_teacher_observes_commanded_motion_with_zero_pose_change(self):
        game = Game(seed=37_500_007, ai_enabled=True)
        teacher = KillFieldTeacher(
            seed=23, ray_count=32, max_bounces=1, horizon=4, hold=2)
        teacher._observe_action_effect(game)
        teacher._emit_action(game, (2, 1, 0), "plan")
        game.frame += 1
        teacher._observe_action_effect(game)
        self.assertTrue(teacher.failed_translation)
        self.assertFalse(teacher.failed_turn)
        self.assertTrue(teacher.action_no_effect)
        self.assertEqual(teacher.no_effect_frames, 1)

    def test_all_macro_action_scores_are_finite(self):
        game = Game(seed=37_500_002, ai_enabled=True)
        teacher = KillFieldTeacher(
            seed=17, ray_count=96, max_bounces=1, horizon=6, hold=2)
        scores = teacher.scores(game)
        self.assertEqual(scores.shape, (18,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertIsInstance(teacher.field, DensityField)

    def test_new_round_invalidates_map_cache(self):
        game = Game(seed=37_500_003, ai_enabled=True)
        teacher = KillFieldTeacher(
            seed=19, ray_count=64, max_bounces=1, horizon=4, hold=2)
        first = teacher._ensure_field(game)
        first_round = teacher.round_number
        game.setup_battle()
        second = teacher._ensure_field(game)
        self.assertNotEqual(first_round, teacher.round_number)
        self.assertIsNot(first, second)
        self.assertEqual(second.counts.shape,
                         (len(game.maze), len(game.maze[0])))

    def test_temporal_guard_blocks_chasing_a_live_own_bullet(self):
        game = Game(seed=38_700_100, ai_enabled=False)
        me = game.tanks[0]
        me.rotation = 90.0
        bullet = Bullet(game, "own_probe", me, game.scale)
        bullet.x, bullet.y = me.x + 15.0, me.y
        bullet.x_speed = (C.BULLETSPEED / C.BULLETHITCHECKINTERVALS
                          * (game.scale / 50.0))
        bullet.y_speed = 0.0
        bullet.just_created = False
        # 引擎已修复"直线追自己刚出膛的子弹会自杀"的 bug：子弹在离开
        # 发射者判定框之前对发射者无害。本用例要测的是"子弹已在飞行中，
        # 再直线追上去"，所以显式标记它已经出过框。
        bullet.has_exited_owner = True
        game.bullets = [bullet]
        me.bullets_fired = 1

        self.assertTrue(action_self_hits(game, (2, 1, 0)))
        self.assertFalse(action_self_hits(game, (1, 1, 0)))
        teacher = KillFieldTeacher(
            seed=1, ray_count=16, max_bounces=1, horizon=2, hold=1)
        teacher._emit_action(game, (2, 1, 0), "test")
        self.assertNotEqual(teacher.last_action, (2, 1, 0))
        self.assertEqual(teacher.own_bullet_guard_events, 1)


class KillFieldStudentTests(unittest.TestCase):
    def test_full_student_action_facts_are_finite_and_914_wide(self):
        game = Game(seed=37_500_008, ai_enabled=True)
        state = KillFieldFeatureState(
            ray_count=32, max_bounces=1, max_flight_frames=12)
        field = state.ensure_field(game)
        preview = action_preview_features(game, field, state.chain, horizon=4)
        self.assertEqual(P37_FULL_OBS_DIM, 914)
        self.assertEqual(preview.shape, (ACTION_PREVIEW_DIM,))
        self.assertTrue(np.isfinite(preview).all())

    def test_student_tail_replaces_coin_tail_without_changing_width(self):
        game = Game(seed=37_500_004, ai_enabled=True)
        state = KillFieldFeatureState(
            ray_count=96, max_bounces=1, max_flight_frames=30)
        features = state.features(game)
        self.assertEqual(P37_OBS_DIM, 801)
        self.assertEqual(KILLFIELD_EXTRA_DIM, 134)
        self.assertEqual(features.shape, (KILLFIELD_EXTRA_DIM,))
        self.assertTrue(np.isfinite(features).all())

        field = state.field
        for item in game.reachable:
            x, y = int(item["x"]), int(item["y"])
            self.assertAlmostEqual(
                float(features[y * 12 + x]), field.guidance_at((x, y)),
                places=6)
        self.assertTrue(np.all(features[:KILLFIELD_MAP_DIM] >= 0.0))
        self.assertTrue(np.all(features[:KILLFIELD_MAP_DIM] <= 1.0))

    def test_laika_cell_change_does_not_award_student_chain(self):
        game = Game(seed=37_500_005, ai_enabled=True)
        state = KillFieldFeatureState(
            ray_count=64, max_bounces=1, max_flight_frames=20)
        state.features(game)
        game.tanks[1].x += game.scale
        gain = state.advance(game)
        self.assertEqual(gain, 0.0)

    def test_collected_guidance_is_signed_without_losing_magnitude(self):
        game = Game(seed=37_500_006, ai_enabled=True)
        state = KillFieldFeatureState(
            ray_count=64, max_bounces=1, max_flight_frames=20)
        positive = state.features(game)[:KILLFIELD_MAP_DIM].copy()
        cells = [
            (int(item["x"]), int(item["y"])) for item in game.reachable
            if state.field.guidance_at(
                (int(item["x"]), int(item["y"]))) > 0.0
        ]
        cell = cells[0]
        state.chain.collected.add((state.field.target_cell, cell))
        signed = state.features(game)[:KILLFIELD_MAP_DIM]
        index = cell[1] * 12 + cell[0]
        self.assertAlmostEqual(float(signed[index]), -float(positive[index]))


if __name__ == "__main__":
    unittest.main()
