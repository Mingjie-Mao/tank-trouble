import time
import types
import unittest

from training.search_profile import (
    CHILDREN,
    PRIMITIVES,
    Profiler,
    format_stages,
)


class Dummy:
    def outer(self):
        self.inner()
        self.inner()
        return "outer"

    def inner(self):
        time.sleep(0.001)
        return "inner"

    def boom(self):
        raise ValueError("boom")


class ProfilerTest(unittest.TestCase):
    def test_counts_and_times_calls(self):
        obj = Dummy()
        with Profiler() as profiler:
            profiler.wrap(Dummy, "inner", "inner")
            obj.inner()
            obj.inner()
            obj.inner()
        self.assertEqual(profiler.stats["inner"]["calls"], 3)
        self.assertGreater(profiler.stats["inner"]["seconds"], 0.0)

    def test_return_value_is_preserved(self):
        obj = Dummy()
        with Profiler() as profiler:
            profiler.wrap(Dummy, "inner", "inner")
            self.assertEqual(obj.inner(), "inner")

    def test_restore_removes_the_wrapper(self):
        original = Dummy.inner
        profiler = Profiler()
        profiler.wrap(Dummy, "inner", "inner")
        self.assertIsNot(Dummy.inner, original)
        profiler.restore()
        self.assertIs(Dummy.inner, original)

    def test_restore_is_idempotent(self):
        original = Dummy.inner
        profiler = Profiler()
        profiler.wrap(Dummy, "inner", "inner")
        profiler.restore()
        profiler.restore()
        self.assertIs(Dummy.inner, original)

    def test_exception_still_records_and_propagates(self):
        obj = Dummy()
        with Profiler() as profiler:
            profiler.wrap(Dummy, "boom", "boom")
            with self.assertRaises(ValueError):
                obj.boom()
        self.assertEqual(profiler.stats["boom"]["calls"], 1)

    def test_context_manager_restores_on_exception(self):
        original = Dummy.inner
        try:
            with Profiler() as profiler:
                profiler.wrap(Dummy, "inner", "inner")
                raise RuntimeError("x")
        except RuntimeError:
            pass
        self.assertIs(Dummy.inner, original)

    def test_module_level_function_wrapping(self):
        module = types.ModuleType("fake")
        module.work = lambda value: value * 2
        with Profiler() as profiler:
            profiler.wrap(module, "work", "work")
            self.assertEqual(module.work(3), 6)
        self.assertEqual(profiler.stats["work"]["calls"], 1)

    def test_nested_labels_are_inclusive(self):
        obj = Dummy()
        with Profiler() as profiler:
            profiler.wrap(Dummy, "inner", "inner")
            profiler.wrap(Dummy, "outer", "outer")
            obj.outer()
            stats = dict(profiler.stats)
        # outer contains two inner calls, so its inclusive time is larger.
        self.assertGreaterEqual(stats["outer"]["seconds"],
                                stats["inner"]["seconds"])
        self.assertEqual(stats["inner"]["calls"], 2)


class ReportTest(unittest.TestCase):
    def test_exclusive_subtracts_declared_children(self):
        profiler = Profiler()
        profiler.stats["full_search"]["calls"] = 2
        profiler.stats["full_search"]["seconds"] = 1.0
        profiler.stats["search"]["calls"] = 2
        profiler.stats["search"]["seconds"] = 0.6
        profiler.stats["long_tail_fire"]["calls"] = 1
        profiler.stats["long_tail_fire"]["seconds"] = 0.3
        rows = profiler.report(total_frames=100)
        self.assertAlmostEqual(rows["full_search"]["inclusive_seconds"], 1.0)
        self.assertAlmostEqual(rows["full_search"]["exclusive_seconds"], 0.1)

    def test_exclusive_never_goes_negative(self):
        profiler = Profiler()
        profiler.stats["full_search"]["calls"] = 1
        profiler.stats["full_search"]["seconds"] = 0.1
        profiler.stats["search"]["calls"] = 1
        profiler.stats["search"]["seconds"] = 0.5
        rows = profiler.report(total_frames=10)
        self.assertEqual(rows["full_search"]["exclusive_seconds"], 0.0)

    def test_missing_children_are_ignored(self):
        profiler = Profiler()
        profiler.stats["full_search"]["calls"] = 1
        profiler.stats["full_search"]["seconds"] = 1.0
        rows = profiler.report(total_frames=10)
        self.assertAlmostEqual(rows["full_search"]["exclusive_seconds"], 1.0)

    def test_per_frame_and_per_call_rates(self):
        profiler = Profiler()
        profiler.stats["search"]["calls"] = 4
        profiler.stats["search"]["seconds"] = 2.0
        rows = profiler.report(total_frames=100)
        self.assertAlmostEqual(rows["search"]["ms_per_call"], 500.0)
        self.assertAlmostEqual(rows["search"]["ms_per_frame"], 20.0)
        self.assertAlmostEqual(rows["search"]["calls_per_frame"], 0.04)

    def test_zero_calls_does_not_divide_by_zero(self):
        profiler = Profiler()
        profiler.stats["search"]["calls"] = 0
        profiler.stats["search"]["seconds"] = 0.0
        rows = profiler.report(total_frames=0)
        self.assertEqual(rows["search"]["ms_per_call"], 0.0)
        self.assertEqual(rows["search"]["ms_per_frame"], 0.0)

    def test_children_map_is_a_tree_without_cycles(self):
        for parent, children in CHILDREN.items():
            self.assertNotIn(parent, children)
        # every child appears under at most one parent
        seen = []
        for children in CHILDREN.values():
            seen.extend(children)
        self.assertEqual(len(seen), len(set(seen)))

    def test_primitives_are_never_subtracted_from_a_parent(self):
        # They are called from several parents; subtracting them anywhere
        # would double-count against the others.
        for children in CHILDREN.values():
            for primitive in PRIMITIVES:
                self.assertNotIn(primitive, children)

    def test_primitive_time_is_reported_standalone(self):
        profiler = Profiler()
        profiler.stats["exact_root_search"]["calls"] = 1
        profiler.stats["exact_root_search"]["seconds"] = 1.0
        profiler.stats["clone_exact_game"]["calls"] = 12
        profiler.stats["clone_exact_game"]["seconds"] = 0.3
        rows = profiler.report(total_frames=10)
        self.assertAlmostEqual(
            rows["exact_root_search"]["exclusive_seconds"], 1.0)
        self.assertAlmostEqual(
            rows["clone_exact_game"]["inclusive_seconds"], 0.3)

    def test_format_stages_renders(self):
        payload = {
            "stages": {"search": {
                "calls": 2, "inclusive_seconds": 1.0,
                "exclusive_seconds": 1.0, "ms_per_call": 500.0,
                "calls_per_frame": 0.1, "ms_per_frame": 50.0}},
            "total_frames": 20, "wall_seconds": 1.5, "ms_per_frame": 75.0,
            "frame_budget_ms_at_60fps": 16.67,
            "total_full_searches": 2, "seconds_per_full_search": 0.5,
            "mean_search_frame_rate": 0.1,
        }
        text = format_stages(payload)
        self.assertIn("search", text)
        self.assertIn("60fps budget", text)


if __name__ == "__main__":
    unittest.main()
