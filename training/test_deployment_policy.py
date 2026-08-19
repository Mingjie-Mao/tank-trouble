import unittest

from training.deployment_policy import ActionRepeatPolicy


class CountingPolicy:
    name = "counting"

    def reset(self):
        self.calls = 0

    def act(self, _game):
        self.calls += 1
        return {"forward": self.calls % 2 == 1, "fire": self.calls == 2}


class ActionRepeatPolicyTest(unittest.TestCase):
    def test_holds_each_action_for_interval(self):
        base = CountingPolicy()
        policy = ActionRepeatPolicy(base, interval=2)

        outputs = [policy.act(None) for _ in range(5)]

        self.assertEqual(base.calls, 3)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[2], outputs[3])
        self.assertNotEqual(outputs[1], outputs[2])
        self.assertEqual(policy.decisions, 3)

    def test_reset_clears_held_action(self):
        base = CountingPolicy()
        policy = ActionRepeatPolicy(base, interval=3)
        policy.act(None)
        policy.act(None)
        policy.reset()

        result = policy.act(None)

        self.assertEqual(base.calls, 1)
        self.assertTrue(result["forward"])

    def test_rejects_invalid_interval(self):
        with self.assertRaises(ValueError):
            ActionRepeatPolicy(CountingPolicy(), interval=0)


if __name__ == "__main__":
    unittest.main()

