"""Small deployment wrappers shared by evaluation and the web runtime."""

from __future__ import annotations


class ActionRepeatPolicy:
    """Run a policy once, then hold its controls for ``interval`` frames.

    Tank Trouble physics still advances at 25 FPS. Only policy inference is
    reduced, matching the fluid P27b browser deployment rather than frame-skip
    training semantics.
    """

    def __init__(self, policy, interval=2):
        if int(interval) < 1:
            raise ValueError("interval must be at least one")
        self.policy = policy
        self.interval = int(interval)
        self.name = f"{getattr(policy, 'name', 'policy')}_repeat{self.interval}"
        self.reset()

    def reset(self):
        self.policy.reset()
        self.frames = 0
        self.decisions = 0
        self.held = None

    def act(self, game):
        if self.held is None or self.frames % self.interval == 0:
            self.held = dict(self.policy.act(game))
            self.decisions += 1
        self.frames += 1
        return dict(self.held)

    def __getattr__(self, name):
        return getattr(self.policy, name)

