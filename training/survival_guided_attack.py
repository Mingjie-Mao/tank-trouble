"""P31: lightweight explicit opportunity-planning teacher.

P30 exposed action-conditioned opportunity facts but expected PPO to infer
their action alignment from sparse rewards.  P31 instead runs the already
validated 360-degree opportunity MPC at a shorter 24-frame horizon.  It
explicitly searches movement, aiming, firing, terminal outcomes, and suicide
paths, producing the guidance trajectories that the next student should
distil.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.opportunity_teacher_v2 import OpportunityMPC360
from training.survival_distill_v2 import legacy_econ


FPS = 25
GUIDE_HORIZON = 24
GUIDE_HOLD = 8


class GuidedAttackPolicy(OpportunityMPC360):
    name = "P31 轻量机会 MPC 导航与瞄准老师"

    def __init__(self, horizon=GUIDE_HORIZON, hold=GUIDE_HOLD, seed=0):
        super().__init__(seed=seed, horizon=horizon, hold=hold)
        self.econ = dict(legacy_econ(), cap=12 * FPS, start=80.0)


def probe_command(args):
    from training.evaluate import play_round_dual_engine

    policy = GuidedAttackPolicy(
        horizon=args.horizon, hold=args.hold, seed=args.planner_seed)
    result = play_round_dual_engine(policy, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["probe"])
    parser.add_argument("--seed", type=int, default=30_000_001)
    parser.add_argument("--planner-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=GUIDE_HORIZON)
    parser.add_argument("--hold", type=int, default=GUIDE_HOLD)
    args = parser.parse_args()
    probe_command(args)


if __name__ == "__main__":
    main()
