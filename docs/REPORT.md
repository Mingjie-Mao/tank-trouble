# Tank Trouble AI — method and results

Two agents play the original game's AI, Laika, on this repository's engine
([ENGINE.md](ENGINE.md)): **Tactical**, a real-time MPC search, and **League
PPO**, a learned policy distilled from an MPC teacher. League PPO is the one the
web demo runs.

## Results

| Agent | True win rate vs Laika | Decision time (p50 / p95) |
|---|---:|---:|
| **League PPO** (deployed) | **95.5%** (10,000 rounds) | 2.3 / 3.3 ms |
| Tactical | 91.0–91.7% (three 2,000-round sets) | 1.6 / 9.2 ms |

**Protocol.** One round per seed, the agent on tank 0 and Laika on tank 1; a
round both tanks survive for 30 s is a draw; only surviving the settlement
window is a win, so a kill followed by death to a bullet still in flight is a
double KO, not a win. Every comparison is paired by seed on never-used seeds and
was pre-registered — arms, seeds, decision rule and SHA-256 of the model and
code fixed before the first round. Latencies are Node measurements on an idle
machine.

| Pre-registered test | Seeds | Result |
|---|---|---|
| PPO before distillation − Tactical | three sets of 2,000 | +2.85, +2.10, +3.65pp (each p < 0.02) |
| MPC teacher − PPO before distillation | 2,000 | +1.80pp, CI [+0.49, +3.11], p = 0.009 |
| Distilled PPO − PPO before distillation | 10,000 | **+1.59pp, CI [+1.02, +2.16], p = 5e-8** |

A single 2,000-round evaluation moves by about ±1pp (actions are sampled), so
effects near 1pp need ~10,000 paired rounds: a 2,000-round test of the
distilled model gave +0.65pp, not significant, before the 10,000-round test
confirmed it.

## Tactical: real-time MPC search

Every frame, each candidate action is rolled forward 36 frames in a sandbox
copy of the game, with Laika re-run by its own script from a fresh state (no
hidden state is read). Candidates are scored by progress toward positions from
which a shot connects (an inverse ricochet density field), shots the engine's
ballistics verifies as hits, and exposure. A bullet-safety search vetoes moves
that die, a settlement audit vetoes shots that would end in a double KO, a
topological chase breaks stalls, and the chosen move is held for a few frames.
Its main failure is stalling: 59–73 of 2,000 rounds end undecided at 30 s.

## League PPO

**Environment** (`web/rl/duel-env.js`). One round per episode on a fresh maze.
Reward only at the end: win +1 (decaying to +0.5 at 30 s), double KO −0.1,
loss or draw −1, plus up to +0.25 for not changing action more often than 13%
of frames.

**Observation: 1030 visible facts.** Maze grid; wall rays; both tanks' pose,
motion, ammo and readiness; BFS navigation; whether each barrel's current shot
hits; every bullet with a wall-exact 75-frame forecast; threat summary; round
phase; own recent actions; `dodge_safety` — for each of nine moves, whether
holding it for 24 frames survives; and how many frames each side must turn
before it holds a hit. Never the seed, the random stream, or the opponent's
buttons or plan.

**Network.** Map MLP (840 → 256 → 128), per-bullet MLP pooled by mean and max
(→ 64), scalar MLP (90 → 128), trunk 256, actor 18 actions (3 throttle × 3
turn × fire), critic 1. The browser runs the same forward pass in plain JS,
matched to PyTorch within 1e-5.

**Training** (`training/js_league_ppo.py`). PPO, 288 environments × 128 steps,
γ 0.999, λ 0.95, clip 0.2. Opponents: 10% Laika, 10% Tactical, 80% frozen
ancestors (a ring of eight). 30M steps from scratch, continued to 100M without
gain, then 20M at learning rate 1e-4 (+1.8pp). About 2,500–3,000 steps/s on a
10-core M4 with nine Node workers.

**Safety shield.** At inference, drop every move `dodge_safety` says dies
(when one survives) and drop fire while the current shot would hit the shooter.
Free to compute and worth +3.4pp (91.5% → 94.85% on development seeds).

## MPC teacher and distillation

**Why.** Replaying each failed round from k frames before the failure and
resampling the policy: from 32 frames (1.3 s) earlier, 97% of double KOs and
79% of losses can be turned into wins, and simply resampling the policy
wins about half of them; from 8 frames earlier almost none can. The policy plays well but
loses coin flips at decisive moments.

**Teacher** (`web/rl/ppo-mpc.js`). In danger — a barrel lines up, a bullet is
on course, or a move is forecast to die (about one frame in seven) — each of
the policy's top actions is played out 24 frames by the shielded policy itself
in the same fair sandbox, and the best is taken. +1.8pp over the policy, but
0.2–0.6 s per decision: too slow for the 40 ms frame.

**Distillation** (`web/rl/teacher-labels.mjs`, `distil` in the trainer). The
policy plays 2,000 rounds; on its 163,086 danger frames the teacher scores its
top actions (4 candidates × 4 rollouts). The target is the policy's own
distribution tilted toward the teacher, π*(a) ∝ π(a)·exp(16·score(a)); three
epochs of KL(π* ‖ π). Same network and cost, +1.59pp. A control with the
teacher's scores shuffled among candidates gains 1.8pp less on development
seeds, so most of the gain is the teacher's choice, not just sharper
sampling. A second round on the distilled policy's own states added nothing:
the student has caught up with this teacher.

## What did not help

| Idea | Result |
|---|---|
| Training past 30M steps | 93.60% → 93.05% at 100M |
| Pricing the double KO at −0.6 instead of −0.1 | no difference |
| Deep shield (veto the sampled action if a sandbox rollout dies) | no gain over the plain shield |
| Handing face-offs to Tactical | fewer double KOs, more losses, no net gain |
| First-strike observation (matched 3-seed ablation) | steadier fine-tuning, no gain over the policy it started from |
| A second distillation round | no gain |

## Reproduction

```bash
# train (r3 from scratch, later continued to 100M; r6 fine-tunes it)
python3 -u training/js_league_ppo.py train --run r3 --workers 9 --total-steps 30000000
python3 -u training/js_league_ppo.py train --run r6 --workers 9 --total-steps 20000000 \
    --lr 0.0001 --entropy-coef 0.003 --critic-warmup-updates 10 \
    --init-from training/runs/js_league/r3/latest.pt

# teacher labels, distillation, export to the browser
cd web && node rl/teacher-labels.mjs ../training/runs/js_league/distill/labels_r1 80000000 2000 9
cd .. && python3 -u training/js_league_ppo.py distil training/runs/js_league/r6/latest.pt \
    --data training/runs/js_league/distill/labels_r1 --run distill/b16 --beta 16
python3 training/js_league_ppo.py export training/runs/js_league/distill/b16/latest.pt

# grade any agent against Laika under the protocol above
cd web && node rl/eval-agent.mjs ppo-league 9000000 5000
cd web && node rl/eval-agent.mjs p27-js-tactical-v2 9000000 5000
```

## Acknowledgements

The duel observation and reward build on
[Cichlider/killfield](https://github.com/Cichlider/killfield) (MIT), as does the
planner under `web/lib/killfield-runtime/` that Tactical and the MPC sandbox
use; its license and notice are retained.
