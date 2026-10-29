# Tank Trouble AI — method and results

One AI plays the original game's AI, Laika, on a faithful port of the game
([ENGINE.md](ENGINE.md)): a policy network trained by league PPO on an
observation of our own design, taught by an MPC search, and backed in play by
the same search within a 30 ms budget. The web demo runs all of it in the
browser, compiled from Rust to WebAssembly, the search shared among Web
Workers. Against Laika the search plays copies of it kept in step with what
it visibly does.

## Results

| Agent | Observation | True win rate vs Laika | Decision time in the browser |
|---|---:|---:|---|
| **Network + 4×4 real-time search against tracked Laika** (in the demo, 8+ cores) | 230 | **98.92%** | 0.3 ms; on danger frames p50 20 ms, max 32 ms |
| Network + 2×2 real-time search against tracked Laika (in the demo, fewer cores) | 230 | 98.53% | 0.3 ms; on danger frames p50 23 ms, max 33 ms |
| Network + real-time search against a fresh Laika | 230 | 97.04% | 0.3 ms; on danger frames p50 25 ms, max 33 ms |
| Network alone | 230 | 96.67% | 0.3 ms |
| Previous network (1030-value observation adapted from a public project) | 1030 | 95.94% | — |

10,000 rounds on never-used seeds (14,000,000 – 14,009,999), one round per
seed, actions sampled, shield on. A kill followed by death to a bullet still
in flight is a double KO, not a win; a round undecided after 30 s is a draw.
Decision times: WebAssembly in Web Workers on an M4 (4×4 with seven helper
workers); natively 1.5× faster. The evaluations play every search in full;
in the browser the 30 ms budget cuts 14–35% of 4×4 searches short across
eight workers (from run to run on the same idle machine), and 41–69% of 2×2
searches on one.

| Pre-registered test (paired by seed) | Rounds | Difference, 95% CI |
|---|---:|---|
| Network + search − previous network | 10,000 | **+1.10pp [+0.59, +1.61]** |
| Network − previous network | 10,000 | +0.73pp [+0.21, +1.25] |
| Network + search − network | 10,000 | +0.37pp [−0.11, +0.85] |
| Long-trained network (before the teacher) − previous network | 10,000 | +0.66pp [+0.15, +1.17] |
| Search against tracked Laika − against a fresh Laika (second registration, same seeds) | 10,000 | **+1.49pp [+1.18, +1.80]** |
| 4×4 − 2×2 search, both against tracked Laika (third registration, same seeds) | 10,000 | **+0.39pp [+0.09, +0.69]** |

**Against KillField's planner** (no Laika to track; the search runs as against a fresh Laika) (a public Tank Trouble AI; our earlier agent
built on it beat Laika 91% of the time on this engine), 1,000 rounds, 500 from each seat, both
engines run in lockstep: **583 wins, 201 losses**, 30 double KOs, 186 draws —
74.4% of decided rounds [71.3%, 77.4%].

A 2,000-round evaluation moves by about ±1pp, and a single 30M-step training
run by much more (below), so every claim above rests on 10,000 paired rounds.

## The observation: 230 values

Facts a player could read off the screen, each with a plain meaning; never the
seed, the random stream, the opponent's buttons or Laika's plan.

| Group | Values | Contents |
|---|---:|---|
| Self | 8 | ammo, ready to fire, heading, velocity in my frame, turn rate, wall contact |
| Local walls | 62 | every wall edge of the 5×5 cells around me, my offset in my cell |
| Opponent | 13 | position and distance in my frame, maze distance, next cell toward it, line of sight, its barrel against my bearing, closing speed, alive |
| Barrels | 6 | for each barrel: would a shot now hit the opponent or the shooter, and when (the engine's own bullet substeps) |
| Laika's intent | 7 | its own aiming model says it is about to fire; frames each side must turn to hold a hit; inside Laika's 7-cell hunting and 2-cell firing range; time since its last shot; its ammo |
| Bullets | 10×10 | each live bullet, most urgent first: position, direction, owner, bounced, lifetime, forecast hit on me and when, closest pass |
| Survival | 9 | for each of the nine moves, does holding it 24 frames survive — with the shot Laika is about to take included |
| Time | 6 | round clock, settlement, last action, time since my last shot |
| Tactics | 19 | the game's own shortest route to the opponent (next waypoint and how far to turn), exits of both cells, how near each barrel's shot passes and whether turning three frames either way passes nearer, the bullets' summed threat, how long the current move has been held |

It was built by ablation. A first compact version (211 values, without the
tactics group) reached 78% where the earlier 1030-value observation reached
93% at 30M steps. Hiding one group of the 1030 values at a time showed what
was missing: without the route signals self-play drifts into mutual 30 s
draws (19 → 317 per 2,000), and without the near-miss aim signals losses
quadruple (68 → 269). The tactics group computes both, with our own engine
functions, and closes the gap.

## Training

**League PPO** (`training/ppo.py`, environments in Rust via `engine/py`):
512 rounds in parallel, 128-step rollouts, about 10,000 steps/s on a 10-core
M4. Each round's opponent is Laika (20%) or one of the learner's last eight
frozen versions (80%). The network: a per-bullet MLP pooled by mean and max,
a scalar MLP, a 256-unit trunk, 18 actions (throttle × turn × fire) and a
critic. Reward only at the end: win +1, loss or draw −1, double KO −0.5.

**Safety shield.** At inference, drop every move the survival lookahead says
dies (when one survives) and drop fire while the current shot would hit the
shooter.

**Recipe.** 30M steps from scratch; 60M more at learning rate 1e-4 and
γ 0.9997; then 20M with the teacher's loss below. Short runs are a lottery:
three seeds of the same configuration end at 92.7%, 91.8% and 79.5% at 30M,
the last stuck in self-play draws. Longer training leaves the trap: the two
continued runs reached 91.6% and 96.0% on development seeds, and the better
one was carried forward.

## The MPC teacher, in training and in play

**Why.** Replaying failed rounds from k frames before the failure: from 32
frames (1.3 s) earlier nearly every double KO and most losses can still be
won; from 8 frames, almost none. The policy loses decisions, not positions.

**Search** (`engine/core/src/teacher.rs`). In danger — a barrel holds a hit or
some move is forecast to die, about a quarter of all frames — the policy's
most likely actions and its own sampled action are each played out 24 frames
by the shielded policy in a fair sandbox: Laika rebuilt from scratch, a fresh
random stream, nothing hidden read. All candidates face the same futures.
Score: win +1, double KO −0.5, death −1.

**Teacher.** The long-trained network plays 6,000 rounds; on its 372,779
danger frames the search scores 4 candidates × 8 rollouts. The target is the
network's own distribution tilted toward the teacher, π*(a) ∝ π(a)·exp(16·score(a)).
For 20M more PPO steps, every minibatch adds KL(π* ‖ π) on teacher frames,
weighted 4× where the teacher has a preference. Against the same 20M steps
without it: +0.67pp [+0.16, +1.18] on development seeds (plain continued
PPO drifted down 0.46pp, not significant).

**In play** (`engine/core/src/agent.rs`). The same search with 2 candidates ×
2 rollouts runs on danger frames under a 30 ms budget: the network's sampled
action is scored first, the others while time remains, and a candidate
replaces it only if strictly better. Against a Laika rebuilt from scratch a
wider search (3×3) was no stronger and twice as slow; against tracked Laika
it is (below).

## Reading Laika

**Where the last 3% went.** With the search above, 97.33% on development
seeds; stronger searches (4 candidates × 8 rollouts, 32 frames) added only
+0.35pp. A diagnostic search that cheats, on the same 10,000 seeds:

| The search's Laika | Win rate | Loss | Double KO | Draw |
|---|---:|---:|---:|---:|
| Rebuilt from scratch, fresh random stream (fair) | 97.33% | 125 | 98 | 44 |
| A copy of the real one's memory, fresh random stream | 98.72% | 43 | 43 | 42 |
| A copy of its memory and its random stream | 98.71% | 44 | 44 | 41 |

Knowing Laika's memory — its goal (hunt, dodge, flee, drive to a cell), its
queue of actions (turn to an angle, fire after a delay) and its timers —
is worth +1.39pp [+1.09, +1.69]; its future coin flips are worth nothing.
Within 24 frames Laika is all but deterministic once its state is known.

**Tracking it fairly** (`engine/core/src/belief.rs`). Its memory is hidden,
but every move it makes is on screen. Sixteen copies of Laika start the round
as the real one does; each frame every copy replays the step on a copy of
the game with a random stream of its own, and the copies whose tank moved
and fired as the real one did survive and are cloned back to sixteen. They
share Laika's program — as the fair search already did — not its memory or
random stream. At least one copy holds the real state exactly on 99.95% of
frames, for 0.1 ms per frame natively, and each rollout plays one of them.
The draws are untouched: a 24-frame search cannot see a 30-second clock.

## A wider search, in parallel

Once the search's Laika is right, compute pays. On the development seeds,
against 2×2 at 98.73% (42 losses, 43 double KOs, 42 draws):

| Search against tracked Laika | Win rate | Loss | Double KO | Draw | − 2×2 |
|---|---:|---:|---:|---:|---|
| 4 candidates × 8 rollouts | 99.06% | 14 | 44 | 36 | +0.33pp [+0.06, +0.60] |
| 4 candidates × 4 rollouts | 99.07% | 18 | 44 | 31 | +0.34pp [+0.07, +0.61] |

4×4 costs half as much as 4×8 for the same gain and was registered; on the
held-out seeds 98.92% (27 losses, 32 double KOs, 49 draws) against 98.53%.
Most of the gain is losses: they fall by a third to three fifths.

Up to 20 rollouts of 2.9 ms each do not fit one thread's 30 ms. The arena
worker's agent splits a decision into `plan` (candidates and rollout seeds),
`rollouts` (any subset of the jobs) and `conclude`; the jobs are dealt
round-robin to the arena worker and up to seven helper workers, each holding
a copy of the arena that replays every command and every frame's moves and
checks a fingerprint of the state. A helper answering more than 5 ms past the
budget counts as unplayed, a candidate missing a rollout is not chosen, and a
copy that drifts drops the search to one worker and 2×2. With no deadline the
shared search plays the single-thread search move for move (tested). Making
it fit also took the rollout from 3.9 to 2.9 ms with bit-identical results:
the network's layers transposed to accumulate a row at a time, the hit test's
rotation computed once behind an exact bounding-box check, the wall grid's
buckets stored flat.

## What did not help

| Idea | Result |
|---|---|
| Comparing observations on single 30M-step runs | seed spread (79–93%) larger than the differences |
| Draw priced at −2 instead of −1 | fewer draws, more losses, same win rate (3 seeds) |
| Fine-tuning the champion 20M steps with draws at −2, then searching against tracked Laika | 98.64%, as the −1 control (98.63%) and the champion (98.73%); draws 41 vs 42 (10,000 rounds) |
| Half the rounds against Laika instead of a fifth | same mean, wider spread (3 seeds) |
| Scoring undecided rollouts by the critic | same student, labelling 2.5× slower |
| A wider real-time search (3×3) against a Laika rebuilt from scratch | no stronger than 2×2 |
| A stronger search (4×8, 32 frames) against a Laika rebuilt from scratch | +0.35pp, not significant (6,000 rounds): the search lacked knowledge of Laika, not compute |
| Against tracked Laika: 4 candidates × 1 rollout | 98.60% vs 98.67% for 2×2 (3,000 rounds) |
| Laika's tracked memory and a 24-frame forecast of it as 18 more inputs (fine-tuned 20–40M steps) | 98.37% at best, vs 98.73%: fewer losses (35 vs 42), twice the draws (84) — it learned to keep out of Laika's way, not to hunt it; with draws at −2, 98.75% (+0.02pp). True memory instead of tracked: same result, so not a tracking error |
| Hunting late in the round: from frame 375, every 4th calm frame searched too (48 frames × 1, or 24 × 2) | 98.73% and 98.67% vs 98.73%: about 33 draws turned into wins and as many wins into draws — the long rounds' results reshuffle with the random stream, no kill is found that was not there |
| A second round of expert iteration, from the improved network | no change |
| A 512-unit trunk | steadier at 30M steps, same after the full recipe; −0.62pp [−1.08, −0.16] with search |

## Reproduction

```bash
cd engine/py && maturin build --release && pip install ../target/wheels/tank_rl-*.whl && cd ../..
# train (see the recipe in training/ppo.py's docstring), then
python3 training/ppo.py export training/runs/rust/champion/latest.pt --out engine/models/champion.policy
python3 training/ppo.py eval training/runs/rust/champion/latest.pt --seeds 970000,990000
# the Laika-memory diagnostic (fair | state | full)
(cd engine && cargo run --release --example oracle -- models/champion.policy 13000000 10000 state 2 2)
# the browser build
cd web && npm run build:wasm
```
