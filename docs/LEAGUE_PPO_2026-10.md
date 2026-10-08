# League PPO — a pure network that matches, and with a search shield beats, Tactical

Checkpoint: 2026-10-09. Code: `web/rl/` (environment, browser agent, hybrids,
evaluator) and `training/js_league_ppo.py` (trainer, evaluator, exporter).

> **Engine note.** Everything from *Result* to *Latency* below was trained and
> graded on the browser engine the arena used at the time, a vendored JS
> snapshot. The repository has since moved to its own engine,
> [`web/lib/engine/`](OWN_ENGINE_2026-10.md), translated from the Python port and
> checked against it frame by frame. The two differ in one rule — when a tank's
> own bullet can hit it — so every number is being re-measured there; the first
> re-baseline is in [*On this repository's engine*](#on-this-repositorys-engine).

## Result

On a pre-registered holdout of 2000 never-used seeds, under one protocol for
every agent (one round per seed, 30 s with both tanks alive is a draw, only
surviving the settlement window is a win):

| Agent | Wins | Losses | Double-KOs | Draws | True win rate |
|---|---:|---:|---:|---:|---:|
| **League PPO + deep shield (h36)** | **1884** | **41** | **23** | 52 | **94.2%** ±1.0 |
| League PPO + move/fire shield | 1876 | 69 | 29 | 26 | 93.8% ±1.1 |
| Tactical v2 (search champion, same physics) | 1837 | 59 | 31 | 73 | 91.8% ±1.2 |

```
Primary (pre-registered)   deep shield - Tactical v2   +2.35pp  95% CI [+0.77, +3.93]  McNemar p = 0.0043
Secondary                  deep shield - move/fire     +0.40pp  95% CI [-1.03, +1.83]  p = 0.63
Exploratory                move/fire   - Tactical v2   +1.95pp  95% CI [+0.40, +3.50]  p = 0.016
```

The primary rule — superiority if the paired CI's lower bound is above zero —
was written down, with SHA-256 hashes of the model and every code file, before
any holdout round was played. It passed. The network was trained for 30M
environment steps; neither shield requires any training.

Tactical reads **91.8–92.5%** here, against the **96.0%** published for it. The
difference is protocol, not regression: the browser arena has no 30 s draw rule,
and under it 66–73 of Tactical's 2000 rounds are still undecided at 30 s
(section *Comparing on equal terms*).

## Design

### Environment

`web/rl/duel-env.js` runs on the browser arena's engine with the original
collision model (no wall sliding). One episode is one round on a fresh
maze; the seed fixes maze, spawns and headings.

**Terminal-only reward.** Nothing is paid before the round's authoritative end
(`round_end`, emitted after the 75-frame settlement window), so a kill followed
by death to a bullet already in flight is scored as the double-KO it is.

| Outcome | Reward |
|---|---|
| Win within 10 s | +1.0, decaying logarithmically to +0.5 at 30 s |
| Double-KO | −0.1 |
| Loss | −1.0 |
| Both alive at 30 s (draw) | −1.0 |
| Every terminal, additionally | up to +0.25 for changing action no more often than 13% of frames |

The ordering is deliberate: a stall is worth exactly a loss, so passivity is
never a result; the slowest win still beats the best-played trade
(0.5 > −0.1 + 0.25); and the smoothness bonus is a *rate*, so dying early cannot
buy it. 13% is the measured action-change rate of competent controllers (Laika
and the MPC planner), not a tuned constant.

**Observation: 1028 values of visible facts, no decision answers.** The policy
gets anything a player could work out from the screen and its own inputs, and
nothing else — never the seed, the RNG, the opponent's buttons or goal stack.

| Block | Size | Content |
|---|---:|---|
| Maze grid | 840 | 12×10 padded cells × {exists, 4 walls, me here, opponent here} |
| Wall rays | 16 | distance to the first wall around the hull, ≤ 4 cells |
| Self / opponent | 12 + 12 | pose, frame-differenced velocity and turn rate, ammo, readiness, alive, wall contact — opponent in my frame |
| Navigation | 10 | BFS path length, the next step's direction in my frame, straight-line bearing, dead-end depth for both |
| Aim, mine and theirs | 5 + 5 | the **current** barrel's shot only: hit / self-hit / nothing, time to hit, closest pass |
| Bullets | 10 × 10 | every live bullet in my frame: position, velocity, owner, whether it can hit its owner yet, lifetime, and a wall-exact 75-frame forecast of whether it reaches me or them and when |
| Threat summary | 3 + 1 | most urgent incoming bullet for each side, worst pass, count of bullets on a course to me |
| Phase and clock | 4 | fighting / settling / frozen, elapsed fraction of 30 s |
| Own recent actions | 3 + 6 + 1 + 1 | last three actions, the round's action-change rate so far, idle streak |
| **`dodge_safety`** | **9** | for each of the nine no-fire moves, roll the world 24 frames holding it, opponent holding its buttons: survives → closest bullet clearance; dies → how long it lasted |

`dodge_safety` is the single most important channel: an action-conditioned
lookahead presented as a fact rather than a decision.

**Action.** Discrete(18) = throttle {back, none, forward} × turn {left, none,
right} × fire, chosen every engine frame (25 Hz).

### Network

| Branch | Input | Layers | Output |
|---|---|---|---|
| Map | 840 | MLP 256 → 128 | 128 |
| Bullets | 10 rows × 10 | shared row MLP 64 → 64, masked mean ‖ max, → 64 | 64 |
| Scalars | 88 | Linear 128, tanh, Linear 128 | 128 |
| Trunk | 320 | Linear 256, ReLU | actor 18, critic 1 |

Bullet rows are pooled because slot order has no meaning. The map branch is an
MLP rather than a convolution: on this machine a 3×3 conv over a 12×10 grid cost
~7× an MLP of the same width on MPS (296 vs 39 ms per 4096-row forward+backward),
and the grid is small enough that position-specific weights lose little.

### Opponent league

Each new episode samples its opponent: **10% Laika, 10% Tactical, 80% frozen
ancestors** of the learner. A frozen generation is published every 1,048,576
steps into a ring of eight; a ring, not a queue, so an ancestor keeps its slot
index while episodes against it are still running. Tactical in the pool uses the
L1 opponent model (Laika's script would be a strong and wrong prior about a
learned opponent) and a 12 ms evasion budget so its rare long searches cannot
stall a worker. Before the first ancestor exists, its share goes to Laika.

### Optimisation

PPO, 288 environments × 128 steps per update, 4 epochs × 8 minibatches, clip
0.2, GAE λ 0.95, **γ 0.999** — the horizon must cover the longest round
(750 + 125 frames) because the only reward arrives at its end. Learning rate
3e-4 and entropy 0.01 both decay linearly to zero over a fixed 30M-step horizon
that a resume continues rather than restarts. The first 20 updates train only
the critic. Every terminal, including the 30 s draw, cuts the bootstrap: the
draw is a real outcome with its own reward, not a truncation.

### Throughput engineering

The environment had to sustain thousands of steps per second with a 24-frame
lookahead in every observation. Each speed-up below is exact — verified
bit-for-bit against the definition it replaces.

| Change | Effect | Verification |
|---|---|---|
| Wall lookups through a dense integer grid instead of string-keyed buckets | 3.5× environment speed (wall tests were 40% of a step) | 3.98M random points, 0 differences |
| `dodge_safety` decomposed: the world is rolled once with my tank removed, then each of the nine moves simulates only its own hull against the recorded hit points | 1.7× on the observation | 43,396 states × 9 values, 0 differences against nine full rollouts |
| Asynchronous collection — each worker is served as soon as it replies, and fast workers may collect more of the update's budget | Lockstep was gated by the slowest of 4 performance + 6 efficiency cores | GAE computed per worker over its own trajectory |
| Rollout inference on CPU, one thread, without `Categorical` overhead | 1.6 → 0.35 ms per 32-row call | — |
| Updates on CPU, 8 threads, while workers are idle | MPS was slower under other GPU tenants (263 vs 96 ms per minibatch) | — |

Sustained throughput: 2.5–5.7k steps/s; 30M steps in roughly five hours.

## Training curve

Graded on fixed seeds from 970000, sampled actions:

| Steps | Generation | True win rate vs Laika |
|---:|---|---:|
| 4.2M | gen003 | 28.0% (200 rounds) |
| 9.4M | gen008 | 56.5% (400) |
| 16.8M | gen015 | 79.0% (400) |
| **30.0M** | **final** | **89.7%** (2000, two seed bases: 89.5% / 89.9%) |

The previous best model-free result in this project was 36.4%.

Sampling beats argmax. At 9.4M steps the greedy policy scored 45.3% against
56.5% sampled on the same 400 seeds: a deterministic policy stalls into 30 s
draws (62 vs 28). Every number here uses sampled actions.

Self-play stalemates appeared and resolved on their own. Between ~4M and ~22M
steps, 20–34% of rounds against frozen ancestors were draws — two increasingly
cautious copies avoiding each other. Pricing the draw as a loss eventually
broke it (12% by 30M), but it consumed much of the 80% self-play share.

## The safety shield

The observation already contains the nine `dodge_safety` values and the current
barrel's self-hit forecast. The shield turns them into a hard rule over the
policy's distribution: drop every action whose movement is forecast to die
when some movement survives (otherwise keep only the longest-lasting one), and
drop fire while the current shot would hit the shooter. Then sample.

| Checkpoint | No shield | Shield | Paired effect (1000 seeds) |
|---|---:|---:|---|
| gen015 (16.8M) | 77.5% | 85.9% | +8.4pp, CI [+5.5, +11.3], p = 1.5e-8 |
| final, seeds 970000 | 89.5% | 93.3% | +3.8pp, p = 3e-4 |
| final, seeds 990000 | 89.9% | 94.8% | +4.9pp, p = 1e-6 |

The network does not exploit information it can see as well as a two-line rule
does, and the gap narrows with training (8.4 → ~4.3pp). The shield costs nothing
at inference: the values are already computed for the observation.

## Comparing on equal terms

`web/rl/eval-agent.mjs` grades any browser policy under the protocol above, on
the same maze and spawns per seed as the trainer's evaluator. Seeds
970000–970999 and 990000–990999:

| Agent | Wins | Losses | Double-KOs | Draws | True win rate |
|---|---:|---:|---:|---:|---:|
| Tactical v2 | 1850 | 53 | 31 | **66** | 92.5% |
| Tactical Smooth (its own sliding physics) | 1840 | 89 | 23 | 48 | 92.0% |
| League PPO + shield | 1866 | 68 | 29 | 37 | 93.3% |

PPO + shield vs Tactical v2: +0.8pp, CI [−0.8, +2.4], p = 0.35 — a tie. The JS
agent's 93.3% also agrees with the Python evaluator's 94.0% on these seeds,
which checks the browser path end to end.

### The two agents fail differently

| Cause of a non-win | PPO + shield | Tactical v2 |
|---|---:|---:|
| Shot by Laika (loss, or died first in a double-KO) | **82** | 65 |
| Undecided at 30 s | 37 | **66** |
| Own ricochet | 6 | 10 |
| Killed Laika, then hit by its bullet in flight | 8 | 6 |
| Other | 1 | 3 |

PPO's self-kills are nearly gone; its residual weakness is **new** shots from
Laika, which `dodge_safety` cannot see — it holds the opponent's buttons, so it
only models bullets already in flight. Tactical's weakness is stalling.

## Search hybrids

Two ways to combine the network with search (`web/rl/ppo-hybrid.js`):

- **Deep shield — PPO decides, search vetoes.** The sampled action is held for
  *h* frames in a sandbox where Laika runs its real script (L2), so its future
  aim and fire are modelled. If that dies, the most likely of the policy's top
  four alternatives that survives is played instead.
- **PPO-pruned Tactical — Tactical decides.** The policy's distribution, folded
  onto Tactical's ten action classes, replaces the learned prior in front of
  Tactical's rollouts; Tactical keeps its top three plus invariants.

Seeds 970000 + 990000:

| Agent | True win rate | vs Tactical v2 | vs PPO + shield |
|---|---:|---|---|
| Deep shield, h24 | 93.5% | +1.0pp, p = 0.23 | +0.2pp, p = 0.84 |
| **Deep shield, h36** | **95.0%** | **+2.5pp, CI [+1.0, +3.9], p = 0.0015** | +1.7pp, p = 0.027 |
| PPO-pruned Tactical, K3 | 92.0% | −0.4pp, p = 0.63 | −1.2pp, p = 0.13 |

Letting Tactical decide inherits its stalling (91 draws). Letting the network
decide and search only veto keeps the network's tempo and removes deaths a
24-frame horizon cannot see. h36 was selected after comparing it with h24 on
these seeds, which is why it was re-tested on a pre-registered holdout before
any claim was made — the result at the top of this report.

On the holdout, the deep shield's advantage **over the plain shield did not
replicate** (+0.40pp, CI spans zero): it converted losses into draws (69 → 41
losses, 26 → 52 draws). Its advantage over Tactical did.

## Latency

Per-round p95 decision time, Node, measured while a training run shared the
machine: PPO + shield 19.9–21.0 ms, deep shield 21.1–26.2 ms, Tactical v2
25.3–29.8 ms; the bare network forward pass is ~1.1 ms. These are Node figures
and comparable only to each other. The browser costs 2.4–2.9× Node for the same
policy ([calibration](BROWSER_LATENCY_CALIBRATION_2026-08-28.md)); a browser
measurement of the PPO agents is the next step.

## On this repository's engine

The same 2000 seeds (970000 + 990000), same protocol, on `web/lib/engine/`:

| Agent | Wins | Losses | Double-KOs | Draws | True win rate |
|---|---:|---:|---:|---:|---:|
| Tactical v2 | 1845 | 51 | 25 | 79 | 92.2% ±1.2 |
| League PPO r1 + move/fire shield | 1887 | 63 | 32 | 18 | 94.3% ±1.0 |

```
PPO + shield - Tactical v2   147 PPO-only wins / 105 Tactical-only wins   +2.1pp
```

The policy here is the one trained on the previous engine, run unchanged: the
ordering between the two agents carries over (+2.1pp against +0.8pp before),
and Tactical's draws remain its main failure. Retraining on this engine is
deferred until one open fidelity question is settled — the per-frame update
order of tanks and bullets in the original, which decides how a tank's own
bullet near the muzzle should be treated ([details](OWN_ENGINE_2026-10.md#open-question-update-order)).

## Limitations

- One training seed. The curve and the final model come from a single run.
- Laika only. Hunter, Dodger and Random have not been graded under this protocol.
- The deep shield, like Tactical, models Laika with its own script (L2). Against
  an unknown opponent that model is wrong; the plain shield does not depend on it.
- The holdout confirms superiority over Tactical v2 under the 30 s draw rule. In
  the arena, which has no draw rule, Tactical's undecided rounds play on.

## Reproduction

```bash
# train (≈5 h on a 10-core M4)
python3 -u training/js_league_ppo.py train --run r1 --workers 9 --total-steps 30000000

# grade a checkpoint (Python), with or without the shield
python3 -u training/js_league_ppo.py eval training/runs/js_league/r1/latest.pt \
    --games 1000 --seed 970000 --sample --shield move+fire

# export to the browser, then grade any browser policy under the same protocol
python3 training/js_league_ppo.py export training/runs/js_league/r1/latest.pt
cd web && node rl/eval-agent.mjs ppo-deep-shield-h36 4000000 1000
```

## Acknowledgements

The duel observation and reward in `web/rl/duel-env.js`, and the planner under
`web/lib/killfield-runtime/` that Tactical builds on, are ported from
[Cichlider/killfield](https://github.com/Cichlider/killfield) (MIT); its
license and notice are retained. The experiments above ran on that project's JS
engine snapshot, which this repository has since replaced with its own.
