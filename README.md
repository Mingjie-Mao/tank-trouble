[![English](https://img.shields.io/badge/ENGLISH-4285F4?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-555555?style=for-the-badge)](README.zh-CN.md)

# Tank Trouble AI

A faithful Python port of the original Flash **Tank Trouble**, plus a deployable
browser AI that combines learning, planning, safety verification, and rigorous
evaluation.

## Play the current champion

**[Tactical vs Laika — open the browser arena](https://tank-trouble-ai.pages.dev/)**

No login or installation is required. The arena supports:

- Tactical vs Laika at the original 25 FPS;
- human vs AI and AI vs AI;
- deterministic maze seeds and score reset;
- live search-rate and p95 latency telemetry;
- the frozen four-opponent promotion report.

The deployed champion is **Tactical Smooth**. It is primarily a browser-native
online search agent, not the historical P27b neural policy. It evaluates legal
actions with the exact game model, then adds visible-bullet safety, post-kill
survival, topology pursuit, and physics-verified moving-target interception.
Before a crowded firing window it verifies that the predicted kill still has a
survivable settlement path and suppresses unsafe shots.

Smooth adds two things to the frozen Tactical champion: **wall-contact physics**,
which removes almost all wall grinding, and **fire continuation**, which scores a
shot together with the movement that follows it instead of assuming the tank
then stands still. Both were adopted from upstream KillField after an ablation
established what each one actually contributes — see
[the ablation report](docs/KILLFIELD_UPSTREAM_ABLATION_2026-08-16.md).

## Current results

| Agent / benchmark | True result | Decision cost | Role |
|---|---:|---:|---|
| **Tactical Smooth vs Laika** | **1912/2000 = 95.6%** | **0.66 ms p50 / 20.05 ms p95** | Deployed champion, fresh holdout |
| Tactical Smooth four-opponent pool | 302/320 = 94.4% | both colors, paired | Laika, Hunter, Dodger, Random |
| Tactical (frozen predecessor) vs Laika | 576/600 = 96.0% | 0.75 ms p50 / 12.57 ms p95 | Previous champion, same protocol |
| KillField vs Laika | 262/300 = 87.3% | 5.92 ms reference p95 | Third-party baseline, 2026-08-12 snapshot |
| KillField four-opponent sample | 76/104 = 73.1% | same seeded protocol | Third-party generalisation baseline |
| P27b | 1330/1500 = 88.7% | 8.6 ms | Historical pure-network baseline |
| Exact-state shielded MPC | 297/300 = 99.0% | seconds | Privileged offline teacher, not deployable |

The KillField rows describe the **vendored 2026-08-12 snapshot** on this
repository's seeds. Upstream has since moved on and reports 897/1000 = 89.7% for
its current agent; that figure was independently reproduced here, exactly, as
part of the ablation. The two numbers come from different seed bases and
different physics, so neither is an effect size for the other.

Smooth was promoted on a **pre-registered non-inferiority test**, not on a
better point estimate. The margin, the endpoint and the decision table were
written down before the holdout ran, on a seed base that had never informed any
tuning decision:

| | Frozen Tactical | Tactical Smooth |
|---|---:|---:|
| True wins (2000 paired rounds) | 1914 (95.70%) | 1912 (95.60%) |
| Double-KOs | 29 | **22** |
| Wall-contact frames | 21.35% | **3.00%** |
| Commanded moves with zero motion | 12.52% | **0.12%** |
| Direction changes per second | 7.23 | **3.60** |
| p50 / p95 | 0.75 / 12.57 ms | 0.66 / 20.05 ms |

```
Difference   -0.10pp    95% CI [-1.34pp, +1.14pp]
McNemar      81 harmed / 79 saved, p = 0.937
Pre-registered margin -1.5pp -> lower bound clears it -> non-inferior
```

So Smooth obtains the movement quality at no measurable cost in win rate: a
two-round difference across 2000 rounds, with harmed and saved rounds almost
perfectly balanced. "Not significant" is not reported as "equivalent" — the
margin was fixed in advance precisely so that the claim has a defined meaning.

The earlier pooled figure for the frozen predecessor remains published for
comparison, and the same discipline produced it:

| Blind seed base | True wins | Losses | Double-KOs | Draws | Win rate |
|---|---:|---:|---:|---:|---:|
| `2900000` | 291 | 5 | 4 | 0 | 97.0% |
| `3200000` | 285 | 10 | 5 | 0 | 95.0% |
| **Pooled** | **576** | **15** | **9** | **0** | **96.0%** |

The two-point spread between those bases is the honest measure of how much a
single 300-game base can move, which is why promotion requires a second base and
why the pooled figure is published rather than the best one. At 25 FPS the frame
budget is 40 ms, so 20.05 ms remains comfortably real-time.

All public scores use **true wins**. Killing Laika first and then dying to a live
bullet is a double-KO, not a win.

## Technical path

The project did not jump directly to search. It established where learning works,
where it fails, and which information planning contributes.

| Stage | Best result | What was learned |
|---|---:|---|
| Random / scripted hunter | 0.5% / 22.5% | Reference difficulty |
| P17 model-free line (BC + PPO) | 36.4% | Bullet prediction and navigation mattered more than reward shaping |
| Leak-free MPC | 96.0% | Short exact planning is exceptionally strong in this game |
| P21–P27 search distillation | up to 88.7% | Regress all action scores; argmax cloning suffers from tied labels |
| P27b pure network | 88.7% | Fast learned baseline, but below fair online planning |
| Tactical | 96.0% | Search + explicit safety + failure-driven anti-evasion is the product path |
| **Tactical Smooth** | **95.6%** | Movement quality and policy strength are separate axes; physics fixed the first at no cost to the second |
| Privileged exact teacher | 99.0% | Useful label oracle, but reads hidden RNG/opponent state |

### Tactical decision stack

```text
visible game state
      ↓
inverse-density attack field
      ↓
10 legal first actions × exact 36-frame rollout
      ↓
visible-bullet and own-bullet safety verification
      ↓
sparse pre-fire settlement audit for crowded bullet states
      ↓
sparse two-stage correction / post-kill survival
      ↓
topology anti-stall pursuit + moving-target ricochet interception
      ↓
action at the original 25 FPS
```

The public champion contains no learned gate. A first MLP action-prior candidate
covered 99.17% of teacher actions in offline Top-7 validation, but it still searched
8.74/10 actions on live games and did not reduce p95 latency. It was rejected rather
than silently replacing Tactical.

## Self-play and model improvement

The web page's **AI vs AI** mode runs two independent agents in the same exact
physics world. The right agent receives a mirrored visible-state view. This is a
live demonstration and diagnostic tool; playing in the browser does **not** train
or mutate the deployed champion.

Improvement happens offline through a guarded flywheel:

```text
frozen Tactical vs candidate, identical seeds and both colors
      ↓
Laika / Hunter / Dodger / Random league
      ↓
save every recovery, regression, double-KO, draw, and hard seed
      ↓
collect search-teacher corrections and train the next candidate
      ↓
gate on win rate, paired regressions, double-KO, draw, color gap, and p95
      ↓
300 fresh Laika seeds → promote or reject
```

The browser displays the last frozen league report. The offline flywheel generates
machine-readable comparison and hard-case files. No candidate is public until it
passes every gate.

A resumable unattended flywheel is scheduled every day at 02:00 local time. It
starts with the permanent regression seeds, stops failed candidates early, and
only unlocks the 300-game blind test after every paired pool gate passes.

```bash
cd web

# Compare any hidden candidate against frozen Tactical on identical cases.
npm run flywheel:selfplay -- \
  --candidate p27-js-two-stage \
  --rounds-per-side 10 --seed 2400000
```

## Quickstart

The current champion is the fixed browser deployment above. The original Python
port requires only Python 3 (`tkinter` provides the local window):

```bash
python3 play_tank_trouble.py
```

Historical Python baselines remain available:

```bash
# Pure-network baseline, not the current browser champion
python3 training/watch.py --policy p27b

# Privileged slow teacher
python3 training/watch.py --policy exact

# RL-line champion
python3 training/watch.py --policy model
```

Verify the port against the decompiled original:

```bash
python3 test_original_port.py
```

Run the Python test suite (`unittest`; pytest is not a dependency):

```bash
python3 -m unittest discover -s training -p "test_*.py" -t .
```

Run the browser product locally:

```bash
cd web
npm install
npm run dev
```

## Evaluation rules

1. **Kill first is not a win.** Settlement continues while live bullets can kill
   their owner. Reports separate wins, losses, double-KOs, and draws.
2. **Candidates use paired seeds and both colors.** A changed result is attributed
   to the policy, not a different maze or spawn side.
3. **Generalisation precedes promotion.** Candidates face Laika, Hunter, Dodger,
   and Random before the final Laika blind run.
4. **Latency is a promotion gate.** Strength alone is insufficient; p95 must remain
   below the 40 ms frame budget.
5. **The champion is frozen.** Training and self-play create candidates, never
   in-place changes to the deployed policy.

## What did not work

| Attempt | Result | Lesson |
|---|---|---|
| Eight reward-shaping variants after P8 | No promotion | Missing information was the bottleneck |
| CNN map head | 22.5% | Four explicit navigation features worked better |
| 60M-step PPO continuation | 36% → 21% | Constant-LR fine-tuning eventually collapses |
| Argmax behavior cloning | 8.9% | Many actions tie; clone the score landscape instead |
| Learned value leaf | teacher 93.5% → 87.5% | On-policy value targets reinforced passivity |
| Longer/deeper search alone | no reliable gain | Tactical exchanges are short and discontinuous |
| First learned gate | 0% safe coverage | Do not skip physics without evidence |
| First learned Top-7 prior | no p95 improvement | Offline action accuracy is not product speed |

## Repository layout

```text
play_tank_trouble.py       Local Python game
test_original_port.py      Fidelity checks against the decompiled Flash source
tank_trouble_original/     Faithful game port
swf_decompiled/            Decompiled ActionScript source of truth
training/                  RL, MPC, distillation, evaluation, and historical P26–P41
docs/                      Technical reports and archived experiment decisions
web/                       Browser-native Tactical product
  app/                     Large-screen arena and promotion monitor
  lib/                     Exact JS physics, Tactical, opponents, self-play flywheel
  scripts/                 League, blind evaluation, teacher collection, promotion
  tests/                   Parity, safety, regression, worker, and league tests
```

The P26–P41 exploration line is preserved under `docs/`. Vendored KillField-derived
runtime code retains its own license and notice in `web/lib/killfield-runtime/`.

## Next steps

1. Use the paired self-play flywheel to grow a permanent hard-case regression set.
2. Improve search scheduling or candidate generation only when live p95 decreases.
3. Add lightweight opponent modelling without assuming every opponent is Laika.
4. Distil only demonstrably beneficial search corrections back into a network.
5. Promote a new Tactical only after the four-opponent gates and 300-game blind run.
