[中文](README.zh-CN.md)

# Tank Trouble AI

AI for the classic Flash game **Tank Trouble**, playing the game's own AI, Laika: a real-time MPC search, and a reinforcement-learning policy distilled from an MPC teacher.

Live demo: [https://tank-trouble-ai.pages.dev/](https://tank-trouble-ai.pages.dev/)

## Results

| Agent | Method | True win rate vs Laika | Decision time (p95) |
|---|---|---:|---:|
| **League PPO** (in the demo) | PPO self-play + safety shield, distilled from an MPC teacher | **95.5%** | 3.3 ms |
| Tactical | Real-time MPC search | 91.0–91.7% | 9.2 ms |

Pre-registered tests on never-used seeds: 10,000 rounds for League PPO, three sets of 2,000 for Tactical. A kill followed by death to a bullet still in flight is a double KO, not a win; a round undecided after 30 s is a draw.

## How it works

- **Engine**: Python and JavaScript ports of the decompiled Flash game, following its original frame order, checked against each other frame by frame.
- **Tactical**: every frame, rolls candidate actions 36 frames forward in a sandbox, vetoes moves and shots that die, and chases when stalled.
- **League PPO**: trained in PyTorch against Laika, Tactical and its own past versions, on visible facts only, with a safety shield over its actions. An MPC teacher that plays out its top actions is then distilled back into the network by expert iteration: 1.6 points more wins at the same per-frame cost.

Details: [method and results](docs/REPORT.md) · [engine](docs/ENGINE.md)

## Repository

| Path | Contents |
|---|---|
| `tank_trouble_original/` | Python port of the game and Laika |
| `web/` | Browser arena; `web/lib/engine/` JS engine; `web/rl/` RL environment, agents and evaluator |
| `training/js_league_ppo.py` | PPO training, evaluation, distillation and export |
| `swf_decompiled/` | Decompiled Flash source |

Tech stack: Python, PyTorch, JavaScript, TypeScript, React, Web Worker.

## Running locally

- Python game: `python3 play_tank_trouble.py`
- Browser arena: `cd web && npm install && npm run dev`
