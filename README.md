[中文](README.zh-CN.md)

# Tank Trouble AI

An AI for the classic Flash game **Tank Trouble** that beats the game's own AI, Laika, 98.9% of the time: a reinforcement-learning policy on an observation of our own design, taught by an MPC search and backed by the same search in real time, which plays against copies of Laika kept in step with what it visibly does. Everything runs in the browser, compiled from Rust to WebAssembly.

Live demo: [https://tank-trouble-ai.pages.dev/](https://tank-trouble-ai.pages.dev/)

## Results

| Agent | True win rate vs Laika | Decision time (browser) |
|---|---:|---|
| **Network + 4×4 real-time search, tracking Laika** (in the demo, 8+ cores) | **98.92%** | 0.3 ms; danger frames p50 20 ms, max 32 ms |
| Network + 2×2 real-time search, tracking Laika (in the demo, fewer cores) | 98.53% | 0.3 ms; danger frames 23 ms |
| Network + real-time search against a fresh Laika | 97.04% | 0.3 ms; danger frames p50 25 ms |
| Network alone (230-value observation) | 96.67% | 0.3 ms |
| Previous network (1030-value observation) | 95.94% | — |

10,000 rounds on held-out seeds, pre-registered and paired by seed: tracking Laika adds +1.49 points, 95% CI [+1.18, +1.80]; the wider search +0.39 [+0.09, +0.69] more; +2.98 [+2.54, +3.42] over the previous network. A kill followed by death to a bullet still in flight is a double KO, not a win; a round undecided after 30 s is a draw. Head to head against KillField's planner, a public Tank Trouble AI: 583 wins to 201 over 1,000 rounds.

## How it works

- **Engine**: Python and Rust ports of the decompiled Flash game in its original frame order, checked against each other frame by frame.
- **Observation**: 230 facts a player could read off the screen — local walls, both tanks, every bullet's forecast, Laika's own aiming model, a survival lookahead for each move, the route to the opponent and how near each shot passes.
- **Training**: league PPO in PyTorch against Laika and the policy's own past versions, with Rust environments at ~10,000 steps/s and a safety shield over the actions.
- **Teacher and search**: in danger, the policy's top actions are played out 24 frames in a fair sandbox; the scores train the network and, with a 30 ms budget, pick its moves in play — in the browser 4 candidates × 4 rollouts shared among up to eight Web Workers, each holding a copy of the game in step.
- **Reading Laika**: Laika's goal and queued actions are hidden, but its moves are not. Sixteen copies of Laika replay every frame and only those that moved as the real one did survive; the search plays them instead of a Laika rebuilt from scratch.

Details: [method and results](docs/REPORT.md) · [engine](docs/ENGINE.md)

## Repository

| Path | Contents |
|---|---|
| `engine/core/` | Rust engine, Laika, observation, search and the playing agent |
| `engine/py/`, `training/ppo.py` | Python bindings; PPO training, teacher labels, evaluation and export |
| `engine/wasm/`, `web/` | The browser arena (WebAssembly in a Web Worker, React UI) |
| `tank_trouble_original/` | Python port of the game and Laika |
| `swf_decompiled/` | Decompiled Flash source |

Tech stack: Rust, WebAssembly, Python, PyTorch, TypeScript, React.

## Running locally

- Browser arena: `cd web && npm install && npm run dev`
- Python game: `python3 play_tank_trouble.py`
