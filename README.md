[中文](README.zh-CN.md)

# Tank Trouble AI

A real-time battle AI built on the classic Flash game **Tank Trouble**, exploring reinforcement learning (PPO), model predictive control (MPC), policy distillation and safe planning in a dynamic adversarial environment.

Live demo: [https://tank-trouble-ai.pages.dev/](https://tank-trouble-ai.pages.dev/)

## Core implementation

- **Game engine**: Python and JavaScript game engines built from the original Flash game, reproducing tank movement, collisions, bullet ricochets and maze generation, with consistency verified by frame-by-frame parity tests.
- **Reinforcement learning**: PPO / Actor-Critic, GAE and league self-play training in PyTorch, combined with action-conditioned survival prediction and safety constraints for a more stable policy.
- **Planning and distillation**: MPC search over exact physics simulation, combined with behaviour cloning, expert iteration and action-score distillation, exploring how to transfer online planning ability into a neural network.
- **Real-time battles**: a browser-based AI battle system with human-vs-AI, AI self-play, live performance monitoring and automated evaluation.

## Results

On a 2,000-round fixed-seed evaluation on the current in-house JavaScript engine:

| Method | True win rate vs Laika |
|---|---:|
| League PPO + safety shield | **94.3%** |
| Tactical v2 (MPC search) | 92.2% |

The earlier Tactical Smooth reached a **95.6%** true win rate on an independent held-out set on the previous engine; since the physics engine has since been replaced, that result does not represent the current version.

See [League PPO](https://github.com/Mingjie-Mao/tank-trouble/blob/main/docs/LEAGUE_PPO_2026-10.md) and the [engine validation report](https://github.com/Mingjie-Mao/tank-trouble/blob/main/docs/OWN_ENGINE_2026-10.md) for details.

## Tech stack

Python, PyTorch, PPO, GAE, MPC, JavaScript, TypeScript, React, Web Worker

## Running locally

**Python game:** `python3 play_tank_trouble.py`

**Browser arena:**

1. `cd web`
2. `npm install`
3. `npm run dev`
