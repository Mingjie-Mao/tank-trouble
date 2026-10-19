[English](README.md)

# Tank Trouble AI

为经典 Flash 游戏 **Tank Trouble** 做的 AI，对手是游戏原版 AI Laika：一个是实时 MPC 搜索，一个是经 MPC 老师蒸馏的强化学习策略。

在线演示：[https://tank-trouble-ai.pages.dev/](https://tank-trouble-ai.pages.dev/)

## 结果

| AI | 方法 | 对 Laika 真胜率 | 决策耗时（p95） |
|---|---|---:|---:|
| **League PPO**（网页演示） | PPO 自博弈 + 安全护盾，经 MPC 老师蒸馏 | **95.5%** | 3.3 ms |
| Tactical | 实时 MPC 搜索 | 91.0–91.7% | 9.2 ms |

在从未使用过的种子上做预注册测试：League PPO 1 万局，Tactical 三组各 2,000 局。先击杀、随后被仍在飞行的子弹打死算双亡，不算胜；30 秒未分胜负算平局。

## 怎么做的

- **引擎**：基于反编译的原版 Flash 实现 Python 和 JavaScript 两版引擎，遵循原版的帧执行顺序，两者逐帧对拍一致。
- **Tactical**：每帧把候选动作在沙盒里前推 36 帧，否决会被打死的移动和开火，僵持时拓扑追击。
- **League PPO**：用 PyTorch 训练，对手为 Laika、Tactical 和自己的历史版本；只观测可见信息，并在动作上加安全护盾。再用 MPC 老师（前瞻模拟策略的候选动作）通过专家迭代蒸馏回网络：胜率提高 1.6 个百分点，每帧耗时不变。

详见：[方法与结果](docs/REPORT.md) · [引擎](docs/ENGINE.md)

## 目录

| 路径 | 内容 |
|---|---|
| `tank_trouble_original/` | 游戏和 Laika 的 Python 移植 |
| `web/` | 浏览器对战；`web/lib/engine/` 为 JS 引擎；`web/rl/` 为 RL 环境、智能体与评测 |
| `training/js_league_ppo.py` | PPO 训练、评测、蒸馏与导出 |
| `swf_decompiled/` | 反编译的 Flash 源码 |

技术栈：Python、PyTorch、JavaScript、TypeScript、React、Web Worker。

## 本地运行

- Python 游戏：`python3 play_tank_trouble.py`
- 浏览器对战：`cd web && npm install && npm run dev`
