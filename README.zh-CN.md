[English](README.md)

# Tank Trouble AI

经典 Flash 游戏 **Tank Trouble** 的 AI，对原版 AI Laika 的真胜率 98.9%：在我们自己设计的观测上用强化学习训练策略网络，由 MPC 搜索当老师指导训练，对战时再用同一个搜索实时辅助决策；搜索里的 Laika 是根据它看得见的动作同步跟踪的副本。全部由 Rust 编译成 WebAssembly，在浏览器里运行。

在线演示：[https://tank-trouble-ai.pages.dev/](https://tank-trouble-ai.pages.dev/)

## 结果

| 智能体 | 对 Laika 真胜率 | 决策耗时（浏览器） |
|---|---:|---|
| **网络 + 4×4 实时搜索，跟踪 Laika**（网页演示，8 核及以上） | **98.92%** | 0.3 ms；危险帧 p50 20 ms，最长 32 ms |
| 网络 + 2×2 实时搜索，跟踪 Laika（网页演示，核数更少时） | 98.53% | 0.3 ms；危险帧 23 ms |
| 网络 + 实时搜索，Laika 从零重建 | 97.04% | 0.3 ms；危险帧 p50 25 ms |
| 仅网络（230 维观测） | 96.67% | 0.3 ms |
| 上一版网络（1030 维观测） | 95.94% | — |

在留出种子上预注册、按种子配对测 1 万局：跟踪 Laika 提高 1.49 个百分点，95% 置信区间 [+1.18, +1.80]；更宽的搜索再提高 0.39 [+0.09, +0.69]；比上一版网络高 2.98 [+2.54, +3.42]。先击杀、随后被仍在飞行的子弹打死算双亡，不算胜；30 秒未分胜负算平局。与公开项目 KillField 的规划器正面对战 1000 局：583 胜 201 负。

## 怎么做的

- **引擎**：把反编译的 Flash 游戏按原版帧顺序移植成 Python 和 Rust，两者逐帧对拍。
- **观测**：230 个玩家能从屏幕上读出的事实——周围的墙、双方坦克、每颗子弹的弹道预测、Laika 自己的瞄准模型、每个移动方向的生存前瞻、通往对手的路线、每一枪离对手最近能擦过多远。
- **训练**：PyTorch 联赛 PPO，对手是 Laika 和自己的历史版本；环境用 Rust 实现，约 1 万步/秒；动作外加安全护盾。
- **老师与搜索**：危险时把网络最想做的几个动作放进公平沙盒各推演 24 帧；打分既用来训练网络，也在对战中以 30 ms 预算实时挑选动作——浏览器里是 4 个候选 × 4 次推演，分给最多 8 个各持一份同步棋局副本的 Web Worker 并行计算。
- **看懂 Laika**：Laika 的目标和排队中的动作看不见，但它的每个动作看得见。16 个 Laika 副本逐帧重放，只留下动作和真 Laika 一致的；搜索就和这些副本对推，而不是从零重建的 Laika。

详细：[方法与结果](docs/REPORT.md) · [引擎](docs/ENGINE.md)

## 仓库结构

| 路径 | 内容 |
|---|---|
| `engine/core/` | Rust 引擎、Laika、观测、搜索与对战智能体 |
| `engine/py/`、`training/ppo.py` | Python 绑定；PPO 训练、老师标注、评测与导出 |
| `engine/wasm/`、`web/` | 浏览器对战场（Web Worker 里的 WebAssembly，React 界面） |
| `tank_trouble_original/` | 游戏与 Laika 的 Python 移植 |
| `swf_decompiled/` | 反编译的 Flash 源码 |

技术栈：Rust、WebAssembly、Python、PyTorch、TypeScript、React。

## 本地运行

- 浏览器对战场：`cd web && npm install && npm run dev`
- Python 版游戏：`python3 play_tank_trouble.py`
