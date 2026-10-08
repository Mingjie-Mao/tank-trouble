[English](README.md)

# Tank Trouble AI

基于经典 Flash 游戏 **Tank Trouble** 的实时对战 AI 项目，探索强化学习（PPO）、模型预测控制（MPC）、策略蒸馏与安全规划在动态对抗环境中的应用。

在线演示：[https://tank-trouble-ai.pages.dev/](https://tank-trouble-ai.pages.dev/)

## 核心实现

- **游戏引擎**：基于原版 Flash 实现 Python 和 JavaScript 游戏引擎，复现坦克运动、碰撞、子弹反弹与迷宫生成，并通过逐帧对拍验证一致性。
- **强化学习**：基于 PyTorch 实现 PPO / Actor-Critic、GAE 和联赛自博弈训练，结合动作条件化的存活预测与安全约束，提高策略稳定性。
- **规划与蒸馏**：实现基于精确物理模拟的 MPC 搜索，结合行为克隆、专家迭代和动作评分蒸馏，探索将在线规划能力迁移至神经网络。
- **实时对战**：实现浏览器端 AI 对战系统，支持人机对战、AI 自博弈、实时性能监控及自动化评测。

## 实验结果

在当前自研 JavaScript 引擎的 2,000 局固定种子测试中：

| 方法 | 对 Laika 真胜率 |
|---|---:|
| League PPO + 安全护盾 | **94.3%** |
| Tactical v2（MPC 搜索） | 92.2% |

历史 Tactical Smooth 在旧引擎的独立留出测试中取得 **95.6%** 真胜率；由于后续更换物理引擎，该结果不代表当前版本表现。

详细实验见 [League PPO](https://github.com/Mingjie-Mao/tank-trouble/blob/main/docs/LEAGUE_PPO_2026-10.md) 和 [引擎验证报告](https://github.com/Mingjie-Mao/tank-trouble/blob/main/docs/OWN_ENGINE_2026-10.md)。

## 技术栈

Python、PyTorch、PPO、GAE、MPC、JavaScript、TypeScript、React、Web Worker

## 本地运行

**Python 游戏：** `python3 play_tank_trouble.py`

**浏览器对战：**

1. `cd web`
2. `npm install`
3. `npm run dev`
