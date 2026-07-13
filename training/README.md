# 训练目录索引

训练入口保留为顶层脚本，便于直接运行，也与现有实验日志中的命令保持一致。二进制模型、
采样数据和 Python 缓存均为本地产物，不进入 Git。

## 当前主线

| 阶段 | 文件 | 用途 |
| --- | --- | --- |
| P22 基线 | `score_distill.py`、`expert_iter.py` | 408 维评分网络与专家迭代 |
| P24 生存课程 | `survival_mode.py`、`survival_distill.py` | 生存计分环境、蒸馏和原版验收 |
| P25v1 机会课程 | `opportunity_distill.py` | 正面有限炮线、射击位势能与先导蒸馏 |
| P25v2 机会老师 | `opportunity_teacher_v2.py` | 360°、75 帧、最多两次反弹的可信炮线 |
| P25v2 蒸馏 | `opportunity_distill_v2.py` | 独立数据、开火门控、P22 引导与多轮 DAgger |

## 目录职责

- `analysis/`：离线行为与击杀模式分析。
- `scripts/`：可复用的长任务启动脚本。
- `logs/`：实验过程日志；关键结论仍写入 `EXPERIMENTS.md`。
- `models/`：本地模型产物，`.pt`、`.zip` 不进入 Git。
- `*_data/`：本地采样分片，不进入 Git；不同观测语义的数据必须隔离。

## 记录与恢复

- `EXPERIMENTS.md`：实验台账、指标与负结果。
- `../docs/HANDOFF_COMPLETE_CONTEXT.md`：跨会话完整上下文与恢复顺序。

Mac 长训练统一使用 `caffeinate -i`，允许屏幕关闭但阻止系统因空闲休眠。
