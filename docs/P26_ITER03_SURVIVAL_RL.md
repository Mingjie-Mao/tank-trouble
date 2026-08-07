# P26：iter03 生存模式 RL 热启动

日期：2026-07-14

## 目的

从 `training/models/p24r530_iter03.pt` 原样继承 530 维观测、18 个联合动作和
三层 1024 宽 actor，只新增价值头，用 PPO 直接优化学生自己闭环到达状态下的
P24v2 生存经济。实验不修改 P24/P25 文件，也不替换 P25v2 原版冠军。

## 实现

- 脚本：`training/survival_rl_warmstart.py`
- 环境：Laika 无敌、我方一发死亡、使用 `legacy_econ()`，不启用失败的 v3
  空弹夹税或命中免疫。
- 动作：`Discrete(18)`，顺序严格复用 `training.mpc_agent.CANDIDATES`，每个
  动作保持 2 帧。
- 观测：408 维原版物理事实 + 122 维账本，共 530 维。
- 热启动检查：新 actor 与 iter03 对随机输入逐元素最大误差为 `0.0`。
- 价值预热：第一轮只更新新价值头，共享 actor 主干保持逐元素不变。
- PPO：`gamma=1.0`，避免折扣鼓励“先刷池、后送死”；低温 `0.05` 将评分头
  转为可探索策略，贪心动作仍与 iter03 相同。
- 每轮保存 actor 快照；导出的 actor 保持 Replica-530 旧加载器兼容。

## 奖励恒等式

每步先奖励池变化 `delta_pool / 50`，终局再修正：

- 死亡：减去当前全池并额外 `-4`，整局未折扣回报恒为 `-6`。
- 流干：额外 `-2`，整局未折扣回报恒为 `-4`。
- 撑满 30 秒：额外 `+2`，整局未折扣回报等于 `final_pool / 50`。

因此局内覆盖、接近或命中不能购买死亡；只有活到结算才保留财富。

## 结果

固定生存种子 `26020000`，每组 40 局：

| 模型 | 决策步 | 死亡 | 流干 | 撑满 | 秒/命中 | 卡墙 | 风格/秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| iter03 | 0 | 40.0% | 60.0% | 0.0% | 17.88 | 19.12% | -0.796 |
| P26 early stop | 8,192 | 30.0% | 70.0% | 0.0% | 21.58 | 13.88% | -0.583 |
| P26 rejected | 24,576 | 25.0% | 72.5% | 2.5% | 26.40 | 27.77% | -1.251 |

8,192 步模型在另一组 100 局原版验收中为 57.0% 真胜率、62.0% 先杀率、
12.0% 双亡；iter03 的历史 400 局原版真胜率为 53.8%。两者不是同规模配对
对照，只能说明短训没有立即摧毁原版能力，不能证明它超过 iter03。

## 判决

本轮证明了 iter03 可以无损热启动到 on-policy RL，且 PPO 确实改变了闭环行为；
但当前训练没有收敛到“持续命中并撑满”。8k 时主要学到降低死亡和卡墙，命中
反而变慢；继续到 24k 后卡墙与风格显著退化。故 24k 模型拒收，恢复 8k 作为
仅供观察的早停候选，P25v2 仍是部署冠军。

后续若继续 P26，应把“撑满率/命中间隔/卡墙”做成定期验证门，而不是按最后一
轮保存；还需要提高成功轨迹密度或采用课程化初始池/时长，不能仅增加训练步数。

## 文件与命令

- 早停候选：`training/models/p26_survival_rl_actor.pt`
- 完整断点：`training/models/p26_survival_rl_checkpoint.pt`
- 24k 拒收 actor：`training/models/p26_survival_rl_actor_iter12_rejected.pt`
- 24k 拒收断点：`training/models/p26_survival_rl_checkpoint_iter12_rejected.pt`

```bash
# 生存回放
python3 training/watch.py --policy p24r530-student-survival \
  --net training/models/p26_survival_rl_actor.pt --seed 26020000

# 原版回放
python3 training/watch.py --policy p24r530-student-original \
  --net training/models/p26_survival_rl_actor.pt --seed 27020000

# 固定验收
python3 training/survival_rl_warmstart.py eval \
  --actor training/models/p26_survival_rl_actor.pt \
  --survival-n 40 --original-n 100 --seed 26020000

# 从 iter03 重新训练
python3 training/survival_rl_warmstart.py train --updates 4

# 从完整断点续训
python3 training/survival_rl_warmstart.py train --resume --updates 4
```
