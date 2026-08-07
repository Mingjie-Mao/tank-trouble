# P28：连续前沿势能人工 Review

日期：2026-07-23

P27 录像暴露出训练/部署错位：随机采样策略偶尔移动并撑满，但确定性 argmax
大多静止；BFS 奖励只在跨格时变化，逐帧存活奖励还给静止提供了即时正反馈；
成功 BC 又复制了成功局中的全部静止动作。

P28 保留 P27 的 656 维完全可观测状态，并做三项修改：

- 沿 BFS 下一路点计算连续距离，每帧奖励距离势能差，往返净额为零；
- 逐帧存活奖励从 `+0.002` 改为 `0`；
- 成功回放只保留首次格、正向推进、命中事件及其前 8 个决策。

每轮额外跑一局固定种子的确定性 argmax 探针，直接暴露录像策略，而不是用随机
训练 batch 代替部署行为。

8,192 决策结果：

| 快照 | 随机 batch 撑满 | 确定性终局 | 确定性访问格 | 命中 |
|---|---:|---|---:|---:|
| iter01 | 13.3% | cap | 2 | 1 |
| iter02 | 18.8% | cap | 2 | 1 |
| iter03 | 11.1% | cap | 2 | 1 |
| iter04 | 16.7% | drain | 5 | 0 |

`iter04` 证明连续移动梯度已经改变 argmax：固定局访问 5 格，不再全程停住；但它
没有把跑图接回命中，最终在 11.96 秒流干，并有 43 帧卡墙。当前结论是“移动子
问题出现进展，主动补给子问题仍未解决”，不能升为最终模型。

```bash
python3 training/watch.py --policy p28-dense-frontier-survival \
  --net training/models/p28_dense_frontier_actor_iter04.pt --seed 831886081
```

保守对照（同图、能撑满但只访问 2 格）：

```bash
python3 training/watch.py --policy p28-dense-frontier-survival \
  --net training/models/p28_dense_frontier_actor_iter02.pt --seed 831886081
```
