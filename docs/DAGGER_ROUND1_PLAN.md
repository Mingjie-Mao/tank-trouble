# DAgger 纠正蒸馏 — 第 1 轮

日期：2026-08-04
状态：**采集已完成。第 4 节的假设被数据否定，见第 0 节。**

---

## 0. 结果：假设被否定

`collect` 跑完 120 局（119 胜 / 1 负，seed 970118），32367 个状态。
门控条件是 `temporal_correction_rate ≥ 10%`。

**实测 7.05%。低于阈值，本方案的前提不成立。**

报告：`training/analysis/dagger/collect_round1.json`
数据：`training/temporal_intent_data/dagger_round1.npz`

### 0.1 GRU 不是搜索负载的来源，而是最便宜的路径

| | 帧数占比 | 搜索率 |
|---|---|---|
| GRU 驱动的帧 | 70.6% | **4.76%** |
| 非 GRU 帧 | 29.4% | **23.23%** |

差了将近 5 倍，方向和假设相反。原因在 `act()` 的门槛本身：GRU 只在
`p27_action[2]==0`（不开火）且 `risk < risk_search_threshold` 时才被允许改写动作。

| 搜索原因 | 该原因下 GRU 驱动比例 |
|---|---|
| fire | **0.0%** |
| risk | **0.0%** |
| proactive | 70.4% |
| followup | 28.0% |

也就是说，**GRU 在结构上就被排除在昂贵的帧之外**。它不可能是那些搜索的原因。
4.76% vs 23.23% 不是相关性，是这个门槛的直接后果。

### 0.2 搜索负载是架构规则，不是学习误差

3298 个搜索帧（占全部帧 10.19%）的触发原因：

| 原因 | 占搜索 | 说明 |
|---|---|---|
| proactive | 32.2% | 固定 24 帧定时器，与危险无关 |
| fire | 21.9% | 每一次开火提议都强制搜索 |
| risk | 17.9% | 入射风险 ≥ 0.18 |
| followup | 15.4% | successor shield 后续审计 |
| network | 7.9% | 审计失败，**这才是"网络提议了危险动作"** |
| topology | 4.7% | 拓扑动作审计失败 |

只有 7.9% 的搜索是网络提议不安全导致的。**即使把 GRU 训练到完美（0 纠正），
搜索率最多从 11.1% 降到约 10%。** 不解决任何问题。

`unsafe_temporal_movement` 有 1562 个状态，但其中只有 349 个（22.3%）真的跑了搜索——
GRU 被否决大多发生在一次**本来就要跑的**搜索内部，并不额外制造搜索。

### 0.3 真正的成本在单次搜索

120 局 wall 658.2s × 6 worker ≈ 3949 CPU-s / 32367 帧 = **约 122 ms/帧**，
而 60fps 的预算是 16.7 ms。

反推单次 `full_search` 成本（对非搜索帧的假设不敏感）：

| 假设非搜索帧成本 | 单次 full_search |
|---|---|
| 2 ms | 1.18 s |
| 3 ms | 1.17 s |
| 5 ms | 1.15 s |

**约 1.2 秒一次。** 如果这个数成立，那么把搜索率从 11% 压到 8% 在感知上没有意义——
卡顿是"10% 的帧上有一次 1 秒级停顿"，不是"搜索太频繁"。

**但这个数字是从 6 进程总 CPU 时间反推的**，MacBook Air 的能效核调度可能让它虚高。
优化之前必须先测准，否则是猜。这就是 `training/search_profile.py` 存在的原因。

### 0.4 走位抖动归因（同一批数据）

目标改为"走位好看 + 胜率高"后，用同一批 32367 个状态归因抖动。
**结论又是反直觉的：抖动几乎全部来自精确搜索，GRU 是唯一平滑的部分。**

| 来源 | 帧数 | 动作切换率 |
|---|---|---|
| GRU 提议被采纳 | 21239 | **0.1%** |
| GRU 提议被否决 | 1612 | **52.2%** |
| 跑了搜索的帧 | 3293 | **37.0%** |
| 没搜索的帧 | 28954 | 3.4% |

按搜索原因：

| reason | 切换率 |
|---|---|
| proactive | **45.7%** |
| fire | 35.1% |
| risk | 25.5% |
| followup | 17.1% |
| topology | 15.7% |
| network | 0.8% |

动作段中位数 4 帧，24.7% 的段只持续 1 帧。

机制：`exact_root_search` 对每个候选独立算 value，**目标函数里没有任何连续性项**。
每次搜索都从零重新挑，没有理由偏向当前正在执行的动作。而 `proactive` 是固定
24 帧定时器、与危险无关——相当于每 24 帧掷一次硬币，45.7% 概率原地变向。

### 0.5 已实现：动作连续性偏好

`training/exact_state_mpc_teacher.py` 新增 `prefer_movement_continuity()`，
与已有的 `prefer_nonfire_secured_kill` / `prefer_nonfire_low_gain` 同构。

安全论证（封闭）：

1. 只从 `row["allowed"]` 为真的行里选——不可能提升被精确 rollout 否决的动作。
2. 只在 value 差 ≤ epsilon 时生效。
3. **不改变开火决策**——候选必须与当前最优解有相同的 fire 位。开火仍归长尾
   自伤检查和那两个 nonfire 偏好管。
4. successor shield 分支内单独再应用一次（shield 会按原始 value 重挑），
   且记账放在 shield 计数之后，不会被误记成 `successor_shield_override`。

`epsilon` 单位是原始 value。参考尺度：`death_penalty * SCORE_SCALE = 180`
是"确定死亡"的代价，所以 epsilon 必须远低于它。
**默认 0.0（关闭）**，必须显式打开，属于 opt-in。

新计数器 `movement_continuity_holds` 已进逐局报告和前台日志。
18 个单测，不依赖 torch，专门钉住"永远不选 unsafe""永远不改开火位"两条不变式。

### 0.6 这批数据还有什么用

不能用来降搜索率，但 32367 个状态里有：

- `unsafe_temporal_movement` 1562
- `hardcase_event` 1222（其中 `missed_fire_window` 类别 884，`stutter_stall` 271，
  `dead_end_stall` 186）
- `terminal_window` 90（970118 的完整死亡前窗口）

这些对**行为**目标（目标 3、4）仍然有效。但必须明确：训练它不会让部署变快。

---

## 1. 前台 GUI 基线（temporal-hybrid EXPERIMENT）

来源：`training/analysis/live/watch_supervision.jsonl`
机器可读汇总：`training/analysis/dagger/live_watch_baseline.json`

原始 291 行，其中 seed 970000–970022 出现两次（两次 GUI 会话各写了一遍）。
按 seed 去重后 **268 局**。下面所有数字都是去重后的。

| 指标 | 值 |
|---|---|
| 结果 | 261 胜 / 5 负 / 2 双亡 = 97.4% |
| 主动击杀局 | 185 / 268 = 69.0% |
| 0 击杀获胜（Laika 自杀） | 77 局 |
| 击杀类型 | bounce 126、direct 58 |
| 搜索率 | 均值 13.4%、中位 10.7%、最大 100% |
| 搜索率 > 25% 的局 | 24 局 |
| 局长 | 均值 271.7、中位 220.5、最大 1196 |
| 开火窗口 | 562 个，抓住 163（29.0%），漏掉 399 |
| 长尾开火检查 | 788 次检查，**451 次否决（57.2%）** |
| 拓扑目标 | 737 请求，263 完成（35.7%），237 中止（32.2%） |

每 1000 帧事件率：

| 事件 | 每 1000 帧 |
|---|---|
| movement_switch | 164.3 |
| throttle_reversal | 70.3 |
| turn_reversal | 65.7 |
| stutter_stall | 7.3 |
| dead_end_stall | 6.1 |
| missed_fire_window | 5.5 |
| passive_map_control | 0.7 |

逐局问题归因（268 局中有该问题的局数）：

| 类别 | 局数 |
|---|---|
| movement_stutter | 153 |
| dead_end_navigation | 139 |
| fire_opportunity_gap | 138 |
| excessive_search_handoff | 43 |
| passive_map_control | 37 |
| terminal_loss | 5 |
| double_death_risk | 2 |

**97.4% 不是正式胜率。** 这是前台演示、非官方复位、不是配对评测，
而且是 EXPERIMENT 策略。正式基线仍然是 P27b 的 88.2% / 89.6% 和精确老师的 99.0%。

### 1.1 非胜局（全部进永久回归集合）

| seed | 结果 | 死因 | 帧 | 搜索率 | 射击 |
|---|---|---|---|---|---|
| 970252 | loss | **self** | 86 | **100%** | 1 |
| 970243 | loss | laika_direct | 126 | 56.9% | 0 |
| 970197 | double_death | laika_unknown | 172 | 37.1% | 2 |
| 970105 | loss | laika_bounce | 185 | 44.6% | 2 |
| 970128 | loss | laika_direct | 228 | 39.9% | 1 |
| 970163 | loss | laika_bounce | 278 | 36.5% | 0 |
| 970170 | double_death | laika_bounce | 533 | 7.9% | 0 |

`970252` 最值得注意：**每一帧都在做精确搜索，最后还是自伤致死。**
这直接否定了"多搜索就更安全"的直觉，也说明问题不在搜索预算。

`970163`、`970243`、`970170` 三局 **0 次射击**。不是没打中，是从头到尾没开过枪。

### 1.2 从数字读出来的机制

1. **长尾开火否决率 57.2%。** 学习栈提出的开火有一半以上被精确长尾自伤检查
   拒绝。每次否决都要重新跑一次 no-fire 搜索。这是最大的单项计算浪费，
   也是最直接的分布不一致证据。
2. **拓扑中止率 32.2%，完成率只有 35.7%。** 拓扑规划器提出目标，安全审计
   否决，意图被清空，下一帧重新提出。这正是 `dead_end_navigation` 和
   `movement_stutter` 同时高的原因——不是网络在原地抖，是目标在反复被推翻。
3. **每 1000 帧 136 次油门/转向反转。** 约 13.6% 的帧在反向。
4. **开火窗口只抓住 29%。** 而且 69% 的局才有主动击杀，77 局是 Laika 自己撞死的。
   胜率里有相当一部分不是我们赢的。

---

## 2. 已定位的根因

`training/temporal_intent_pipeline.py` 的 `_make_teacher()` **没有传
`temporal_intent_net`**。所以 `topology_temporal_*.npz` 里每一个状态都来自
P27b + 拓扑的分布，而部署时 GRU 在回路里驱动移动。

同一个文件的 `_collect_seed()` 还有：

```python
"rows": rows if true_result == "win" else [],
```

失败局的行被整个丢掉，失败窗口从来没有进入过训练集。

这两点合起来就是经典的 DAgger 缺口：网络在自己访问不到的状态上训练，
在自己真正访问的状态上提议危险动作，安全层被迫高频纠正。
**卡顿是这个缺口的症状，不是 top-k 的症状。**

这和已知事实一致：top-12 → top-8 → top-4 只是把搜索次数在不同 seed 上搬来搬去
（27.3% / 9.4% / 34.1%），top-4 还在 996004 直接自杀。

---

## 3. 本轮改了什么

### `training/dagger_correction_recorder.py`（新增，纯 numpy）

- `build_correction_record()` — 归一化单帧决策轨迹：
  网络动作 / GRU 动作 / 拓扑动作 / 迟滞后动作 / 提议动作 / 实际执行动作，
  加上 reason、category、safe_root_count、interventions、audit_failed、
  长尾开火否决、风险。
- `classify_correction()` — 把每帧分到一个纠正通道，优先级从高到低：
  `terminal_window` → `successor_shield_override` → `fire_rejected` →
  `unsafe_temporal_movement` → `search_override` → `topology_abort` →
  `hardcase_event` → `accepted`。
- `correction_weight()` — 通道权重 × 局部难度（safe_root ≤ 2 时 ×1.5，
  拓扑活跃 ×1.2）。权重刻意压得低：把纠正样本权重拉太高会让移动头塌陷到
  安全老师的保守动作上，反而复现我们要消除的抖动。上界有单测守着（≤ 12）。
- `tag_round()` — 非胜局保留死亡前 90 帧作为 `terminal_window`。
- `build_dataset()` — 输出 npz，是 `temporal_intent_pipeline` 格式的超集，
  `temporal_intent_pipeline.py train` 可以直接读，不用改。

### `training/sparse_exact_safety_policy.py`（修改）

`act()` 现在额外记录纠正轨迹到 `last_temporal_sample`。没有新增任何 rollout，
没人读的时候完全惰性。修正点：

- `executed_fire` 用 `action[2]`，不用局部变量 `fire`——后者在敌方已死时被强制
  归零，会被误判成开火纠正。
- 新增 `audit_failed` / `topology_aborted` 标志，因为审计失败走的是
  `reason="committed"/"topology"` 分支，从 `reason` 看不出来。

### `training/dagger_distill.py`（新增）

`collect` 子命令：**带着当前 GRU 跑 rollout**（on-policy），精确老师做标签，
失败局的行保留。逐局写监督报告，不只写失败局。

### `training/test_dagger_correction_recorder.py`（新增）

41 个单测，全部不依赖 torch。覆盖通道优先级、权重上界、失败窗口、
标签必须是执行动作而不是 GRU 被否决的提议、npz dtype 安全。

---

## 4. 怎么跑

```bash
cd /Users/mingjie/Desktop/tank-trouble

# 0) 先验证
python3 -m unittest discover -s training -p 'test_*.py'
python3 -m py_compile training/*.py
git diff --check

# 1) 用当前 GRU 采 on-policy 纠正数据
python3 training/dagger_distill.py collect \
  --temporal-intent-net training/models/temporal_intent_topology_v1.pt \
  --seed-list 970000:120 --workers 6 \
  --out training/temporal_intent_data/dagger_round1.npz \
  --report training/analysis/dagger/collect_round1.json

# 2) 训练（复用现有 trainer，数据格式是超集）
python3 training/temporal_intent_pipeline.py train \
  --data training/temporal_intent_data/dagger_round1.npz \
  --out training/models/temporal_intent_dagger_r1.pt \
  --report training/analysis/dagger/train_r1.json
```

**第 1 步跑完先不要训练。** 先看 `collect_round1.json` 里的
`temporal_correction_rate`。如果它低于 ~10%，说明 GRU 其实很少被否决，
那卡顿的根因就不是分布不一致，本方案的前提不成立，应该停下来重新归因。

---

## 4b. 门控怎么跑

`training/dagger_gate.py` 读两份 `sparse_exact_safety_policy.py --out` 的报告，
机械地做全指标比较。基线和候选走同一套评测代码，不引入新的评测路径。

```bash
SEEDS=$(python3 training/dagger_gate.py seeds)

# 基线：当前 GRU 在永久回归种子上的表现（必须先跑，否则没有比较对象）
python3 training/sparse_exact_safety_policy.py \
  --seed-list "$SEEDS" --workers 6 --topology-assist \
  --network-move-hold-frames 4 --audit-interval 6 --temporal-confidence 0.60 \
  --temporal-intent-net training/models/temporal_intent_topology_v1.pt \
  --out training/analysis/dagger/base_permanent.json

# 候选：同样的命令，只换 --temporal-intent-net
python3 training/sparse_exact_safety_policy.py \
  --seed-list "$SEEDS" --workers 6 --topology-assist \
  --network-move-hold-frames 4 --audit-interval 6 --temporal-confidence 0.60 \
  --temporal-intent-net training/models/temporal_intent_dagger_r1.pt \
  --out training/analysis/dagger/cand_permanent.json

# 比较
python3 training/dagger_gate.py compare \
  --baseline training/analysis/dagger/base_permanent.json \
  --candidate training/analysis/dagger/cand_permanent.json \
  --level permanent \
  --out training/analysis/dagger/gate_permanent.json
```

判定规则（写在 `gate_verdict()` 里，有单测守着）：

- `permanent` 级别**只要有一局非胜就直接失败**，不看比率。
- 任何 `hard` 指标（win_rate / nonwin_rate / double_death_rate）回退直接失败，
  容差为 0。
- `objective` 指标（搜索率 / wall time / 高搜索局占比）**至少要有一个改善**，
  否则这一轮什么都没买到，不值得承担风险。
- `watch` 指标回退不自动失败，但会列进 `requires_explanation`，必须解释。

第三条是专门防"保守塌陷"的：网络学会什么都不提议，搜索率当然会掉，
但 `active_kill_rate` 和 `fire_capture_rate` 会同时掉。单测
`test_conservative_collapse_is_visible` 就是钉这个场景。

后面每一级只要改 `--seed-list` 和 `--level`。

---

## 5. 门控顺序（按项目规则）

1. 永久失败种子必须全过（`python3 training/dagger_gate.py seeds`）：
   `979000, 996004, 998002, 970105, 970128, 970163, 970170, 970197, 970243, 970252`
2. 12 局配对
3. 100 个未见种子
4. 300 局
5. 最后才做正式 1000@970000 / 500@990000

每一级都要同时比较（不能只看胜率）：

| 指标 | 当前基线 | 期望方向 |
|---|---|---|
| 胜率 | 97.4%（非正式） | 不下降 |
| 双亡 + 失败 | 7 / 268 | 不上升 |
| 搜索率均值 | 13.4% | 下降 |
| 搜索率 > 25% 的局 | 24 / 268 | 下降 |
| 实际 wall time | — | 下降 |
| 局长均值 / 最大 | 271.7 / 1196 | 不上升 |
| throttle + turn reversal / 1000 帧 | 136.0 | 下降 |
| stutter / dead-end / 1000 帧 | 7.3 / 6.1 | 下降 |
| missed_fire_window / 1000 帧 | 5.5 | 不上升 |
| 开火窗口抓取率 | 29.0% | 不下降 |
| 主动击杀局 | 185 / 268 | 不下降 |
| 长尾开火否决率 | 57.2% | 下降 |
| 拓扑中止率 | 32.2% | 下降 |

---

## 6. 这个方案会怎么失败

不做"一定会提升"的承诺。已知的失败模式：

1. **纠正样本太少。** 如果 `temporal_correction_rate` 很低，
   数据集里就没有足够的分歧状态，训练等于重复现有数据。前提不成立。
2. **保守塌陷。** 纠正样本权重偏高时，移动头会学成"总是选安全老师的动作"，
   连续性丢失，`movement_switch` 和 reversal 反而上升。
   这就是权重压低并加上界单测的原因。看 reversal 指标能直接发现。
3. **一轮不够。** DAgger 通常要 2–3 轮才收敛：新模型访问新状态，
   又产生新的分歧。第 1 轮搜索率下降但没到目标是正常的，不是失败。
4. **移动改好了但开火没改。** 时序网络按设计不管开火，
   所以 `missed_fire_window` 和 57.2% 的长尾否决率**大概率不会**被这轮修好。
   那需要单独的 fire-opportunity 头，必须另外训，不能和移动 GRU 混训。
5. **搜索率下降但胜率也下降。** 说明安全层的纠正本来就是必要的，
   网络只是学会了不去提议那些动作，同时也不去提议正确的进攻动作。
   看主动击杀局数能区分这两种情况。

---

## 7. 下一轮之后

- 独立 fire-opportunity 头（用 `training/space_control_teacher.py` 的
  fire/no-fire 精确反事实标签）。
- space-control 头，学反弹弹压缩 Laika 活动空间。
- 两者都不能和当前移动 GRU 一次性混训。
