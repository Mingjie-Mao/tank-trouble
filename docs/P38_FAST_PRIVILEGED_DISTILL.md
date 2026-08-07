# P38 Fast Privileged Distillation

## Purpose

P37 called its deployed policy a single network, but its 914-dimensional
observation still contained online planning:

- a 512-ray inverse kill-field build when Laika entered a new cell;
- nine 36-frame movement-conditioned sandboxes;
- the inherited P29 block also contained nine 24-frame sandboxes.

P38 moves those computations to the offline teacher. Deployment receives a
308-dimensional cheap vector and an 8x10x12 spatial tensor. A 3,511,123
parameter network predicts 18 action scores. Auxiliary heads reconstruct the
134 kill-field facts and 90 movement-preview facts during training, but neither
target is computed at runtime.

## Eight-question audit

### 1. Goal

The desired terminal ordering is:

1. active kill and alive at the original scoring frame;
2. opponent self-kill and alive at scoring, with less credit;
3. double death;
4. own death or timeout.

P37 violated this goal by returning an active-kill score immediately when
Laika died. P38 continues the real episode to `round_end`. During the 75 live
post-kill frames, a survival teacher scores movement by exact sandbox survival.
Fire is unavailable because the target is already dead.

Remaining limitation: the pre-kill P37 rollout still uses a short-horizon kill
score. Outcome weighting and the learned post-kill controller reduce the
mismatch, but they do not provide a globally optimal pre-kill survival value.

### 2. Information

Runtime inputs contain the maze, both poses, local wall rays, up to ten bullets,
bullet trajectory facts, weapon state, exploration state, previous action,
actual displacement/rotation, failed-action flags, enemy-alive state, frozen
state, and time remaining to scoring freeze.

The previous hard early return after enemy death is removed. The same network
observes and acts throughout the live post-kill window.

### 3. Representation

The network has a spatial CNN tower and a vector MLP tower, fused before four
heads: action scores, kill-field reconstruction, action-preview reconstruction,
and survival prediction. It can represent global maze geometry and continuous
entity facts without online planning. No recurrent state is required for the
first version because action effects and bullets are explicitly observed.

### 4. Experience

The completed run used 64 teacher rounds and two 32-round DAgger corrections:

- 19,281 aggregate decision states;
- 3,207 post-kill states;
- teacher bootstrap outcomes: 45 active wins, 9 opponent self-wins, 2 doubles,
  6 timeouts, 1 enemy loss, and 1 self loss;
- DAgger deliberately retained the student's failure distribution.

This is enough to establish the pipeline, not enough to claim saturation.
Threatened post-kill states are rarer than safe post-kill frames and remain the
highest-value target for future collection.

### 5. Learning signal

Every planning state receives the complete 18-action score landscape rather
than only an argmax. The expensive 134 field facts and 90 previews are auxiliary
targets. Episode survival supplies an additional diagnostic target.

Teacher action margins are often extremely small, so exact top-1 agreement is
not a reliable objective. Checkpoint selection uses teacher regret, balanced
between pre-kill and post-kill phases. High-spread movement landscapes receive
extra weight because they contain the states where a wrong move is costly.

### 6. Optimization

Validation is split by complete episode seed/map. The final checkpoint is the
minimum balanced-regret epoch, including epoch zero as a protected candidate,
so later overfitting cannot overwrite a better source model.

The selected aggregate checkpoint used epoch 9 of the hard-score correction;
later soft-distribution fine-tuning did not beat its balanced regret and was
therefore rejected automatically.

### 7. Generalization

No frame from a validation episode appears in training. This removes the old
adjacent-frame leakage. The current result therefore measures held-out random
mazes/seeds from the same generator, not different maze sizes, physics, or a
different opponent policy.

### 8. Evaluation and optimality

The deployed path was verified to call neither inverse field construction nor
`make_sandbox`. A 75-frame post-kill execution continued planning, never fired,
and reached freeze alive in the smoke seed. Measured full planned-action latency
was approximately 0.83 ms mean and 1.36 ms p95 on the development machine.

On the strict held-out map group, the chosen checkpoint had balanced normalized
teacher regret 0.03293. Exact top-1 is intentionally lower because the median
teacher top-two margin is near zero. This does not establish gameplay
optimality; it establishes that the latency bug is removed, post-kill decisions
exist, and the remaining measurable gap is concentrated in pre-kill student
states.

## Artifacts

- Trainer and policy: `training/killfield_fast_distill.py`
- Tests: `training/test_killfield_fast_distill.py`
- Aggregate data: `training/killfield_fast_data/`
- Selected model: `training/models/p38_killfield_fast.pt`
- Playback policy: `p38-killfield-fast`

Playback:

```bash
python3 training/watch.py --policy p38-killfield-fast --seed 38500077
```
