"""
P24v4 生存老师重蒸馏：完整账本观测 + 移动/开火双头 + 多轮 DAgger。

旧 P24v2.1 的老师评分依赖 ledger.visited，但学生只看到分数池和剩余时间，
导致同一观测对应冲突标签。本版加入 12x10 覆盖冷却图，使观测包含老师决策
所需的完整账本状态；网络将 18 动作拆成 9 路移动价值和 9 路条件开火头。
开火头在均匀训练之外使用机会状态平衡重放，避免稀少开火标签被淹没。

数据、模型与旧 P24 完全隔离：
  training/survival_data_v2/
  training/models/p24v4_survival_best.pt

用法：

  python3 training/survival_distill_v2.py pipeline --fresh \
    --teacher-rounds 128 --bootstrap-rounds 128 \
    --dagger-rounds 3 --rounds-per-dagger 128 --workers 8
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.score_distill import FULL_OBS_DIM, full_obs


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "survival_data_v2")
MODELS_DIR = os.path.join(HERE, "models")
BEST_NET = os.path.join(MODELS_DIR, "p24v4_survival_best.pt")
P22_NET = os.path.join(MODELS_DIR, "scorenet_best.pt")
MAP_W = 12
MAP_H = 10
VISIT_DIM = MAP_W * MAP_H
LEDGER_DIM = 2 + VISIT_DIM
OBS_DIM = FULL_OBS_DIM + LEDGER_DIM
MOVE_COUNT = 9
SCALE = 300.0
DECIDE_EVERY = 2
FIRE_ADV_MARGIN = 2.0 / SCALE
MOVE_SOFTMAX_TEMPERATURE = 6.0 / SCALE
ORIGINAL_WIN_FLOOR = 0.65


def legacy_econ():
    from training.survival_mode import ECON
    return dict(ECON, empty_mag=0.0, hit_immunity=0)


def bind_env(env, game, frames):
    if env.game is not game:
        env.game = game
        env._build_wall_boxes()
    env._frames = frames
    env._prev_phi = env._phi()


def ledger_features(ledger, econ):
    """老师决策充分账本：池、剩余时间、每格覆盖奖励冷却。"""
    features = np.zeros(LEDGER_DIM, dtype=np.float32)
    features[0] = np.clip(ledger.pool / 300.0, 0.0, 3.0)
    features[1] = np.clip(
        (econ["cap"] - ledger.frames) / max(econ["cap"], 1), 0.0, 1.0)
    cooldown = max(float(econ["cover_cd"]), 1.0)
    for (cell_x, cell_y), last_frame in ledger.visited.items():
        if 0 <= cell_x < MAP_W and 0 <= cell_y < MAP_H:
            age = max(0.0, ledger.frames - last_frame)
            active = max(0.0, 1.0 - age / cooldown)
            features[2 + cell_y * MAP_W + cell_x] = active
    return features


def build_observation(env, game, ledger, econ):
    bind_env(env, game, ledger.frames)
    return np.concatenate([full_obs(env), ledger_features(ledger, econ)])


def _apply(tank, action):
    throttle, turn, fire = action
    tank.forward, tank.backup = throttle == 2, throttle == 0
    tank.turn_left, tank.turn_right = turn == 0, turn == 2
    tank.fire = fire == 1


def _dict_to_action(action):
    throttle = 2 if action.get("forward") else 0 if action.get("backup") else 1
    turn = 0 if action.get("turn_left") else 2 if action.get("turn_right") else 1
    return throttle, turn, int(bool(action.get("fire")))


def _teacher_labels(game, ledger, rng_seed, econ):
    from training.mpc_agent import CANDIDATES, make_sandbox
    from training.survival_mode import survival_rollout

    scores = np.empty(len(CANDIDATES), dtype=np.float32)
    for index, action in enumerate(CANDIDATES):
        sandbox = make_sandbox(game, "L2", rng_seed=rng_seed)
        scores[index] = survival_rollout(
            sandbox, action, ledger.pool, ledger.visited, ledger.frames,
            econ=econ, style=True)
    paired = scores.reshape(MOVE_COUNT, 2)
    move_targets = paired.max(axis=1) / SCALE
    fire_advantage = (paired[:, 1] - paired[:, 0]) / SCALE
    fire_targets = fire_advantage > FIRE_ADV_MARGIN
    return scores, move_targets.astype(np.float32), fire_targets.astype(np.uint8)


def _load_actor(actor_kind, actor_path):
    if actor_kind == "p22":
        from training.score_distill import ScoreNetPolicy
        return ScoreNetPolicy(actor_path)
    if actor_kind == "student":
        return SurvivalTwoHeadPolicy(actor_path)
    return None


def _collect_worker(job):
    (phase, worker, rounds, seed0, epsilon, actor_kind, actor_path,
     data_dir) = job
    import torch
    torch.set_num_threads(1)
    from tank_trouble_original.game import Game
    from training.mpc_agent import CANDIDATES
    from training.survival_mode import Ledger
    from training.tt_gym_env import TankTroubleGym

    econ = legacy_econ()
    actor = _load_actor(actor_kind, actor_path)
    env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
    rng = random.Random(seed0 + worker * 104729 + 419)
    observations = []
    move_labels = []
    fire_labels = []
    stats = dict(games=0, death=0, drain=0, cap=0, hits=0, frames=0,
                 opportunity_states=0, actor_missed=0, decisions=0,
                 regret=0.0)
    for episode in range(rounds):
        game = Game(seed=seed0 + episode, ai_enabled=True, invincible={1})
        ledger = Ledger(game, econ)
        action = (1, 1, 0)
        while True:
            if ledger.frames % DECIDE_EVERY == 0:
                observation = build_observation(env, game, ledger, econ)
                scores, move_target, fire_target = _teacher_labels(
                    game, ledger, rng.randrange(1 << 30), econ)
                observations.append(observation)
                move_labels.append(move_target)
                fire_labels.append(fire_target)
                opportunity = bool(fire_target.any())
                stats["opportunity_states"] += int(opportunity)

                if actor_kind == "teacher":
                    chosen = CANDIDATES[int(scores.argmax())]
                elif actor_kind == "student":
                    chosen = _dict_to_action(actor.act_ctx(game, ledger))
                else:
                    chosen = _dict_to_action(actor.act(game))
                if rng.random() < epsilon:
                    chosen = (rng.randrange(3), rng.randrange(3), 0)
                action_index = CANDIDATES.index(chosen)
                movement = action_index // 2
                stats["regret"] += float(scores.max() - scores[action_index])
                stats["actor_missed"] += int(
                    opportunity and not chosen[2] and fire_target[movement])
                stats["decisions"] += 1
                action = chosen

            _apply(game.tanks[0], action)
            events = game.step()
            end = ledger.on_frame(game, events)
            if end != "alive":
                break
        stats["games"] += 1
        stats[end] += 1
        stats["hits"] += ledger.hits
        stats["frames"] += ledger.frames

    phase_dir = os.path.join(data_dir, phase)
    os.makedirs(phase_dir, exist_ok=True)
    path = os.path.join(phase_dir, f"shard_{worker}.npz")
    np.savez_compressed(
        path,
        X=np.asarray(observations, dtype=np.float32),
        M=np.asarray(move_labels, dtype=np.float32),
        F=np.asarray(fire_labels, dtype=np.uint8),
    )
    return path, stats


def collect_phase(phase, total_rounds, workers, seed, actor_kind,
                  actor_path=None, epsilon=0.03, data_dir=DATA_DIR):
    base, remainder = divmod(total_rounds, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        rounds = base + (1 if worker < remainder else 0)
        if rounds:
            jobs.append((phase, worker, rounds, seed + offset, epsilon,
                         actor_kind, actor_path, data_dir))
            offset += rounds
    started = time.time()
    print(f"===== {phase}: {total_rounds}局 actor={actor_kind} "
          f"eps={epsilon:.0%} =====", flush=True)
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        outputs = pool.map(_collect_worker, jobs)
    totals = {}
    samples = 0
    for path, stats in outputs:
        samples += len(np.load(path)["X"])
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
    games = max(totals["games"], 1)
    alive_seconds = totals["frames"] / 25.0
    decisions = max(totals["decisions"], 1)
    opportunities = max(totals["opportunity_states"], 1)
    print(f"  完成 {samples}样本 / {totals['games']}局 / "
          f"{time.time() - started:.0f}s", flush=True)
    print(f"  命中间隔 {alive_seconds / max(totals['hits'], 1):.1f}s  "
          f"终局 死{totals['death']/games:.1%} "
          f"干{totals['drain']/games:.1%} 满{totals['cap']/games:.1%}",
          flush=True)
    print(f"  机会状态 {totals['opportunity_states']/decisions:.1%}  "
          f"actor错失 {totals['actor_missed']/opportunities:.1%}  "
          f"平均老师后悔 {totals['regret']/decisions:.1f}", flush=True)
    return totals


def load_aggregate(data_dir=DATA_DIR):
    paths = sorted(glob.glob(os.path.join(data_dir, "*", "shard_*.npz")))
    if not paths:
        raise RuntimeError(f"没有训练数据：{data_dir}")
    observations, move_labels, fire_labels = [], [], []
    for path in paths:
        data = np.load(path)
        observations.append(data["X"])
        move_labels.append(data["M"])
        fire_labels.append(data["F"])
    X = np.concatenate(observations)
    M = np.concatenate(move_labels)
    F = np.concatenate(fire_labels)
    print(f"数据聚合: {len(paths)} shard -> {len(X)}样本  "
          f"机会状态 {F.any(axis=1).mean():.1%}  "
          f"开火动作 {F.mean():.2%}", flush=True)
    return X, M, F


def build_network(input_dim=OBS_DIM, width=1024):
    import torch.nn as nn

    class SurvivalTwoHeadNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.move_trunk = nn.Sequential(
                nn.Linear(input_dim, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
            )
            self.move_head = nn.Linear(width, MOVE_COUNT)
            self.fire_trunk = nn.Sequential(
                nn.Linear(input_dim, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
            )
            self.fire_head = nn.Sequential(
                nn.Linear(width, width // 2), nn.ReLU(),
                nn.Linear(width // 2, MOVE_COUNT),
            )

        def forward(self, observation):
            move_latent = self.move_trunk(observation)
            fire_latent = self.fire_trunk(observation)
            return (self.move_head(move_latent),
                    self.fire_head(fire_latent))

    return SurvivalTwoHeadNet()


def train_model(X, M, F, epochs=12, batch_size=4096, learning_rate=3e-4):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(0)
    generator = np.random.default_rng(0)
    order = generator.permutation(len(X))
    validation_count = max(1, int(len(X) * 0.05))
    validation = order[:validation_count]
    training = order[validation_count:]
    X_tensor = torch.as_tensor(X)
    M_tensor = torch.as_tensor(M)
    move_policy_targets = torch.softmax(
        M_tensor / MOVE_SOFTMAX_TEMPERATURE, dim=1)
    F_tensor = torch.as_tensor(F.astype(np.float32))
    network = build_network(X.shape[1])
    move_optimizer = torch.optim.Adam(
        list(network.move_trunk.parameters())
        + list(network.move_head.parameters()),
        lr=learning_rate)
    fire_optimizer = torch.optim.Adam(
        list(network.fire_trunk.parameters())
        + list(network.fire_head.parameters()), lr=learning_rate)

    opportunity_indices = training[F[training].any(axis=1)]
    ordinary_indices = training[~F[training].any(axis=1)]
    opportunity_action_rate = (
        float(F[opportunity_indices].mean()) if len(opportunity_indices)
        else float(F[training].mean()))
    balanced_positive_rate = max(0.5 * opportunity_action_rate, 1e-4)
    positive_weight = min(
        5.0, math.sqrt((1.0 - balanced_positive_rate)
                       / balanced_positive_rate))
    pos_weight = torch.full((MOVE_COUNT,), positive_weight)
    replay_steps = max(2, math.ceil(len(training) / batch_size))

    def validation_metrics():
        network.eval()
        with torch.no_grad():
            move_prediction, fire_logits = network(X_tensor[validation])
            move_mse = -(move_policy_targets[validation]
                         * functional.log_softmax(move_prediction, dim=1)) \
                .sum(dim=1).mean().item()
            move_top1 = (move_prediction.argmax(1)
                         == M_tensor[validation].argmax(1)).float().mean().item()
            probabilities = torch.sigmoid(fire_logits)
            chosen_move = move_prediction.argmax(1)
            rows = torch.arange(len(validation))
            probabilities = probabilities[rows, chosen_move]
            truth = F_tensor[validation, chosen_move].bool()
            best_threshold, best_f1 = 0.5, -1.0
            best_precision = best_recall = 0.0
            for threshold in np.arange(0.10, 0.91, 0.05):
                predicted = probabilities >= threshold
                true_positive = (predicted & truth).sum().item()
                precision = true_positive / max(predicted.sum().item(), 1)
                recall = true_positive / max(truth.sum().item(), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                if f1 > best_f1:
                    best_threshold, best_f1 = float(threshold), f1
                    best_precision, best_recall = precision, recall
        network.train()
        return (move_mse, move_top1, best_threshold,
                best_precision, best_recall, best_f1)

    started = time.time()
    for epoch in range(epochs):
        shuffled = training[torch.randperm(len(training)).numpy()]
        total_move = total_fire = 0.0
        move_batches = fire_batches = 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            move_prediction, _ = network(X_tensor[indices])
            move_loss = -(move_policy_targets[indices]
                          * functional.log_softmax(move_prediction, dim=1)) \
                .sum(dim=1).mean()
            move_optimizer.zero_grad()
            move_loss.backward()
            move_optimizer.step()
            total_move += move_loss.item()
            move_batches += 1

        # 火控使用独立编码器，稀少事件不会覆盖移动表征。
        if len(opportunity_indices) and len(ordinary_indices):
            half = max(1, batch_size // 2)
            for _ in range(replay_steps):
                positive = generator.choice(
                    opportunity_indices, size=half, replace=True)
                negative = generator.choice(
                    ordinary_indices, size=half, replace=True)
                indices = np.concatenate([positive, negative])
                fire_latent = network.fire_trunk(X_tensor[indices])
                fire_logits = network.fire_head(fire_latent)
                fire_loss = functional.binary_cross_entropy_with_logits(
                    fire_logits, F_tensor[indices], pos_weight=pos_weight)
                fire_optimizer.zero_grad()
                fire_loss.backward()
                fire_optimizer.step()
                total_fire += fire_loss.item()
                fire_batches += 1

        metrics = validation_metrics()
        print(f"  epoch {epoch + 1}/{epochs}  "
              f"move {total_move/max(move_batches,1):.4f} "
              f"fire {total_fire/max(fire_batches,1):.4f}  "
              f"val_move {metrics[0]:.4f} top1 {metrics[1]:.1%}  "
              f"fire P/R/F1 {metrics[3]:.1%}/{metrics[4]:.1%}/{metrics[5]:.1%} "
              f"thr {metrics[2]:.2f}  {time.time()-started:.0f}s", flush=True)
    return network, metrics


def train_candidate(iteration, epochs, data_dir=DATA_DIR,
                    models_dir=MODELS_DIR):
    import torch
    X, M, F = load_aggregate(data_dir)
    network, metrics = train_model(X, M, F, epochs=epochs)
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"p24v4_survival_iter{iteration:02d}.pt")
    torch.save({
        "state_dict": network.state_dict(),
        "in_dim": X.shape[1],
        "fire_threshold": metrics[2],
        "ledger_dim": LEDGER_DIM,
        "version": "p24v4_two_head",
    }, path)
    print(f"候选已保存 {path}", flush=True)
    return path


class SurvivalTwoHeadPolicy:
    name = "p24v4_survival_two_head"

    def __init__(self, model_path):
        import torch
        from training.tt_gym_env import TankTroubleGym
        self.torch = torch
        payload = torch.load(model_path, weights_only=True)
        self.network = build_network(payload["in_dim"])
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.fire_threshold = float(payload.get("fire_threshold", 0.5))
        self.env = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
        self.econ = legacy_econ()
        self.game = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.last_movement = 4
        self.desire_fire = False

    def reset(self):
        self.game = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.last_movement = 4
        self.desire_fire = False

    def _decide(self, game, ledger):
        observation = build_observation(self.env, game, ledger, self.econ)
        with self.torch.no_grad():
            move_scores, fire_logits = self.network(
                self.torch.as_tensor(observation).unsqueeze(0))
        self.last_movement = int(move_scores[0].argmax())
        self.desire_fire = bool(
            self.torch.sigmoid(fire_logits[0, self.last_movement]).item()
            >= self.fire_threshold)

    def _action(self, game):
        # 产生开火脉冲，避免持续按住扳机后无法再次触发。
        fire = self.desire_fire and game.tanks[0].trigger_released
        throttle, turn = MOVE_OPTIONS[self.last_movement]
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire}

    def act_ctx(self, game, ledger):
        if not game.tanks[0].alive:
            return {}
        if (game is not self.context_game
                or game.round_number != self.context_round):
            self.context_game = game
            self.context_round = game.round_number
            self.context_step = 0
            self.last_movement = 4
            self.desire_fire = False
        if self.context_step % DECIDE_EVERY == 0:
            self._decide(game, ledger)
        self.context_step += 1
        return self._action(game)

    def act(self, game):
        from training.survival_mode import Ledger
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.ledger = Ledger(game, self.econ)
        else:
            end = self.ledger.on_frame(game, game.events)
            if end in ("drain", "cap"):
                self.ledger = Ledger(game, self.econ)
        return self.act_ctx(game, self.ledger)


MOVE_OPTIONS = [
    (throttle, turn) for throttle in (0, 1, 2) for turn in (0, 1, 2)
]


def _survival_eval_worker(job):
    model_path, seed, count = job
    from training.survival_mode import run_survival
    policy = SurvivalTwoHeadPolicy(model_path)
    econ = legacy_econ()
    return [run_survival(policy, seed + index, econ=econ)
            for index in range(count)]


def evaluate_survival(model_path, n, seed, workers):
    from training.survival_mode import _agg
    base, remainder = divmod(n, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count:
            jobs.append((model_path, seed + offset, count))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        parts = pool.map(_survival_eval_worker, jobs)
    aggregate = _agg([result for part in parts for result in part])
    print(f"===== P24v4学生 生存验收 {aggregate['n']}局 @{seed} "
          f"({time.time()-started:.0f}s) =====")
    print(f"  存活 {aggregate['alive_s']:.1f}s  "
          f"命中间隔 {aggregate['hit_iv']:.1f}s  "
          f"风格 {aggregate['style_rate']:+.2f}/s  "
          f"卡墙 {aggregate['stuck_pct']:.1f}%")
    return aggregate


def _original_eval_worker(job):
    model_path, seed, count = job
    from training.evaluate import play_round_dual_engine
    policy = SurvivalTwoHeadPolicy(model_path)
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def evaluate_original(model_path, n, seed, workers):
    base, remainder = divmod(n, workers)
    jobs = []
    offset = 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count:
            jobs.append((model_path, seed + offset, count))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rounds = [item for part in pool.map(_original_eval_worker, jobs)
                  for item in part]
    total = len(rounds)
    count = lambda key: sum(result["true_result"] == key for result in rounds)
    shots = sum(result["shots"] for result in rounds)
    win = count("win") / total
    print(f"===== P24v4学生 原版验收 {total}局 @{seed} "
          f"({time.time()-started:.0f}s) =====")
    print(f"  真胜率 {win:.1%}  负 {count('loss')/total:.1%}  "
          f"双亡 {count('double_death')/total:.1%}  "
          f"平 {count('draw')/total:.1%}")
    print(f"  场均开火 {shots/total:.1f}  "
          f"命中率 {sum(result['kills'] for result in rounds)/max(shots,1):.1%}  "
          f"平均局长 {sum(result['frames'] for result in rounds)/total/25:.1f}s")
    return win


def survival_style_score(aggregate):
    hit_interval = aggregate["hit_iv"]
    hit_component = (min(1.0, 2.0 / hit_interval)
                     if math.isfinite(hit_interval) and hit_interval > 0
                     else 0.0)
    stuck_component = 1.0 - min(aggregate["stuck_pct"] / 20.0, 1.0)
    style_component = np.clip(
        (aggregate["style_rate"] + 1.5) / 3.0, 0.0, 1.0)
    return float(0.50 * hit_component
                 + 0.30 * stuck_component
                 + 0.20 * style_component)


def run_pipeline(args):
    if args.fresh and os.path.isdir(args.data_dir):
        shutil.rmtree(args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    if not args.skip_initial_collect:
        collect_phase("teacher", args.teacher_rounds, args.workers, 11_000_000,
                      "teacher", epsilon=args.epsilon, data_dir=args.data_dir)
        collect_phase("bootstrap_p22", args.bootstrap_rounds, args.workers,
                      11_200_000, "p22", P22_NET, args.epsilon,
                      data_dir=args.data_dir)

    best_win = -1.0
    best_style = -1.0
    best_qualifies = False
    best_path = None
    for iteration in range(args.dagger_rounds + 1):
        candidate = train_candidate(
            iteration, args.epochs, args.data_dir, args.models_dir)
        survival = evaluate_survival(
            candidate, args.survival_gate_n, args.survival_gate_seed,
            args.workers)
        style_score = survival_style_score(survival)
        gate_win = evaluate_original(
            candidate, args.gate_n, args.gate_seed, args.workers)
        qualifies = gate_win >= ORIGINAL_WIN_FLOOR
        promote = (
            best_path is None
            or (qualifies and not best_qualifies)
            or (qualifies and best_qualifies
                and (style_score, gate_win) > (best_style, best_win))
            or (not qualifies and not best_qualifies and gate_win > best_win)
        )
        if promote:
            best_win = gate_win
            best_style = style_score
            best_qualifies = qualifies
            best_path = candidate
            shutil.copyfile(candidate, args.best_net)
            print(f"  评测门晋升: 胜率 {best_win:.1%}  "
                  f"风格分 {best_style:.3f}  "
                  f"生存命中间隔 {survival['hit_iv']:.1f}s -> {args.best_net}",
                  flush=True)
        else:
            print(f"  评测门不晋升: 候选胜率 {gate_win:.1%} "
                  f"风格 {style_score:.3f} | 当前 {best_win:.1%}/"
                  f"{best_style:.3f}",
                  flush=True)
        if iteration < args.dagger_rounds:
            collect_phase(
                f"dagger_{iteration + 1:02d}", args.rounds_per_dagger,
                args.workers, 11_400_000 + iteration * 200_000,
                "student", args.best_net, args.epsilon,
                data_dir=args.data_dir)

    print(f"===== 最终冠军: {best_path} 门胜率 {best_win:.1%} "
          f"风格 {best_style:.3f} =====")
    evaluate_survival(
        args.best_net, args.final_survival_n, args.final_survival_seed,
        args.workers)
    evaluate_original(
        args.best_net, args.eval_n, args.eval_seed, args.workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pipeline", "eval", "eval-original"])
    parser.add_argument("--teacher-rounds", type=int, default=128)
    parser.add_argument("--bootstrap-rounds", type=int, default=128)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--rounds-per-dagger", type=int, default=128)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--gate-n", type=int, default=80)
    parser.add_argument("--gate-seed", type=int, default=9_840_000)
    parser.add_argument("--survival-gate-n", type=int, default=40)
    parser.add_argument("--survival-gate-seed", type=int, default=8_840_000)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=986000)
    parser.add_argument("--final-survival-n", type=int, default=80)
    parser.add_argument("--final-survival-seed", type=int, default=886000)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--best-net", default=BEST_NET)
    parser.add_argument("--net", default=BEST_NET)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=987000)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-initial-collect", action="store_true")
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline(args)
    elif args.mode == "eval":
        evaluate_survival(args.net, args.n, args.seed, args.workers)
    else:
        evaluate_original(args.net, args.n, args.seed, args.workers)


if __name__ == "__main__":
    main()
