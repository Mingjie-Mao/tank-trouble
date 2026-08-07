"""
P26 amortized MPC.

Goal: move MPC's expensive per-action rollout knowledge into a pure network.
Training may use slow offline rollouts; deployment is still one network forward
pass and does not run online search.

The model predicts:
  - 18 action scores
  - 18 x auxiliary rollout facts: kill, death, double death, survival horizons
  - 9 fire/no-fire calibration logits

Usage examples:

  python3 training/p26_amortized_mpc.py collect \
    --phase smoke --rounds 1 --workers 1 --actor-kind p25v3

  python3 training/p26_amortized_mpc.py train --epochs 2

  python3 training/p26_amortized_mpc.py eval \
    --net training/models/p26_amortized_mpc_iter00.pt --n 20
"""

import argparse
import glob
import multiprocessing as mp
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.opportunity_distill import (  # noqa: E402
    GOOD_FIRE_BONUS,
    HOLD,
    HORIZON,
    OBS_DIM,
    PRESSURE_BONUS,
    SCORE_SCALE,
    SUICIDE_FIRE_PENALTY,
    _shot_event,
)
from training.opportunity_distill_v2 import (  # noqa: E402
    BEST_NET as P25V2_NET,
    P22_NET,
    _load_network,
    build_observation,
)
from training.opportunity_distill_v3 import (  # noqa: E402
    margin_gated_action,
)
from training.opportunity_teacher_v2 import OpportunityAnalyzer360  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "p26_amortized_data")
MODELS_DIR = os.path.join(HERE, "models")
BEST_NET = os.path.join(MODELS_DIR, "p26_amortized_mpc_best.pt")

AUX_HORIZONS = (24, 48, 72)
AUX_NAMES = (
    "kill",
    "death",
    "double_death",
    "survive_24",
    "survive_48",
    "survive_72",
)
AUX_DIM = len(AUX_NAMES)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stack_observation(history, frame_stack):
    if frame_stack <= 1:
        return history[-1]
    if len(history) >= frame_stack:
        frames = history[-frame_stack:]
    else:
        frames = [history[0]] * (frame_stack - len(history)) + list(history)
    return np.concatenate(frames).astype(np.float32, copy=False)


def rollout_targets(sandbox, first_action, analyzer, start_metrics,
                    score_horizon=HORIZON, hold=HOLD):
    """Return the teacher score plus compact rollout facts for one action."""
    max_horizon = max(score_horizon, max(AUX_HORIZONS))
    me, enemy = sandbox.tanks[0], sandbox.tanks[1]
    throttle, turn, fire = first_action
    shot = _shot_event(sandbox) if fire == 1 else None
    fired = False
    score = None
    me_dead_frame = None
    enemy_dead_frame = None
    true_result = None
    survive_at = {horizon: None for horizon in AUX_HORIZONS}

    for frame in range(max_horizon):
        if frame == 0:
            me.forward, me.backup = throttle == 2, throttle == 0
            me.turn_left, me.turn_right = turn == 0, turn == 2
            me.fire = fire == 1
        elif frame == hold:
            me.fire = False

        events = sandbox.step()
        fired = fired or any(event[0] == "fire" and event[1] == 0
                             for event in events)

        if me_dead_frame is None and not me.alive:
            me_dead_frame = frame
        if enemy_dead_frame is None and not enemy.alive:
            enemy_dead_frame = frame

        if score is None:
            if not me.alive:
                score = -1000.0 + frame
            elif not enemy.alive and frame >= hold:
                score = 1000.0 - frame
            elif frame == score_horizon - 1:
                end_metrics = analyzer.metrics(sandbox)
                score = analyzer.potential(end_metrics) - analyzer.potential(
                    start_metrics)
                if fired and shot is not None:
                    result = shot["result"]
                    if result == "HIT" and start_metrics[0] >= 0.60:
                        score += GOOD_FIRE_BONUS
                    elif result == "SUICIDE":
                        score -= SUICIDE_FIRE_PENALTY
                    elif shot.get("closest", float("inf")) <= 0.75 * sandbox.scale:
                        score += PRESSURE_BONUS

        step_count = frame + 1
        if step_count in survive_at:
            survive_at[step_count] = 1.0 if me.alive else 0.0

        for event in events:
            if event[0] == "round_end":
                winner = event[1]
                true_result = ("win" if winner == 0 else
                               "loss" if winner == 1 else "double_death")
        if true_result is not None:
            break

    if score is None:
        end_metrics = analyzer.metrics(sandbox)
        score = analyzer.potential(end_metrics) - analyzer.potential(
            start_metrics)

    if true_result == "win":
        kill, death, double_death = 1.0, 0.0, 0.0
    elif true_result == "loss":
        kill, death, double_death = 0.0, 1.0, 0.0
    elif true_result == "double_death":
        kill, death, double_death = 1.0, 1.0, 1.0
    else:
        kill = 1.0 if enemy_dead_frame is not None else 0.0
        death = 1.0 if me_dead_frame is not None else 0.0
        double_death = 1.0 if kill and death else 0.0

    survival = []
    for horizon in AUX_HORIZONS:
        if survive_at[horizon] is None:
            survive_at[horizon] = 0.0 if death else 1.0
        survival.append(survive_at[horizon])

    aux = np.asarray([kill, death, double_death, *survival], dtype=np.float32)
    return float(score), aux


def label_actions(game, analyzer, metrics, step_seed, score_horizon,
                  score_samples=1):
    from training.mpc_agent import CANDIDATES, make_sandbox

    scores = np.empty(len(CANDIDATES), dtype=np.float32)
    aux = np.empty((len(CANDIDATES), AUX_DIM), dtype=np.float32)
    rng = random.Random(step_seed)
    for index, action in enumerate(CANDIDATES):
        total_score = 0.0
        total_aux = np.zeros(AUX_DIM, dtype=np.float32)
        for _ in range(max(1, int(score_samples))):
            sandbox = make_sandbox(game, "L2",
                                   rng_seed=rng.randrange(1 << 30))
            score, facts = rollout_targets(
                sandbox, action, analyzer, metrics, score_horizon)
            total_score += score
            total_aux += facts
        samples = max(1, int(score_samples))
        scores[index] = total_score / samples
        aux[index] = total_aux / samples
    return scores, aux


def _load_legacy_actor(path):
    network, input_dim = _load_network(path)
    return network, input_dim


def legacy_actor_action(network, input_dim, observation, metrics, candidates,
                        actor_kind, fire_margin):
    import torch

    actor_input = observation[:input_dim]
    with torch.no_grad():
        predicted = network(torch.as_tensor(actor_input).unsqueeze(0))[0].numpy()
    if actor_kind in ("p25v2", "p25v3") and input_dim == OBS_DIM:
        margin = fire_margin if actor_kind == "p25v3" else 0.0
        return margin_gated_action(predicted, metrics, candidates, margin)
    return candidates[int(predicted.argmax())]


def fire_targets(scores, fire_target_margin):
    values = np.asarray(scores)
    if values.ndim == 1:
        paired = values.reshape(9, 2)
        return (paired[:, 1] - paired[:, 0] > fire_target_margin).astype(
            np.float32)
    paired = values.reshape(-1, 9, 2)
    return (paired[:, :, 1] - paired[:, :, 0] > fire_target_margin).astype(
        np.float32)


def sample_weight(scores, chosen, aux):
    regret = max(0.0, float(scores.max() - scores[chosen])) / SCORE_SCALE
    lethal = 1.0 if scores[chosen] < -500.0 and scores.max() > -500.0 else 0.0
    chosen_aux = aux[chosen]
    bad_fire = 1.0 if chosen % 2 == 1 and chosen_aux[1] > 0.5 else 0.0
    return 1.0 + min(2.0, 2.0 * regret) + 2.0 * lethal + bad_fire


def _collect_worker(job):
    (phase, worker, rounds, seed0, actor_kind, actor_path, epsilon,
     frame_stack, score_horizon, fire_margin, fire_target_margin,
     fire_threshold, kill_weight, death_weight, double_death_weight,
     survive_weight, fire_prob_weight, result_win_weight,
     result_loss_weight, result_double_death_weight,
     result_draw_weight) = job
    import torch
    torch.set_num_threads(1)
    from training.mpc_agent import CANDIDATES
    from training.tt_gym_env import TankTroubleGym

    actor = actor_dim = p26_policy = None
    if actor_kind == "p22":
        actor, actor_dim = _load_legacy_actor(actor_path or P22_NET)
    elif actor_kind in ("p25v2", "p25v3"):
        actor, actor_dim = _load_legacy_actor(actor_path or P25V2_NET)
    elif actor_kind == "p26":
        p26_policy = P26Policy(
            actor_path or BEST_NET, fire_margin=fire_margin,
            fire_threshold=fire_threshold, kill_weight=kill_weight,
            death_weight=death_weight,
            double_death_weight=double_death_weight,
            survive_weight=survive_weight,
            fire_prob_weight=fire_prob_weight)

    env = TankTroubleGym(seed=0, reward_version=1, terminal_mode="score",
                         obs_traj=True, obs_nav=True)
    rng = random.Random(worker * 104729 + seed0 + 2603)
    xs, y_score, y_aux, y_fire, weights = [], [], [], [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    regret, lethal, decisions = 0.0, 0, 0

    for episode in range(rounds):
        env._base_seed = seed0 + episode
        env._episode = 0
        env.reset()
        analyzer = OpportunityAnalyzer360(env.game)
        history = []
        episode_start = len(weights)
        if p26_policy is not None:
            p26_policy.reset()
        while True:
            observation, metrics = build_observation(
                env, env.game, analyzer, env._frames)
            history.append(observation)
            stacked = stack_observation(history, frame_stack)
            step_seed = rng.randrange(1 << 30)
            scores, aux = label_actions(
                env.game, analyzer, metrics, step_seed, score_horizon)

            if actor_kind == "teacher":
                action = CANDIDATES[int(scores.argmax())]
            elif actor_kind in ("p22", "p25v2", "p25v3"):
                action = legacy_actor_action(
                    actor, actor_dim, observation, metrics, CANDIDATES,
                    actor_kind, fire_margin)
            elif actor_kind == "p26":
                action = p26_policy.act(env.game)
                action = (
                    2 if action.get("forward") else 0 if action.get("backup") else 1,
                    0 if action.get("turn_left") else 2 if action.get("turn_right") else 1,
                    1 if action.get("fire") else 0,
                )
            else:
                action = (rng.randrange(3), rng.randrange(3), 0)

            if rng.random() < epsilon:
                action = (rng.randrange(3), rng.randrange(3), 0)

            chosen = CANDIDATES.index(action)
            regret += float(scores.max() - scores[chosen])
            if scores[chosen] < -500.0 and scores.max() > -500.0:
                lethal += 1
            decisions += 1

            xs.append(stacked)
            y_score.append(scores / SCORE_SCALE)
            y_aux.append(aux)
            y_fire.append(fire_targets(scores / SCORE_SCALE,
                                       fire_target_margin))
            weights.append(sample_weight(scores, chosen, aux))

            _, _, terminated, truncated, info = env.step(np.asarray(action))
            if terminated or truncated:
                result = info.get("result", "draw")
                stats[result] = stats.get(result, 0) + 1
                multiplier = {
                    "win": result_win_weight,
                    "loss": result_loss_weight,
                    "double_death": result_double_death_weight,
                    "draw": result_draw_weight,
                }.get(result, 1.0)
                if multiplier != 1.0:
                    for index in range(episode_start, len(weights)):
                        weights[index] *= multiplier
                break

    phase_dir = os.path.join(DATA_DIR, phase)
    os.makedirs(phase_dir, exist_ok=True)
    path = os.path.join(phase_dir, f"shard_{worker}.npz")
    np.savez_compressed(
        path,
        X=np.asarray(xs, np.float32),
        Y_score=np.asarray(y_score, np.float32),
        Y_aux=np.asarray(y_aux, np.float32),
        Y_fire=np.asarray(y_fire, np.float32),
        W=np.asarray(weights, np.float32),
        frame_stack=np.asarray([frame_stack], np.int32),
        aux_names=np.asarray(AUX_NAMES),
    )
    return path, stats, regret, lethal, decisions


def collect_phase(phase, rounds, workers, seed_base, actor_kind,
                  actor_path=None, epsilon=0.03, frame_stack=4,
                  score_horizon=HORIZON, fire_margin=0.08,
                  fire_target_margin=0.08, fire_threshold=0.0,
                  kill_weight=0.10, death_weight=0.10,
                  double_death_weight=0.20, survive_weight=0.0,
                  fire_prob_weight=0.30, result_win_weight=1.0,
                  result_loss_weight=1.0, result_double_death_weight=1.0,
                  result_draw_weight=1.0):
    if rounds <= 0:
        return
    base, remainder = divmod(rounds, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((phase, worker, count, seed_base + offset,
                         actor_kind, actor_path, epsilon, frame_stack,
                         score_horizon, fire_margin, fire_target_margin,
                         fire_threshold, kill_weight, death_weight,
                         double_death_weight, survive_weight,
                         fire_prob_weight, result_win_weight,
                         result_loss_weight, result_double_death_weight,
                         result_draw_weight))
            offset += count
    print(f"===== P26 collect {phase}: {rounds} games actor={actor_kind} "
          f"eps={epsilon:.0%} stack={frame_stack} "
          f"fire_margin={fire_margin:g} "
          f"result_weights={result_win_weight:g}/{result_loss_weight:g}/"
          f"{result_double_death_weight:g}/{result_draw_weight:g} =====",
          flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_collect_worker, jobs)

    stats, regret, lethal, decisions = {}, 0.0, 0, 0
    for _, worker_stats, worker_regret, worker_lethal, worker_decisions in results:
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
        regret += worker_regret
        lethal += worker_lethal
        decisions += worker_decisions
    total = sum(stats.values())
    print(f"  done {decisions} samples / {total} games / "
          f"{time.time()-started:.0f}s", flush=True)
    print(f"  on-policy win {stats.get('win', 0)/max(total, 1):.1%}  "
          f"double death {stats.get('double_death', 0)/max(total, 1):.1%}  "
          f"teacher regret {regret/max(decisions, 1):.1f}  "
          f"lethal mistakes {lethal/max(decisions, 1):.2%}", flush=True)


def build_p26_net(in_dim, width=1024, aux_dim=AUX_DIM):
    import torch.nn as nn

    class _P26Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
            )
            self.score = nn.Linear(width, 18)
            self.aux = nn.Linear(width, 18 * aux_dim)
            self.fire = nn.Linear(width, 9)
            self.aux_dim = aux_dim

        def forward(self, x):
            h = self.trunk(x)
            return {
                "score": self.score(h),
                "aux": self.aux(h).view(-1, 18, self.aux_dim),
                "fire": self.fire(h),
            }

    return _P26Net()


def _csv_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _phase_weight_map(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): float(weight) for key, weight in value.items()}
    weights = {}
    for item in _csv_items(value):
        if "=" not in item:
            raise ValueError(f"phase weight must be phase=weight, got {item!r}")
        phase, weight = item.split("=", 1)
        weights[phase.strip()] = float(weight)
    return weights


def _category_weight_array(data, category_weights):
    multipliers = _phase_weight_map(category_weights)
    if not multipliers or "category" not in data.files:
        return 1.0
    categories = data["category"].astype(str)
    weights = np.ones(len(categories), dtype=np.float32)
    for category, multiplier in multipliers.items():
        weights[categories == category] = float(multiplier)
    return weights


def load_data(frame_stack, fire_target_margin=None, include_phases=None,
              exclude_phase_prefixes=None, phase_weights=None,
              category_weights=None):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*", "shard_*.npz")))
    xs, y_score, y_aux, y_fire, weights, used = [], [], [], [], [], []
    include = set(_csv_items(include_phases))
    exclude_prefixes = tuple(_csv_items(exclude_phase_prefixes))
    multipliers = _phase_weight_map(phase_weights)
    phase_counts = {}
    for path in paths:
        phase = os.path.basename(os.path.dirname(path))
        if phase.startswith("smoke"):
            continue
        if include and phase not in include:
            continue
        if any(phase.startswith(prefix) for prefix in exclude_prefixes):
            continue
        data = np.load(path)
        if data["X"].shape[1] != OBS_DIM * frame_stack:
            continue
        if data["Y_aux"].shape[2] != AUX_DIM:
            continue
        xs.append(data["X"])
        y_score.append(data["Y_score"])
        y_aux.append(data["Y_aux"])
        if fire_target_margin is None:
            y_fire.append(data["Y_fire"])
        else:
            y_fire.append(fire_targets(data["Y_score"], fire_target_margin))
        category_multiplier = _category_weight_array(data, category_weights)
        weights.append(data["W"] * multipliers.get(phase, 1.0)
                       * category_multiplier)
        used.append(path)
        phase_counts[phase] = phase_counts.get(phase, 0) + len(data["X"])
    if not xs:
        raise RuntimeError(f"no P26 data for frame_stack={frame_stack}")
    X = np.concatenate(xs)
    Ys = np.concatenate(y_score)
    Ya = np.concatenate(y_aux)
    Yf = np.concatenate(y_fire)
    W = np.concatenate(weights)
    print(f"P26 data: {len(used)} shards -> {len(X)} samples", flush=True)
    for phase, count in sorted(phase_counts.items()):
        multiplier = multipliers.get(phase, 1.0)
        print(f"  phase {phase}: {count} samples x{multiplier:g}",
              flush=True)
    return X, Ys, Ya, Yf, W


def _weighted_mean(loss, weights):
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def ranking_loss(pred, target, weights, rank_margin):
    import torch

    best = target.argmax(dim=1)
    pred_best = pred.gather(1, best.unsqueeze(1))
    losses = torch.relu(rank_margin - pred_best + pred)
    mask = torch.ones_like(losses)
    mask.scatter_(1, best.unsqueeze(1), 0.0)
    losses = losses * mask
    return _weighted_mean(losses, weights)


def train_model(frame_stack=4, epochs=12, batch=4096, width=1024,
                lr=3e-4, val_frac=0.05, rank_margin=0.03,
                aux_weight=0.25, fire_weight=0.35, rank_weight=0.20,
                fire_target_margin=0.0, fire_pos_weight_cap=50.0,
                init_path=None, include_phases=None,
                exclude_phase_prefixes=None, phase_weights=None,
                category_weights=None):
    import torch
    import torch.nn.functional as F

    X, Ys, Ya, Yf, W = load_data(
        frame_stack, fire_target_margin,
        include_phases=include_phases,
        exclude_phase_prefixes=exclude_phase_prefixes,
        phase_weights=phase_weights,
        category_weights=category_weights)
    n = len(X)
    n_val = max(1, min(n - 1, int(n * val_frac)))
    perm = np.random.default_rng(2600).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xv = torch.as_tensor(X[val_idx])
    Ysv = torch.as_tensor(Ys[val_idx])
    Yav = torch.as_tensor(Ya[val_idx])
    Yfv = torch.as_tensor(Yf[val_idx])
    Wv = torch.as_tensor(W[val_idx])
    Xt = torch.as_tensor(X[train_idx])
    Yst = torch.as_tensor(Ys[train_idx])
    Yat = torch.as_tensor(Ya[train_idx])
    Yft = torch.as_tensor(Yf[train_idx])
    Wt = torch.as_tensor(W[train_idx])

    net = build_p26_net(X.shape[1], width)
    if init_path:
        payload = torch.load(init_path, weights_only=False)
        init_dim = int(payload["in_dim"])
        init_width = int(payload.get("width", 1024))
        if init_dim != X.shape[1] or init_width != width:
            raise RuntimeError(
                f"init net shape mismatch: {init_path} has "
                f"in_dim={init_dim} width={init_width}, expected "
                f"in_dim={X.shape[1]} width={width}")
        net.load_state_dict(payload["state_dict"])
        print(f"initialized from {init_path}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    fire_pos = Yft.float().mean(dim=0).clamp_min(1e-4)
    fire_pos_weight = ((1.0 - fire_pos) / fire_pos).clamp(
        min=1.0, max=fire_pos_weight_cap)
    print(f"P26 fire target margin {fire_target_margin:g}  "
          f"positive rate {Yft.float().mean().item():.2%}  "
          f"pos_weight mean {fire_pos_weight.mean().item():.1f}",
          flush=True)

    def batch_loss(x, y_score, y_aux, y_fire, weights):
        out = net(x)
        score_loss = _weighted_mean(
            F.mse_loss(out["score"], y_score, reduction="none"), weights)
        aux_loss = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                out["aux"], y_aux, reduction="none"), weights)
        fire_loss = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                out["fire"], y_fire, reduction="none",
                pos_weight=fire_pos_weight), weights)
        rank = ranking_loss(out["score"], y_score, weights, rank_margin)
        loss = (score_loss + aux_weight * aux_loss
                + fire_weight * fire_loss + rank_weight * rank)
        return loss, score_loss, aux_loss, fire_loss, rank, out

    def metrics():
        with torch.no_grad():
            loss, score_loss, aux_loss, fire_loss, rank, out = batch_loss(
                Xv, Ysv, Yav, Yfv, Wv)
            pred = out["score"]
            top1 = (pred.argmax(1) == Ysv.argmax(1)).float().mean().item()
            top3_idx = pred.topk(3, dim=1).indices
            hit3 = (top3_idx == Ysv.argmax(1, keepdim=True)).any(1)
            top3 = hit3.float().mean().item()
            fire_acc = (
                (torch.sigmoid(out["fire"]) > 0.5) == (Yfv > 0.5)
            ).float().mean().item()
            aux_acc = (
                (torch.sigmoid(out["aux"]) > 0.5) == (Yav > 0.5)
            ).float().mean().item()
        return (loss.item(), score_loss.item(), aux_loss.item(),
                fire_loss.item(), rank.item(), top1, top3, fire_acc, aux_acc)

    t0 = time.time()
    n_train = len(Xt)
    for epoch in range(epochs):
        order = torch.randperm(n_train)
        total, batches = 0.0, 0
        for start in range(0, n_train, batch):
            idx = order[start:start + batch]
            loss, *_ = batch_loss(
                Xt[idx], Yst[idx], Yat[idx], Yft[idx], Wt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        vals = metrics()
        print(f"  epoch {epoch+1}/{epochs} train {total/max(batches,1):.4f} "
              f"val {vals[0]:.4f} score {vals[1]:.4f} aux {vals[2]:.4f} "
              f"fire {vals[3]:.4f} rank {vals[4]:.4f} "
              f"top1 {vals[5]:.1%} top3 {vals[6]:.1%} "
              f"fire_acc {vals[7]:.1%} aux_acc {vals[8]:.1%} "
              f"{time.time()-t0:.0f}s", flush=True)
    return net, X.shape[1]


def save_model(net, path, in_dim, frame_stack, width):
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": in_dim,
        "frame_stack": frame_stack,
        "width": width,
        "aux_names": AUX_NAMES,
    }, path)
    print(f"saved {path}", flush=True)


def load_p26_network(path):
    import torch

    payload = torch.load(path, weights_only=False)
    in_dim = int(payload["in_dim"])
    width = int(payload.get("width", 1024))
    frame_stack = int(payload.get("frame_stack", max(1, in_dim // OBS_DIM)))
    network = build_p26_net(in_dim, width)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network, frame_stack


def select_action(outputs, candidates, fire_margin=0.0, fire_threshold=0.0,
                  kill_weight=0.10, death_weight=0.10,
                  double_death_weight=0.20, survive_weight=0.0,
                  fire_prob_weight=0.30):
    score = np.asarray(outputs["score"], dtype=np.float32)
    aux = _sigmoid(np.asarray(outputs["aux"], dtype=np.float32))
    fire_prob = _sigmoid(np.asarray(outputs["fire"], dtype=np.float32))
    value = (score
             + kill_weight * aux[:, 0]
             - death_weight * aux[:, 1]
             - double_death_weight * aux[:, 2]
             + survive_weight * aux[:, 4]
             + survive_weight * aux[:, 5])
    value[1::2] += fire_prob_weight * (fire_prob - 0.5)
    paired_score = score.reshape(9, 2)
    fire_ok = (paired_score[:, 1] - paired_score[:, 0]) > fire_margin
    if fire_threshold > 0.0:
        fire_ok = fire_ok & (fire_prob > fire_threshold)
    adjusted = value.copy()
    for movement in range(9):
        if not fire_ok[movement]:
            adjusted[movement * 2 + 1] = -1e9
    return candidates[int(adjusted.argmax())]


class P26Policy:
    name = "p26_amortized_mpc"

    def __init__(self, net_path=BEST_NET, fire_margin=0.0,
                 fire_threshold=0.0, kill_weight=0.10,
                 death_weight=0.10, double_death_weight=0.20,
                 survive_weight=0.0, fire_prob_weight=0.30,
                 fire_assist_line=0.0, fire_assist_max_risk=0.35,
                 fire_assist_min_delta=-0.03,
                 suppress_blind_fire_line=0.0):
        import torch
        from training.mpc_agent import CANDIDATES
        from training.tt_gym_env import TankTroubleGym

        self.torch = torch
        self.candidates = CANDIDATES
        self.network, self.frame_stack = load_p26_network(net_path)
        self.fire_margin = float(fire_margin)
        self.fire_threshold = float(fire_threshold)
        self.kill_weight = float(kill_weight)
        self.death_weight = float(death_weight)
        self.double_death_weight = float(double_death_weight)
        self.survive_weight = float(survive_weight)
        self.fire_prob_weight = float(fire_prob_weight)
        self.fire_assist_line = float(fire_assist_line)
        self.fire_assist_max_risk = float(fire_assist_max_risk)
        self.fire_assist_min_delta = float(fire_assist_min_delta)
        self.suppress_blind_fire_line = float(suppress_blind_fire_line)
        self.env = TankTroubleGym(seed=0, reward_version=1,
                                  obs_traj=True, obs_nav=True)
        self.game = None
        self.analyzer = None
        self.frames = 0
        self.history = []

    def reset(self):
        self.game = None
        self.analyzer = None
        self.frames = 0
        self.history = []

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer360(game)
            self.frames = 0
            self.history = []
        observation, metrics = build_observation(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        self.history.append(observation)
        stacked = stack_observation(self.history, self.frame_stack)
        with self.torch.no_grad():
            out = self.network(self.torch.as_tensor(stacked).unsqueeze(0))
        outputs = {
            "score": out["score"][0].numpy(),
            "aux": out["aux"][0].numpy(),
            "fire": out["fire"][0].numpy(),
        }
        throttle, turn, fire = select_action(
            outputs, self.candidates, self.fire_margin, self.fire_threshold,
            self.kill_weight, self.death_weight, self.double_death_weight,
            self.survive_weight, self.fire_prob_weight)
        if len(game.tanks) > 1 and game.tanks[1].alive:
            line, _, risk = [float(value) for value in metrics[:3]]
            movement = throttle * 3 + turn
            paired_score = outputs["score"].reshape(9, 2)
            fire_delta = float(paired_score[movement, 1]
                               - paired_score[movement, 0])
            if (fire == 1 and self.suppress_blind_fire_line > 0.0
                    and line < self.suppress_blind_fire_line):
                fire = 0
            if (fire == 0 and self.fire_assist_line > 0.0
                    and line >= self.fire_assist_line
                    and risk <= self.fire_assist_max_risk
                    and fire_delta >= self.fire_assist_min_delta):
                fire = 1
        elif len(game.tanks) > 1:
            fire = 0
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _eval_worker(job):
    (worker, net_path, seed, count, fire_margin, fire_threshold,
     kill_weight, death_weight, double_death_weight, survive_weight,
     fire_prob_weight, fire_assist_line, fire_assist_max_risk,
     fire_assist_min_delta, suppress_blind_fire_line) = job
    import torch
    torch.set_num_threads(1)
    from training.evaluate import play_round_dual_engine

    policy = P26Policy(net_path, fire_margin, fire_threshold,
                       kill_weight, death_weight, double_death_weight,
                       survive_weight, fire_prob_weight, fire_assist_line,
                       fire_assist_max_risk, fire_assist_min_delta,
                       suppress_blind_fire_line)
    return [play_round_dual_engine(policy, seed + index)
            for index in range(count)]


def evaluate(net_path, n, seed, workers, fire_margin=0.0,
             fire_threshold=0.0, kill_weight=0.10, death_weight=0.10,
             double_death_weight=0.20, survive_weight=0.0,
             fire_prob_weight=0.30, fire_assist_line=0.0,
             fire_assist_max_risk=0.35, fire_assist_min_delta=-0.03,
             suppress_blind_fire_line=0.0):
    base, remainder = divmod(n, workers)
    jobs, offset = [], 0
    for worker in range(workers):
        count = base + (1 if worker < remainder else 0)
        if count > 0:
            jobs.append((worker, net_path, seed + offset, count,
                         fire_margin, fire_threshold, kill_weight,
                         death_weight, double_death_weight, survive_weight,
                         fire_prob_weight, fire_assist_line,
                         fire_assist_max_risk, fire_assist_min_delta,
                         suppress_blind_fire_line))
            offset += count
    started = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rounds = [item for part in pool.map(_eval_worker, jobs)
                  for item in part]
    total = len(rounds)
    count = lambda key: sum(result["true_result"] == key
                            for result in rounds)
    shots = sum(result["shots"] for result in rounds)
    win = count("win") / total
    print(f"===== P26 {os.path.basename(net_path)} {total} games @{seed} "
          f"({time.time()-started:.0f}s) =====", flush=True)
    print(f"  true win {win:.1%}  loss {count('loss')/total:.1%}  "
          f"double death {count('double_death')/total:.1%}  "
          f"draw {count('draw')/total:.1%}", flush=True)
    print(f"  shots/game {shots/total:.1f}  "
          f"hit rate {sum(result['kills'] for result in rounds)/max(shots, 1):.1%}  "
          f"avg length {sum(result['frames'] for result in rounds)/total/25:.1f}s",
          flush=True)
    return win


def train_candidate(index, args):
    path = os.path.join(MODELS_DIR, f"p26_amortized_mpc_iter{index:02d}.pt")
    net, in_dim = train_model(
        frame_stack=args.frame_stack,
        epochs=args.epochs,
        batch=args.batch,
        width=args.width,
        lr=args.lr,
        rank_margin=args.rank_margin,
        aux_weight=args.aux_weight,
        fire_weight=args.fire_weight,
        rank_weight=args.rank_weight,
        fire_target_margin=args.fire_target_margin,
        fire_pos_weight_cap=args.fire_pos_weight_cap,
        init_path=args.init_net,
        include_phases=args.include_phases,
        exclude_phase_prefixes=args.exclude_phase_prefixes,
        phase_weights=args.phase_weights,
        category_weights=args.category_weights,
    )
    save_model(net, path, in_dim, args.frame_stack, args.width)
    return path


def run_pipeline(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    collect_phase("teacher", args.teacher_rounds, args.workers, 12_600_000,
                  "teacher", epsilon=args.epsilon,
                  frame_stack=args.frame_stack,
                  score_horizon=args.score_horizon,
                  fire_margin=args.fire_margin,
                  fire_target_margin=args.fire_target_margin,
                  fire_threshold=args.fire_threshold,
                  kill_weight=args.kill_weight,
                  death_weight=args.death_weight,
                  double_death_weight=args.double_death_weight,
                  survive_weight=args.survive_weight,
                  fire_prob_weight=args.fire_prob_weight,
                  result_win_weight=args.result_win_weight,
                  result_loss_weight=args.result_loss_weight,
                  result_double_death_weight=(
                      args.result_double_death_weight),
                  result_draw_weight=args.result_draw_weight)
    collect_phase("bootstrap_p25v3", args.p25v3_rounds, args.workers,
                  12_800_000, "p25v3", args.p25v3_net,
                  args.epsilon, args.frame_stack, args.score_horizon,
                  args.fire_margin, args.fire_target_margin,
                  args.fire_threshold, args.kill_weight, args.death_weight,
                  args.double_death_weight, args.survive_weight,
                  args.fire_prob_weight, args.result_win_weight,
                  args.result_loss_weight,
                  args.result_double_death_weight,
                  args.result_draw_weight)
    collect_phase("bootstrap_p22", args.p22_rounds, args.workers,
                  13_000_000, "p22", args.p22_net,
                  args.epsilon, args.frame_stack, args.score_horizon,
                  args.fire_margin, args.fire_target_margin,
                  args.fire_threshold, args.kill_weight, args.death_weight,
                  args.double_death_weight, args.survive_weight,
                  args.fire_prob_weight, args.result_win_weight,
                  args.result_loss_weight,
                  args.result_double_death_weight,
                  args.result_draw_weight)

    best_win, best_path = -1.0, None
    for iteration in range(args.dagger_rounds + 1):
        candidate = train_candidate(iteration, args)
        gate_win = evaluate(candidate, args.gate_n, args.gate_seed,
                            args.workers, args.fire_margin,
                            args.fire_threshold, args.kill_weight,
                            args.death_weight, args.double_death_weight,
                            args.survive_weight, args.fire_prob_weight)
        if gate_win > best_win:
            best_win, best_path = gate_win, candidate
            shutil.copyfile(candidate, BEST_NET)
            print(f"  promoted {best_win:.1%} -> {BEST_NET}", flush=True)
        else:
            print(f"  not promoted: candidate {gate_win:.1%} "
                  f"< best {best_win:.1%}", flush=True)
        if iteration < args.dagger_rounds:
            collect_phase(
                f"dagger_{iteration + 1:02d}", args.rounds_per_dagger,
                args.workers, 13_200_000 + iteration * 200_000,
                "p26", BEST_NET, args.epsilon, args.frame_stack,
                args.score_horizon, args.fire_margin,
                args.fire_target_margin, args.fire_threshold,
                args.kill_weight, args.death_weight,
                args.double_death_weight, args.survive_weight,
                args.fire_prob_weight, args.result_win_weight,
                args.result_loss_weight, args.result_double_death_weight,
                args.result_draw_weight)

    print(f"===== P26 final champion: {best_path} gate {best_win:.1%} =====",
          flush=True)
    evaluate(BEST_NET, args.eval_n, args.eval_seed, args.workers,
             args.fire_margin, args.fire_threshold, args.kill_weight,
             args.death_weight, args.double_death_weight,
             args.survive_weight, args.fire_prob_weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "train", "eval",
                                         "pipeline"])
    parser.add_argument("--phase", default="manual")
    parser.add_argument("--actor-kind",
                        choices=["teacher", "p22", "p25v2", "p25v3",
                                 "p26", "random"],
                        default="p25v3")
    parser.add_argument("--actor-net", default=None)
    parser.add_argument("--p22-net", default=P22_NET)
    parser.add_argument("--p25v3-net", default=P25V2_NET)
    parser.add_argument("--net", default=BEST_NET)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--teacher-rounds", type=int, default=64)
    parser.add_argument("--p25v3-rounds", type=int, default=128)
    parser.add_argument("--p22-rounds", type=int, default=64)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--rounds-per-dagger", type=int, default=128)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--score-horizon", type=int, default=HORIZON)
    parser.add_argument("--fire-margin", type=float, default=0.0)
    parser.add_argument("--fire-threshold", type=float, default=0.0)
    parser.add_argument("--fire-assist-line", type=float, default=0.0,
                        help="If >0, force fire on clear safe lines.")
    parser.add_argument("--fire-assist-max-risk", type=float, default=0.35)
    parser.add_argument("--fire-assist-min-delta", type=float, default=-0.03)
    parser.add_argument("--suppress-blind-fire-line", type=float, default=0.0,
                        help="If >0, suppress fire below this line score.")
    parser.add_argument("--fire-target-margin", type=float, default=0.0)
    parser.add_argument("--fire-pos-weight-cap", type=float, default=50.0)
    parser.add_argument("--init-net", default=None)
    parser.add_argument("--include-phases", default=None,
                        help="Comma-separated phase names to train on.")
    parser.add_argument("--exclude-phase-prefixes", default=None,
                        help="Comma-separated phase prefixes to skip.")
    parser.add_argument("--phase-weights", default=None,
                        help="Comma-separated phase=weight multipliers.")
    parser.add_argument("--category-weights", default=None,
                        help="Comma-separated failure_category=weight "
                             "multipliers for shards that include category.")
    parser.add_argument("--kill-weight", type=float, default=0.10)
    parser.add_argument("--death-weight", type=float, default=0.10)
    parser.add_argument("--double-death-weight", type=float, default=0.20)
    parser.add_argument("--survive-weight", type=float, default=0.0)
    parser.add_argument("--fire-prob-weight", type=float, default=0.30)
    parser.add_argument("--result-win-weight", type=float, default=1.0)
    parser.add_argument("--result-loss-weight", type=float, default=1.0)
    parser.add_argument("--result-double-death-weight", type=float,
                        default=1.0)
    parser.add_argument("--result-draw-weight", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=0.03)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--fire-weight", type=float, default=0.35)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=973000)
    parser.add_argument("--gate-n", type=int, default=80)
    parser.add_argument("--gate-seed", type=int, default=9_700_000)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=973000)
    args = parser.parse_args()

    if args.mode == "collect":
        collect_phase(args.phase, args.rounds, args.workers, 12_500_000,
                      args.actor_kind, args.actor_net, args.epsilon,
                      args.frame_stack, args.score_horizon,
                      args.fire_margin, args.fire_target_margin,
                      args.fire_threshold, args.kill_weight,
                      args.death_weight, args.double_death_weight,
                      args.survive_weight, args.fire_prob_weight,
                      args.result_win_weight, args.result_loss_weight,
                      args.result_double_death_weight,
                      args.result_draw_weight)
    elif args.mode == "train":
        train_candidate(args.index, args)
    elif args.mode == "eval":
        evaluate(args.net, args.n, args.seed, args.workers,
                 args.fire_margin, args.fire_threshold, args.kill_weight,
                 args.death_weight, args.double_death_weight,
                 args.survive_weight, args.fire_prob_weight,
                 args.fire_assist_line, args.fire_assist_max_risk,
                 args.fire_assist_min_delta, args.suppress_blind_fire_line)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
