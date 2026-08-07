"""P26: warm-start PPO from the P24 Replica-530 iter03 actor.

The survival economy is the proven P24v2 variant.  The 530-dimensional
observation and the 18 joint actions stay unchanged, so the initial greedy
policy is exactly iter03.  PPO adds only a value head and then optimizes the
actor on states reached by the actor itself.

Examples:
  python3 training/survival_rl_warmstart.py smoke
  python3 training/survival_rl_warmstart.py train --updates 8
  python3 training/survival_rl_warmstart.py eval \
    --actor training/models/p26_survival_rl_actor.pt --survival-n 40 \
    --original-n 100
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.distributions import Categorical

from training.mpc_agent import CANDIDATES
from training.survival_distill_v2 import OBS_DIM, legacy_econ
from training.survival_expert_iter_530 import (
    DECIDE_EVERY,
    SurvivalReplica530Policy,
    apply_action,
    build_observation,
)


HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_WARMSTART = os.path.join(MODELS_DIR, "p24r530_iter03.pt")
DEFAULT_ACTOR = os.path.join(MODELS_DIR, "p26_survival_rl_actor.pt")
DEFAULT_CHECKPOINT = os.path.join(
    MODELS_DIR, "p26_survival_rl_checkpoint.pt")
REWARD_SCALE = 50.0
DEATH_PENALTY = 4.0
DRAIN_PENALTY = 2.0
CAP_BONUS = 2.0
EXPLORE_TERMINAL_PENALTY = 8.0
SURVIVAL_REWARD_PER_FRAME = 0.002
UNIQUE_CELL_REWARD = 0.08
NOVEL_PROGRESS_REWARD = 0.06


class WarmStartActorCritic(nn.Module):
    def __init__(self, in_dim=OBS_DIM, width=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.actor = nn.Linear(width, len(CANDIDATES))
        self.value = nn.Linear(width, 1)

    def features(self, observation):
        hidden = torch.relu(self.fc1(observation))
        hidden = torch.relu(self.fc2(hidden))
        return torch.relu(self.fc3(hidden))

    def forward(self, observation):
        hidden = self.features(observation)
        return self.actor(hidden), self.value(hidden).squeeze(-1)

    def actor_state_dict(self):
        return {
            "0.weight": self.fc1.weight.detach().cpu(),
            "0.bias": self.fc1.bias.detach().cpu(),
            "2.weight": self.fc2.weight.detach().cpu(),
            "2.bias": self.fc2.bias.detach().cpu(),
            "4.weight": self.fc3.weight.detach().cpu(),
            "4.bias": self.fc3.bias.detach().cpu(),
            "6.weight": self.actor.weight.detach().cpu(),
            "6.bias": self.actor.bias.detach().cpu(),
        }

    def load_actor_state_dict(self, state):
        with torch.no_grad():
            self.fc1.weight.copy_(state["0.weight"])
            self.fc1.bias.copy_(state["0.bias"])
            self.fc2.weight.copy_(state["2.weight"])
            self.fc2.bias.copy_(state["2.bias"])
            self.fc3.weight.copy_(state["4.weight"])
            self.fc3.bias.copy_(state["4.bias"])
            self.actor.weight.copy_(state["6.weight"])
            self.actor.bias.copy_(state["6.bias"])


def load_warmstart(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    in_dim = int(payload.get("in_dim", OBS_DIM))
    if in_dim != OBS_DIM:
        raise ValueError(f"expected {OBS_DIM} inputs, got {in_dim}")
    model = WarmStartActorCritic(in_dim=in_dim).to(device)
    model.load_actor_state_dict(payload["state_dict"])
    nn.init.orthogonal_(model.value.weight, gain=1.0)
    nn.init.zeros_(model.value.bias)
    return model, payload


def save_models(model, optimizer, args, update, total_steps, metrics):
    os.makedirs(os.path.dirname(args.actor), exist_ok=True)
    saved_args = {key: value for key, value in vars(args).items()
                  if key != "func"}
    actor_payload = {
        "state_dict": model.actor_state_dict(),
        "in_dim": OBS_DIM,
        "ledger_dim": 122,
        "version": "p26_survival_rl_actor",
        "warmstart": os.path.basename(args.warmstart),
        "update": update,
        "total_steps": total_steps,
        "metrics": metrics,
    }
    torch.save(actor_payload, args.actor)
    if update % args.snapshot_every == 0:
        actor_root, actor_extension = os.path.splitext(args.actor)
        snapshot_path = f"{actor_root}_iter{update:02d}{actor_extension}"
        torch.save(actor_payload, snapshot_path)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": saved_args,
        "update": update,
        "total_steps": total_steps,
        "metrics": metrics,
    }, args.checkpoint)


class SurvivalRLEnv:
    """One survival episode with an 18-way action every two frames."""

    def __init__(self, seed, reward_profile="explore",
                 terminal_penalty=EXPLORE_TERMINAL_PENALTY):
        from training.tt_gym_env import TankTroubleGym

        self.seed_rng = random.Random(seed)
        self.encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        self.econ = legacy_econ()
        self.reward_profile = reward_profile
        self.terminal_penalty = terminal_penalty
        self.game = None
        self.ledger = None
        self.explored_cells = None
        self.cell_entries = None
        self.max_cell_progress = None
        self.shaping_total = 0.0
        self.episode_return = 0.0

    def reset(self):
        from tank_trouble_original.game import Game
        from training.survival_mode import Ledger

        episode_seed = self.seed_rng.randrange(1 << 30)
        self.game = Game(
            seed=episode_seed, ai_enabled=True, invincible={1},
            hit_immunity_frames={1: self.econ["hit_immunity"]})
        self.ledger = Ledger(self.game, self.econ)
        tank = self.game.tanks[0]
        cell = self._cell(tank)
        self.explored_cells = {cell}
        self.cell_entries = {cell: (tank.x, tank.y)}
        self.max_cell_progress = {cell: 0.0}
        self.shaping_total = 0.0
        self.episode_return = 0.0
        return build_observation(
            self.encoder, self.game, self.ledger, self.econ)

    def _cell(self, tank):
        return int(tank.x // self.game.scale), int(tank.y // self.game.scale)

    def _exploration_reward(self):
        tank = self.game.tanks[0]
        if not tank.alive:
            return 0.0
        reward = SURVIVAL_REWARD_PER_FRAME
        cell = self._cell(tank)
        if cell not in self.explored_cells:
            self.explored_cells.add(cell)
            self.cell_entries[cell] = (tank.x, tank.y)
            self.max_cell_progress[cell] = 0.0
            reward += UNIQUE_CELL_REWARD

        entry_x, entry_y = self.cell_entries[cell]
        progress = min(1.0, math.hypot(
            tank.x - entry_x, tank.y - entry_y) / self.game.scale)
        previous = self.max_cell_progress[cell]
        if progress > previous and not tank.hit_something:
            reward += NOVEL_PROGRESS_REWARD * (progress - previous)
            self.max_cell_progress[cell] = progress
        return reward

    def step(self, action_index):
        pool_before = self.ledger.pool
        end = "alive"
        shaping_reward = 0.0
        action = CANDIDATES[int(action_index)]
        for _ in range(DECIDE_EVERY):
            apply_action(self.game, action)
            events = self.game.step()
            end = self.ledger.on_frame(self.game, events)
            if self.reward_profile == "explore":
                shaping_reward += self._exploration_reward()
            if end != "alive":
                break

        reward = (self.ledger.pool - pool_before) / REWARD_SCALE
        reward += shaping_reward
        if self.reward_profile == "explore":
            if end == "death":
                reward -= self.ledger.pool / REWARD_SCALE \
                    + self.terminal_penalty
            elif end == "drain":
                reward -= self.terminal_penalty
            elif end == "cap":
                reward += CAP_BONUS
        else:
            if end == "death":
                reward -= self.ledger.pool / REWARD_SCALE + DEATH_PENALTY
            elif end == "drain":
                reward -= DRAIN_PENALTY
            elif end == "cap":
                reward += CAP_BONUS
        self.shaping_total += shaping_reward
        self.episode_return += reward

        done = end != "alive"
        observation = None if done else build_observation(
            self.encoder, self.game, self.ledger, self.econ)
        info = None
        if done:
            info = {
                "end": end,
                "episode_return": self.episode_return,
                "frames": self.ledger.frames,
                "hits": self.ledger.hits,
                "stuck_frames": self.ledger.stuck_frames,
                "style": self.ledger.style,
                "pool": self.ledger.pool,
                "unique_cells": len(self.explored_cells),
                "shaping_total": self.shaping_total,
            }
        return observation, float(reward), done, info


def distribution(model, observations, temperature):
    actor_scores, values = model(observations)
    return Categorical(logits=actor_scores / temperature), values


def collect_rollout(model, envs, observations, args, device):
    obs_buffer = []
    action_buffer = []
    log_prob_buffer = []
    reward_buffer = []
    done_buffer = []
    value_buffer = []
    episodes = []

    for _ in range(args.rollout_steps):
        observation_tensor = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device)
        with torch.no_grad():
            dist, values = distribution(
                model, observation_tensor, args.temperature)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        rewards = np.empty(len(envs), dtype=np.float32)
        dones = np.empty(len(envs), dtype=np.float32)
        next_observations = []
        for index, env in enumerate(envs):
            next_obs, reward, done, info = env.step(actions[index].item())
            rewards[index] = reward
            dones[index] = float(done)
            if done:
                episodes.append(info)
                next_obs = env.reset()
            next_observations.append(next_obs)

        obs_buffer.append(np.stack(observations))
        action_buffer.append(actions.cpu().numpy())
        log_prob_buffer.append(log_probs.cpu().numpy())
        reward_buffer.append(rewards)
        done_buffer.append(dones)
        value_buffer.append(values.cpu().numpy())
        observations = next_observations

    with torch.no_grad():
        final_obs = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device)
        _, final_values = model(final_obs)

    rewards = np.stack(reward_buffer)
    dones = np.stack(done_buffer)
    values = np.stack(value_buffer)
    advantages = np.zeros_like(rewards)
    last_advantage = np.zeros(len(envs), dtype=np.float32)
    next_values = final_values.cpu().numpy()
    for step in reversed(range(args.rollout_steps)):
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + args.gamma * next_values * nonterminal \
            - values[step]
        last_advantage = delta + args.gamma * args.gae_lambda \
            * nonterminal * last_advantage
        advantages[step] = last_advantage
        next_values = values[step]
    returns = advantages + values

    rollout = {
        "observations": np.concatenate(obs_buffer),
        "actions": np.concatenate(action_buffer),
        "log_probs": np.concatenate(log_prob_buffer),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
        "values": values.reshape(-1),
    }
    return rollout, observations, episodes


def update_ppo(model, optimizer, rollout, args, device, value_only=False):
    observations = torch.as_tensor(
        rollout["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(
        rollout["actions"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(
        rollout["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(
        rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(
        rollout["returns"], dtype=torch.float32, device=device)
    old_values = torch.as_tensor(
        rollout["values"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / \
        (advantages.std() + 1e-8)

    size = len(observations)
    totals = dict(policy=0.0, value=0.0, entropy=0.0, kl=0.0, batches=0)
    for _ in range(args.epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, args.minibatch):
            indices = order[start:start + args.minibatch]
            if value_only:
                with torch.no_grad():
                    hidden = model.features(observations[indices])
                    actor_scores = model.actor(hidden)
                dist = Categorical(logits=actor_scores / args.temperature)
                values = model.value(hidden.detach()).squeeze(-1)
            else:
                dist, values = distribution(
                    model, observations[indices], args.temperature)
            new_log_probs = dist.log_prob(actions[indices])
            log_ratio = new_log_probs - old_log_probs[indices]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(
                ratio, 1.0 - args.clip, 1.0 + args.clip) \
                * advantages[indices]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_delta = values - old_values[indices]
            clipped_values = old_values[indices] + torch.clamp(
                value_delta, -args.value_clip, args.value_clip)
            value_loss = 0.5 * torch.max(
                (values - returns[indices]).pow(2),
                (clipped_values - returns[indices]).pow(2)).mean()
            entropy = dist.entropy().mean()
            loss = args.value_coef * value_loss if value_only else \
                policy_loss + args.value_coef * value_loss \
                - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            totals["kl"] += approximate_kl.item()
            totals["batches"] += 1
        if not value_only and totals["kl"] / totals["batches"] > args.target_kl:
            break
    batches = max(totals.pop("batches"), 1)
    return {key: value / batches for key, value in totals.items()}


def summarize_episodes(episodes):
    if not episodes:
        return {"episodes": 0}
    frames = sum(item["frames"] for item in episodes)
    hits = sum(item["hits"] for item in episodes)
    return {
        "episodes": len(episodes),
        "death_pct": sum(x["end"] == "death" for x in episodes)
        / len(episodes),
        "drain_pct": sum(x["end"] == "drain" for x in episodes)
        / len(episodes),
        "cap_pct": sum(x["end"] == "cap" for x in episodes)
        / len(episodes),
        "seconds_per_hit": frames / 25.0 / max(hits, 1),
        "stuck_pct": sum(x["stuck_frames"] for x in episodes)
        / max(frames, 1),
        "style_per_second": sum(x["style"] for x in episodes)
        / max(frames / 25.0, 1e-6),
        "mean_return": float(np.mean([
            x["episode_return"] for x in episodes])),
        "mean_unique_cells": float(np.mean([
            x["unique_cells"] for x in episodes])),
        "mean_shaping": float(np.mean([
            x["shaping_total"] for x in episodes])),
    }


def verify_warmstart(model, warmstart_path, device):
    from training.score_distill import build_net

    payload = torch.load(warmstart_path, map_location=device,
                         weights_only=True)
    reference = build_net(OBS_DIM).to(device)
    reference.load_state_dict(payload["state_dict"])
    probe = torch.randn(7, OBS_DIM, device=device)
    with torch.no_grad():
        expected = reference(probe)
        actual, _ = model(probe)
    difference = (expected - actual).abs().max().item()
    if difference != 0.0:
        raise RuntimeError(f"warm-start actor mismatch: {difference}")
    return difference


def train_command(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    start_update = 0
    total_steps = 0
    if args.resume:
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False)
        model = WarmStartActorCritic().to(device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_update = int(checkpoint.get("update", 0))
        total_steps = int(checkpoint.get("total_steps", 0))
        print(f"resume: update={start_update} steps={total_steps}", flush=True)
    else:
        model, _ = load_warmstart(args.warmstart, device)
        difference = verify_warmstart(model, args.warmstart, device)
        print(f"warm-start exact: max_diff={difference:.1f}", flush=True)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate)
    envs = [SurvivalRLEnv(
        args.seed + 1009 * index, reward_profile=args.reward_profile,
        terminal_penalty=args.terminal_penalty)
            for index in range(args.envs)]
    observations = [env.reset() for env in envs]
    started = time.time()
    final_update = start_update + args.updates
    for local_update in range(1, args.updates + 1):
        update = start_update + local_update
        rollout, observations, episodes = collect_rollout(
            model, envs, observations, args, device)
        total_steps += args.envs * args.rollout_steps
        value_only = not args.resume \
            and update <= args.value_warmup_updates
        losses = update_ppo(
            model, optimizer, rollout, args, device, value_only=value_only)
        episode_metrics = summarize_episodes(episodes)
        metrics = {**losses, **episode_metrics}
        phase = "value" if value_only else "ppo"
        print(
            f"update {update}/{final_update} [{phase}] steps={total_steps} "
            f"episodes={episode_metrics.get('episodes', 0)} "
            f"hit_s={episode_metrics.get('seconds_per_hit', math.inf):.2f} "
            f"cells={episode_metrics.get('mean_unique_cells', 0.0):.1f} "
            f"cap={episode_metrics.get('cap_pct', 0.0):.1%} "
            f"death={episode_metrics.get('death_pct', 0.0):.1%} "
            f"entropy={losses['entropy']:.3f} kl={losses['kl']:.5f} "
            f"elapsed={time.time()-started:.0f}s", flush=True)
        save_models(model, optimizer, args, update, total_steps, metrics)
    print(f"actor: {args.actor}\ncheckpoint: {args.checkpoint}", flush=True)


def survival_summary(actor_path, n, seed):
    from training.survival_mode import run_survival

    policy = SurvivalReplica530Policy(actor_path)
    econ = legacy_econ()
    rounds = [run_survival(policy, seed + index, econ=econ)
              for index in range(n)]
    frames = sum(item["frames"] for item in rounds)
    hits = sum(item["hits"] for item in rounds)
    return {
        "n": n,
        "death_pct": sum(x["end"] == "death" for x in rounds) / n,
        "drain_pct": sum(x["end"] == "drain" for x in rounds) / n,
        "cap_pct": sum(x["end"] == "cap" for x in rounds) / n,
        "seconds_per_hit": frames / 25.0 / max(hits, 1),
        "stuck_pct": sum(x["stuck_frames"] for x in rounds)
        / max(frames, 1),
        "style_per_second": sum(x["style"] for x in rounds)
        / max(frames / 25.0, 1e-6),
        "mean_settle": float(np.mean([x["settle"] for x in rounds])),
    }


def original_summary(actor_path, n, seed):
    from training.evaluate import evaluate_dual

    policy = SurvivalReplica530Policy(actor_path)
    return evaluate_dual(policy, n=n, base_seed=seed, verbose=True)


def eval_command(args):
    result = {
        "actor": args.actor,
        "survival": survival_summary(
            args.actor, args.survival_n, args.seed),
        "original": original_summary(
            args.actor, args.original_n, args.seed + 1_000_000),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def smoke_command(args):
    device = torch.device(args.device)
    model, _ = load_warmstart(args.warmstart, device)
    difference = verify_warmstart(model, args.warmstart, device)
    env = SurvivalRLEnv(
        args.seed, reward_profile=args.reward_profile,
        terminal_penalty=args.terminal_penalty)
    observation = env.reset()
    episode_reward = 0.0
    for _ in range(args.steps):
        tensor = torch.as_tensor(
            observation[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            scores, value = model(tensor)
        observation, reward, done, info = env.step(scores.argmax(1).item())
        episode_reward += reward
        if done:
            print("episode", info, "reward", episode_reward, flush=True)
            episode_reward = 0.0
            observation = env.reset()
    actor_path = "/tmp/p26_survival_rl_smoke_actor.pt"
    torch.save({
        "state_dict": model.actor_state_dict(),
        "in_dim": OBS_DIM,
        "ledger_dim": 122,
        "version": "p26_survival_rl_actor_smoke",
    }, actor_path)
    reloaded = SurvivalReplica530Policy(actor_path)
    reloaded.reset()
    print(
        f"smoke ok: obs={observation.shape} value={value.item():.3f} "
        f"warmstart_diff={difference:.1f} actor={actor_path}", flush=True)


def add_common_paths(parser):
    parser.add_argument("--warmstart", default=DEFAULT_WARMSTART)
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=26_000_003)
    parser.add_argument(
        "--reward-profile", choices=("explore", "settle"),
        default="explore")
    parser.add_argument(
        "--terminal-penalty", type=float,
        default=EXPLORE_TERMINAL_PENALTY)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    add_common_paths(smoke)
    smoke.add_argument("--steps", type=int, default=16)
    smoke.set_defaults(func=smoke_command)

    train = subparsers.add_parser("train")
    add_common_paths(train)
    train.add_argument("--updates", type=int, default=8)
    train.add_argument("--envs", type=int, default=8)
    train.add_argument("--rollout-steps", type=int, default=256)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--temperature", type=float, default=0.05)
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--gae-lambda", type=float, default=0.97)
    train.add_argument("--clip", type=float, default=0.1)
    train.add_argument("--value-clip", type=float, default=0.2)
    train.add_argument("--value-coef", type=float, default=0.5)
    train.add_argument("--entropy-coef", type=float, default=0.003)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--target-kl", type=float, default=0.02)
    train.add_argument("--value-warmup-updates", type=int, default=1)
    train.add_argument("--torch-threads", type=int, default=4)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--snapshot-every", type=int, default=1)
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("eval")
    add_common_paths(evaluate)
    evaluate.add_argument("--survival-n", type=int, default=40)
    evaluate.add_argument("--original-n", type=int, default=100)
    evaluate.set_defaults(func=eval_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
