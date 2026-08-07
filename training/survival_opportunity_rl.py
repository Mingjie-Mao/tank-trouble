"""P29: survival opportunity RL with disciplined conditional firing.

P28 taught movement but not why to move.  P29 retains its frontier state and
adds observable opportunity facts from P25.  The 18 legacy scores are decoded
as a 9-way movement distribution plus a conditional fire distribution.  Fire
is masked unless a credible, non-suicidal shot is visible.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mpc_agent import CANDIDATES
from training.opportunity_distill_v2 import ready_line
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.score_distill import build_net
from training.survival_distill_v2 import legacy_econ
from training.survival_expert_iter_530 import DECIDE_EVERY, apply_action
from training.survival_frontier_rl import (
    FPS,
    FRONTIER_OBS_DIM,
    FrontierState,
    frontier_observation,
)
from training.survival_rl_warmstart import WarmStartActorCritic


HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_WARMSTART = os.path.join(
    MODELS_DIR, "p28_dense_frontier_actor_iter04.pt")
DEFAULT_ACTOR = os.path.join(MODELS_DIR, "p29_opportunity_actor.pt")
DEFAULT_CHECKPOINT = os.path.join(
    MODELS_DIR, "p29_opportunity_checkpoint.pt")
DEFAULT_REPLAY = os.path.join(HERE, "p29_attack_replay.npz")
DEFAULT_TEACHER = os.path.join(MODELS_DIR, "best_model.pt")

OPPORTUNITY_METRICS_DIM = 5
FIRE_FACTS_DIM = 6
OPPORTUNITY_DIM = OPPORTUNITY_METRICS_DIM + FIRE_FACTS_DIM
P29_OBS_DIM = FRONTIER_OBS_DIM + OPPORTUNITY_DIM
FIRE_FACTS_OFFSET = FRONTIER_OBS_DIM + OPPORTUNITY_METRICS_DIM

REWARD_SCALE = 50.0
TERMINAL_PENALTY = 20.0
FRONTIER_PROGRESS_REWARD = 0.05
UNIQUE_CELL_REWARD = 0.10
OPPORTUNITY_POTENTIAL_REWARD = 0.02
HIT_EVENT_REWARD = 2.0
GOOD_FIRE_REWARD = 0.40
PRESSURE_FIRE_REWARD = 0.10
WASTED_FIRE_PENALTY = 0.10
SUICIDE_FIRE_PENALTY = 1.0
FIRE_READY_THRESHOLD = 0.55
PRESSURE_THRESHOLD = 0.25
ATTACK_CONTEXT_STEPS = 10


def shot_facts(game, metrics):
    from tank_trouble_original.laika import LaikaAI

    tank = game.tanks[0]
    weapon_ready = bool(game.weapon_ready(tank))
    trigger_ready = bool(tank.trigger_released)
    assessment = None
    if tank.alive and weapon_ready and trigger_ready:
        assessment = LaikaAI(game, tank).check_bullet_path(tank.rotation)
    result = None if assessment is None else assessment.get("result")
    closest = float("inf") if assessment is None else float(
        assessment.get("closest", float("inf")))
    pressure = max(0.0, 1.0 - closest / max(0.75 * game.scale, 1e-9))
    return np.asarray([
        ready_line(metrics),
        float(weapon_ready),
        float(trigger_ready),
        float(result == "HIT"),
        float(result == "SUICIDE"),
        pressure,
    ], dtype=np.float32)


def opportunity_observation(encoder, game, ledger, econ, frontier, analyzer):
    base = frontier_observation(encoder, game, ledger, econ, frontier)
    metrics = analyzer.metrics(game)
    fire_facts = shot_facts(game, metrics)
    return np.concatenate([base, metrics, fire_facts]), metrics, fire_facts


def load_expanded_warmstart(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    source_dim = int(payload.get("in_dim", FRONTIER_OBS_DIM))
    if source_dim != FRONTIER_OBS_DIM:
        raise ValueError(f"P29 warm-start expected {FRONTIER_OBS_DIM}, got {source_dim}")
    source = payload["state_dict"]
    model = WarmStartActorCritic(in_dim=P29_OBS_DIM).to(device)
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.weight[:, :source_dim].copy_(source["0.weight"])
        model.fc1.bias.copy_(source["0.bias"])
        model.fc2.weight.copy_(source["2.weight"])
        model.fc2.bias.copy_(source["2.bias"])
        model.fc3.weight.copy_(source["4.weight"])
        model.fc3.bias.copy_(source["4.bias"])
        model.actor.weight.copy_(source["6.weight"])
        model.actor.bias.copy_(source["6.bias"])
        nn.init.orthogonal_(model.value.weight, gain=1.0)
        nn.init.zeros_(model.value.bias)
    return model


def verify_warmstart(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    reference = build_net(FRONTIER_OBS_DIM).to(device)
    reference.load_state_dict(payload["state_dict"])
    base = torch.randn(8, FRONTIER_OBS_DIM, device=device)
    extra = torch.randn(8, OPPORTUNITY_DIM, device=device)
    with torch.no_grad():
        expected = reference(base)
        actual, _ = model(torch.cat([base, extra], dim=1))
    difference = (expected - actual).abs().max().item()
    if difference != 0.0:
        raise RuntimeError(f"P29 warm-start mismatch: {difference}")
    return difference


def factorized_distribution(model, observations, temperature):
    hidden = model.features(observations)
    scores = model.actor(hidden)
    values = model.value(hidden.detach()).squeeze(-1)
    paired = scores.reshape(-1, 9, 2)
    facts = observations[:, FIRE_FACTS_OFFSET:]
    fire_allowed = (
        (facts[:, 0] >= FIRE_READY_THRESHOLD)
        & (facts[:, 1] > 0.5)
        & (facts[:, 2] > 0.5)
        & (facts[:, 4] < 0.5)
        & ((facts[:, 3] > 0.5) | (facts[:, 5] >= PRESSURE_THRESHOLD)))
    force_fire = facts[:, 3] > 0.5
    fire_logits = paired.clone()
    fire_logits[:, :, 1] = torch.where(
        fire_allowed[:, None], fire_logits[:, :, 1],
        torch.full_like(fire_logits[:, :, 1], -1e9))
    fire_logits[:, :, 0] = torch.where(
        force_fire[:, None],
        torch.full_like(fire_logits[:, :, 0], -1e9),
        fire_logits[:, :, 0])
    move_logits = paired.max(dim=2).values / temperature
    move_log_probs = torch.log_softmax(move_logits, dim=1)
    fire_logits = fire_logits / temperature
    fire_log_probs = torch.log_softmax(fire_logits, dim=2)
    joint_log_probs = move_log_probs[:, :, None] + fire_log_probs
    return Categorical(logits=joint_log_probs.reshape(-1, 18)), values


class OpportunitySurvivalEnv:
    def __init__(self, seed, cap_seconds, start_pool,
                 terminal_penalty=TERMINAL_PENALTY):
        from training.tt_gym_env import TankTroubleGym

        self.seed_rng = random.Random(seed)
        self.encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        self.econ = dict(
            legacy_econ(), cap=int(cap_seconds * FPS), start=float(start_pool))
        self.terminal_penalty = terminal_penalty
        self.game = None
        self.ledger = None
        self.frontier = None
        self.analyzer = None
        self.observation = None
        self.metrics = None
        self.fire_facts = None
        self.episode_return = 0.0
        self.shots = 0
        self.good_shots = 0
        self.wasted_shots = 0
        self.max_ready_line = 0.0

    def reset(self):
        from tank_trouble_original.game import Game
        from training.survival_mode import Ledger

        self.game = Game(
            seed=self.seed_rng.randrange(1 << 30), ai_enabled=True,
            invincible={1}, hit_immunity_frames={1: 0})
        self.ledger = Ledger(self.game, self.econ)
        self.frontier = FrontierState(self.game, dense=True)
        self.analyzer = OpportunityAnalyzer360(self.game)
        self.episode_return = 0.0
        self.shots = self.good_shots = self.wasted_shots = 0
        self.max_ready_line = 0.0
        return self._observe()

    def _observe(self):
        self.observation, self.metrics, self.fire_facts = \
            opportunity_observation(
                self.encoder, self.game, self.ledger, self.econ,
                self.frontier, self.analyzer)
        self.max_ready_line = max(
            self.max_ready_line, float(self.fire_facts[0]))
        return self.observation

    def step(self, action_index):
        pool_before = self.ledger.pool
        hits_before = self.ledger.hits
        reward_events = 0.0
        frontier_progress = 0.0
        opportunity_progress = 0.0
        first_visit_event = False
        fired_event = False
        good_fire_event = False
        end = "alive"
        action = CANDIDATES[int(action_index)]
        for _ in range(DECIDE_EVERY):
            latched_target = self.frontier.target
            frontier_before = self.frontier.continuous_distance(
                self.game, latched_target)
            metrics_before = self.analyzer.metrics(self.game)
            potential_before = self.analyzer.potential(metrics_before)
            assessment = shot_facts(self.game, metrics_before)
            apply_action(self.game, action)
            events = self.game.step()
            end = self.ledger.on_frame(self.game, events)
            actual_fire = any(
                event[0] == "fire" and event[1] == 0 for event in events)
            if actual_fire:
                fired_event = True
                self.shots += 1
                if assessment[4] > 0.5:
                    reward_events -= SUICIDE_FIRE_PENALTY
                    self.wasted_shots += 1
                elif assessment[3] > 0.5:
                    reward_events += GOOD_FIRE_REWARD
                    self.good_shots += 1
                    good_fire_event = True
                elif assessment[5] > 0.0:
                    reward_events += PRESSURE_FIRE_REWARD
                    self.good_shots += 1
                    good_fire_event = True
                else:
                    reward_events -= WASTED_FIRE_PENALTY
                    self.wasted_shots += 1
            if end != "death":
                frontier_after = self.frontier.continuous_distance(
                    self.game, latched_target)
                local_frontier = frontier_before - frontier_after
                _, first_visit = self.frontier.observe_position(self.game)
                metrics_after = self.analyzer.metrics(self.game)
                local_opportunity = self.analyzer.potential(metrics_after) \
                    - potential_before
                if local_frontier > 0 and self.game.tanks[0].hit_something:
                    local_frontier = 0.0
                reward_events += FRONTIER_PROGRESS_REWARD * local_frontier
                reward_events += UNIQUE_CELL_REWARD * float(first_visit)
                reward_events += OPPORTUNITY_POTENTIAL_REWARD \
                    * local_opportunity
                frontier_progress += local_frontier
                opportunity_progress += local_opportunity
                first_visit_event = first_visit_event or first_visit
            if end != "alive":
                break

        hit_event = self.ledger.hits > hits_before
        if hit_event:
            reward_events += HIT_EVENT_REWARD
        reward = (self.ledger.pool - pool_before) / REWARD_SCALE + reward_events
        failure_adjustment = self.terminal_penalty \
            - self.econ["start"] / REWARD_SCALE
        if end == "death":
            reward -= self.ledger.pool / REWARD_SCALE + failure_adjustment
        elif end == "drain":
            reward -= failure_adjustment
        elif end == "cap":
            reward += self.econ["start"] / REWARD_SCALE
        self.episode_return += reward

        done = end != "alive"
        observation = None if done else self._observe()
        info = {
            "frontier_progress": frontier_progress,
            "opportunity_progress": opportunity_progress,
            "first_visit": first_visit_event,
            "fired": fired_event,
            "good_fire": good_fire_event,
            "hit": hit_event,
        }
        if done:
            info.update({
                "end": end,
                "return": self.episode_return,
                "frames": self.ledger.frames,
                "hits": self.ledger.hits,
                "shots": self.shots,
                "good_shots": self.good_shots,
                "wasted_shots": self.wasted_shots,
                "max_ready_line": self.max_ready_line,
                "stuck_frames": self.ledger.stuck_frames,
                "unique_cells": int(self.frontier.visited.sum()),
                "pool": self.ledger.pool,
            })
        return observation, float(reward), done, info


class AttackReplay:
    def __init__(self, path, max_steps=50_000):
        self.path = path
        self.max_steps = max_steps
        self.observations = np.empty((0, P29_OBS_DIM), dtype=np.float32)
        self.actions = np.empty(0, dtype=np.int64)
        if path and os.path.exists(path):
            data = np.load(path)
            if data["observations"].shape[1] == P29_OBS_DIM:
                self.observations = data["observations"].astype(np.float32)
                self.actions = data["actions"].astype(np.int64)

    def __len__(self):
        return len(self.actions)

    def add(self, trajectory, repeats=1):
        useful = [index for index, item in enumerate(trajectory) if item[2]]
        if not useful:
            return
        selected = set()
        for index in useful:
            selected.update(range(max(0, index - ATTACK_CONTEXT_STEPS), index + 1))
        items = [trajectory[index] for index in sorted(selected)]
        observations = np.stack([item[0] for item in items])
        actions = np.asarray([item[1] for item in items], dtype=np.int64)
        for _ in range(repeats):
            self.observations = np.concatenate([self.observations, observations])
            self.actions = np.concatenate([self.actions, actions])
        if len(self) > self.max_steps:
            self.observations = self.observations[-self.max_steps:]
            self.actions = self.actions[-self.max_steps:]

    def sample(self, count, device):
        if not len(self):
            return None
        indices = np.random.randint(0, len(self), min(count, len(self)))
        return (
            torch.as_tensor(
                self.observations[indices], dtype=torch.float32, device=device),
            torch.as_tensor(
                self.actions[indices], dtype=torch.long, device=device),
        )

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez_compressed(
            self.path, observations=self.observations, actions=self.actions)


def action_index(action):
    throttle = 2 if action.get("forward") else 0 if action.get("backup") else 1
    turn = 0 if action.get("turn_left") else 2 if action.get("turn_right") else 1
    return CANDIDATES.index((throttle, turn, int(bool(action.get("fire")))))


def bootstrap_attack_replay(replay, args):
    if args.bootstrap_rounds <= 0:
        return
    from training.opportunity_distill_v2 import OpportunityScoreNetPolicyV2

    episodes_with_hits = 0
    retained_before = len(replay)
    for episode in range(args.bootstrap_rounds):
        env = OpportunitySurvivalEnv(
            args.seed + 70_000_000 + episode,
            args.cap_seconds, args.start_pool, args.terminal_penalty)
        observation = env.reset()
        teacher = OpportunityScoreNetPolicyV2(args.teacher_net)
        teacher.reset()
        trajectory = []
        while True:
            chosen = action_index(teacher.act(env.game))
            next_observation, _, done, info = env.step(chosen)
            useful = bool(
                info["opportunity_progress"] > 0.05
                or info["good_fire"] or info["hit"])
            trajectory.append([observation.copy(), chosen, useful])
            if done:
                if info["hits"] > 0:
                    episodes_with_hits += 1
                    replay.add(trajectory, repeats=3)
                break
            observation = next_observation
    replay.save()
    print(
        f"teacher bootstrap: rounds={args.bootstrap_rounds} "
        f"hit_episodes={episodes_with_hits} "
        f"replay_added={len(replay)-retained_before}", flush=True)


def collect_rollout(model, envs, observations, pending, replay, args, device):
    buffers = {key: [] for key in (
        "observations", "actions", "log_probs", "rewards", "dones", "values")}
    episodes = []
    retained = 0
    for _ in range(args.rollout_steps):
        tensor = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device)
        with torch.no_grad():
            dist, values = factorized_distribution(
                model, tensor, args.temperature)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        rewards = np.empty(len(envs), dtype=np.float32)
        dones = np.empty(len(envs), dtype=np.float32)
        next_observations = []
        for index, env in enumerate(envs):
            action = actions[index].item()
            pending[index].append([observations[index].copy(), action, False])
            next_obs, reward, done, info = env.step(action)
            pending[index][-1][2] = bool(
                info["opportunity_progress"] > 0.05
                or info["good_fire"] or info["hit"])
            rewards[index] = reward
            dones[index] = float(done)
            if done:
                episodes.append(info)
                if info["hits"] > 0:
                    repeats = 3 if info["end"] == "cap" and info["hits"] > 0 \
                        else 2 if info["hits"] > 0 else 1
                    before = len(replay)
                    replay.add(pending[index], repeats)
                    retained += len(replay) - before
                pending[index] = []
                next_obs = env.reset()
            next_observations.append(next_obs)
        buffers["observations"].append(np.stack(observations))
        buffers["actions"].append(actions.cpu().numpy())
        buffers["log_probs"].append(log_probs.cpu().numpy())
        buffers["rewards"].append(rewards)
        buffers["dones"].append(dones)
        buffers["values"].append(values.cpu().numpy())
        observations = next_observations

    with torch.no_grad():
        final_tensor = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device)
        _, final_values = model(final_tensor)
    rewards = np.stack(buffers["rewards"])
    dones = np.stack(buffers["dones"])
    values = np.stack(buffers["values"])
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
    rollout = {
        "observations": np.concatenate(buffers["observations"]),
        "actions": np.concatenate(buffers["actions"]),
        "log_probs": np.concatenate(buffers["log_probs"]),
        "advantages": advantages.reshape(-1),
        "returns": (advantages + values).reshape(-1),
        "values": values.reshape(-1),
    }
    return rollout, observations, pending, episodes, retained


def update_ppo(model, optimizer, rollout, replay, args, device, value_only):
    observations = torch.as_tensor(
        rollout["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(
        rollout["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(
        rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(
        rollout["returns"], dtype=torch.float32, device=device)
    old_values = torch.as_tensor(
        rollout["values"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    totals = dict(policy=0.0, value=0.0, entropy=0.0, bc=0.0, kl=0.0, n=0)
    size = len(observations)
    for _ in range(args.epochs):
        order = torch.randperm(size, device=device)
        for start in range(0, size, args.minibatch):
            indices = order[start:start + args.minibatch]
            if value_only:
                with torch.no_grad():
                    hidden = model.features(observations[indices])
                    scores = model.actor(hidden)
                    dist, _ = factorized_distribution(
                        model, observations[indices], args.temperature)
                values = model.value(hidden.detach()).squeeze(-1)
            else:
                dist, values = factorized_distribution(
                    model, observations[indices], args.temperature)
            new_log_probs = dist.log_prob(actions[indices])
            log_ratio = new_log_probs - old_log_probs[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(
                ratio * advantages[indices],
                torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                * advantages[indices]).mean()
            clipped_values = old_values[indices] + torch.clamp(
                values - old_values[indices], -args.value_clip, args.value_clip)
            value_loss = 0.5 * torch.max(
                (values - returns[indices]).pow(2),
                (clipped_values - returns[indices]).pow(2)).mean()
            entropy = dist.entropy().mean()
            bc_loss = torch.zeros((), device=device)
            if not value_only:
                replay_batch = replay.sample(args.bc_batch, device)
                if replay_batch is not None:
                    replay_obs, replay_actions = replay_batch
                    replay_dist, _ = factorized_distribution(
                        model, replay_obs, args.temperature)
                    bc_loss = -replay_dist.log_prob(replay_actions).mean()
            loss = args.value_coef * value_loss if value_only else \
                policy_loss + args.value_coef * value_loss \
                - args.entropy_coef * entropy + args.bc_coef * bc_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = ((ratio - 1) - log_ratio).mean()
            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            totals["bc"] += bc_loss.item()
            totals["kl"] += approximate_kl.item()
            totals["n"] += 1
        if not value_only and totals["kl"] / totals["n"] > args.target_kl:
            break
    count = max(totals.pop("n"), 1)
    return {key: value / count for key, value in totals.items()}


def summarize(episodes):
    if not episodes:
        return {"episodes": 0}
    frames = sum(x["frames"] for x in episodes)
    return {
        "episodes": len(episodes),
        "cap_pct": sum(x["end"] == "cap" for x in episodes) / len(episodes),
        "death_pct": sum(x["end"] == "death" for x in episodes) / len(episodes),
        "mean_cells": float(np.mean([x["unique_cells"] for x in episodes])),
        "mean_hits": float(np.mean([x["hits"] for x in episodes])),
        "mean_shots": float(np.mean([x["shots"] for x in episodes])),
        "wasted_shot_pct": sum(x["wasted_shots"] for x in episodes)
        / max(sum(x["shots"] for x in episodes), 1),
        "stuck_pct": sum(x["stuck_frames"] for x in episodes) / max(frames, 1),
    }


def greedy_probe(model, cap_seconds, start_pool, terminal_penalty, seed, device):
    env = OpportunitySurvivalEnv(seed, cap_seconds, start_pool, terminal_penalty)
    observation = env.reset()
    while True:
        tensor = torch.as_tensor(
            observation[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            dist, _ = factorized_distribution(model, tensor, 0.01)
            action = dist.probs.argmax(1).item()
        observation, _, done, info = env.step(action)
        if done:
            return info


def save_models(model, optimizer, replay, args, update, steps, metrics):
    econ = dict(legacy_econ(), cap=args.cap_seconds * FPS,
                start=float(args.start_pool))
    payload = {
        "state_dict": model.actor_state_dict(),
        "in_dim": P29_OBS_DIM,
        "version": "p29_survival_opportunity_rl",
        "econ": econ,
        "update": update,
        "total_steps": steps,
        "attack_replay_steps": len(replay),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(args.actor), exist_ok=True)
    torch.save(payload, args.actor)
    root, extension = os.path.splitext(args.actor)
    torch.save(payload, f"{root}_iter{update:02d}{extension}")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update": update,
        "total_steps": steps,
    }, args.checkpoint)
    replay.save()


def train_command(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    model = load_expanded_warmstart(args.warmstart, device)
    difference = verify_warmstart(model, args.warmstart, device)
    print(f"P29 warm-start exact: max_diff={difference:.1f}", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc1.weight.requires_grad_(True)
    model.value.weight.requires_grad_(True)
    model.value.bias.requires_grad_(True)

    def freeze_legacy_columns(gradient):
        gradient = gradient.clone()
        gradient[:, :FRONTIER_OBS_DIM] = 0.0
        return gradient

    model.fc1.weight.register_hook(freeze_legacy_columns)
    optimizer = torch.optim.Adam(
        [model.fc1.weight, model.value.weight, model.value.bias],
        lr=args.learning_rate)
    replay = AttackReplay(args.attack_replay, args.max_replay_steps)
    if not len(replay):
        bootstrap_attack_replay(replay, args)
    envs = [OpportunitySurvivalEnv(
        args.seed + index * 1009, args.cap_seconds, args.start_pool,
        args.terminal_penalty) for index in range(args.envs)]
    observations = [env.reset() for env in envs]
    pending = [[] for _ in envs]
    started = time.time()
    total_steps = 0
    for update in range(1, args.updates + 1):
        rollout, observations, pending, episodes, retained = collect_rollout(
            model, envs, observations, pending, replay, args, device)
        total_steps += args.envs * args.rollout_steps
        value_only = update <= args.value_warmup_updates
        losses = update_ppo(
            model, optimizer, rollout, replay, args, device, value_only)
        metrics = {**losses, **summarize(episodes)}
        greedy = greedy_probe(
            model, args.cap_seconds, args.start_pool,
            args.terminal_penalty, args.seed + 91_000_000, device)
        metrics.update({
            "greedy_end": greedy["end"],
            "greedy_cells": greedy["unique_cells"],
            "greedy_hits": greedy["hits"],
            "greedy_shots": greedy["shots"],
            "greedy_wasted_shots": greedy["wasted_shots"],
        })
        print(
            f"update {update}/{args.updates} steps={total_steps} "
            f"episodes={metrics.get('episodes', 0)} "
            f"hits={metrics.get('mean_hits', 0):.2f} "
            f"shots={metrics.get('mean_shots', 0):.2f} "
            f"waste={metrics.get('wasted_shot_pct', 0):.1%} "
            f"attack_replay={len(replay)}(+{retained}) "
            f"greedy={greedy['end']}/{greedy['unique_cells']}c/"
            f"{greedy['hits']}h/{greedy['shots']}s "
            f"bc={losses['bc']:.3f} kl={losses['kl']:.4f} "
            f"elapsed={time.time()-started:.0f}s", flush=True)
        save_models(model, optimizer, replay, args, update, total_steps, metrics)
    print(f"actor: {args.actor}\nattack replay: {args.attack_replay}", flush=True)


class OpportunityRLPolicy:
    name = "p29_survival_opportunity"

    def __init__(self, model_path):
        from training.tt_gym_env import TankTroubleGym

        payload = torch.load(model_path, weights_only=True)
        self.network = build_net(int(payload["in_dim"]))
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.econ = payload.get("econ", legacy_econ())
        self.encoder = TankTroubleGym(seed=0, obs_traj=True, obs_nav=True)
        self.game = None
        self.round_number = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.frontier = None
        self.analyzer = None
        self.last_action = (1, 1, 0)

    def reset(self):
        self.game = self.round_number = self.ledger = None
        self.context_game = self.context_round = None
        self.context_step = 0
        self.frontier = self.analyzer = None
        self.last_action = (1, 1, 0)

    def _dict(self):
        throttle, turn, fire = self.last_action
        return {
            "forward": throttle == 2, "backup": throttle == 0,
            "turn_left": turn == 0, "turn_right": turn == 2,
            "fire": fire == 1,
        }

    def act_ctx(self, game, ledger):
        if not game.tanks[0].alive:
            return {}
        if game is not self.context_game \
                or game.round_number != self.context_round:
            self.context_game = game
            self.context_round = game.round_number
            self.context_step = 0
            self.frontier = FrontierState(game, dense=True)
            self.analyzer = OpportunityAnalyzer360(game)
            self.last_action = (1, 1, 0)
        else:
            self.frontier.observe_position(game)
        if self.context_step % DECIDE_EVERY == 0:
            observation, _, _ = opportunity_observation(
                self.encoder, game, ledger, self.econ,
                self.frontier, self.analyzer)
            tensor = torch.as_tensor(observation)[None]
            with torch.no_grad():
                scores = self.network(tensor)
                paired = scores.reshape(1, 9, 2)
                fire_facts = tensor[:, FIRE_FACTS_OFFSET:]
                allowed = bool(
                    fire_facts[0, 0] >= FIRE_READY_THRESHOLD
                    and fire_facts[0, 1] > 0.5
                    and fire_facts[0, 2] > 0.5
                    and fire_facts[0, 4] < 0.5
                    and (fire_facts[0, 3] > 0.5
                         or fire_facts[0, 5] >= PRESSURE_THRESHOLD))
                movement = int(paired.max(2).values.argmax(1))
                forced = bool(fire_facts[0, 3] > 0.5)
                fire = int(forced or (
                    allowed and paired[0, movement, 1]
                    > paired[0, movement, 0]))
            self.last_action = CANDIDATES[movement * 2 + fire]
        self.context_step += 1
        return self._dict()

    def act(self, game):
        from training.survival_mode import Ledger

        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self.ledger = Ledger(game, self.econ)
        else:
            end = self.ledger.on_frame(game, game.events)
            if end in ("drain", "cap"):
                self.ledger = Ledger(game, self.econ)
        return self.act_ctx(game, self.ledger)


def smoke_command(args):
    device = torch.device(args.device)
    model = load_expanded_warmstart(args.warmstart, device)
    difference = verify_warmstart(model, args.warmstart, device)
    result = greedy_probe(
        model, args.cap_seconds, args.start_pool,
        args.terminal_penalty, args.seed, device)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print(f"smoke ok: obs={P29_OBS_DIM} warmstart_diff={difference:.1f}")


def add_common(parser):
    parser.add_argument("--warmstart", default=DEFAULT_WARMSTART)
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--attack-replay", default=DEFAULT_REPLAY)
    parser.add_argument("--cap-seconds", type=int, default=12)
    parser.add_argument("--start-pool", type=float, default=80.0)
    parser.add_argument("--terminal-penalty", type=float, default=TERMINAL_PENALTY)
    parser.add_argument("--seed", type=int, default=29_000_001)
    parser.add_argument("--device", default="cpu")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    add_common(smoke)
    smoke.set_defaults(func=smoke_command)
    train = subparsers.add_parser("train")
    add_common(train)
    train.add_argument("--updates", type=int, default=4)
    train.add_argument("--envs", type=int, default=8)
    train.add_argument("--rollout-steps", type=int, default=256)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--temperature", type=float, default=0.05)
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--gae-lambda", type=float, default=0.97)
    train.add_argument("--clip", type=float, default=0.1)
    train.add_argument("--value-clip", type=float, default=0.2)
    train.add_argument("--value-coef", type=float, default=0.05)
    train.add_argument("--entropy-coef", type=float, default=0.003)
    train.add_argument("--bc-coef", type=float, default=0.08)
    train.add_argument("--bc-batch", type=int, default=256)
    train.add_argument("--max-replay-steps", type=int, default=50_000)
    train.add_argument("--bootstrap-rounds", type=int, default=16)
    train.add_argument("--teacher-net", default=DEFAULT_TEACHER)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--target-kl", type=float, default=0.02)
    train.add_argument("--value-warmup-updates", type=int, default=1)
    train.add_argument("--torch-threads", type=int, default=4)
    train.set_defaults(func=train_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
