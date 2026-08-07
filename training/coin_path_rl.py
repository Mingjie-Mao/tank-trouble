"""P35: PPO on the shared one-shot shortest-path coin economy."""

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.coin_path_rules import (
    CHAIN_WINDOW_FRAMES,
    HIT_LOSS,
    WALL_FINE,
    advance_coin_chains,
    build_coins,
    collect_coins,
    kill_bonus,
    neighbors,
)
from training.mpc_agent import CANDIDATES
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.score_distill import build_net
from training.survival_distill_v2 import legacy_econ
from training.survival_expert_iter_530 import DECIDE_EVERY, apply_action
from training.survival_frontier_rl import FPS, MAP_H, MAP_W, FrontierState
from training.survival_opportunity_rl import (
    P29_OBS_DIM,
    OpportunitySurvivalEnv,
    factorized_distribution,
    opportunity_observation,
    shot_facts,
    update_ppo,
)
from training.survival_rl_warmstart import WarmStartActorCritic


HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_WARMSTART = os.path.join(
    MODELS_DIR, "p29c_feature_only_actor_iter02.pt")
DEFAULT_ACTOR = os.path.join(MODELS_DIR, "p35_coin_actor.pt")
DEFAULT_CHECKPOINT = os.path.join(MODELS_DIR, "p35_coin_checkpoint.pt")
DEFAULT_REPLAY = os.path.join(HERE, "p35_coin_success_replay.npz")

COIN_MAP_DIM = MAP_W * MAP_H
LEGACY_COIN_ECON_DIM = 5
COIN_CHAIN_DIM = 4
COIN_ECON_DIM = LEGACY_COIN_ECON_DIM + COIN_CHAIN_DIM
COIN_ROUTE_DIM = 5
P35_V1_OBS_DIM = P29_OBS_DIM + COIN_MAP_DIM + LEGACY_COIN_ECON_DIM
P35_WAYPOINT_OBS_DIM = P35_V1_OBS_DIM + COIN_ROUTE_DIM
P35_EXTRA_DIM = COIN_MAP_DIM + COIN_ECON_DIM + COIN_ROUTE_DIM
P35_OBS_DIM = P29_OBS_DIM + P35_EXTRA_DIM
PASS_SCORE = 50.0
REWARD_SCALE = 50.0
OPPONENT_GAIN_WEIGHT = 1.0
DEFICIT_TIME_WEIGHT = 0.10
ACTIVE_KILL_BONUS = 0.8
OPPONENT_SELF_BONUS = 0.2
DEATH_PENALTY = 0.2
DOUBLE_DEATH_PENALTY = 1.0
TIMEOUT_PENALTY = 0.1
WALL_EVENT_PENALTY = 0.0
SAFE_SETTLE_FRAMES = FPS
COIN_ROUTE_REWARD = 0.12


class CoinLedger:
    def __init__(self, game, cap):
        self.pool = 0.0
        self.frames = 0
        self.cap = cap
        self.visited = {}
        cell = self._cell(game)
        self.visited[cell] = 0

    @staticmethod
    def _cell(game):
        tank = game.tanks[0]
        return int(tank.x // game.scale), int(tank.y // game.scale)

    def advance(self, game, bank):
        self.frames += 1
        self.pool = bank
        self.visited[self._cell(game)] = self.frames


def coin_features(coins, banks, initial_pool, frames, cap,
                  chain_counts, chain_timers):
    grid = np.zeros(COIN_MAP_DIM, dtype=np.float32)
    for (cell_x, cell_y), value in coins.items():
        if 0 <= cell_x < MAP_W and 0 <= cell_y < MAP_H:
            grid[cell_y * MAP_W + cell_x] = value
    remaining = sum(coins.values())
    econ = np.asarray([
        np.clip(banks[0] / 200.0, 0.0, 2.0),
        np.clip(banks[1] / 200.0, 0.0, 2.0),
        remaining / max(initial_pool, 1.0),
        frames / max(cap, 1),
        np.clip((banks[0] - banks[1]) / 100.0, -2.0, 2.0),
        np.clip(chain_counts[0] / 10.0, 0.0, 1.0),
        np.clip(chain_timers[0] / CHAIN_WINDOW_FRAMES, 0.0, 1.0),
        np.clip(chain_counts[1] / 10.0, 0.0, 1.0),
        np.clip(chain_timers[1] / CHAIN_WINDOW_FRAMES, 0.0, 1.0),
    ], dtype=np.float32)
    return np.concatenate([grid, econ])


def coin_route_features(game, target, distance, value):
    if target is None:
        return np.zeros(COIN_ROUTE_DIM, dtype=np.float32)
    tank = game.tanks[0]
    target_x = (target[0] + 0.5) * game.scale
    target_y = (target[1] + 0.5) * game.scale
    delta_x, delta_y = target_x - tank.x, target_y - tank.y
    forward = (tank.rotation - 90.0) * math.pi / 180.0
    cosine, sine = math.cos(forward), math.sin(forward)
    local_forward = (delta_x * cosine + delta_y * sine) / game.scale
    local_right = (-delta_x * sine + delta_y * cosine) / game.scale
    return np.asarray([
        np.clip(local_forward / 12.0, -1.0, 1.0),
        np.clip(local_right / 12.0, -1.0, 1.0),
        np.clip(distance / 20.0, 0.0, 1.0),
        value,
        1.0,
    ], dtype=np.float32)


def load_expanded_warmstart(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    source_dim = int(payload.get("in_dim", P29_OBS_DIM))
    source = payload["state_dict"]
    model = WarmStartActorCritic(in_dim=P35_OBS_DIM).to(device)
    if source_dim not in (
            P29_OBS_DIM, P35_V1_OBS_DIM,
            P35_WAYPOINT_OBS_DIM, P35_OBS_DIM):
        raise ValueError(
            f"P35 warm-start has unsupported input size {source_dim}")
    with torch.no_grad():
        model.fc1.weight.zero_()
        if source_dim == P35_WAYPOINT_OBS_DIM:
            route_start = P35_V1_OBS_DIM + COIN_CHAIN_DIM
            model.fc1.weight[:, :P35_V1_OBS_DIM].copy_(
                source["0.weight"][:, :P35_V1_OBS_DIM])
            model.fc1.weight[:, route_start:route_start + COIN_ROUTE_DIM].copy_(
                source["0.weight"][:, P35_V1_OBS_DIM:])
        else:
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
    source_dim = int(payload.get("in_dim", P29_OBS_DIM))
    reference = build_net(source_dim).to(device)
    reference.load_state_dict(payload["state_dict"])
    base = torch.randn(8, source_dim, device=device)
    with torch.no_grad():
        expected = reference(base)
        if source_dim == P35_OBS_DIM:
            model_input = base
        else:
            model_input = torch.randn(8, P35_OBS_DIM, device=device)
            if source_dim == P35_WAYPOINT_OBS_DIM:
                route_start = P35_V1_OBS_DIM + COIN_CHAIN_DIM
                model_input[:, :P35_V1_OBS_DIM] = \
                    base[:, :P35_V1_OBS_DIM]
                model_input[:, route_start:route_start + COIN_ROUTE_DIM] = \
                    base[:, P35_V1_OBS_DIM:]
            else:
                model_input[:, :source_dim] = base
        actual, _ = model(model_input)
    difference = (expected - actual).abs().max().item()
    if difference != 0.0:
        raise RuntimeError(f"P35 warm-start mismatch: {difference}")
    return difference


class CoinPathEnv(OpportunitySurvivalEnv):
    def __init__(self, seed, cap_seconds=30):
        from training.tt_gym_env import TankTroubleGym

        self.seed_rng = random.Random(seed)
        self.encoder = TankTroubleGym(
            seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
        self.cap = cap_seconds * FPS
        self.econ = dict(legacy_econ(), cap=self.cap, start=0.0)
        self.game = None
        self.ledger = None
        self.frontier = None
        self.analyzer = None
        self.observation = None

    def reset(self):
        from tank_trouble_original.game import Game

        self.game = Game(
            seed=self.seed_rng.randrange(1 << 30), ai_enabled=True)
        self.coins, self.path = build_coins(self.game)
        self.initial_pool = sum(self.coins.values())
        self.banks = [0.0, 0.0]
        self.counts = [0, 0]
        self.picked_values = [0.0, 0.0]
        self.chain_counts = [0, 0]
        self.chain_timers = [0, 0]
        self.max_chains = [0, 0]
        collect_coins(
            self.game, self.coins, self.banks,
            self.counts, self.picked_values,
            self.chain_counts, self.chain_timers, self.max_chains)
        self.ledger = CoinLedger(self.game, self.cap)
        self.ledger.pool = self.banks[0]
        self.frontier = FrontierState(self.game, dense=True)
        self.analyzer = OpportunityAnalyzer360(self.game)
        self.episode_return = 0.0
        self.shots = 0
        self.wasted_shots = 0
        self.wall_frames = 0
        self.safe_kill = False
        self.double_death = False
        self.hit_enemy = 0
        self.hit_self = 0
        self.enemy_dead_pending = False
        self.kill_credit = False
        self.settle_remaining = 0
        self.coin_target = None
        self._ensure_coin_target()
        return self._observe()

    def _route_state(self, target):
        if target is None:
            return 0.0, None
        tank = self.game.tanks[0]
        current = (
            int(tank.x // self.game.scale),
            int(tank.y // self.game.scale))
        distances = self.game.dist_map(target[0], target[1])
        if distances is None:
            return 0.0, None
        steps = distances[current[0]][current[1]]
        if steps is None or steps != steps:
            return 0.0, None
        if current == target:
            center_x = (target[0] + 0.5) * self.game.scale
            center_y = (target[1] + 0.5) * self.game.scale
            distance = math.hypot(
                tank.x - center_x, tank.y - center_y) / self.game.scale
            return distance, target
        candidates = []
        for adjacent in neighbors(self.game, current):
            value = distances[adjacent[0]][adjacent[1]]
            if value is not None and value == value and value < steps:
                candidates.append((float(value), adjacent[1], adjacent[0]))
        waypoint = target if not candidates else (
            min(candidates)[2], min(candidates)[1])
        center_x = (waypoint[0] + 0.5) * self.game.scale
        center_y = (waypoint[1] + 0.5) * self.game.scale
        local = math.hypot(
            tank.x - center_x, tank.y - center_y) / self.game.scale
        return max(float(steps) - 1.0, 0.0) + local, waypoint

    def _path_distance(self, target):
        return self._route_state(target)[0]

    def _ensure_coin_target(self):
        if self.coin_target in self.coins:
            return
        tank = self.game.tanks[0]
        current = (
            int(tank.x // self.game.scale),
            int(tank.y // self.game.scale))
        candidates = []
        for cell, value in self.coins.items():
            distances = self.game.dist_map(cell[0], cell[1])
            if distances is None:
                continue
            distance = distances[current[0]][current[1]]
            if distance is not None and distance == distance:
                candidates.append((
                    (float(distance) + 0.5) / max(value, 1.0),
                    -value, cell[1], cell[0]))
        self.coin_target = None if not candidates else (
            min(candidates)[3], min(candidates)[2])

    def _observe(self):
        self._ensure_coin_target()
        base, self.metrics, self.fire_facts = opportunity_observation(
            self.encoder, self.game, self.ledger, self.econ,
            self.frontier, self.analyzer)
        extra = coin_features(
            self.coins, self.banks, self.initial_pool,
            self.ledger.frames, self.cap,
            self.chain_counts, self.chain_timers)
        route_distance, waypoint = self._route_state(self.coin_target)
        route = coin_route_features(
            self.game, waypoint, route_distance,
            self.coins.get(self.coin_target, 0.0))
        self.observation = np.concatenate([base, extra, route])
        return self.observation

    def step(self, action_index):
        own_before, opponent_before = self.banks
        wall_before = self.wall_frames
        action = CANDIDATES[int(action_index)]
        coin_gain = 0.0
        opponent_coin_gain = 0.0
        route_shaping = 0.0
        end = "alive"
        for _ in range(DECIDE_EVERY):
            self._ensure_coin_target()
            target_before = self.coin_target
            route_before = self._path_distance(target_before)
            assessment = shot_facts(
                self.game, self.analyzer.metrics(self.game))
            picked_before = self.picked_values.copy()
            apply_action(self.game, action)
            events = self.game.step()
            advance_coin_chains(self.chain_counts, self.chain_timers)
            collect_coins(
                self.game, self.coins, self.banks,
                self.counts, self.picked_values,
                self.chain_counts, self.chain_timers, self.max_chains)
            coin_gain += self.picked_values[0] - picked_before[0]
            opponent_coin_gain += self.picked_values[1] - picked_before[1]
            if target_before in self.coins:
                route_after = self._path_distance(target_before)
                route_shaping += COIN_ROUTE_REWARD * (
                    route_before - route_after)
            for index, tank in enumerate(self.game.tanks):
                if tank.hit_something:
                    fine = min(self.banks[index], WALL_FINE)
                    self.banks[index] -= fine
                    if index == 0:
                        self.wall_frames += 1
            for event in events:
                if event[0] == "fire" and event[1] == 0:
                    self.shots += 1
                    if assessment[4] > 0.5 or (
                            assessment[3] < 0.5 and assessment[5] <= 0.0):
                        self.wasted_shots += 1
                if event[0] != "hit":
                    continue
                attacker, victim = event[1], event[2]
                loss = min(self.banks[victim], HIT_LOSS)
                self.banks[victim] -= loss
                if attacker != victim:
                    self.banks[attacker] += kill_bonus(
                        self.game, self.game.tanks[attacker],
                        self.game.tanks[victim])
                if attacker == 0 and victim == 1:
                    self.hit_enemy += 1
                    self.kill_credit = True
                if victim == 0:
                    self.hit_self += 1
            self.ledger.advance(self.game, self.banks[0])
            self.frontier.observe_position(self.game)
            alive = [tank.alive for tank in self.game.tanks]
            if not alive[0]:
                self.double_death = not alive[1] \
                    or self.enemy_dead_pending
                end = "double" if self.double_death else "death"
                break
            if not alive[1]:
                if not self.enemy_dead_pending:
                    self.enemy_dead_pending = True
                    self.settle_remaining = SAFE_SETTLE_FRAMES
                self.settle_remaining -= 1
                if self.settle_remaining <= 0:
                    self.safe_kill = self.kill_credit
                    end = "kill" if self.safe_kill else "opponent_self"
                    break
            if self.ledger.frames >= self.cap:
                end = "cap"
                break

        own_delta = self.banks[0] - own_before
        opponent_delta = self.banks[1] - opponent_before
        reward = (own_delta - OPPONENT_GAIN_WEIGHT * opponent_delta) \
            / REWARD_SCALE
        score_deficit = max(0.0, self.banks[1] - self.banks[0]) \
            / REWARD_SCALE
        reward -= DEFICIT_TIME_WEIGHT * score_deficit \
            * (DECIDE_EVERY / FPS)
        reward += route_shaping
        reward -= WALL_EVENT_PENALTY * (self.wall_frames - wall_before)
        if end == "kill":
            reward += ACTIVE_KILL_BONUS
        elif end == "opponent_self":
            reward += OPPONENT_SELF_BONUS
        elif end == "double":
            reward -= DOUBLE_DEATH_PENALTY
        elif end == "death":
            reward -= DEATH_PENALTY
        elif end == "cap":
            reward -= TIMEOUT_PENALTY
        self.episode_return += reward
        done = end != "alive"
        info = {
            "coin_gain": coin_gain,
            "opponent_coin_gain": opponent_coin_gain,
            "route_shaping": route_shaping,
            "wall_delta": self.wall_frames - wall_before,
            "hit": self.hit_enemy > 0,
        }
        if done:
            info.update({
                "end": end,
                "return": self.episode_return,
                "frames": self.ledger.frames,
                "bank": self.banks[0],
                "opponent_bank": self.banks[1],
                "coins": self.picked_values[0],
                "opponent_coins": self.picked_values[1],
                "chain": self.chain_counts[0],
                "opponent_chain": self.chain_counts[1],
                "max_chain": self.max_chains[0],
                "opponent_max_chain": self.max_chains[1],
                "qualified": self.banks[0] >= PASS_SCORE,
                "course_success": self.safe_kill,
                "safe_kill": self.safe_kill,
                "double_death": self.double_death,
                "shots": self.shots,
                "wasted_shots": self.wasted_shots,
                "wall_frames": self.wall_frames,
                "unique_cells": len(self.ledger.visited),
                "hits": self.hit_enemy,
                "stuck_frames": self.wall_frames,
            })
        observation = None if done else self._observe()
        return observation, float(reward), done, info


class SuccessReplay:
    def __init__(self, path, max_steps=50_000):
        self.path = path
        self.max_steps = max_steps
        self.observations = np.empty((0, P35_OBS_DIM), dtype=np.float32)
        self.actions = np.empty(0, dtype=np.int64)
        if path and os.path.exists(path):
            data = np.load(path)
            if data["observations"].shape[1] == P35_OBS_DIM:
                self.observations = data["observations"].astype(np.float32)
                self.actions = data["actions"].astype(np.int64)

    def __len__(self):
        return len(self.actions)

    def add(self, trajectory, repeats=2):
        if not trajectory:
            return
        observations = np.stack([item[0] for item in trajectory])
        actions = np.asarray([item[1] for item in trajectory], dtype=np.int64)
        for _ in range(repeats):
            self.observations = np.concatenate([self.observations, observations])
            self.actions = np.concatenate([self.actions, actions])
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
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez_compressed(
            self.path, observations=self.observations, actions=self.actions)


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
            action = int(actions[index].item())
            next_obs, reward, done, info = env.step(action)
            clean_progress = info["wall_delta"] == 0 and (
                info["coin_gain"] > 0.0 or info["route_shaping"] > 1e-5)
            pending[index].append(
                (observations[index].copy(), action, clean_progress))
            rewards[index] = reward
            dones[index] = float(done)
            if done:
                episodes.append(info)
                before = len(replay)
                navigation = [
                    (observation, selected_action)
                    for observation, selected_action, keep in pending[index]
                    if keep
                ]
                replay.add(navigation, repeats=2)
                if info["course_success"]:
                    combat_tail = [
                        (observation, selected_action)
                        for observation, selected_action, _
                        in pending[index][-8:]
                    ]
                    replay.add(combat_tail, repeats=1)
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
        final = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=device)
        _, final_values = factorized_distribution(model, final, args.temperature)
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
    return {
        "observations": np.concatenate(buffers["observations"]),
        "actions": np.concatenate(buffers["actions"]),
        "log_probs": np.concatenate(buffers["log_probs"]),
        "advantages": advantages.reshape(-1),
        "returns": (advantages + values).reshape(-1),
        "values": values.reshape(-1),
    }, observations, pending, episodes, retained


def summarize(episodes):
    if not episodes:
        return {"episodes": 0}
    return {
        "episodes": len(episodes),
        "mean_bank": float(np.mean([item["bank"] for item in episodes])),
        "mean_opponent_bank": float(np.mean(
            [item["opponent_bank"] for item in episodes])),
        "qualified_pct": float(np.mean(
            [item["qualified"] for item in episodes])),
        "safe_kill_pct": float(np.mean(
            [item["safe_kill"] for item in episodes])),
        "course_success_pct": float(np.mean(
            [item["course_success"] for item in episodes])),
        "double_death_pct": float(np.mean(
            [item["double_death"] for item in episodes])),
        "death_pct": float(np.mean([
            item["end"] in ("death", "double") for item in episodes])),
        "mean_cells": float(np.mean(
            [item["unique_cells"] for item in episodes])),
        "wall_pct": sum(item["wall_frames"] for item in episodes)
        / max(sum(item["frames"] for item in episodes), 1),
        "wasted_shot_pct": sum(item["wasted_shots"] for item in episodes)
        / max(sum(item["shots"] for item in episodes), 1),
        "mean_max_chain": float(np.mean(
            [item["max_chain"] for item in episodes])),
        "mean_opponent_max_chain": float(np.mean(
            [item["opponent_max_chain"] for item in episodes])),
    }


def greedy_probe(model, seed, cap_seconds, device):
    env = CoinPathEnv(seed, cap_seconds)
    observation = env.reset()
    while True:
        tensor = torch.as_tensor(
            observation[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            dist, _ = factorized_distribution(model, tensor, 0.01)
            action = int(dist.probs.argmax(1))
        observation, _, done, info = env.step(action)
        if done:
            return info


class CoinPathRLPolicy:
    """Greedy viewer adapter for a saved P35 actor."""

    name = "P35 金币课程 RL 学生"

    def __init__(self, model_path=DEFAULT_ACTOR):
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        in_dim = int(payload.get("in_dim", P35_OBS_DIM))
        if in_dim == P35_OBS_DIM:
            state = payload["state_dict"]
        else:
            expanded = load_expanded_warmstart(model_path, torch.device("cpu"))
            state = expanded.actor_state_dict()
        self.network = build_net(P35_OBS_DIM)
        self.network.load_state_dict(state)
        self.network.eval()

    def reset(self):
        pass

    def act_index(self, env):
        observation = env.observation
        if observation is None:
            observation = env._observe()
        with torch.no_grad():
            logits = self.network(torch.as_tensor(observation)[None])[0]
        return int(logits.argmax())


def save_models(model, optimizer, replay, args, update, steps, metrics):
    payload = {
        "state_dict": model.actor_state_dict(),
        "in_dim": P35_OBS_DIM,
        "version": "p35_shared_path_coin_rl",
        "update": update,
        "total_steps": steps,
        "success_replay_steps": len(replay),
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
    print(f"P35 warm-start exact: max_diff={difference:.1f}", flush=True)
    if args.coin_adapter_only:
        for parameter in (
                model.fc1.bias, *model.fc2.parameters(),
                *model.fc3.parameters(), *model.actor.parameters()):
            parameter.requires_grad_(False)
        old_dim = P29_OBS_DIM

        def mask_old_inputs(gradient):
            masked = gradient.clone()
            masked[:, :old_dim] = 0.0
            return masked

        model.fc1.weight.register_hook(mask_old_inputs)
        print("P35 adapter-only: old policy frozen; training coin columns + value",
              flush=True)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=args.learning_rate)
    replay = SuccessReplay(args.success_replay, args.max_replay_steps)
    envs = [CoinPathEnv(
        args.seed + index * 1009, args.cap_seconds)
        for index in range(args.envs)]
    observations = [env.reset() for env in envs]
    pending = [[] for _ in envs]
    total_steps = 0
    started = time.time()
    for update in range(1, args.updates + 1):
        rollout, observations, pending, episodes, retained = collect_rollout(
            model, envs, observations, pending, replay, args, device)
        total_steps += args.envs * args.rollout_steps
        losses = update_ppo(
            model, optimizer, rollout, replay, args, device, False)
        metrics = {**losses, **summarize(episodes)}
        greedy = None
        if not args.skip_greedy_probe:
            greedy = greedy_probe(
                model, args.seed + 95_000_000,
                args.cap_seconds, device)
            metrics.update({
                "greedy_end": greedy["end"],
                "greedy_bank": greedy["bank"],
                "greedy_safe_kill": greedy["safe_kill"],
                "greedy_double": greedy["double_death"],
                "greedy_cells": greedy["unique_cells"],
            })
        greedy_text = "" if greedy is None else (
            f"greedy={greedy['end']}/{greedy['bank']:.0f}g/"
            f"{greedy['unique_cells']}c ")
        print(
            f"update {update}/{args.updates} steps={total_steps} "
            f"episodes={metrics.get('episodes', 0)} "
            f"bank={metrics.get('mean_bank', 0):.1f} "
            f"success={metrics.get('course_success_pct', 0):.2f} "
            f"qualified={metrics.get('qualified_pct', 0):.2f} "
            f"kill={metrics.get('safe_kill_pct', 0):.2f} "
            f"double={metrics.get('double_death_pct', 0):.2f} "
            f"replay={len(replay)}(+{retained}) "
            f"{greedy_text}"
            f"kl={losses['kl']:.4f} elapsed={time.time()-started:.0f}s",
            flush=True)
        save_models(
            model, optimizer, replay, args,
            update, total_steps, metrics)
    print(f"actor: {args.actor}\nsuccess replay: {args.success_replay}")


def smoke_command(args):
    device = torch.device(args.device)
    model = load_expanded_warmstart(args.warmstart, device)
    difference = verify_warmstart(model, args.warmstart, device)
    env = CoinPathEnv(args.seed, args.cap_seconds)
    observation = env.reset()
    print(json.dumps({
        "obs": int(observation.shape[0]),
        "warmstart_diff": difference,
        "coins": len(env.coins),
        "pool": env.initial_pool,
        "path_length": len(env.path),
    }, ensure_ascii=False))


def add_common(parser):
    parser.add_argument("--warmstart", default=DEFAULT_WARMSTART)
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--success-replay", default=DEFAULT_REPLAY)
    parser.add_argument("--cap-seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=35_000_001)
    parser.add_argument("--device", default="cpu")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    add_common(smoke)
    smoke.set_defaults(func=smoke_command)
    train = subparsers.add_parser("train")
    add_common(train)
    train.add_argument("--updates", type=int, default=3)
    train.add_argument("--envs", type=int, default=4)
    train.add_argument("--rollout-steps", type=int, default=256)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--minibatch", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-5)
    train.add_argument("--temperature", type=float, default=0.08)
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--gae-lambda", type=float, default=0.97)
    train.add_argument("--clip", type=float, default=0.1)
    train.add_argument("--value-clip", type=float, default=0.2)
    train.add_argument("--value-coef", type=float, default=0.5)
    train.add_argument("--entropy-coef", type=float, default=0.005)
    train.add_argument("--bc-coef", type=float, default=0.2)
    train.add_argument("--bc-batch", type=int, default=256)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--target-kl", type=float, default=0.02)
    train.add_argument("--max-replay-steps", type=int, default=50_000)
    train.add_argument("--torch-threads", type=int, default=4)
    train.add_argument("--value-warmup-updates", type=int, default=0)
    train.add_argument("--skip-greedy-probe", action="store_true",
                       help="Do not run the post-update greedy episode.")
    train.add_argument("--coin-adapter-only", action="store_true",
                       help="Freeze the old P29 policy and train only the "
                            "new coin input columns plus value head.")
    train.set_defaults(func=train_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
