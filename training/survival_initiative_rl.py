"""P30: counterfactual initiative curriculum.

For each decision, compare every movement against a no-op sandbox with the
same freshly seeded Laika.  Only opportunity improvement caused by our action
earns initiative.  Positive initiative creates a short token; hits with a
token keep the full survival income, while reactive hits receive only crumbs.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mpc_agent import CANDIDATES, make_sandbox
from training.opportunity_teacher_v2 import OpportunityAnalyzer360
from training.score_distill import build_net
from training.survival_distill_v2 import legacy_econ
from training.survival_expert_iter_530 import DECIDE_EVERY, apply_action
from training.survival_frontier_rl import FPS, FrontierState
from training.survival_opportunity_rl import (
    FIRE_FACTS_OFFSET,
    FIRE_READY_THRESHOLD,
    FRONTIER_PROGRESS_REWARD,
    GOOD_FIRE_REWARD,
    HIT_EVENT_REWARD,
    OPPORTUNITY_POTENTIAL_REWARD,
    P29_OBS_DIM,
    PRESSURE_THRESHOLD,
    PRESSURE_FIRE_REWARD,
    SUICIDE_FIRE_PENALTY,
    TERMINAL_PENALTY,
    UNIQUE_CELL_REWARD,
    WASTED_FIRE_PENALTY,
    OpportunityRLPolicy,
    OpportunitySurvivalEnv,
    factorized_distribution,
    opportunity_observation,
    shot_facts,
    summarize,
    update_ppo,
)
from training.survival_rl_warmstart import WarmStartActorCritic


HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_WARMSTART = os.path.join(
    MODELS_DIR, "p29c_feature_only_actor_iter02.pt")
DEFAULT_ACTOR = os.path.join(MODELS_DIR, "p30_initiative_actor.pt")
DEFAULT_CHECKPOINT = os.path.join(MODELS_DIR, "p30_initiative_checkpoint.pt")
DEFAULT_REPLAY = os.path.join(HERE, "p30_initiative_replay.npz")

MOVE_COUNT = 9
TOKEN_DIM = 3
P30_EXTRA_DIM = MOVE_COUNT + TOKEN_DIM
P30_OBS_DIM = P29_OBS_DIM + P30_EXTRA_DIM
NOOP_MOVE_INDEX = 4
COUNTERFACTUAL_HORIZON = 8
COUNTERFACTUAL_SCALE = 50.0
INITIATIVE_THRESHOLD = 0.04
INITIATIVE_REWARD = 1.0
INITIATIVE_PEAK_MARGIN = INITIATIVE_THRESHOLD * COUNTERFACTUAL_SCALE
TOKEN_FRAMES = 3 * FPS
REACTIVE_HIT_INCOME = 5.0
REACTIVE_HIT_REWARD = 0.20


def simulate_opportunity(game, analyzer, movement, rng_seed=0):
    sandbox = make_sandbox(game, "L2", rng_seed=rng_seed)
    throttle, turn = movement
    tank = sandbox.tanks[0]
    tank.forward, tank.backup = throttle == 2, throttle == 0
    tank.turn_left, tank.turn_right = turn == 0, turn == 2
    tank.fire = False
    for _ in range(COUNTERFACTUAL_HORIZON):
        sandbox.step()
        if not tank.alive:
            return -100.0, False
    return analyzer.potential(analyzer.metrics(sandbox)), True


def initiative_analysis(game, analyzer):
    movements = [(throttle, turn)
                 for throttle in (0, 1, 2) for turn in (0, 1, 2)]
    baseline, baseline_alive = simulate_opportunity(
        game, analyzer, movements[NOOP_MOVE_INDEX], rng_seed=0)
    output = np.zeros(MOVE_COUNT, dtype=np.float32)
    projections = np.full(MOVE_COUNT, baseline, dtype=np.float32)
    for index, movement in enumerate(movements):
        if index == NOOP_MOVE_INDEX:
            continue
        potential, alive = simulate_opportunity(
            game, analyzer, movement, rng_seed=0)
        projections[index] = potential
        if alive and not baseline_alive:
            advantage = 2.0
        elif not alive and baseline_alive:
            advantage = -2.0
        else:
            advantage = (potential - baseline) / COUNTERFACTUAL_SCALE
        output[index] = np.clip(advantage, -2.0, 2.0)
    return output, projections


def initiative_previews(game, analyzer):
    return initiative_analysis(game, analyzer)[0]


def initiative_credit(advantage, projection, opportunity_peak):
    unlocked = advantage >= INITIATIVE_THRESHOLD \
        and projection >= opportunity_peak + INITIATIVE_PEAK_MARGIN
    if not unlocked:
        return False, 0.0
    gain = min(
        2.0, (projection - opportunity_peak) / COUNTERFACTUAL_SCALE)
    return True, gain


def initiative_observation(encoder, game, ledger, econ, frontier, analyzer,
                           token_frames, token_armed, opportunity_peak):
    base, metrics, fire_facts = opportunity_observation(
        encoder, game, ledger, econ, frontier, analyzer)
    previews, projections = initiative_analysis(game, analyzer)
    current_potential = analyzer.potential(metrics)
    peak_gap = max(0.0, opportunity_peak - current_potential) \
        / COUNTERFACTUAL_SCALE
    token = np.asarray([
        token_frames / max(TOKEN_FRAMES, 1),
        float(token_armed),
        np.clip(peak_gap, 0.0, 2.0),
    ], dtype=np.float32)
    return (np.concatenate([base, previews, token]), metrics, fire_facts,
            previews, projections, current_potential)


def load_expanded_warmstart(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if int(payload.get("in_dim", P29_OBS_DIM)) != P29_OBS_DIM:
        raise ValueError(f"P30 warm-start must have {P29_OBS_DIM} inputs")
    source = payload["state_dict"]
    model = WarmStartActorCritic(in_dim=P30_OBS_DIM).to(device)
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.weight[:, :P29_OBS_DIM].copy_(source["0.weight"])
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
    reference = build_net(P29_OBS_DIM).to(device)
    reference.load_state_dict(payload["state_dict"])
    base = torch.randn(8, P29_OBS_DIM, device=device)
    extra = torch.randn(8, P30_EXTRA_DIM, device=device)
    with torch.no_grad():
        expected = reference(base)
        actual, _ = model(torch.cat([base, extra], dim=1))
    difference = (expected - actual).abs().max().item()
    if difference != 0.0:
        raise RuntimeError(f"P30 warm-start mismatch: {difference}")
    return difference


class InitiativeSurvivalEnv(OpportunitySurvivalEnv):
    def __init__(self, seed, cap_seconds, start_pool,
                 terminal_penalty=TERMINAL_PENALTY):
        self.token_frames = 0
        self.token_strength = 0.0
        self.token_armed = False
        self.last_initiative = 0.0
        self.opportunity_peak = float("-inf")
        self.initiative_previews = np.zeros(MOVE_COUNT, dtype=np.float32)
        self.initiative_projections = np.zeros(MOVE_COUNT, dtype=np.float32)
        self.initiative_hits = 0
        self.reactive_hits = 0
        super().__init__(seed, cap_seconds, start_pool, terminal_penalty)

    def reset(self):
        self.token_frames = 0
        self.token_strength = 0.0
        self.token_armed = False
        self.last_initiative = 0.0
        self.opportunity_peak = float("-inf")
        self.initiative_hits = 0
        self.reactive_hits = 0
        return super().reset()

    def _observe(self):
        result = initiative_observation(
                self.encoder, self.game, self.ledger, self.econ,
                self.frontier, self.analyzer, self.token_frames,
                self.token_armed, self.opportunity_peak)
        self.observation, self.metrics, self.fire_facts, \
            self.initiative_previews, self.initiative_projections, \
            current_potential = result
        if not np.isfinite(self.opportunity_peak):
            self.opportunity_peak = current_potential
        else:
            self.opportunity_peak = max(
                self.opportunity_peak, current_potential)
        self.max_ready_line = max(
            self.max_ready_line, float(self.fire_facts[0]))
        return self.observation

    def step(self, action_index):
        movement_index = int(action_index) // 2
        initiative = float(self.initiative_previews[movement_index])
        projection = float(self.initiative_projections[movement_index])
        self.last_initiative = initiative
        unlocked, initiative_gain = initiative_credit(
            initiative, projection, self.opportunity_peak)
        if unlocked:
            self.opportunity_peak = projection
            self.token_frames = TOKEN_FRAMES
            self.token_strength = initiative
            self.token_armed = False

        pool_before = self.ledger.pool
        reward_events = INITIATIVE_REWARD * initiative_gain
        frontier_progress = 0.0
        opportunity_progress = 0.0
        first_visit_event = False
        fired_event = False
        good_fire_event = False
        initiative_hit_event = False
        reactive_hit_event = False
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
            token_active = self.token_frames > 0
            end = self.ledger.on_frame(self.game, events)
            actual_fire = any(
                event[0] == "fire" and event[1] == 0 for event in events)
            actual_hit = any(
                event[0] == "hit" and event[1] == 0 and event[2] == 1
                for event in events)
            if actual_fire:
                fired_event = True
                self.shots += 1
                if token_active:
                    self.token_armed = True
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
            if actual_hit:
                if token_active and self.token_armed:
                    self.initiative_hits += 1
                    initiative_hit_event = True
                    reward_events += HIT_EVENT_REWARD
                    self.token_frames = 0
                    self.token_strength = 0.0
                    self.token_armed = False
                    self.opportunity_peak = self.analyzer.potential(
                        self.analyzer.metrics(self.game))
                else:
                    self.reactive_hits += 1
                    reactive_hit_event = True
                    removed_income = self.econ["hit"] - REACTIVE_HIT_INCOME
                    self.ledger.pool -= removed_income
                    self.ledger.led["hit"] -= removed_income
                    reward_events += REACTIVE_HIT_REWARD
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
            if self.token_frames > 0:
                self.token_frames -= 1
                if self.token_frames == 0:
                    self.token_strength = 0.0
                    self.token_armed = False
            if end == "alive" and self.ledger.pool <= 0:
                self.ledger.pool = 0.0
                end = "drain"
            if end == "cap" and self.ledger.pool <= 0:
                end = "drain"
            if end != "alive":
                break

        reward = (self.ledger.pool - pool_before) / 50.0 + reward_events
        failure_adjustment = self.terminal_penalty \
            - self.econ["start"] / 50.0
        if end == "death":
            reward -= self.ledger.pool / 50.0 + failure_adjustment
        elif end == "drain":
            reward -= failure_adjustment
        elif end == "cap":
            reward += self.econ["start"] / 50.0
        self.episode_return += reward

        done = end != "alive"
        observation = None if done else self._observe()
        info = {
            "frontier_progress": frontier_progress,
            "opportunity_progress": opportunity_progress,
            "initiative": initiative,
            "initiative_gain": initiative_gain,
            "initiative_unlocked": unlocked,
            "first_visit": first_visit_event,
            "fired": fired_event,
            "good_fire": good_fire_event,
            "hit": initiative_hit_event,
            "reactive_hit": reactive_hit_event,
        }
        if done:
            info.update({
                "end": end,
                "return": self.episode_return,
                "frames": self.ledger.frames,
                "hits": self.initiative_hits,
                "total_hits": self.initiative_hits + self.reactive_hits,
                "initiative_hits": self.initiative_hits,
                "reactive_hits": self.reactive_hits,
                "shots": self.shots,
                "good_shots": self.good_shots,
                "wasted_shots": self.wasted_shots,
                "max_ready_line": self.max_ready_line,
                "stuck_frames": self.ledger.stuck_frames,
                "unique_cells": int(self.frontier.visited.sum()),
                "pool": self.ledger.pool,
            })
        return observation, float(reward), done, info


class InitiativeReplay:
    def __init__(self, path, max_steps=50_000):
        self.path = path
        self.max_steps = max_steps
        self.observations = np.empty((0, P30_OBS_DIM), dtype=np.float32)
        self.actions = np.empty(0, dtype=np.int64)
        if path and os.path.exists(path):
            data = np.load(path)
            if data["observations"].shape[1] == P30_OBS_DIM:
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
            selected.update(range(max(0, index - 12), index + 1))
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
                info["initiative"] >= INITIATIVE_THRESHOLD
                or info["hit"])
            rewards[index] = reward
            dones[index] = float(done)
            if done:
                episodes.append(info)
                if info["initiative_hits"] > 0:
                    before = len(replay)
                    replay.add(pending[index], repeats=3)
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


def greedy_probe(model, cap_seconds, start_pool, penalty, seed, device):
    env = InitiativeSurvivalEnv(seed, cap_seconds, start_pool, penalty)
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
    payload = {
        "state_dict": model.actor_state_dict(),
        "in_dim": P30_OBS_DIM,
        "version": "p30_counterfactual_initiative_rl",
        "econ": dict(legacy_econ(), cap=args.cap_seconds * FPS,
                     start=float(args.start_pool)),
        "update": update,
        "total_steps": steps,
        "initiative_replay_steps": len(replay),
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
    print(f"P30 warm-start exact: max_diff={difference:.1f}", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc1.weight.requires_grad_(True)
    model.value.weight.requires_grad_(True)
    model.value.bias.requires_grad_(True)

    def freeze_old_columns(gradient):
        gradient = gradient.clone()
        gradient[:, :P29_OBS_DIM] = 0.0
        return gradient

    model.fc1.weight.register_hook(freeze_old_columns)
    optimizer = torch.optim.Adam(
        [model.fc1.weight, model.value.weight, model.value.bias],
        lr=args.learning_rate)
    replay = InitiativeReplay(args.initiative_replay, args.max_replay_steps)
    envs = [InitiativeSurvivalEnv(
        args.seed + index * 1009, args.cap_seconds, args.start_pool,
        args.terminal_penalty) for index in range(args.envs)]
    observations = [env.reset() for env in envs]
    pending = [[] for _ in envs]
    total_steps = 0
    started = time.time()
    for update in range(1, args.updates + 1):
        rollout, observations, pending, episodes, retained = collect_rollout(
            model, envs, observations, pending, replay, args, device)
        total_steps += args.envs * args.rollout_steps
        losses = update_ppo(
            model, optimizer, rollout, replay, args, device,
            update <= args.value_warmup_updates)
        metrics = {**losses, **summarize(episodes)}
        initiative_rate = sum(x["initiative_hits"] for x in episodes) \
            / max(len(episodes), 1)
        reactive_rate = sum(x["reactive_hits"] for x in episodes) \
            / max(len(episodes), 1)
        greedy = greedy_probe(
            model, args.cap_seconds, args.start_pool,
            args.terminal_penalty, args.seed + 92_000_000, device)
        metrics.update({
            "initiative_hits_per_episode": initiative_rate,
            "reactive_hits_per_episode": reactive_rate,
            "greedy_initiative_hits": greedy["initiative_hits"],
            "greedy_reactive_hits": greedy["reactive_hits"],
            "greedy_cells": greedy["unique_cells"],
            "greedy_end": greedy["end"],
        })
        print(
            f"update {update}/{args.updates} steps={total_steps} "
            f"episodes={metrics.get('episodes', 0)} "
            f"initiative_hit={initiative_rate:.2f} reactive={reactive_rate:.2f} "
            f"replay={len(replay)}(+{retained}) "
            f"greedy={greedy['end']}/{greedy['unique_cells']}c/"
            f"{greedy['initiative_hits']}I/{greedy['reactive_hits']}R "
            f"bc={losses['bc']:.3f} kl={losses['kl']:.4f} "
            f"elapsed={time.time()-started:.0f}s", flush=True)
        save_models(model, optimizer, replay, args, update, total_steps, metrics)
    print(f"actor: {args.actor}\ninitiative replay: {args.initiative_replay}")


class InitiativeRLPolicy(OpportunityRLPolicy):
    name = "p30_counterfactual_initiative"

    def __init__(self, model_path):
        super().__init__(model_path)
        self.token_frames = 0
        self.token_strength = 0.0
        self.token_armed = False
        self.last_initiative = 0.0
        self.opportunity_peak = float("-inf")

    def reset(self):
        super().reset()
        self.token_frames = 0
        self.token_strength = 0.0
        self.token_armed = False
        self.last_initiative = 0.0
        self.opportunity_peak = float("-inf")

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
            self.token_frames = 0
            self.token_strength = 0.0
            self.token_armed = False
            self.last_initiative = 0.0
            self.opportunity_peak = float("-inf")
            self.last_action = (1, 1, 0)
        else:
            self.frontier.observe_position(game)
            if self.token_frames > 0:
                self.token_frames -= 1
                if self.token_frames == 0:
                    self.token_strength = 0.0
                    self.token_armed = False
            if any(event[0] == "fire" and event[1] == 0
                   for event in game.events) and self.token_frames > 0:
                self.token_armed = True
            if any(event[0] == "hit" and event[1] == 0 and event[2] == 1
                   for event in game.events):
                self.token_frames = 0
                self.token_strength = 0.0
                self.token_armed = False
                self.opportunity_peak = float("-inf")
        if self.context_step % DECIDE_EVERY == 0:
            result = initiative_observation(
                self.encoder, game, ledger, self.econ, self.frontier,
                self.analyzer, self.token_frames, self.token_armed,
                self.opportunity_peak)
            observation, _, _, previews, projections, current_potential = \
                result
            if not np.isfinite(self.opportunity_peak):
                self.opportunity_peak = current_potential
            else:
                self.opportunity_peak = max(
                    self.opportunity_peak, current_potential)
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
                action_index = movement * 2 + fire
            self.last_action = CANDIDATES[action_index]
            movement = action_index // 2
            self.last_initiative = float(previews[movement])
            projection = float(projections[movement])
            unlocked, _ = initiative_credit(
                self.last_initiative, projection, self.opportunity_peak)
            if unlocked:
                self.opportunity_peak = projection
                self.token_frames = TOKEN_FRAMES
                self.token_strength = self.last_initiative
                self.token_armed = False
        self.context_step += 1
        return self._dict()


def smoke_command(args):
    device = torch.device(args.device)
    model = load_expanded_warmstart(args.warmstart, device)
    difference = verify_warmstart(model, args.warmstart, device)
    env = InitiativeSurvivalEnv(
        args.seed, args.cap_seconds, args.start_pool, args.terminal_penalty)
    observation = env.reset()
    noop = float(observation[P29_OBS_DIM + NOOP_MOVE_INDEX])
    repeat, projections = initiative_analysis(env.game, env.analyzer)
    best = int(repeat.argmax())
    unlocked, gain = initiative_credit(
        float(repeat[best]), float(projections[best]), env.opportunity_peak)
    repeat_unlocked, repeat_gain = initiative_credit(
        float(repeat[best]), float(projections[best]),
        max(env.opportunity_peak, float(projections[best])))
    noop_unlocked, _ = initiative_credit(
        noop, float(projections[NOOP_MOVE_INDEX]), env.opportunity_peak)
    if noop != 0.0 or noop_unlocked or repeat_unlocked or repeat_gain != 0.0:
        raise RuntimeError("P30 initiative anti-cycle invariant failed")
    print("preview", np.round(repeat, 4).tolist(), flush=True)
    print(f"noop={noop:.6f} repeat_diff="
          f"{np.max(np.abs(repeat - env.initiative_previews)):.6f}")
    print(f"best={best} unlock={unlocked} gain={gain:.4f} "
          f"repeat_unlock={repeat_unlocked}")
    result = greedy_probe(
        model, args.cap_seconds, args.start_pool,
        args.terminal_penalty, args.seed, device)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print(f"smoke ok: obs={P30_OBS_DIM} warmstart_diff={difference:.1f}")


def add_common(parser):
    parser.add_argument("--warmstart", default=DEFAULT_WARMSTART)
    parser.add_argument("--actor", default=DEFAULT_ACTOR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--initiative-replay", default=DEFAULT_REPLAY)
    parser.add_argument("--cap-seconds", type=int, default=12)
    parser.add_argument("--start-pool", type=float, default=80.0)
    parser.add_argument("--terminal-penalty", type=float, default=TERMINAL_PENALTY)
    parser.add_argument("--seed", type=int, default=30_000_001)
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
    train.add_argument("--rollout-steps", type=int, default=192)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--minibatch", type=int, default=192)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--temperature", type=float, default=0.05)
    train.add_argument("--gamma", type=float, default=0.995)
    train.add_argument("--gae-lambda", type=float, default=0.97)
    train.add_argument("--clip", type=float, default=0.1)
    train.add_argument("--value-clip", type=float, default=0.2)
    train.add_argument("--value-coef", type=float, default=0.05)
    train.add_argument("--entropy-coef", type=float, default=0.003)
    train.add_argument("--bc-coef", type=float, default=0.20)
    train.add_argument("--bc-batch", type=int, default=192)
    train.add_argument("--max-replay-steps", type=int, default=50_000)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--target-kl", type=float, default=0.02)
    train.add_argument("--value-warmup-updates", type=int, default=1)
    train.add_argument("--torch-threads", type=int, default=4)
    train.set_defaults(func=train_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
