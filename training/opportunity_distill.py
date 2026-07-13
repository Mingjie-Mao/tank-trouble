"""
P25 原版机会课程：主动把局面推进到安全、可信的击杀机会。

规则始终是原版 Tank Trouble。塑形只进入老师的候选动作评分：

    score = 原版终局价值 + Phi(s_after) - Phi(s_before) + 出手机会事件

Phi 由可信炮线、到最近直接射击位的路径进度和来弹风险组成。所有位置项
都是有正有负的势能差，往返或绕圈净额为零。学生观测在 P21b 的 408 维
物理观测上增加 32 维机会事实，避免老师标签使用学生看不到的信息。

用法：

  python3 training/opportunity_distill.py teacher-eval --n 20
  python3 training/opportunity_distill.py pipeline --rounds 160 --workers 8
  python3 training/opportunity_distill.py eval --n 200
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.score_distill import FULL_OBS_DIM, build_net, full_obs, train

FPS = 25
HORIZON = 48
HOLD = 16
RISK_HORIZON = 30
FIRE_DISTANCE_CAP = 8.0
LINE_WEIGHT = 45.0
FIRE_POSITION_WEIGHT = 35.0
RISK_WEIGHT = 25.0
GOOD_FIRE_BONUS = 12.0
PRESSURE_BONUS = 4.0
SUICIDE_FIRE_PENALTY = 12.0
OPPORTUNITY_DIM = 5 + 9 * 3
OBS_DIM = FULL_OBS_DIM + OPPORTUNITY_DIM
SCORE_SCALE = 1000.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "opportunity_data")
MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_NET = os.path.join(MODELS_DIR, "p25_opportunity_scorenet.pt")
DEFAULT_STUDENT = os.path.join(MODELS_DIR, "scorenet_best.pt")
MOVE_OPTIONS = [(throttle, turn)
                for throttle in (0, 1, 2) for turn in (0, 1, 2)]


def _cell(game, tank):
    return int(tank.x // game.scale), int(tank.y // game.scale)


def _finite_distance(game, source, target):
    distances = game.dist_map(source[0], source[1])
    if distances is None:
        return None
    tx, ty = target
    if not (0 <= tx < len(distances)
            and 0 <= ty < len(distances[tx])):
        return None
    value = distances[tx][ty]
    if value is None or value != value:
        return None
    return float(value)


class OpportunityAnalyzer:
    """从当前可见物理状态提取机会势能和动作条件机会预演。"""

    def __init__(self, game):
        thickness = float(game.wall_half_t)
        self.boxes = np.asarray([
            (min(x1, x2) - thickness, min(y1, y2) - thickness,
             max(x1, x2) + thickness, max(y1, y2) + thickness)
            for x1, y1, x2, y2 in game.walls
        ], dtype=np.float64)

    def _line_clear(self, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        for xmin, ymin, xmax, ymax in self.boxes:
            low, high = 0.0, 1.0
            for origin, delta, bound_low, bound_high in (
                    (x0, dx, xmin, xmax), (y0, dy, ymin, ymax)):
                if abs(delta) < 1e-12:
                    if origin < bound_low or origin > bound_high:
                        low, high = 1.0, 0.0
                        break
                    continue
                first = (bound_low - origin) / delta
                second = (bound_high - origin) / delta
                low = max(low, min(first, second))
                high = min(high, max(first, second))
                if low > high:
                    break
            if low <= high and high > 1e-6 and low < 1.0 - 1e-6:
                return False
        return True

    def line_quality(self, game):
        from tank_trouble_original import constants as constants
        from training.tt_gym_env import (
            HIT_RADIUS_SCALE, SHOT_FAN_DEG, SHOT_SIM_FRAMES,
            _reflective_closest_batch)

        me, enemy = game.tanks[0], game.tanks[1]
        if not (me.alive and enemy.alive):
            return 0.0
        forward = (me.rotation - 90.0) * math.pi / 180.0
        angles = forward + np.asarray(SHOT_FAN_DEG) * math.pi / 180.0
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        spawn_distance = game.scale * 4.5 / 16.0
        origins = np.stack([
            me.x + directions[:, 0] * spawn_distance,
            me.y + directions[:, 1] * spawn_distance,
        ], axis=1)
        speed = constants.BULLETSPEED * (game.scale / 50.0)
        result = _reflective_closest_batch(
            origins, directions, np.full(len(angles), speed),
            np.full(len(angles), float(SHOT_SIM_FRAMES)), 2,
            self.boxes, enemy.x, enemy.y)
        hit = result[:, 0] <= HIT_RADIUS_SCALE * game.scale
        if not hit.any():
            return 0.0
        time_quality = 1.0 - np.minimum(
            result[:, 1] / SHOT_SIM_FRAMES, 1.0)
        bounce_quality = 1.0 - 0.10 * np.minimum(result[:, 2], 2.0)
        quality = (0.65 + 0.35 * time_quality) * bounce_quality
        return float(np.max(np.where(hit, quality, 0.0)))

    def incoming_risk(self, game):
        from tank_trouble_original import constants as constants
        from training.tt_gym_env import HIT_RADIUS_SCALE, _reflective_closest_batch

        me = game.tanks[0]
        bullets = [bullet for bullet in game.bullets if not bullet.removed]
        if not bullets or not me.alive:
            return 0.0
        origins, directions, speeds, horizons = [], [], [], []
        for bullet in bullets:
            frame_vx = bullet.x_speed * constants.BULLETHITCHECKINTERVALS
            frame_vy = bullet.y_speed * constants.BULLETHITCHECKINTERVALS
            speed = math.hypot(frame_vx, frame_vy)
            if speed < 1e-9:
                continue
            origins.append((bullet.x, bullet.y))
            directions.append((frame_vx / speed, frame_vy / speed))
            speeds.append(speed)
            horizons.append(min(RISK_HORIZON, max(0, bullet.lifetime)))
        if not origins:
            return 0.0
        result = _reflective_closest_batch(
            np.asarray(origins), np.asarray(directions), np.asarray(speeds),
            np.asarray(horizons, dtype=np.float64), 3, self.boxes,
            me.x, me.y)
        hit = result[:, 0] <= HIT_RADIUS_SCALE * game.scale
        if not hit.any():
            return 0.0
        urgency = 1.0 - np.minimum(result[:, 1] / RISK_HORIZON, 1.0)
        return float(np.max(np.where(hit, urgency, 0.0)))

    def nearest_firing_position(self, game):
        me, enemy = game.tanks[0], game.tanks[1]
        me_cell = _cell(game, me)
        best_distance, best_cell = float("inf"), None
        for item in game.reachable:
            candidate = (item["x"], item["y"])
            cx = (candidate[0] + 0.5) * game.scale
            cy = (candidate[1] + 0.5) * game.scale
            if math.hypot(cx - enemy.x, cy - enemy.y) < 0.75 * game.scale:
                continue
            distance = _finite_distance(game, me_cell, candidate)
            if distance is None or distance >= best_distance:
                continue
            if self._line_clear(cx, cy, enemy.x, enemy.y):
                best_distance, best_cell = distance, candidate
        if best_cell is None:
            best_cell = _cell(game, enemy)
            fallback = _finite_distance(game, me_cell, best_cell)
            best_distance = FIRE_DISTANCE_CAP if fallback is None else fallback
        return min(best_distance, FIRE_DISTANCE_CAP), best_cell

    def _next_direction(self, game, target_cell):
        from tank_trouble_original.maze import h_open, v_open

        me = game.tanks[0]
        current = _cell(game, me)
        distances = game.dist_map(target_cell[0], target_cell[1])
        if distances is None or current == target_cell:
            return 0.0, 0.0

        def value(cell):
            x, y = cell
            if 0 <= x < len(distances) and 0 <= y < len(distances[x]):
                distance = distances[x][y]
                if distance is not None and distance == distance:
                    return float(distance)
            return float("inf")

        x, y = current
        width, height = len(game.maze), len(game.maze[0])
        neighbors = []
        if x > 0 and v_open(game.maze, x, y):
            neighbors.append((x - 1, y))
        if x < width - 1 and v_open(game.maze, x + 1, y):
            neighbors.append((x + 1, y))
        if y > 0 and h_open(game.maze, x, y - 1):
            neighbors.append((x, y - 1))
        if y < height - 1 and h_open(game.maze, x, y):
            neighbors.append((x, y + 1))
        if not neighbors:
            return 0.0, 0.0
        next_cell = min(neighbors, key=value)
        if value(next_cell) >= value(current):
            return 0.0, 0.0
        dx = (next_cell[0] + 0.5) * game.scale - me.x
        dy = (next_cell[1] + 0.5) * game.scale - me.y
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return 0.0, 0.0
        dx, dy = dx / length, dy / length
        forward = (me.rotation - 90.0) * math.pi / 180.0
        cos_forward, sin_forward = math.cos(forward), math.sin(forward)
        return (dx * cos_forward + dy * sin_forward,
                -dx * sin_forward + dy * cos_forward)

    def metrics(self, game):
        line = self.line_quality(game)
        distance, target = self.nearest_firing_position(game)
        reach = 1.0 - min(distance, FIRE_DISTANCE_CAP) / FIRE_DISTANCE_CAP
        reach = max(reach, line)
        risk = self.incoming_risk(game)
        direction = self._next_direction(game, target)
        return np.asarray([line, reach, risk, direction[0], direction[1]],
                          dtype=np.float32)

    def potential(self, metrics):
        line, reach, risk = [float(value) for value in metrics[:3]]
        safety = 1.0 - 0.60 * risk
        return (LINE_WEIGHT * line * safety
                + FIRE_POSITION_WEIGHT * reach * safety
                - RISK_WEIGHT * risk)

    def action_previews(self, game, base_metrics):
        from training.mpc_agent import make_sandbox

        output = np.empty(27, dtype=np.float32)
        for index, (throttle, turn) in enumerate(MOVE_OPTIONS):
            sandbox = make_sandbox(game, "L1", rng_seed=0)
            enemy = sandbox.tanks[1]
            enemy.forward = enemy.backup = False
            enemy.turn_left = enemy.turn_right = False
            enemy.fire = False
            me = sandbox.tanks[0]
            me.forward, me.backup = throttle == 2, throttle == 0
            me.turn_left, me.turn_right = turn == 0, turn == 2
            me.fire = False
            for _ in range(HOLD):
                sandbox.step()
                if not me.alive:
                    break
            if not me.alive:
                values = (-1.0, -1.0, 1.0)
            else:
                after = self.metrics(sandbox)
                values = (after[0] - base_metrics[0],
                          after[1] - base_metrics[1], after[2])
            output[index * 3:(index + 1) * 3] = values
        return output


def bind_env(env, game, frames):
    if env.game is not game:
        env.game = game
        env._build_wall_boxes()
    env._frames = frames
    env._prev_phi = env._phi()


def opportunity_obs(env, game, analyzer, frames):
    bind_env(env, game, frames)
    metrics = analyzer.metrics(game)
    previews = analyzer.action_previews(game, metrics)
    return np.concatenate([full_obs(env), metrics, previews]), metrics


def _shot_event(game):
    from tank_trouble_original.laika import LaikaAI

    me = game.tanks[0]
    if not (me.trigger_released and game.weapon_ready(me)):
        return None
    return LaikaAI(game, me).check_bullet_path(me.rotation)


def opportunity_rollout(sandbox, first_action, analyzer, start_metrics,
                        hold=HOLD, horizon=HORIZON):
    me, enemy = sandbox.tanks[0], sandbox.tanks[1]
    throttle, turn, fire = first_action
    shot = _shot_event(sandbox) if fire == 1 else None
    fired = False
    for frame in range(horizon):
        if frame == 0:
            me.forward, me.backup = throttle == 2, throttle == 0
            me.turn_left, me.turn_right = turn == 0, turn == 2
            me.fire = fire == 1
        elif frame == hold:
            me.fire = False
        events = sandbox.step()
        fired = fired or any(event[0] == "fire" and event[1] == 0
                             for event in events)
        if not me.alive:
            return -1000.0 + frame
        if not enemy.alive and frame >= hold:
            return 1000.0 - frame
    end_metrics = analyzer.metrics(sandbox)
    score = analyzer.potential(end_metrics) - analyzer.potential(start_metrics)
    if fired and shot is not None:
        result = shot["result"]
        if result == "HIT" and start_metrics[0] >= 0.60:
            score += GOOD_FIRE_BONUS
        elif result == "SUICIDE":
            score -= SUICIDE_FIRE_PENALTY
        elif shot.get("closest", float("inf")) <= 0.75 * sandbox.scale:
            score += PRESSURE_BONUS
    return score


class OpportunityMPC:
    name = "p25_opportunity_mpc"

    def __init__(self, seed=0, horizon=HORIZON, hold=HOLD):
        from training.mpc_agent import CANDIDATES
        self.candidates = CANDIDATES
        self.rng = random.Random(seed)
        self.horizon = horizon
        self.hold = hold
        self.game = None
        self.analyzer = None

    def reset(self):
        self.game = None
        self.analyzer = None

    def act(self, game):
        from training.mpc_agent import make_sandbox

        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer(game)
        start = self.analyzer.metrics(game)
        step_seed = self.rng.randrange(1 << 30)
        scores = []
        for action in self.candidates:
            sandbox = make_sandbox(game, "L2", rng_seed=step_seed)
            scores.append(opportunity_rollout(
                sandbox, action, self.analyzer, start,
                self.hold, self.horizon))
        throttle, turn, fire = self.candidates[int(np.argmax(scores))]
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _collect_worker(job):
    worker, rounds, seed0, epsilon = job
    import torch
    torch.set_num_threads(1)
    from training.mpc_agent import CANDIDATES, make_sandbox
    from training.tt_gym_env import TankTroubleGym

    env = TankTroubleGym(seed=0, reward_version=1, terminal_mode="score",
                         obs_traj=True, obs_nav=True)
    rng = random.Random(worker * 104729 + 71)
    observations, labels = [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    opportunity_gain, decisions = 0.0, 0
    for episode in range(rounds):
        env._base_seed = seed0 + episode
        env._episode = 0
        env.reset()
        analyzer = OpportunityAnalyzer(env.game)
        while True:
            observation, start = opportunity_obs(
                env, env.game, analyzer, env._frames)
            step_seed = rng.randrange(1 << 30)
            scores = np.empty(len(CANDIDATES), dtype=np.float32)
            for index, action in enumerate(CANDIDATES):
                sandbox = make_sandbox(env.game, "L2", rng_seed=step_seed)
                scores[index] = opportunity_rollout(
                    sandbox, action, analyzer, start)
            observations.append(observation)
            labels.append(scores / SCORE_SCALE)
            best_index = int(scores.argmax())
            opportunity_gain += float(scores[best_index])
            decisions += 1
            if rng.random() < epsilon:
                action = np.asarray(
                    [rng.randrange(3), rng.randrange(3), 0])
            else:
                action = np.asarray(CANDIDATES[best_index])
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                result = info.get("result", "draw")
                stats[result] = stats.get(result, 0) + 1
                break
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"opportunity_shard_{worker}.npz")
    np.savez_compressed(path, X=np.asarray(observations, np.float32),
                        Y=np.asarray(labels, np.float32))
    return path, stats, opportunity_gain, decisions


def collect(total_rounds, workers, epsilon=0.05, seed_base=9_000_000):
    per_worker = max(1, total_rounds // workers)
    jobs = [(worker, per_worker, seed_base + worker * per_worker, epsilon)
            for worker in range(workers)]
    print(f"采集: {workers}进程 x {per_worker}局, 原版机会老师重标18动作",
          flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_collect_worker, jobs)
    xs, ys, stats = [], [], {}
    opportunity_gain, decisions = 0.0, 0
    for path, worker_stats, gain, count in results:
        data = np.load(path)
        xs.append(data["X"])
        ys.append(data["Y"])
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
        opportunity_gain += gain
        decisions += count
    X, Y = np.concatenate(xs), np.concatenate(ys)
    games = sum(stats.values())
    print(f"采集完成: {len(X)}样本 / {games}局 / {time.time()-started:.0f}s",
          flush=True)
    print(f"  老师含噪声真胜率 {stats.get('win', 0)/max(games, 1):.1%}  "
          f"双亡 {stats.get('double_death', 0)/max(games, 1):.1%}  "
          f"平均候选最优分 {opportunity_gain/max(decisions, 1):+.1f}",
          flush=True)
    return X, Y


def _load_score_network(path):
    import torch
    payload = torch.load(path, weights_only=True)
    input_dim = payload.get("in_dim", FULL_OBS_DIM)
    network = build_net(input_dim)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network


def _dagger_worker(job):
    worker, rounds, seed0, epsilon, student_path = job
    import torch
    torch.set_num_threads(1)
    from training.mpc_agent import CANDIDATES, make_sandbox
    from training.tt_gym_env import TankTroubleGym

    student = _load_score_network(student_path)
    env = TankTroubleGym(seed=0, reward_version=1, terminal_mode="score",
                         obs_traj=True, obs_nav=True)
    rng = random.Random(worker * 104729 + 97)
    observations, labels = [], []
    stats = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    regret, lethal, decisions = 0.0, 0, 0
    for episode in range(rounds):
        env._base_seed = seed0 + episode
        env._episode = 0
        env.reset()
        analyzer = OpportunityAnalyzer(env.game)
        while True:
            observation, start = opportunity_obs(
                env, env.game, analyzer, env._frames)
            step_seed = rng.randrange(1 << 30)
            scores = np.empty(len(CANDIDATES), dtype=np.float32)
            for index, action in enumerate(CANDIDATES):
                sandbox = make_sandbox(env.game, "L2", rng_seed=step_seed)
                scores[index] = opportunity_rollout(
                    sandbox, action, analyzer, start)
            observations.append(observation)
            labels.append(scores / SCORE_SCALE)
            with torch.no_grad():
                student_scores = student(
                    torch.as_tensor(observation[:FULL_OBS_DIM]).unsqueeze(0))[0]
            pick = int(student_scores.argmax())
            regret += float(scores.max() - scores[pick])
            if scores[pick] < -500.0 and scores.max() > -500.0:
                lethal += 1
            decisions += 1
            if rng.random() < epsilon:
                action = np.asarray(
                    [rng.randrange(3), rng.randrange(3), 0])
            else:
                action = np.asarray(CANDIDATES[pick])
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                result = info.get("result", "draw")
                stats[result] = stats.get(result, 0) + 1
                break
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"opportunity_dagger_shard_{worker}.npz")
    np.savez_compressed(path, X=np.asarray(observations, np.float32),
                        Y=np.asarray(labels, np.float32))
    return path, stats, regret, lethal, decisions


def collect_dagger(total_rounds, workers, student_path, epsilon=0.05,
                   seed_base=9_200_000):
    per_worker = max(1, total_rounds // workers)
    jobs = [(worker, per_worker, seed_base + worker * per_worker,
             epsilon, student_path) for worker in range(workers)]
    print(f"DAgger采集: {workers}进程 x {per_worker}局, "
          f"学生={os.path.basename(student_path)}", flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_dagger_worker, jobs)
    stats, regret, lethal, decisions = {}, 0.0, 0, 0
    for _, worker_stats, worker_regret, worker_lethal, worker_decisions in results:
        for key, value in worker_stats.items():
            stats[key] = stats.get(key, 0) + value
        regret += worker_regret
        lethal += worker_lethal
        decisions += worker_decisions
    games = sum(stats.values())
    print(f"DAgger完成: {decisions}样本 / {games}局 / "
          f"{time.time()-started:.0f}s", flush=True)
    print(f"  P22现场真胜率 {stats.get('win', 0)/max(games, 1):.1%}  "
          f"平均机会后悔 {regret/max(decisions, 1):.1f}  "
          f"致死误判 {lethal/max(decisions, 1):.2%}", flush=True)


def load_opportunity_data():
    import glob
    paths = sorted(glob.glob(os.path.join(DATA_DIR,
                                          "opportunity*_shard_*.npz")))
    xs, ys = [], []
    for path in paths:
        data = np.load(path)
        if data["X"].shape[1] != OBS_DIM:
            continue
        xs.append(data["X"])
        ys.append(data["Y"])
    if not xs:
        raise RuntimeError("没有可用的 P25 机会数据")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    print(f"机会数据聚合: {len(paths)} shard -> {len(X)}样本", flush=True)
    return X, Y


class OpportunityScoreNetPolicy:
    name = "p25_opportunity_scorenet"

    def __init__(self, net_path=DEFAULT_NET):
        import torch
        from training.mpc_agent import CANDIDATES
        from training.tt_gym_env import TankTroubleGym

        self.torch = torch
        self.candidates = CANDIDATES
        payload = torch.load(net_path, weights_only=True)
        input_dim = payload.get("in_dim", OBS_DIM)
        self.net = build_net(input_dim)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self.env = TankTroubleGym(seed=0, reward_version=1,
                                  obs_traj=True, obs_nav=True)
        self.game = None
        self.analyzer = None
        self.frames = 0

    def reset(self):
        self.game = None
        self.analyzer = None
        self.frames = 0

    def act(self, game):
        if not game.tanks[0].alive:
            return {}
        if game is not self.game:
            self.game = game
            self.analyzer = OpportunityAnalyzer(game)
            self.frames = 0
        observation, _ = opportunity_obs(
            self.env, game, self.analyzer, self.frames)
        self.frames += 1
        with self.torch.no_grad():
            scores = self.net(
                self.torch.as_tensor(observation).unsqueeze(0))[0]
        throttle, turn, fire = self.candidates[int(scores.argmax())]
        return {"forward": throttle == 2, "backup": throttle == 0,
                "turn_left": turn == 0, "turn_right": turn == 2,
                "fire": fire == 1}


def _eval_worker(job):
    net_path, seed0, count = job
    import torch
    torch.set_num_threads(1)
    from training.evaluate import play_round_dual_engine

    policy = OpportunityScoreNetPolicy(net_path)
    results = {"win": 0, "loss": 0, "double_death": 0, "draw": 0}
    for index in range(count):
        result = play_round_dual_engine(policy, seed0 + index)
        key = result["true_result"]
        results[key] = results.get(key, 0) + 1
    return results


def evaluate(net_path, n, workers, seed=970000):
    per_worker = max(1, n // workers)
    jobs = [(net_path, seed + worker * per_worker, per_worker)
            for worker in range(workers)]
    started = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_eval_worker, jobs)
    results = {}
    for part in parts:
        for key, value in part.items():
            results[key] = results.get(key, 0) + value
    total = sum(results.values())
    print(f"===== P25 原版验收 {total}局 @{seed} ({time.time()-started:.0f}s) =====")
    print(f"  真胜率 {results.get('win', 0)/total:.1%}  "
          f"负 {results.get('loss', 0)/total:.1%}  "
          f"双亡 {results.get('double_death', 0)/total:.1%}  "
          f"平 {results.get('draw', 0)/total:.1%}")
    print("  参照: P22冠军约68% | P24v2.1 58.7% | Laika镜像40.2%")
    return results


def teacher_evaluate(n, seed):
    from training.evaluate import evaluate_dual
    return evaluate_dual(OpportunityMPC(), n=n, base_seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pipeline", "collect", "dagger",
                                         "eval", "teacher-eval"])
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--workers", type=int,
                        default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--net", default=DEFAULT_NET)
    parser.add_argument("--student-net", default=DEFAULT_STUDENT)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=970000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()

    if args.mode == "teacher-eval":
        teacher_evaluate(args.n, args.seed)
        return
    if args.mode == "eval":
        evaluate(args.net, args.n, args.workers, args.seed)
        return
    if args.mode == "dagger":
        collect_dagger(args.rounds, args.workers, args.student_net,
                       args.epsilon)
        X, Y = load_opportunity_data()
        print("===== P25 DAgger 聚合重训 =====", flush=True)
        import torch
        network, metrics = train(X, Y, epochs=args.epochs)
        torch.save({"state_dict": network.state_dict(),
                    "in_dim": X.shape[1]}, args.net)
        mse, top1, top3 = metrics
        print(f"模型已保存 {args.net}: MSE {mse:.4f} "
              f"top1 {top1:.1%} top3 {top3:.1%}", flush=True)
        evaluate(args.net, args.n, args.workers, args.seed)
        return
    X, Y = collect(args.rounds, args.workers, args.epsilon)
    if args.mode == "collect":
        return
    print("===== P25 评分网络训练 =====", flush=True)
    import torch
    network, metrics = train(X, Y, epochs=args.epochs)
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": network.state_dict(), "in_dim": X.shape[1]},
               args.net)
    mse, top1, top3 = metrics
    print(f"模型已保存 {args.net}: MSE {mse:.4f} "
          f"top1 {top1:.1%} top3 {top3:.1%}", flush=True)
    evaluate(args.net, args.n, args.workers, args.seed)


if __name__ == "__main__":
    main()
