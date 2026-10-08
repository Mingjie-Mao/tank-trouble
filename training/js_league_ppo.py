"""League PPO on the JS engine, after KillField's Hybrid.

The environment is `web/rl/duel-env.js`: this repository's own physics and
Laika, one round per episode, paid once by the round's true result (a kill
followed by death to a bullet already in flight is a double death). Node
workers own the environments; this process owns the networks and talks to
them over pipes in lockstep.

Recipe, taken from Hybrid (`cichlider.github.io/killfield/paper`) unless noted:

* observation: schema 24, 1028 values, including the nine-move `dodge_safety`
  lookahead that Hybrid's matched ablation found worth +9.5 pp vs Laika;
* action: Discrete(18) = 3 throttle x 3 turn x fire, every engine frame;
* reward: terminal only. Win +1 inside 10 s decaying to +0.5 at 30 s, double
  death -0.1, loss and 30 s draw -1, plus up to +0.25 for changing action no
  more often than Laika does;
* opponents: 10% Laika / 10% Tactical (our search champion, standing in for
  Hybrid's 512-ray planner) / 80% frozen ancestors. Until the first ancestor
  exists the frozen share goes to Laika;
* PPO: 256 envs x 128 steps, 4 epochs x 8 minibatches, gamma 0.999, lambda
  0.95, clip 0.2, lr 3e-4 and entropy 0.01 both decaying linearly to zero
  over the *lineage* horizon (a resume continues the schedule, it does not
  restart it), value-only critic warm-up for the first 20 updates.

Usage:
    python3 -u training/js_league_ppo.py train --run r1 --total-steps 30000000
    python3 -u training/js_league_ppo.py train --run r1 --resume
    python3 -u training/js_league_ppo.py eval training/runs/js_league/r1/latest.pt \
        --opponent laika --games 1000 --seed 970000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import struct
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "web" / "rl" / "env-worker.mjs"
RUNS = ROOT / "training" / "runs" / "js_league"

# Layout of web/rl/duel-env.js, schema "tt-duel-24".
OBS_SCHEMA = "tt-duel-24"
OBS_DIM = 1028
MAP_W, MAP_H, MAP_C = 12, 10, 7
MAP_DIM = MAP_W * MAP_H * MAP_C
REST = OBS_DIM - MAP_DIM
BULLET_OFFSET = 900  # within the full observation
BULLET_SLOTS, BULLET_DIM = 10, 10
ACTIONS = 18
OUTCOMES = ("running", "win", "loss", "double", "draw")
SLOT_NAMES = ("laika", "tactical")

TRAIN_SEED_BASE = 50_000_000  # disjoint from the 970000/990000 grading bases
WORKER_SEED_STRIDE = 10_000_000


# ------------------------------------------------------------------ workers

class WorkerPool:
    """`workers` Node processes, `envs_per_worker` environments each.

    `send`/`recv` drive every worker in lockstep (evaluation); training drives
    them one at a time with `send_one`/`recv_one`, so a worker on an efficiency
    core or a slow opponent search never holds the others back.
    """

    def __init__(self, workers: int, envs_per_worker: int, seed: int, weights, wall_sliding=False):
        self.n_workers = workers
        self.e = envs_per_worker
        self.n = workers * envs_per_worker
        e = self.e
        self.layout = [
            ("rest", np.float32, 2 * e * REST), ("reward", np.float32, e),
            ("slot", np.int16, e), ("ended_slot", np.int16, e),
            ("frames", np.uint16, e), ("shots", np.uint16, e), ("hits", np.uint16, e),
            ("map", np.uint8, 2 * e * MAP_DIM), ("mask", np.uint8, 2 * e * BULLET_SLOTS),
            ("done", np.uint8, e), ("outcome", np.uint8, e),
        ]
        self.reply_bytes = sum(np.dtype(t).itemsize * n for _, t, n in self.layout)
        self.procs = []
        for w in range(workers):
            proc = subprocess.Popen(
                ["node", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                cwd=str(ROOT / "web"), bufsize=0,
            )
            self.procs.append(proc)
        for w, proc in enumerate(self.procs):
            self._send_json(proc, 1, {
                "envs": e, "seed": seed + w * WORKER_SEED_STRIDE,
                "weights": list(weights), "wallSliding": wall_sliding,
            })
        self.last = self._gather()

    @staticmethod
    def _send_json(proc, cmd, payload):
        body = json.dumps(payload).encode()
        proc.stdin.write(bytes([cmd]) + struct.pack("<I", len(body)) + body)

    def _read(self, proc, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            r = proc.stdout.readinto(view[got:])
            if not r:
                raise RuntimeError("env worker exited")
            got += r
        return buf

    def _parse(self, raw):
        out, at = {}, 0
        for name, dtype, count in self.layout:
            size = np.dtype(dtype).itemsize * count
            out[name] = np.frombuffer(raw, dtype=dtype, count=count, offset=at)
            at += size
        return out

    def _decode(self, part):
        """One worker's reply as per-env arrays."""
        e = self.e
        res = {}
        for seat in (0, 1):
            obs = np.empty((e, OBS_DIM), dtype=np.float32)
            obs[:, :MAP_DIM] = part["map"][seat * e * MAP_DIM:(seat + 1) * e * MAP_DIM].reshape(e, MAP_DIM)
            obs[:, MAP_DIM:] = part["rest"][seat * e * REST:(seat + 1) * e * REST].reshape(e, REST)
            res[f"obs{seat}"] = obs
            res[f"mask{seat}"] = part["mask"][seat * e * BULLET_SLOTS:(seat + 1) * e * BULLET_SLOTS] \
                .reshape(e, BULLET_SLOTS).astype(np.bool_)
        for key in ("reward", "slot", "ended_slot", "frames", "shots", "hits", "done", "outcome"):
            res[key] = part[key].copy()
        return res

    def fileno(self, w):
        return self.procs[w].stdout.fileno()

    def send_one(self, w, actions: np.ndarray, opponent_actions: np.ndarray):
        self.procs[w].stdin.write(
            b"\x02" + actions.astype(np.uint8).tobytes() + opponent_actions.astype(np.uint8).tobytes())

    def recv_one(self, w):
        return self._decode(self._parse(self._read(self.procs[w], self.reply_bytes)))

    def _gather(self):
        parts = [self.recv_one(w) for w in range(self.n_workers)]
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    def send(self, actions: np.ndarray, opponent_actions: np.ndarray):
        for w in range(self.n_workers):
            s = slice(w * self.e, (w + 1) * self.e)
            self.send_one(w, actions[s], opponent_actions[s])

    def recv(self):
        self.last = self._gather()
        return self.last

    def step(self, actions: np.ndarray, opponent_actions: np.ndarray):
        self.send(actions, opponent_actions)
        return self.recv()

    def set_weights(self, weights):
        for proc in self.procs:
            self._send_json(proc, 3, {"weights": list(weights)})
        for proc in self.procs:
            self._read(proc, 1)

    def close(self):
        for proc in self.procs:
            try:
                proc.stdin.write(b"\x04")
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.wait(timeout=10)


# -------------------------------------------------------------------- model

class ActorCritic(nn.Module):
    """Hybrid's encoder layout: map -> 128, per-bullet MLP pooled -> 64, scalars -> 128, trunk 256.

    One deliberate departure: the map branch is an MLP rather than Hybrid's
    two-layer CNN. On this machine's MPS backend a 3x3 conv over a 12x10 grid
    costs ~7x an MLP of the same output width (296 ms vs 39 ms per 4096-row
    forward+backward), which would make the update, not the environment, the
    bottleneck. The grid is small and position matters, so an MLP loses little.
    """

    def __init__(self):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(MAP_DIM, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
        )
        self.bullets = nn.Sequential(
            nn.Linear(BULLET_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        )
        self.bullet_out = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        n_scalars = OBS_DIM - MAP_DIM - BULLET_SLOTS * BULLET_DIM
        self.scalars = nn.Sequential(nn.Linear(n_scalars, 128), nn.Tanh(), nn.Linear(128, 128), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(128 + 64 + 128, 256), nn.ReLU())
        self.actor = nn.Linear(256, ACTIONS)
        self.critic = nn.Linear(256, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs, mask):
        n = obs.shape[0]
        m = self.map(obs[:, :MAP_DIM])
        rows = obs[:, BULLET_OFFSET:BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM].reshape(n, BULLET_SLOTS, BULLET_DIM)
        h = self.bullets(rows)
        w = mask.unsqueeze(-1).to(h.dtype)
        count = w.sum(1).clamp(min=1.0)
        mean = (h * w).sum(1) / count
        mx = (h * w + (w - 1.0) * 1e4).max(1).values
        mx = torch.where(w.sum(1) > 0, mx, torch.zeros_like(mx))
        b = self.bullet_out(torch.cat([mean, mx], 1))
        scalars = torch.cat([obs[:, MAP_DIM:BULLET_OFFSET], obs[:, BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM:]], 1)
        s = self.scalars(scalars)
        z = self.trunk(torch.cat([m, b, s], 1))
        return self.actor(z), self.critic(z).squeeze(-1)


@torch.inference_mode()
def act(model, obs, mask, device, greedy=False):
    """Sample (or argmax) an action per row. Called once per worker reply, so
    it avoids `Categorical`'s per-call validation overhead."""
    logits, value = model(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device))
    logp_all = torch.log_softmax(logits, -1)
    if greedy:
        action = logits.argmax(-1)
    else:
        action = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
    logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1)
    return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()


# --------------------------------------------------------------------- config

@dataclass
class Config:
    run: str = "r1"
    workers: int = 8
    envs_per_worker: int = 32
    rollout: int = 128
    total_steps: int = 30_000_000  # the lineage horizon the schedules decay over
    epochs: int = 4
    minibatches: int = 8
    lr: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    critic_warmup_updates: int = 20
    w_laika: float = 0.10
    w_tactical: float = 0.10
    w_frozen: float = 0.80
    # Within the frozen share, weight each ancestor by how often it actually
    # beats us (loss 1, double death 0.5) over its last `threat_window`
    # rounds, plus a floor. Not PFSP's (1 - win)^2: that also favours the
    # ancestors we merely stall against, which is the failure it should fix.
    threat_weighting: bool = False
    threat_floor: float = 0.05
    threat_window: int = 200
    snapshot_every: int = 1_048_576  # steps between frozen ancestors
    pool_size: int = 8  # most recent ancestors kept in the league
    seed: int = 1
    wall_sliding: bool = False
    # CPU by default: with other GPU tenants on this machine MPS ran the
    # update slower than eight CPU threads (263 ms vs 96 ms per minibatch).
    device: str = "cpu"
    update_threads: int = 8


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def league_weights(cfg: Config, pool_len: int, threat=None):
    """Slot weights: Laika, Tactical, then each frozen ancestor.

    `threat[k]` is ancestor k's recent outcomes (deque of outcome codes);
    used only when `cfg.threat_weighting` is on.
    """
    if pool_len == 0:
        return [cfg.w_laika + cfg.w_frozen, cfg.w_tactical]
    if not cfg.threat_weighting or threat is None:
        scores = [1.0] * pool_len
    else:
        scores = []
        for k in range(pool_len):
            seen = threat[k]
            beaten = sum(1.0 if o == 2 else 0.5 if o == 3 else 0.0 for o in seen)
            # An ancestor with no history yet is treated as fully threatening.
            scores.append(cfg.threat_floor + (beaten / len(seen) if seen else 1.0))
    total = sum(scores)
    return [cfg.w_laika, cfg.w_tactical] + [cfg.w_frozen * x / total for x in scores]


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------- train

def train(cfg: Config, resume: bool):
    run_dir = RUNS / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = ActorCritic().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    trained_steps, updates, pool = 0, 0, []
    latest = run_dir / "latest.pt"
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        assert state["schema"] == OBS_SCHEMA, f"checkpoint schema {state['schema']} != {OBS_SCHEMA}"
        model.load_state_dict(state["model"])
        # Adam state and the schedule position travel together, or the
        # learning rate silently restarts at its initial value.
        optimiser.load_state_dict(state["optimiser"])
        trained_steps, updates = state["trained_steps"], state["updates"]
        pool = [run_dir / name for name in state["pool"]]
        print(f"resumed at {trained_steps:,} steps, update {updates}, pool {len(pool)}")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Frozen opponents run on CPU: small batches, no device sync per frame.
    frozen = []
    for path in pool:
        net = ActorCritic()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
        net.eval()
        frozen.append(net)
    # Recent outcomes against each ancestor slot, for threat weighting. Not
    # checkpointed: after a resume every ancestor starts as fully threatening.
    threat = [deque(maxlen=cfg.threat_window) for _ in frozen]

    envs = WorkerPool(cfg.workers, cfg.envs_per_worker,
                      TRAIN_SEED_BASE + cfg.seed * 1_000_003 + trained_steps,
                      league_weights(cfg, len(frozen), threat), cfg.wall_sliding)
    n_workers, e = envs.n_workers, envs.e
    budget = envs.n * cfg.rollout
    # Workers advance independently, so a worker on a performance core may
    # collect more than its share of the budget before the update starts.
    cap = 2 * cfg.rollout
    bufs = [{
        "obs": np.zeros((cap, e, OBS_DIM), np.float32),
        "mask": np.zeros((cap, e, BULLET_SLOTS), np.bool_),
        "act": np.zeros((cap, e), np.int64), "logp": np.zeros((cap, e), np.float32),
        "val": np.zeros((cap, e), np.float32), "rew": np.zeros((cap, e), np.float32),
        "done": np.zeros((cap, e), np.float32),
    } for _ in range(n_workers)]
    current = [{k: v[w * e:(w + 1) * e] for k, v in envs.last.items()} for w in range(n_workers)]
    fd_worker = {envs.fileno(w): w for w in range(n_workers)}

    # Rollout inference runs on a CPU copy: batches of `e` rows, one per worker
    # reply, are too small to be worth a device round trip.
    torch.set_num_threads(1)
    actor = ActorCritic()
    actor.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    actor.eval()

    log_path = run_dir / "log.jsonl"
    next_snapshot = (trained_steps // cfg.snapshot_every + 1) * cfg.snapshot_every
    print(f"{n_workers} workers x {e} envs, {budget:,} steps per update, training on {device}; "
          f"pool {len(frozen)}; horizon {cfg.total_steps:,}")

    try:
        while trained_steps < cfg.total_steps:
            t0 = time.time()
            stats = {"episodes": 0}
            by_slot: dict[str, list[int]] = {}
            ep_frames, ep_shots, ep_hits = [], [], []
            lens = [0] * n_workers
            collected = 0
            inflight = set()

            def issue(w):
                s, t, b = current[w], lens[w], bufs[w]
                a, logp, v = act(actor, s["obs0"], s["mask0"], "cpu")
                opp = np.zeros(e, np.int64)
                for k, net in enumerate(frozen):
                    idx = np.nonzero(s["slot"] == k + 2)[0]
                    if idx.size:
                        opp[idx] = act(net, s["obs1"][idx], s["mask1"][idx], "cpu")[0]
                b["obs"][t], b["mask"][t], b["act"][t] = s["obs0"], s["mask0"], a
                b["logp"][t], b["val"][t] = logp, v
                envs.send_one(w, a, opp)
                inflight.add(w)

            for w in range(n_workers):
                issue(w)
            while inflight:
                ready, _, _ = select.select([envs.fileno(w) for w in inflight], [], [])
                for fd in ready:
                    w = fd_worker[fd]
                    s = envs.recv_one(w)
                    inflight.discard(w)
                    t = lens[w]
                    bufs[w]["rew"][t] = s["reward"]
                    bufs[w]["done"][t] = s["done"]
                    lens[w] += 1
                    collected += e
                    current[w] = s
                    for i in np.nonzero(s["done"])[0]:
                        slot = int(s["ended_slot"][i])
                        name = SLOT_NAMES[slot] if slot < 2 else "frozen"
                        by_slot.setdefault(name, [0] * 5)[int(s["outcome"][i])] += 1
                        if slot >= 2 and slot - 2 < len(threat):
                            threat[slot - 2].append(int(s["outcome"][i]))
                        ep_frames.append(int(s["frames"][i]))
                        ep_shots.append(int(s["shots"][i]))
                        ep_hits.append(int(s["hits"][i]))
                        stats["episodes"] += 1
                    if collected + len(inflight) * e < budget and lens[w] < cap:
                        issue(w)
            collect_time = time.time() - t0
            if cfg.threat_weighting and frozen:
                # Every worker is drained here, so the new weights apply to
                # the next episode each env starts.
                envs.set_weights(league_weights(cfg, len(frozen), threat))

            # GAE per worker over however many steps it collected. Every
            # terminal, including the 30 s draw, is a real terminal with its
            # own reward, so bootstrap is cut on all of them.
            parts = {k: [] for k in ("obs", "mask", "act", "logp", "adv", "ret")}
            for w in range(n_workers):
                t_len, b = lens[w], bufs[w]
                next_v = act(actor, current[w]["obs0"], current[w]["mask0"], "cpu")[2]
                adv = np.zeros((t_len, e), np.float32)
                last = np.zeros(e, np.float32)
                for t in reversed(range(t_len)):
                    alive = 1.0 - b["done"][t]
                    nv = next_v if t == t_len - 1 else b["val"][t + 1]
                    delta = b["rew"][t] + cfg.gamma * nv * alive - b["val"][t]
                    last = delta + cfg.gamma * cfg.gae_lambda * alive * last
                    adv[t] = last
                parts["obs"].append(b["obs"][:t_len].reshape(-1, OBS_DIM))
                parts["mask"].append(b["mask"][:t_len].reshape(-1, BULLET_SLOTS))
                parts["act"].append(b["act"][:t_len].reshape(-1))
                parts["logp"].append(b["logp"][:t_len].reshape(-1))
                parts["adv"].append(adv.reshape(-1))
                parts["ret"].append((adv + b["val"][:t_len]).reshape(-1))
            flat = {k: np.concatenate(v) for k, v in parts.items()}
            batch = flat["act"].shape[0]

            progress = min(1.0, trained_steps / cfg.total_steps)
            lr = cfg.lr * (1.0 - progress)
            ent_coef = cfg.entropy_coef * (1.0 - progress)
            for g in optimiser.param_groups:
                g["lr"] = lr
            warmup = updates < cfg.critic_warmup_updates

            # The workers are idle during the update, so it may take the cores.
            torch.set_num_threads(cfg.update_threads)
            model.train()
            b_obs = torch.from_numpy(flat["obs"]).to(device)
            b_mask = torch.from_numpy(flat["mask"]).to(device)
            b_act = torch.from_numpy(flat["act"]).to(device)
            b_logp = torch.from_numpy(flat["logp"]).to(device)
            b_adv = torch.from_numpy(flat["adv"]).to(device)
            b_ret = torch.from_numpy(flat["ret"]).to(device)
            mb = batch // cfg.minibatches
            clipfracs, kls, ents, vlosses, plosses = [], [], [], [], []
            tu = time.time()
            for _ in range(cfg.epochs):
                perm = torch.from_numpy(rng.permutation(batch)).to(device)
                for k in range(cfg.minibatches):
                    idx = perm[k * mb:(k + 1) * mb]
                    logits, value = model(b_obs[idx], b_mask[idx])
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(b_act[idx])
                    entropy = dist.entropy().mean()
                    ratio = (logp - b_logp[idx]).exp()
                    a_mb = b_adv[idx]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    v_loss = 0.5 * F.mse_loss(value, b_ret[idx])
                    if warmup:
                        loss = cfg.value_coef * v_loss
                    else:
                        loss = pg + cfg.value_coef * v_loss - ent_coef * entropy
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimiser.step()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - (logp - b_logp[idx])).mean().item())
                        clipfracs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    ents.append(entropy.item())
                    vlosses.append(v_loss.item())
                    plosses.append(pg.item())
            update_time = time.time() - tu
            actor.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            torch.set_num_threads(1)

            trained_steps += batch
            updates += 1
            elapsed = time.time() - t0
            totals = [0] * 5
            for counts in by_slot.values():
                totals = [x + y for x, y in zip(totals, counts)]
            row = {
                "update": updates, "steps": trained_steps, "lr": lr, "ent_coef": ent_coef,
                "warmup": warmup, "sps": round(batch / elapsed), "collect_s": round(collect_time, 1),
                "update_s": round(update_time, 1), "pool": len(frozen),
                "worker_steps": lens,
                "league_weights": [round(x, 4) for x in league_weights(cfg, len(frozen), threat)],
                "entropy": float(np.mean(ents)), "v_loss": float(np.mean(vlosses)),
                "pg_loss": float(np.mean(plosses)), "kl": float(np.mean(kls)),
                "clipfrac": float(np.mean(clipfracs)), "episodes": stats["episodes"],
                "frames": float(np.mean(ep_frames)) if ep_frames else None,
                "shots": float(np.mean(ep_shots)) if ep_shots else None,
                "hits": float(np.mean(ep_hits)) if ep_hits else None,
                "by_opponent": {k: dict(zip(OUTCOMES[1:], v[1:])) for k, v in by_slot.items()},
            }
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            laika = by_slot.get("laika")
            laika_txt = (f"laika win {laika[1] / max(1, sum(laika)):.1%} of {sum(laika)}"
                         if laika else "laika -")
            print(f"u{updates} {trained_steps / 1e6:.2f}M sps {row['sps']} "
                  f"(collect {collect_time:.0f}s upd {update_time:.0f}s) ent {row['entropy']:.2f} "
                  f"v {row['v_loss']:.3f} kl {row['kl']:.4f} | {laika_txt} | "
                  f"all W/L/D/T {totals[1]}/{totals[2]}/{totals[3]}/{totals[4]}"
                  + (" [critic warm-up]" if warmup else ""), flush=True)

            checkpoint = {
                "schema": OBS_SCHEMA, "model": model.state_dict(),
                "optimiser": optimiser.state_dict(), "trained_steps": trained_steps,
                "updates": updates, "pool": [p.name for p in pool], "config": asdict(cfg),
            }
            atomic_save(checkpoint, latest)
            if trained_steps >= next_snapshot:
                made = len(list(run_dir.glob("gen*.pt")))
                gen = run_dir / f"gen{made:03d}.pt"
                atomic_save({"schema": OBS_SCHEMA, "model": model.state_dict(),
                             "trained_steps": trained_steps}, gen)
                net = ActorCritic()
                net.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
                net.eval()
                # A ring, not a queue: an ancestor keeps its slot index for as
                # long as it is in the league, so an episode already running
                # against slot k is not handed to a different network.
                if len(frozen) < cfg.pool_size:
                    frozen.append(net)
                    threat.append(deque(maxlen=cfg.threat_window))
                    pool.append(gen)
                else:
                    frozen[made % cfg.pool_size] = net
                    threat[made % cfg.pool_size] = deque(maxlen=cfg.threat_window)
                    pool[made % cfg.pool_size] = gen
                envs.set_weights(league_weights(cfg, len(frozen), threat))
                next_snapshot += cfg.snapshot_every
                print(f"  snapshot {gen.name}; league = laika, tactical + {len(frozen)} ancestors")
                checkpoint["pool"] = [p.name for p in pool]
                atomic_save(checkpoint, latest)
    finally:
        envs.close()


# ----------------------------------------------------------------------- eval

DODGE_OFFSET = 1018  # nine dodge_safety values, one per [throttle, turn] move
AIM_SELF_SUICIDE = 891  # "the current barrel's shot would hit me"
SHIELDS = ("none", "move", "move+fire")


@torch.inference_mode()
def shielded_action(model, obs, mask, rng, greedy=False, shield="none"):
    """The policy's action, optionally filtered by the observation's own lookahead.

    `move`: drop every action whose movement `dodge_safety` predicts dies
    within 24 frames, when at least one movement survives; if none does,
    keep only the movement that lasts longest. `move+fire` also drops the
    fire actions when the current barrel's shot is forecast to hit the shooter.
    Both read channels the policy already sees, so this is a test of whether
    the network uses them as well as a hard rule would.
    """
    logits, _ = model(torch.from_numpy(obs), torch.from_numpy(mask))
    probs = torch.softmax(logits, -1).numpy().astype(np.float64)
    overrides = 0
    if shield != "none":
        move_of = np.arange(ACTIONS) // 2
        fire_of = np.arange(ACTIONS) % 2
        for i in range(obs.shape[0]):
            dodge = obs[i, DODGE_OFFSET:DODGE_OFFSET + 9]
            keep = dodge[move_of] >= 0 if (dodge >= 0).any() else dodge[move_of] == dodge.max()
            if shield == "move+fire" and obs[i, AIM_SELF_SUICIDE] > 0.5:
                keep = keep & (fire_of == 0)
            before = probs[i].argmax()
            masked = np.where(keep, probs[i], 0.0)
            if masked.sum() > 0:
                probs[i] = masked / masked.sum()
            overrides += int(not keep[before])
    if greedy:
        return probs.argmax(1), overrides
    u = rng.random((probs.shape[0], 1))
    return np.minimum((probs.cumsum(1) < u).sum(1), ACTIONS - 1), overrides


def evaluate(path: Path, opponent: str, games: int, seed: int, workers: int, greedy: bool,
             wall_sliding: bool, shield: str = "none", out: Path | None = None):
    """True result on seeds seed .. seed+games-1, one round each, in parallel envs.

    A worker given base seed s hands its envs seeds s+1, s+2, ... at start, so
    worker w is started at seed - 1 + w * per and every env is retired after
    its first episode. Per-seed outcomes are kept so two arms over the same
    seeds can be compared pairwise.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["schema"] == OBS_SCHEMA
    model = ActorCritic()
    model.load_state_dict(state["model"])
    model.eval()
    slot = {"laika": 0, "tactical": 1}[opponent]
    weights = [0.0, 0.0]
    weights[slot] = 1.0
    per = math.ceil(games / workers)
    counts = [0] * 5
    frames = []
    t0 = time.time()
    # One worker per seed block; its envs take consecutive seeds at init.
    procs = []
    for w in range(workers):
        start = seed + w * per
        n = min(per, seed + games - start)
        if n <= 0:
            break
        procs.append(WorkerPool(1, n, start - 1, weights, wall_sliding))
    live = [np.ones(p.n, bool) for p in procs]
    rng = np.random.default_rng(seed)
    per_seed = {}
    overrides = 0
    while any(alive.any() for alive in live):
        active = [(w, p, alive) for w, (p, alive) in enumerate(zip(procs, live)) if alive.any()]
        for _, p, _ in active:
            a, n_over = shielded_action(model, p.last["obs0"], p.last["mask0"], rng, greedy, shield)
            overrides += n_over
            p.send(a, np.zeros(p.n, np.int64))
        for w, p, alive in active:
            st = p.recv()
            for i in np.nonzero(st["done"].astype(bool) & alive)[0]:
                counts[int(st["outcome"][i])] += 1
                frames.append(int(st["frames"][i]))
                per_seed[seed + w * per + int(i)] = OUTCOMES[int(st["outcome"][i])]
                alive[i] = False
    for p in procs:
        p.close()
    played = sum(counts)
    wins = counts[1]
    se = math.sqrt(max(wins / played * (1 - wins / played), 1e-12) / played)
    result = {
        "checkpoint": str(path), "opponent": opponent, "seed": seed, "games": played,
        "greedy": greedy, "win": wins, "loss": counts[2], "double": counts[3], "draw": counts[4],
        "win_rate": wins / played, "ci95": [wins / played - 1.96 * se, wins / played + 1.96 * se],
        "mean_frames": float(np.mean(frames)), "seconds": round(time.time() - t0, 1),
        "shield": shield, "shield_overrides": overrides,
    }
    print(json.dumps(result, indent=2))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**result, "per_seed": per_seed}))
    return result


# --------------------------------------------------------------------- export

def export(path: Path, out: Path, label: str):
    """Write a checkpoint as an ES module for `web/rl/ppo-agent.js`, plus a parity case.

    Weights are float32, base64-encoded per tensor. The parity case is one
    real observation from the environment with PyTorch's logits for it, which
    the browser forward pass must reproduce.
    """
    import base64

    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["schema"] == OBS_SCHEMA
    model = ActorCritic()
    model.load_state_dict(state["model"])
    model.eval()
    tensors = {
        name: {"shape": list(t.shape),
               "data": base64.b64encode(t.detach().float().contiguous().numpy().tobytes()).decode()}
        for name, t in model.state_dict().items() if not name.startswith("critic")
    }
    meta = {"schema": OBS_SCHEMA, "label": label, "checkpoint": path.name,
            "run": path.parent.name, "trainedSteps": int(state["trained_steps"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "// Generated by training/js_league_ppo.py export. Do not edit.\n"
        f"export const PPO_META = {json.dumps(meta)};\n"
        f"export const PPO_TENSORS = {json.dumps(tensors)};\n")

    # Parity: a mid-round state with bullets in flight, from the real env.
    pool = WorkerPool(1, 4, 7, [1.0, 0.0])
    rng = np.random.default_rng(0)
    st = pool.last
    for _ in range(60):
        st = pool.step(rng.integers(0, ACTIONS, pool.n), np.zeros(pool.n, np.int64))
    pool.close()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(st["obs0"]), torch.from_numpy(st["mask0"]))
    parity = {"obs": st["obs0"].tolist(), "mask": st["mask0"].astype(int).tolist(),
              "logits": logits.numpy().tolist()}
    parity_path = out.with_suffix(".parity.json")
    parity_path.write_text(json.dumps(parity))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) and {parity_path.name}; "
          f"{meta['trainedSteps']:,} steps")


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    for f_ in Config.__dataclass_fields__.values():
        flag = "--" + f_.name.replace("_", "-")
        if f_.type in ("bool", bool):
            tr.add_argument(flag, action="store_true", default=f_.default)
        else:
            tr.add_argument(flag, type=type(f_.default), default=f_.default)
    tr.add_argument("--resume", action="store_true")
    ev = sub.add_parser("eval")
    ev.add_argument("checkpoint", type=Path)
    ev.add_argument("--opponent", choices=("laika", "tactical"), default="laika")
    ev.add_argument("--games", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=970000)
    ev.add_argument("--workers", type=int, default=8)
    ev.add_argument("--sample", action="store_true", help="sample actions instead of argmax")
    ev.add_argument("--wall-sliding", action="store_true")
    ev.add_argument("--shield", choices=SHIELDS, default="none")
    ev.add_argument("--out", type=Path, default=None, help="write the result with per-seed outcomes")
    ex = sub.add_parser("export")
    ex.add_argument("checkpoint", type=Path)
    ex.add_argument("--out", type=Path, default=ROOT / "web" / "rl" / "models" / "ppo-league.js")
    ex.add_argument("--label", default="League PPO")
    args = parser.parse_args()
    if args.command == "train":
        cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
        train(cfg, args.resume)
    elif args.command == "export":
        export(args.checkpoint, args.out, args.label)
    else:
        evaluate(args.checkpoint, args.opponent, args.games, args.seed, args.workers,
                 not args.sample, args.wall_sliding, args.shield, args.out)


if __name__ == "__main__":
    main()
