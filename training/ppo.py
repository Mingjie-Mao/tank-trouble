"""League PPO on the Rust environment (`engine/py`, module `tank_rl`).

Rounds are stepped in parallel in Rust with observations written straight
into NumPy; this process owns the networks. Each round's opponent is Laika
with probability `laika_share`, otherwise a frozen ancestor of the learner
(a ring of `pool_size`, one published every `snapshot_every` steps). The only
reward is the round's true result, paid at its end. With teacher labels
(`label`), every update also pulls the policy toward the search's choices.

The champion's recipe:
    python3 -u training/ppo.py train --run s3 --seed 3 --total-steps 30000000
    python3 -u training/ppo.py train --run s3_long --seed 3 --init training/runs/rust/s3/latest.pt \
        --lr 1e-4 --gamma 0.9997 --total-steps 60000000
    python3 training/ppo.py export training/runs/rust/s3_long/latest.pt --out s3_long.policy
    python3 training/ppo.py label s3_long.policy --out labels.npz --seed 82000000 --rounds 6000 --rollouts 8
    python3 -u training/ppo.py train --run champion --seed 3 --init training/runs/rust/s3_long/latest.pt \
        --lr 5e-5 --gamma 0.9997 --total-steps 20000000 --aux-data labels.npz --aux-coef 1
    python3 training/ppo.py eval training/runs/rust/champion/latest.pt --seeds 970000,990000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tank_rl

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "training" / "runs" / "rust"
ACTIONS = 18
OUTCOMES = ("running", "win", "loss", "double", "draw")
# ------------------------------------------------------------------ model

class ActorCritic(nn.Module):
    """Two branches over the observation's layout, then a shared trunk.

    bullets: one row MLP shared by every slot, pooled by masked mean and max so
    slot order carries no meaning; scalars: everything else. `width` is the
    trunk's; the scalar branch is half as wide.
    """

    def __init__(self, obs_dim: int, layout: tuple, width: int = 256):
        super().__init__()
        _, _, b_start, slots, b_dim, *_ = layout
        self.obs_dim, self.layout = obs_dim, tuple(layout)
        self.b_slice = slice(b_start, b_start + slots * b_dim)
        self.slots, self.b_dim = slots, b_dim
        keep = np.ones(obs_dim, bool)
        keep[self.b_slice] = False
        self.register_buffer("scalar_idx", torch.from_numpy(np.nonzero(keep)[0]), persistent=False)
        self.bullets = nn.Sequential(nn.Linear(b_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.bullet_out = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        half = width // 2
        self.scalars = nn.Sequential(nn.Linear(int(keep.sum()), half), nn.Tanh(), nn.Linear(half, half), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 + half, width), nn.ReLU())
        self.actor = nn.Linear(width, ACTIONS)
        self.critic = nn.Linear(width, 1)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs, mask):
        n = obs.shape[0]
        parts = []
        h = self.bullets(obs[:, self.b_slice].reshape(n, self.slots, self.b_dim))
        w = mask.unsqueeze(-1).to(h.dtype)
        mean = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        mx = torch.where(w.sum(1) > 0, (h * w + (w - 1.0) * 1e4).max(1).values, torch.zeros_like(mean))
        parts.append(self.bullet_out(torch.cat([mean, mx], 1)))
        parts.append(self.scalars(obs[:, self.scalar_idx]))
        z = self.trunk(torch.cat(parts, 1))
        return self.actor(z), self.critic(z).squeeze(-1)


def width_of(state) -> int:
    """A checkpoint's trunk width, read off its weights."""
    return state["model"]["trunk.0.weight"].shape[0]


def shield_probs(probs: np.ndarray, obs: np.ndarray, layout: tuple) -> np.ndarray:
    """Drop moves the nine-move lookahead says die (if any survives) and fire
    while the current shot would hit the shooter."""
    dodge_start, self_hit = layout[5], layout[6]
    move_of = np.arange(ACTIONS) // 2
    fire_of = np.arange(ACTIONS) % 2
    dodge = obs[:, dodge_start:dodge_start + 9]
    safe = dodge >= 0
    any_safe = safe.any(1, keepdims=True)
    keep = np.where(any_safe, safe[:, move_of], dodge[:, move_of] == dodge.max(1, keepdims=True))
    keep &= ~((obs[:, self_hit] > 0.5)[:, None] & (fire_of == 1)[None, :])
    masked = np.where(keep, probs, 0.0)
    total = masked.sum(1, keepdims=True)
    return np.where(total > 0, masked / np.maximum(total, 1e-12), probs)


def sample(probs: np.ndarray, rng) -> np.ndarray:
    u = rng.random((probs.shape[0], 1))
    return np.minimum((probs.cumsum(1) < u).sum(1), ACTIONS - 1)


# ------------------------------------------------------------------ config

@dataclass
class Config:
    run: str = "v2"
    seed: int = 1
    width: int = 256
    envs: int = 512
    rollout: int = 128
    total_steps: int = 30_000_000
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
    laika_share: float = 0.2
    snapshot_every: int = 1_048_576
    pool_size: int = 8
    # The round's result only: win +1, loss and 30 s draw -1, double KO -0.5.
    double_ko: float = -0.5
    draw: float = -1.0
    env_threads: int = 9
    torch_threads: int = 8
    # Start from these weights (a fresh run otherwise).
    init: str = ""
    # Expert iteration inside PPO: comma-separated teacher label files, whose
    # scored frames add aux_coef * KL(teacher target || policy) to every minibatch.
    aux_data: str = ""
    aux_coef: float = 0.0
    aux_beta: float = 16.0
    aux_pref_weight: float = 4.0


TRAIN_SEED_BASE = 50_000_000  # disjoint from the grading bases


def load_aux(cfg):
    """Scored frames of the label files, as (obs, mask, target, weight)."""
    parts = [np.load(d) for d in cfg.aux_data.split(",")]
    scores = torch.from_numpy(np.concatenate([p["scores"] for p in parts]))
    keep = (~torch.isnan(scores)).any(1)
    obs = torch.from_numpy(np.concatenate([p["obs"] for p in parts]))[keep]
    mask = torch.from_numpy(np.concatenate([p["mask"] for p in parts]).astype(bool))[keep]
    probs = torch.from_numpy(np.concatenate([p["probs"] for p in parts]).astype(np.float32))[keep]
    scores = scores[keep]
    target = teacher_targets(probs, scores, cfg.aux_beta)
    hi = torch.where(torch.isnan(scores), torch.full_like(scores, -9.0), scores).max(1).values
    lo = torch.where(torch.isnan(scores), torch.full_like(scores, 9.0), scores).min(1).values
    weight = torch.where(hi > lo, torch.tensor(cfg.aux_pref_weight), torch.tensor(1.0))
    print(f"aux: {len(obs):,} scored frames, {int((hi > lo).sum()):,} with a preference")
    return obs, mask, target, weight


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------ train

def train(cfg: Config, resume: bool):
    run_dir = RUNS / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    rng = np.random.default_rng(cfg.seed)

    trained, updates, pool = 0, 0, []
    latest = run_dir / "latest.pt"
    env = tank_rl.VecEnv(cfg.envs, TRAIN_SEED_BASE + cfg.seed * 10_000_019, laika_share=cfg.laika_share,
                         threads=cfg.env_threads, double_ko=cfg.double_ko, draw=cfg.draw)
    model = ActorCritic(env.obs_dim, env.layout, cfg.width)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    if cfg.init and not resume:
        model.load_state_dict(torch.load(cfg.init, map_location="cpu", weights_only=False)["model"])
        print(f"initialised from {cfg.init}")
    aux = load_aux(cfg) if cfg.aux_data and cfg.aux_coef > 0 else None
    if resume:
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        trained, updates, pool = state["trained_steps"], state["updates"], state["pool"]
        # A fresh environment stream after a resume: skip the seeds already played.
        env = tank_rl.VecEnv(cfg.envs, TRAIN_SEED_BASE + cfg.seed * 10_000_019 + trained, laika_share=cfg.laika_share,
                             threads=cfg.env_threads, double_ko=cfg.double_ko, draw=cfg.draw)
        print(f"resumed at {trained:,} steps, update {updates}, pool {len(pool)}")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    def frozen_net(name):
        net = ActorCritic(env.obs_dim, env.layout, cfg.width)
        net.load_state_dict(torch.load(run_dir / name, map_location="cpu", weights_only=False)["model"])
        net.eval()
        return net

    if not pool:
        name = "gen000.pt"
        atomic_save({"model": model.state_dict(), "layout": env.layout, "obs_dim": env.obs_dim}, run_dir / name)
        pool = [name]
    frozen = [frozen_net(name) for name in pool]
    n, T = cfg.envs, cfg.rollout
    ancestor = rng.integers(0, len(frozen), n)
    obs0, mask0, obs1, mask1 = env.observe()
    policy = env.policy_slots().astype(bool)
    budget = n * T
    log = open(run_dir / "log.jsonl", "a")
    recent = deque(maxlen=400)  # outcomes against Laika
    recent_all = deque(maxlen=2000)
    next_snapshot = (trained // cfg.snapshot_every + 1) * cfg.snapshot_every

    buf_obs = np.zeros((T, n, env.obs_dim), np.float32)
    buf_mask = np.zeros((T, n, mask0.shape[1]), bool)
    buf_act = np.zeros((T, n), np.int64)
    buf_logp = np.zeros((T, n), np.float32)
    buf_val = np.zeros((T, n), np.float32)
    buf_rew = np.zeros((T, n), np.float32)
    buf_done = np.zeros((T, n), np.float32)

    while trained < cfg.total_steps:
        progress = trained / cfg.total_steps
        lr = cfg.lr * (1.0 - progress)
        ent_coef = cfg.entropy_coef * (1.0 - progress)
        for group in optimiser.param_groups:
            group["lr"] = lr
        t0 = time.time()
        model.eval()
        for t in range(T):
            o = torch.from_numpy(obs0)
            m = torch.from_numpy(mask0.astype(bool))
            with torch.inference_mode():
                logits, value = model(o, m)
                logp_all = torch.log_softmax(logits, -1)
                act = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                logp = logp_all.gather(1, act[:, None]).squeeze(1)
            opp = np.zeros(n, np.int64)
            if policy.any():
                for k in np.unique(ancestor[policy]):
                    rows = np.nonzero(policy & (ancestor == k))[0]
                    with torch.inference_mode():
                        ol, _ = frozen[k](torch.from_numpy(obs1[rows]), torch.from_numpy(mask1[rows].astype(bool)))
                    opp[rows] = torch.multinomial(torch.softmax(ol, -1), 1).squeeze(1).numpy()
            buf_obs[t], buf_mask[t] = obs0, mask0.astype(bool)
            buf_act[t], buf_logp[t], buf_val[t] = act.numpy(), logp.numpy(), value.numpy()
            reward, outcome, frames, seeds = env.step(act.numpy(), opp)
            ended = outcome > 0
            buf_rew[t], buf_done[t] = reward, ended
            for i in np.nonzero(ended)[0]:
                recent_all.append(int(outcome[i]))
                if not policy[i]:
                    recent.append(int(outcome[i]))
            if ended.any():
                policy = env.policy_slots().astype(bool)
                ancestor[ended] = rng.integers(0, len(frozen), int(ended.sum()))
            obs0, mask0, obs1, mask1 = env.observe()
        collect = time.time() - t0

        # GAE over each slot's trajectory; a terminal cuts the bootstrap.
        with torch.inference_mode():
            _, last_v = model(torch.from_numpy(obs0), torch.from_numpy(mask0.astype(bool)))
        adv = np.zeros((T, n), np.float32)
        gae = np.zeros(n, np.float32)
        next_v = last_v.numpy()
        for t in reversed(range(T)):
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + cfg.gamma * next_v * nonterm - buf_val[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
            adv[t] = gae
            next_v = buf_val[t]
        ret = adv + buf_val

        t1 = time.time()
        model.train()
        b_obs = torch.from_numpy(buf_obs.reshape(budget, -1))
        b_mask = torch.from_numpy(buf_mask.reshape(budget, -1))
        b_act = torch.from_numpy(buf_act.reshape(budget))
        b_logp = torch.from_numpy(buf_logp.reshape(budget))
        b_adv = torch.from_numpy(adv.reshape(budget))
        b_ret = torch.from_numpy(ret.reshape(budget))
        warmup = updates < cfg.critic_warmup_updates
        mb = budget // cfg.minibatches
        stats = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(budget)
            for k in range(cfg.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                logits, value = model(b_obs[idx], b_mask[idx])
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, b_act[idx][:, None]).squeeze(1)
                entropy = -(logp_all.exp() * logp_all).sum(1).mean()
                ratio = (logp - b_logp[idx]).exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * F.mse_loss(value, b_ret[idx])
                loss = cfg.value_coef * v_loss if warmup else pg + cfg.value_coef * v_loss - ent_coef * entropy
                aux_kl = torch.zeros(())
                if aux is not None and not warmup:
                    j = torch.randint(0, len(aux[0]), (mb,))
                    t = aux[2][j]
                    aux_logp = torch.log_softmax(model(aux[0][j], aux[1][j])[0], -1)
                    kl_rows = (t * (torch.log(t.clamp_min(1e-12)) - aux_logp)).sum(1)
                    aux_kl = (aux[3][j] * kl_rows).sum() / aux[3][j].sum()
                    loss = loss + cfg.aux_coef * aux_kl
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimiser.step()
                with torch.no_grad():
                    stats.append((entropy.item(), v_loss.item(), ((ratio - 1) - (logp - b_logp[idx])).mean().item(),
                                  aux_kl.item()))
        update_time = time.time() - t1
        trained += budget
        updates += 1

        ent, vl, kl, akl = np.mean(stats, 0)
        laika = recent.count(1) / max(1, len(recent))
        tally = {name: recent_all.count(code) for code, name in enumerate(OUTCOMES) if code}
        sps = budget / (collect + update_time)
        print(f"u{updates} {trained/1e6:.2f}M sps {sps:,.0f} (collect {collect:.1f}s upd {update_time:.1f}s) "
              f"ent {ent:.2f} v {vl:.3f} kl {kl:.4f}" + (f" aux {akl:.3f}" if aux is not None else "")
              + f" | laika win {laika:.1%} of {len(recent)} | all {tally}"
              + (" [critic warm-up]" if warmup else ""), flush=True)
        log.write(json.dumps({"update": updates, "steps": trained, "sps": sps, "entropy": ent, "value_loss": vl,
                              "kl": kl, "aux_kl": akl, "laika_win": laika, "laika_games": len(recent), **tally}) + "\n")
        log.flush()

        if trained >= next_snapshot:
            next_snapshot += cfg.snapshot_every
            name = f"gen{trained // cfg.snapshot_every:03d}.pt"
            atomic_save({"model": model.state_dict(), "layout": env.layout, "obs_dim": env.obs_dim}, run_dir / name)
            pool = (pool + [name])[-cfg.pool_size:]
            frozen = [frozen_net(p) for p in pool]
            ancestor = rng.integers(0, len(frozen), n)
            print(f"  snapshot {name}; pool {len(pool)}", flush=True)
        if updates % 10 == 0 or trained >= cfg.total_steps:
            atomic_save({"model": model.state_dict(), "optimiser": optimiser.state_dict(), "trained_steps": trained,
                         "updates": updates, "pool": pool, "layout": env.layout, "obs_dim": env.obs_dim,
                         "config": asdict(cfg)}, latest)


# ------------------------------------------------------------------ eval

def evaluate(path: Path, seed_bases: list[int], games: int, shield: bool, out: Path | None):
    """True win rate against Laika, one round per seed, sampled actions."""
    torch.set_num_threads(8)
    state = torch.load(path, map_location="cpu", weights_only=False)
    seeds = [s for base in seed_bases for s in range(base, base + games)]
    env = tank_rl.VecEnv(250, 0, threads=9, seeds=seeds)
    model = ActorCritic(env.obs_dim, env.layout, width_of(state))
    model.load_state_dict(state["model"])
    model.eval()
    rng = np.random.default_rng(0)
    per_seed = {}
    t0 = time.time()
    with torch.inference_mode():
        while not env.all_done():
            obs0, mask0, _, _ = env.observe()
            logits, _ = model(torch.from_numpy(obs0), torch.from_numpy(mask0.astype(bool)))
            probs = torch.softmax(logits, -1).numpy().astype(np.float64)
            if shield:
                probs = shield_probs(probs, obs0, env.layout)
            _, outcome, _, seed = env.step(sample(probs, rng), np.zeros(len(probs), np.int64))
            for i in np.nonzero(outcome)[0]:
                per_seed[str(int(seed[i]))] = OUTCOMES[int(outcome[i])]
    counts = {name: list(per_seed.values()).count(name) for name in OUTCOMES[1:]}
    p = counts["win"] / len(per_seed)
    se = math.sqrt(p * (1 - p) / len(per_seed))
    result = {"checkpoint": str(path), "seeds": seed_bases, "games": len(per_seed), "shield": shield, **counts,
              "win_rate": p, "ci95": [p - 1.96 * se, p + 1.96 * se], "seconds": round(time.time() - t0, 1)}
    print(json.dumps(result, indent=2))
    if out:
        out.write_text(json.dumps({**result, "per_seed": per_seed}))


# ------------------------------------------------------------------ teacher

def label(policy_file: Path, out: Path, seed: int, rounds: int, candidates: int, rollouts: int, horizon: int, calm: float):
    """Teacher labels from the student's own rounds against Laika (Rust, in parallel)."""
    t0 = time.time()
    obs, mask, probs, scores, danger, outcomes = tank_rl.teacher_labels(
        str(policy_file), list(range(seed, seed + rounds)), candidates, rollouts, horizon, calm)
    np.savez_compressed(out, obs=obs, mask=mask, probs=probs, scores=scores, danger=danger)
    spread = np.nan_to_num(np.nanmax(scores, 1) - np.nanmin(scores, 1)) if danger.any() else np.zeros(0)
    print(f"{rounds} rounds ({(outcomes == 1).mean():.1%} won): {len(danger):,} frames, {int(danger.sum()):,} scored, "
          f"{int((spread > 0).sum()):,} with a preference, in {time.time() - t0:.0f}s -> {out}")


def teacher_targets(p_old: torch.Tensor, scores: torch.Tensor, beta: float) -> torch.Tensor:
    """The student's distribution tilted by exp(beta * score); an unscored
    action is charged the frame's worst score; unscored frames keep p_old."""
    scored = ~torch.isnan(scores)
    worst = torch.where(scored, scores, torch.full_like(scores, float("inf"))).min(1, keepdim=True).values
    s = torch.where(scored, scores, worst)
    s = torch.where(scored.any(1, keepdim=True), s, torch.zeros_like(s))
    return torch.softmax(torch.log(p_old.clamp_min(1e-30)) + beta * (s - s.mean(1, keepdim=True)), 1)


def export(path: Path, out: Path):
    """Write the network for the Rust/WASM runtime: u32 header length, a JSON
    header (layout, tensor shapes and offsets, parity cases), then f32 data.
    The parity cases are real observations with PyTorch's logits and value."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    env = tank_rl.VecEnv(8, 123_456, threads=2)
    obs_dim, layout = env.obs_dim, tuple(env.layout)
    model = ActorCritic(obs_dim, layout, width_of(state))
    model.load_state_dict(state["model"])
    model.eval()
    rng = np.random.default_rng(0)
    for _ in range(60):  # into the round, with bullets in flight
        env.observe()
        env.step(rng.integers(0, ACTIONS, 8), rng.integers(0, ACTIONS, 8))
    obs0, mask0, _, _ = env.observe()
    with torch.inference_mode():
        logits, value = model(torch.from_numpy(obs0), torch.from_numpy(mask0.astype(bool)))
    tensors, blobs, offset = [], [], 0
    for name, t in model.state_dict().items():
        a = t.detach().float().contiguous().numpy().ravel()
        tensors.append({"name": name, "shape": list(t.shape), "offset": offset})
        blobs.append(a.astype("<f4").tobytes())
        offset += a.size
    header = json.dumps({"obs_dim": obs_dim, "layout": list(layout), "tensors": tensors, "source": str(path),
                         "trained_steps": int(state.get("trained_steps", 0)),
                         "parity": [{"obs": obs0[i].tolist(), "mask": mask0[i].tolist(), "logits": logits[i].tolist(),
                                     "value": float(value[i])}
                                    for i in range(len(obs0))]}).encode()
    out.write_bytes(len(header).to_bytes(4, "little") + header + b"".join(blobs))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), {offset:,} weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    for f in Config.__dataclass_fields__.values():
        tr.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    tr.add_argument("--resume", action="store_true")
    ev = sub.add_parser("eval")
    ev.add_argument("checkpoint", type=Path)
    ev.add_argument("--seeds", default="970000,990000", help="comma-separated seed bases")
    ev.add_argument("--games", type=int, default=1000, help="rounds per seed base")
    ev.add_argument("--no-shield", action="store_true")
    ev.add_argument("--out", type=Path)
    ex = sub.add_parser("export")
    ex.add_argument("checkpoint", type=Path)
    ex.add_argument("--out", type=Path, required=True)
    lb = sub.add_parser("label")
    lb.add_argument("policy", type=Path, help="an exported policy file")
    lb.add_argument("--out", type=Path, required=True)
    lb.add_argument("--seed", type=int, default=80_000_000)
    lb.add_argument("--rounds", type=int, default=2000)
    lb.add_argument("--candidates", type=int, default=4)
    lb.add_argument("--rollouts", type=int, default=4)
    lb.add_argument("--horizon", type=int, default=24)
    lb.add_argument("--calm", type=float, default=0.25)
    args = parser.parse_args()
    if args.command == "train":
        cfg = Config(**{k: getattr(args, k) for k in Config.__dataclass_fields__})
        train(cfg, args.resume)
    elif args.command == "export":
        export(args.checkpoint, args.out)
    elif args.command == "label":
        label(args.policy, args.out, args.seed, args.rounds, args.candidates, args.rollouts, args.horizon, args.calm)
    else:
        evaluate(args.checkpoint, [int(s) for s in args.seeds.split(",")], args.games, not args.no_shield, args.out)


if __name__ == "__main__":
    main()
