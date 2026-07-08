"""
P24 生存模式 v2 —— 衰减计分板课程: 逼出"主动狩猎 + 观赏性"

规则 (用户策划定稿 2026-07-08):
  - Laika 无敌 (打不死, 命中仍触发 hit 事件); 我方一发即死 (含自伤)
  - 分数池不封顶: 起始 100 | 衰减 -10/s | 命中 Laika +50
    | 追击势能 +3/净靠近一格 (BFS, 守恒: 绕圈净额零)
    | 首访新格 +2 (4s 冷却) | 卡墙额外 -5/s
  - 结算: 被打死 -> 结算分清零 | 分数流干 -> 结算 0 | 30s 上限 -> 按现值
  - 三本账遥测 (死亡不抹账): 命中账 / 风格账(势能+覆盖-卡墙) / 时间税

设计约束:
  风格收入上限 (~+5-7/s) < 衰减 (10/s)  => 纯跑图必死, 命中是唯一主粮
  死亡代价 = 全池 (风格分一并没收)      => 生存字典序压倒观赏性

铁律: 生存模式只是训练课程; 验收永远在原版游戏 (真胜率 + 进攻性)。

用法:
  python3 training/survival_mode.py measure --n 40             # 四方对照
  python3 training/survival_mode.py measure --kinds hunt,style # 指定策略
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FPS = 25
ECON = dict(
    start=100.0,
    decay=10.0 / FPS,       # -0.4/帧
    hit=50.0,               # 断粮线 = 每 5s 必须一中
    approach=3.0,           # /净靠近一格, 势能守恒 (战术后撤免费)
    cover=2.0,              # /首访新格
    cover_cd=4 * FPS,       # 首访冷却 4s
    stuck=5.0 / FPS,        # 卡墙额外 -0.2/帧
    cap=30 * FPS,           # 30s 结算
)
DEATH_K = 200.0             # rollout 内死亡的期货代价 (丢掉全部未来收入)
DRAIN_K = 150.0             # rollout 内流干的期货代价


def _cell(g, t):
    return int(t.x // g.scale), int(t.y // g.scale)


def _bfs_dist(g, me, en):
    """me 到 en 的 BFS 路径格数 (distances_for_maze 预计算表, O(1) 查询)"""
    return _cell_dist(g, _cell(g, me), _cell(g, en))


def _cell_dist(g, ca, cb):
    dm = g.dist_map(ca[0], ca[1])
    if dm is None:
        return None
    ex, ey = cb
    if 0 <= ex < len(dm) and 0 <= ey < len(dm[ex]):
        v = dm[ex][ey]
        if v is not None and v == v:
            return float(v)
    return None


# ================================================================ 账本

class Ledger:
    """生存模式经济的唯一实现 —— 采集/评测/回放共用, 保证规则一致。"""

    def __init__(self, g, econ=ECON):
        self.econ = econ
        self.pool = econ["start"]
        self.led = dict(hit=0.0, approach=0.0, cover=0.0,
                        stuck=0.0, decay=0.0)
        self.hits = 0
        self.stuck_frames = 0
        self.cells = 0
        self.frames = 0
        self.visited = {}
        c = _cell(g, g.tanks[0])
        self.visited[c] = 0
        self._prev_cell = c
        self._prev_en_cell = _cell(g, g.tanks[1])

    @property
    def style(self):
        return (self.led["approach"] + self.led["cover"]
                - self.led["stuck"])

    def on_frame(self, g, events):
        """记账一帧, 返回 'alive' | 'death' | 'drain' | 'cap'。"""
        econ = self.econ
        self.frames += 1
        t0 = g.tanks[0]
        for ev in events:
            if ev[0] == "hit" and ev[1] == 0 and ev[2] == 1:
                self.pool += econ["hit"]
                self.led["hit"] += econ["hit"]
                self.hits += 1
        if not t0.alive:
            return "death"                    # 结算清零; pool 保留作"清零前"
        self.pool -= econ["decay"]
        self.led["decay"] += econ["decay"]
        if t0.hit_something:
            self.pool -= econ["stuck"]
            self.led["stuck"] += econ["stuck"]
            self.stuck_frames += 1
        # 追击势能: 以 Laika 上一帧位置为锚 —— 只有我方移动拨动指针
        # (空闲者恒得零, 被追近不计收入); 我方原路折返仍净额为零
        c = _cell(g, t0)
        if c != self._prev_cell:
            d_a = _cell_dist(g, self._prev_cell, self._prev_en_cell)
            d_b = _cell_dist(g, c, self._prev_en_cell)
            if d_a is not None and d_b is not None:
                inc = econ["approach"] * (d_a - d_b)
                self.pool += inc
                self.led["approach"] += inc
        self._prev_en_cell = _cell(g, g.tanks[1])
        if c != self._prev_cell:
            last = self.visited.get(c)
            if last is None or self.frames - last > econ["cover_cd"]:
                self.pool += econ["cover"]
                self.led["cover"] += econ["cover"]
                self.cells += 1
            self._prev_cell = c
        self.visited[c] = self.frames         # 驻留也刷新, 防出格回步刷分
        if self.pool <= 0:
            self.pool = 0.0
            return "drain"
        if self.frames >= econ["cap"]:
            return "cap"
        return "alive"


def run_survival(policy, seed, econ=ECON, mirror=False):
    """跑一局, 返回遥测 dict。mirror=True 时忽略 policy, 玩家位挂 Laika AI。"""
    from tank_trouble_original.game import Game
    g = Game(seed=seed, ai_enabled=True, invincible={1})
    if mirror:
        from tank_trouble_original.laika import LaikaAI
        g.tanks[0].ai = LaikaAI(g, g.tanks[0])
    else:
        policy.reset()
    ledger = Ledger(g, econ)
    end = "cap"
    while True:
        if not mirror:
            if hasattr(policy, "act_ctx"):
                inp = policy.act_ctx(g, ledger)
            else:
                inp = policy.act(g)
            t0 = g.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
        events = g.step()
        end = ledger.on_frame(g, events)
        if end != "alive":
            break
    settle = 0.0 if end == "death" else ledger.pool
    alive_s = ledger.frames / FPS
    return dict(settle=settle, pool_final=ledger.pool, end=end,
                frames=ledger.frames, alive_s=alive_s, hits=ledger.hits,
                stuck_frames=ledger.stuck_frames, cells=ledger.cells,
                led=dict(ledger.led), style=ledger.style)


# ================================================================ 老师

def survival_rollout(sandbox, first_action, pool, visited, frame_now,
                     hold=16, horizon=48, econ=ECON, style=True):
    """老师目标 = 在沙盒里原样模拟计分经济, 返回 Δ池 (相对现值)。

    死亡/流干会终结整局 (没收全池 + 丢掉全部未来收入), 用常数期货代价
    DEATH_K / DRAIN_K 代理视野外的损失。势能项望远镜求和: 只算首末两次
    BFS 查表。style=False 时退化为纯狩猎经济 (命中+衰减+死亡), 无风格项。
    """
    me = sandbox.tanks[0]
    en = sandbox.tanks[1]
    th, tu, f = first_action
    en_cell0 = _cell(sandbox, en)        # 锚定决策时刻的 Laika 位置
    d0 = _cell_dist(sandbox, _cell(sandbox, me), en_cell0)
    local_seen = {}
    prev_cell = _cell(sandbox, me)
    delta = 0.0
    for t in range(horizon):
        if t == 0:
            me.forward, me.backup = th == 2, th == 0
            me.turn_left, me.turn_right = tu == 0, tu == 2
            me.fire = f == 1
        elif t == hold:
            me.fire = False
        events = sandbox.step()
        for ev in events:
            if ev[0] == "hit" and ev[1] == 0 and ev[2] == 1:
                delta += econ["hit"]
        if not me.alive:
            return -pool - DEATH_K + 0.05 * t     # 全池没收+期货, 晚死略好
        delta -= econ["decay"]
        if style:
            if me.hit_something:
                delta -= econ["stuck"]
            c = _cell(sandbox, me)
            if c != prev_cell:
                tt = frame_now + t
                last_real = visited.get(c)
                last_loc = local_seen.get(c)
                fresh_real = last_real is None or tt - last_real > \
                    econ["cover_cd"]
                fresh_loc = last_loc is None or t - last_loc > \
                    econ["cover_cd"]
                if fresh_real and fresh_loc:
                    delta += econ["cover"]
                prev_cell = c
            local_seen[c] = t
        if pool + delta <= 0:
            # 流干终局丢未来, 但保留导航梯度: 贴着 Laika 破产下一决策
            # 还有救, 墙角破产没有 —— 否则贫困线附近所有候选并列, 老师失明
            grad = 0.0
            d1 = _cell_dist(sandbox, _cell(sandbox, me), en_cell0)
            if d1 is not None:
                if style and d0 is not None:
                    grad = econ["approach"] * (d0 - d1)
                elif not style:
                    grad = -0.5 * d1
            return -pool - DRAIN_K + 0.05 * t + grad
    if style and d0 is not None:
        d1 = _cell_dist(sandbox, _cell(sandbox, me), en_cell0)
        if d1 is not None:
            delta += econ["approach"] * (d0 - d1)
    elif not style:
        d1 = _bfs_dist(sandbox, me, en)
        if d1 is not None:
            delta -= 0.5 * d1                     # 纯狩猎档的导航引导 (v1 同款)
    return delta


class SurvivalMPC:
    """生存模式 MPC 老师: 18 候选各前推, survival_rollout 打分取最优。

    style=True 带风格项 (势能/覆盖/卡墙), False 纯狩猎经济。
    """

    def __init__(self, horizon=48, hold=16, seed=0, style=True, econ=ECON):
        import random
        from training.mpc_agent import CANDIDATES, make_sandbox
        self.horizon, self.hold = horizon, hold
        self.style = style
        self.econ = econ
        self.rng = random.Random(seed)
        self._cands = CANDIDATES
        self._make_sandbox = make_sandbox
        self.name = "style_mpc" if style else "hunt_mpc"

    def reset(self):
        pass

    def act_ctx(self, game, ledger):
        if not game.tanks[0].alive:
            return {}
        best_a, best_s = (1, 1, 0), -1e18
        for a in self._cands:
            sb = self._make_sandbox(game, "L2",
                                    rng_seed=self.rng.randrange(1 << 30))
            s = survival_rollout(sb, a, ledger.pool, ledger.visited,
                                 ledger.frames, self.hold, self.horizon,
                                 self.econ, self.style)
            if s > best_s:
                best_s, best_a = s, a
        th, tu, f = best_a
        return {"forward": th == 2, "backup": th == 0,
                "turn_left": tu == 0, "turn_right": tu == 2,
                "fire": f == 1}

    def act(self, game):
        """无外部账本时的兼容入口 (回放器外的临时调用)"""
        from tank_trouble_original.game import Game  # noqa: F401
        dummy = Ledger(game, self.econ)
        return self.act_ctx(game, dummy)


# ================================================================ 四方对照

KINDS = ("idle", "laika", "hunt", "style")


def _make_policy(kind):
    if kind == "idle":
        from training.baselines import IdlePolicy
        return IdlePolicy(), False
    if kind == "laika":
        return None, True
    if kind == "hunt":
        return SurvivalMPC(style=False), False
    if kind == "style":
        return SurvivalMPC(style=True), False
    raise ValueError(kind)


def _worker(job):
    kind, seed0, count = job
    policy, mirror = _make_policy(kind)
    out = []
    for i in range(count):
        out.append(run_survival(policy, seed0 + i, mirror=mirror))
    return out


def _agg(results):
    n = len(results)
    tot_alive = sum(r["alive_s"] for r in results)
    hits = sum(r["hits"] for r in results)
    style = sum(r["style"] for r in results)
    stuck_f = sum(r["stuck_frames"] for r in results)
    frames = sum(r["frames"] for r in results)
    deaths = [r for r in results if r["end"] == "death"]
    return dict(
        n=n,
        settle=sum(r["settle"] for r in results) / n,
        alive_s=tot_alive / n,
        hits=hits / n,
        hit_iv=(tot_alive / hits) if hits else float("inf"),
        style_rate=style / tot_alive if tot_alive else 0.0,
        stuck_pct=100.0 * stuck_f / frames if frames else 0.0,
        cells=sum(r["cells"] for r in results) / n,
        end_death=100.0 * len(deaths) / n,
        end_drain=100.0 * sum(r["end"] == "drain" for r in results) / n,
        end_cap=100.0 * sum(r["end"] == "cap" for r in results) / n,
        pool_at_death=(sum(r["pool_final"] for r in deaths) / len(deaths))
        if deaths else 0.0,
    )


def measure(kinds, n, workers):
    print("===== [P24v2] 衰减计分板: 四方对照 =====", flush=True)
    print(f"经济: 起始{ECON['start']:.0f} 衰减{ECON['decay']*FPS:.0f}/s "
          f"命中+{ECON['hit']:.0f} 势能+{ECON['approach']:.0f}/格 "
          f"覆盖+{ECON['cover']:.0f} 卡墙-{ECON['stuck']*FPS:.0f}/s "
          f"上限{ECON['cap']//FPS}s", flush=True)
    aggs = {}
    for kind in kinds:
        per = max(1, n // workers)
        jobs = [(kind, 880000 + w * per, per) for w in range(workers)]
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers) as pool:
            outs = pool.map(_worker, jobs)
        results = [r for o in outs for r in o]
        a = _agg(results)
        aggs[kind] = a
        print(f"\n--- {kind}: {a['n']}局 ({time.time()-t0:.0f}s) ---",
              flush=True)
        print(f"  结算分 {a['settle']:.1f}  存活 {a['alive_s']:.1f}s  "
              f"命中 {a['hits']:.1f}次/局 (间隔 {a['hit_iv']:.1f}s)",
              flush=True)
        print(f"  风格速率 {a['style_rate']:+.2f}/s  卡墙帧 "
              f"{a['stuck_pct']:.1f}%  覆盖 {a['cells']:.1f}格/局",
              flush=True)
        print(f"  终局: 死{a['end_death']:.0f}% 流干{a['end_drain']:.0f}% "
              f"到时{a['end_cap']:.0f}%  死亡时身价 "
              f"{a['pool_at_death']:.0f}", flush=True)
    if "laika" in aggs:
        base = aggs["laika"]["style_rate"]
        print("\n===== 风格效率指数 (÷ Laika 镜像) =====", flush=True)
        for kind in kinds:
            if kind in ("idle", "laika") or base == 0:
                continue
            print(f"  {kind}: {aggs[kind]['style_rate']/base:.2f}",
                  flush=True)
    print("\n提醒: 训练课程的机制验证; 真验收永远在原版游戏。", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["measure"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kinds", default="idle,laika,hunt,style")
    ap.add_argument("--workers", type=int,
                    default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        if k not in KINDS:
            raise SystemExit(f"未知策略: {k} (可选 {KINDS})")
    measure(kinds, args.n, args.workers)


if __name__ == "__main__":
    main()
