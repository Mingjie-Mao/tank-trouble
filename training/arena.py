"""头对头擂台：两个策略直接对打，用于飞轮的逐轮判定。

为什么需要它
------------
对 Laika 的胜率已经饱和（P37 在 98% 附近），新旧两版都是 98%，那个数字
没有量程，无法判断一轮迭代是变好还是变坏。飞轮每转一圈都必须能判定
接受还是回滚，所以头对头是地基而不是可选项。

镜像的必要性
------------
所有老师都写死了"我是 tanks[0]"，包括 `make_sandbox` 里
`sb.tanks[1].ai = LaikaAI(...)` 和 `rollout` 里 `me = sandbox.tanks[0]`。
要让一个策略扮演 tank1，就得给它一个 tanks 已交换的世界视图。

`_copy_bullet` 用 `sandbox.tanks[b.owner.number]` 按 **number** 而不是
列表下标重定向，所以镜像必须同时交换 `number`，否则子弹归属会错位。

镜像对象在整局内保持同一个 Python 身份并原地刷新，因为策略是有状态的
（`_ensure_field` 用 `game is not self.game` 判回合切换，每帧换新对象
会导致场缓存和承诺计数每帧重置）。

自检
----
`selftest` 让同一个策略自己打自己。交换先后手之后胜负必须接近 50/50；
若明显偏斜，说明镜像有 bug，此时任何擂台结论都不可信。
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tank_trouble_original.game import Game, Tank, Bullet  # noqa: E402


class MirrorView:
    """把真实 game 呈现为"tank1 是我"的世界视图。

    只读：策略从不修改 game，推进由擂台在真实 game 上做。
    每帧 `refresh()` 原地刷新，保持 Python 身份稳定。
    """

    _SWAPPED_LISTS = ("scores", "hit_immunity_remaining", "tank_fields")

    def __init__(self, game):
        self._game = game
        self.tanks = [object.__new__(Tank), object.__new__(Tank)]
        self.bullets = []
        self.refresh()

    def refresh(self):
        game = self._game
        # 交换坦克，并把 number 一起换掉（子弹按 number 重定向）
        for slot, source in enumerate(reversed(game.tanks)):
            target = self.tanks[slot]
            target.__dict__.update(source.__dict__)
            target.number = slot
            target.game = self
            target.ai = None
        # 子弹：owner 重定向到镜像坦克
        self.bullets = []
        for source in game.bullets:
            copy = object.__new__(Bullet)
            copy.__dict__.update(source.__dict__)
            copy.game = self
            copy.owner = self.tanks[1 - source.owner.number]
            self.bullets.append(copy)
        for name in self._SWAPPED_LISTS:
            if hasattr(game, name):
                value = getattr(game, name)
                setattr(self, name, list(reversed(value)))

    def __getattr__(self, name):
        # tanks / bullets / 交换过的列表已是实例属性，走不到这里。
        # 其余（maze / walls / scale / frame / frozen / round_number /
        # dist_map / weapon_ready ...）直接透传。
        return getattr(self._game, name)


class SideAdapter:
    """让一个"只会当 tank0"的策略去控制指定的坦克。"""

    def __init__(self, policy, index):
        self.policy = policy
        self.index = int(index)
        self.mirror = None

    def reset(self):
        self.mirror = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def act(self, game):
        if self.index == 0:
            return self.policy.act(game)
        if self.mirror is None or self.mirror._game is not game:
            self.mirror = MirrorView(game)
        else:
            self.mirror.refresh()
        return self.policy.act(self.mirror)

    def close(self):
        for target in (self.policy,):
            if hasattr(target, "close"):
                target.close()


class _Chain:
    """单方的追猎链状态。刻意写成显式的游戏规则，而不是借用老师内部的
    HuntChainState——上限要能独立配置，而且规则应当可读。"""

    def __init__(self, max_exponent, window):
        self.max_exponent = int(max_exponent)
        self.window = int(window)
        self.count = 0
        self.timer = 0
        self.collected = set()
        self.total = 0.0

    def advance(self):
        self.timer = max(0, self.timer - 1)
        if self.timer == 0:
            self.count = 0

    def collect(self, field, previous_cell, current_cell):
        if previous_cell == current_cell:
            return 0.0
        before = field.guidance_at(previous_cell)
        after = field.guidance_at(current_cell)
        if after <= before + 1e-7:
            return 0.0
        key = (field.target_cell, tuple(current_cell))
        if key in self.collected:
            return 0.0
        reward = float(2 ** min(self.count, self.max_exponent))
        self.count = min(self.count + 1, self.max_exponent)
        self.timer = self.window
        self.collected.add(key)
        self.total += reward
        return reward


class ChainScoreboard:
    """双方对称的追猎链计分板。

    每一方的场以**对手所在格**为目标，所以"上坡"就等于"朝能打到对手的
    位置移动"。场只依赖迷宫和目标格，因此两方共用一个 builder 和一份缓存。

    对手一换格，目标格就变，此前领过的格子重新可领——这是它相对金币的
    关键优势：不会吃光，残局不会退回对峙。
    """

    def __init__(self, game, rays=256, bounces=2, flight=75,
                 max_exponent=4, window=75):
        from training.killfield_teacher import InverseDensityFieldBuilder

        self._cell = _grid_cell
        self.builder = InverseDensityFieldBuilder(
            game, rays, bounces, flight)
        self.cache = {}
        self.chains = [_Chain(max_exponent, window) for _ in range(2)]
        self.cells = [self._cell(game, tank) for tank in game.tanks]
        self.field_builds = 0

    def _field(self, target_cell):
        if target_cell not in self.cache:
            self.cache[target_cell] = self.builder.build(target_cell)
            self.field_builds += 1
        return self.cache[target_cell]

    def on_frame(self, game):
        for index in (0, 1):
            chain = self.chains[index]
            chain.advance()
            if not game.tanks[index].alive:
                continue
            field = self._field(self._cell(game, game.tanks[1 - index]))
            current = self._cell(game, game.tanks[index])
            chain.collect(field, self.cells[index], current)
            self.cells[index] = current

    @property
    def totals(self):
        return [chain.total for chain in self.chains]


def _grid_cell(game, tank):
    return int(tank.x // game.scale), int(tank.y // game.scale)


def _apply(tank, action):
    tank.forward = bool(action.get("forward", False))
    tank.backup = bool(action.get("backup", False))
    tank.turn_left = bool(action.get("turn_left", False))
    tank.turn_right = bool(action.get("turn_right", False))
    tank.fire = bool(action.get("fire", False))


def play_match(make_a, make_b, seed, a_side=0, max_frames=1500):
    """一局对抗。返回 'a' / 'b' / 'double' / 'timeout'。

    判定用原版口径：先杀死对手、并且自己活到回合结束才算赢；先杀后被余弹
    击中是双亡，不算胜利。
    """
    game = Game(seed=seed, ai_enabled=False)
    side_a = SideAdapter(make_a(), a_side)
    side_b = SideAdapter(make_b(), 1 - a_side)
    side_a.reset()
    side_b.reset()
    try:
        for _ in range(max_frames):
            # 必须跑到 frozen 才判定：首杀后引擎还有 125-50=75 帧存活期，
            # 余弹可以把幸存者也打死。提前 break 会把双亡误判成胜利。
            if game.frozen:
                break
            if game.tanks[a_side].alive:
                _apply(game.tanks[a_side], side_a.act(game))
            if game.tanks[1 - a_side].alive:
                _apply(game.tanks[1 - a_side], side_b.act(game))
            game.step()
        alive_a = game.tanks[a_side].alive
        alive_b = game.tanks[1 - a_side].alive
    finally:
        side_a.close()
        side_b.close()
    if alive_a and not alive_b:
        return "a"
    if alive_b and not alive_a:
        return "b"
    if not alive_a and not alive_b:
        return "double"
    return "timeout"


# 结局价值。击杀 1.0 远高于超时胜 0.4：进攻成功率超过约 25% 就该出手，
# 所以"打分数战"永远只是保底，不构成替代击杀的路径。
OUTCOME_VALUE = {
    "kill": 1.0,
    "timeout_ahead": 0.4,
    "timeout_behind": 0.2,
    "double": 0.1,
    "death": 0.0,
}
RANKED_CAP_FRAMES = 30 * 25          # 30 秒结算


def play_ranked_match(make_a, make_b, seed, a_side=0,
                      cap_frames=RANKED_CAP_FRAMES, score_rays=256,
                      max_exponent=4):
    """新规则一局：击杀（并活到结算）直接获胜；到时按追猎链分裁决。

    返回 (结果, A分, B分, 帧数)。结果 ∈
    {'a', 'b', 'double', 'a_score', 'b_score', 'draw'}
    """
    game = Game(seed=seed, ai_enabled=False)
    board = ChainScoreboard(
        game, rays=score_rays, max_exponent=max_exponent)
    side_a = SideAdapter(make_a(), a_side)
    side_b = SideAdapter(make_b(), 1 - a_side)
    side_a.reset()
    side_b.reset()
    frames = 0
    try:
        while True:
            if game.frozen:
                break
            if frames >= cap_frames and \
                    game.tanks[0].alive and game.tanks[1].alive:
                break                      # 双方都活着 -> 到时结算
            if game.tanks[a_side].alive:
                _apply(game.tanks[a_side], side_a.act(game))
            if game.tanks[1 - a_side].alive:
                _apply(game.tanks[1 - a_side], side_b.act(game))
            game.step()
            board.on_frame(game)
            frames += 1
            if frames > cap_frames + 200:  # 死亡后的 75 帧结算窗口留够
                break
        alive_a = game.tanks[a_side].alive
        alive_b = game.tanks[1 - a_side].alive
    finally:
        side_a.close()
        side_b.close()
    score_a = board.totals[a_side]
    score_b = board.totals[1 - a_side]
    if alive_a and not alive_b:
        return "a", score_a, score_b, frames
    if alive_b and not alive_a:
        return "b", score_a, score_b, frames
    if not alive_a and not alive_b:
        return "double", score_a, score_b, frames
    if score_a > score_b:
        return "a_score", score_a, score_b, frames
    if score_b > score_a:
        return "b_score", score_a, score_b, frames
    return "draw", score_a, score_b, frames


def run_ranked_ladder(make_a, make_b, seeds, label_a="A", label_b="B",
                      cap_frames=RANKED_CAP_FRAMES, score_rays=256,
                      verbose=True):
    tally = {k: 0 for k in
             ("a", "b", "double", "a_score", "b_score", "draw")}
    scores_a, scores_b, lengths = [], [], []
    started = time.time()
    for seed in seeds:
        for a_side in (0, 1):
            result, sa, sb, frames = play_ranked_match(
                make_a, make_b, seed, a_side=a_side,
                cap_frames=cap_frames, score_rays=score_rays)
            tally[result] += 1
            scores_a.append(sa)
            scores_b.append(sb)
            lengths.append(frames)
    total = sum(tally.values())
    by_kill = tally["a"] + tally["b"] + tally["double"]
    if verbose:
        print(f"\n  === {label_a} vs {label_b}  ({total} 局，"
              f"{time.time() - started:.0f}s) ===")
        print(f"  击杀判定 {by_kill}  ({100 * by_kill / total:.0f}%)"
              f"   —— {label_a} 胜 {tally['a']} | {label_b} 胜 {tally['b']}"
              f" | 双亡 {tally['double']}")
        print(f"  分数判定 {tally['a_score'] + tally['b_score'] + tally['draw']}"
              f"  —— {label_a} {tally['a_score']} | {label_b} "
              f"{tally['b_score']} | 平 {tally['draw']}")
        print(f"  平均链分  {label_a} {statistics.mean(scores_a):.1f}  |  "
              f"{label_b} {statistics.mean(scores_b):.1f}")
        print(f"  平均局长  {statistics.mean(lengths) / 25:.1f}s")
    value_a = (
        OUTCOME_VALUE["kill"] * tally["a"]
        + OUTCOME_VALUE["timeout_ahead"] * tally["a_score"]
        + OUTCOME_VALUE["timeout_behind"] * tally["b_score"]
        + OUTCOME_VALUE["double"] * tally["double"]
        + 0.5 * (OUTCOME_VALUE["timeout_ahead"]
                 + OUTCOME_VALUE["timeout_behind"]) * tally["draw"]
    ) / max(total, 1)
    if verbose:
        print(f"  {label_a} 期望价值 {value_a:.3f}  (击杀1.0/超时胜0.4/"
              f"超时负0.2/双亡0.1/死0.0)")
    return {"tally": tally, "value_a": value_a,
            "scores": (scores_a, scores_b), "lengths": lengths}


def run_ladder(make_a, make_b, seeds, label_a="A", label_b="B",
               max_frames=1500, verbose=True):
    """每个种子打两局，交换先后手，抵消出生点优势。"""
    tally = {"a": 0, "b": 0, "double": 0, "timeout": 0}
    started = time.time()
    for order, seed in enumerate(seeds):
        for a_side in (0, 1):
            result = play_match(
                make_a, make_b, seed, a_side=a_side, max_frames=max_frames)
            tally[result] += 1
        if verbose and (order + 1) % 8 == 0:
            done = 2 * (order + 1)
            print(f"    [{done} 局] {label_a} {tally['a']} : "
                  f"{tally['b']} {label_b}   双亡 {tally['double']}  "
                  f"超时 {tally['timeout']}", flush=True)
    total = sum(tally.values())
    decisive = tally["a"] + tally["b"]
    rate = tally["a"] / decisive if decisive else 0.5
    # 只在分胜负的局上算二项标准误
    stderr = ((rate * (1 - rate) / decisive) ** 0.5) if decisive else 0.0
    if verbose:
        print(f"\n  === {label_a} vs {label_b} ===")
        print(f"  总计 {total} 局，用时 {time.time() - started:.0f}s")
        print(f"  {label_a} 胜 {tally['a']}  |  {label_b} 胜 {tally['b']}"
              f"  |  双亡 {tally['double']}  |  超时 {tally['timeout']}")
        print(f"  {label_a} 在分胜负局中占 {100 * rate:.1f}% "
              f"± {100 * 1.96 * stderr:.1f}  (95%)")
    return {"tally": tally, "rate": rate, "stderr": stderr, "total": total}


# ------------------------------------------------------------------ 工厂

def killfield_factory(horizon=36, rays=512, bounces=2, flight=75,
                      weights=None, seed=373):
    """构造 P37 击杀场老师；weights 可覆盖评分权重（用于权重探针）。

    `density_rollout` 在调用时读取模块全局权重，所以覆盖必须**局部生效**：
    直接 setattr 会污染同进程里的对照组，让两个变体用同一组权重。
    这里在 `scores()` 前后临时改写并还原。
    """
    import training.killfield_teacher as kt

    class Weighted(kt.KillFieldTeacher):
        def scores(self, game):
            if not weights:
                return super().scores(game)
            saved = {k: getattr(kt, k) for k in weights}
            for key, value in weights.items():
                setattr(kt, key, value)
            try:
                return super().scores(game)
            finally:
                for key, value in saved.items():
                    setattr(kt, key, value)

    def make():
        return Weighted(
            seed=seed, ray_count=rays, max_bounces=bounces,
            max_flight_frames=flight, horizon=horizon)
    return make


def selftest(seeds, horizon=36, rays=512):
    """同一策略自己打自己：交换先后手后必须接近 50/50。"""
    print("=== 镜像自检：P37 老师 vs 自己 ===")
    print("    交换先后手后若明显偏离 50%，说明镜像有 bug\n")
    make = killfield_factory(horizon=horizon, rays=rays)
    result = run_ladder(make, make, seeds, label_a="镜像A", label_b="镜像B")
    rate = result["rate"]
    margin = 1.96 * result["stderr"]
    ok = abs(rate - 0.5) <= max(margin, 0.02)
    print(f"\n  自检判定: {'✅ 通过' if ok else '❌ 不通过——镜像可能有 bug'}"
          f"  (偏离 50% 达 {100 * abs(rate - 0.5):.1f} 点，"
          f"允许 {100 * max(margin, 0.02):.1f})")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=["selftest", "depth", "weights", "ranked"])
    parser.add_argument("--chain-weights", default="12,60,180",
                        help="老师的 HUNT_CHAIN_GAIN_WEIGHT 档位")
    parser.add_argument("--cap-seconds", type=float, default=30.0)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=40_000_000)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=1500)
    args = parser.parse_args()

    seeds = [args.seed_base + i for i in range(args.seeds)]

    if args.command == "selftest":
        selftest(seeds, rays=args.rays)
        return

    if args.command == "depth":
        print("=== 深度探针：horizon 72 vs 36 ===")
        print("    72 明显赢 → 更远视野有价值 → 价值网络有头顶\n")
        run_ladder(
            killfield_factory(horizon=72, rays=args.rays),
            killfield_factory(horizon=36, rays=args.rays),
            seeds, label_a="H72", label_b="H36",
            max_frames=args.max_frames)
        return

    if args.command == "ranked":
        print("=== 新规则：30 秒结算 + 追猎链分裁决 ===")
        print("    规则只改判定；要让老师真的去争分，得抬高它内部的")
        print("    HUNT_CHAIN_GAIN_WEIGHT（原值 12，对比 ALIGNMENT=190）。")
        print("    看的是：击杀判定的比例会不会随权重上升。\n")
        cap = int(args.cap_seconds * 25)
        for raw in args.chain_weights.split(","):
            weight = float(raw)
            print(f"\n--- 链权重 {weight:g} （双方同权重，自己打自己）---")
            make = killfield_factory(
                horizon=36, rays=args.rays,
                weights={"HUNT_CHAIN_GAIN_WEIGHT": weight})
            run_ranked_ladder(
                make, make, seeds,
                label_a=f"链{weight:g}A", label_b=f"链{weight:g}B",
                cap_frames=cap, score_rays=args.rays)
        return

    if args.command == "weights":
        print("=== 权重探针：扰动权重 vs 原版手调权重 ===")
        print("    扰动版能赢 → 手调权重不在局部最优 → 学习有头顶\n")
        variants = [
            ("对准加倍", {"ALIGNMENT_WEIGHT": 380.0}),
            ("风险减半", {"RISK_WEIGHT": 160.0}),
            ("导航加倍", {"GUIDANCE_PROGRESS_WEIGHT": 240.0}),
        ]
        for name, override in variants:
            print(f"\n--- {name} ---")
            run_ladder(
                killfield_factory(horizon=36, rays=args.rays,
                                  weights=override),
                killfield_factory(horizon=36, rays=args.rays),
                seeds, label_a=name, label_b="原版",
                max_frames=args.max_frames)
        return


if __name__ == "__main__":
    main()
