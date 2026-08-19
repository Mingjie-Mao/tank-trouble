"""
回放器 — 在 tkinter 渲染器里观看策略对战 Laika

用法:
  python training/watch.py --policy hunter          # 观看手写猎杀脚本
  python training/watch.py --policy model           # 观看训练好的模型
  python training/watch.py --policy model --model training/models/best_model.zip
  python training/watch.py --policy hunter --seed 910007   # 复现指定局
  python training/watch.py --policy survival               # P24 生存老师狩猎回放
  python training/watch.py --policy p27b                   # 现任网络冠军 (P26+P27b)
  python training/watch.py --policy exact                  # 现任搜索冠军 (精确状态老师)
  python training/watch.py --policy key-hybrid             # 关键事件稀疏搜索实验
  python training/watch.py --policy topology-hybrid        # 持续追击与动作迟滞实验
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play_tank_trouble import App, Game  # noqa: E402
from training.baselines import IdlePolicy, RandomPolicy, HunterPolicy  # noqa: E402
from training.battle_supervision import append_supervision  # noqa: E402
from training.evaluate import RoundTracker, _round_stats  # noqa: E402
from training.tt_gym_env import TRUNCATE_FRAMES  # noqa: E402


class PolicyApp(App):
    """让策略接管 tank0 的渲染窗口 (R 键换局仍可用)"""

    def __init__(self, policy, seed=None, model_env=None, model=None,
                 self_harm_immune=None, invincible=None,
                 hit_immunity_frames=None, official_reset=False,
                 supervision_log=None, max_round_frames=TRUNCATE_FRAMES,
                 run_config=None):
        # App.__init__ sets self_harm_immune / invincible / hit_immunity_frames.
        self.max_round_frames = max(1, int(max_round_frames))
        # Stamped on every logged round.  Without it two GUI sessions with
        # different settings are indistinguishable in the log, and the monitor
        # (which de-duplicates by seed, keeping the newest) silently lets the
        # later run overwrite the earlier baseline.
        self.run_config = dict(run_config or {})
        self.policy = policy
        self.model_env = model_env    # ModelPolicy 用: 独立观测环境
        self.model = model
        self.official_reset = official_reset
        self._official_seed = seed
        self._official_scores = [0, 0]
        self._official_rounds = 0
        self.supervision_log = supervision_log
        super().__init__(seed=seed, two_players=False,
                         self_harm_immune=self_harm_immune,
                         invincible=invincible,
                         hit_immunity_frames=hit_immunity_frames)
        tag = " [Laika免疫自伤]" if self_harm_immune else ""
        tag += " [official-reset]" if official_reset else ""
        self.root.title(f"Tank Trouble — {policy_name(policy)} vs Laika{tag}")
        self._start_supervision_round()

    def _start_supervision_round(self):
        self._round_tracker = RoundTracker(self.game)
        self._supervision_frames = 0
        self._round_recorded = False

    def _record_supervision_round(self, winner):
        if self._round_recorded or not self.supervision_log:
            return
        self._round_recorded = True
        result = ("truncated" if winner is None else
                  "win" if winner == 0 else
                  "loss" if winner == 1 else "double_death")
        if hasattr(self.policy, "event_tracker"):
            self.policy.event_tracker.finish(self._supervision_frames)
        row = _round_stats(
            self._round_tracker, result, self._supervision_frames)
        row.update({
            "source": "watch",
            "policy": policy_name(self.policy),
            "seed": (None if self._official_seed is None else
                     self._official_seed + self._official_rounds),
        })
        row.update(self.run_config)
        if hasattr(self.policy, "event_tracker"):
            row["event_metrics"] = self.policy.event_tracker.summary()
        for name in (
                "exact_searches", "policy_frames", "temporal_overrides",
                "long_tail_fire_checks", "long_tail_fire_rejections",
                "topology_requests", "topology_completions",
                "topology_aborts", "movement_continuity_holds"):
            if hasattr(self.policy, name):
                row[name] = int(getattr(self.policy, name))
        if row.get("policy_frames"):
            row["search_frame_rate"] = (
                row.get("exact_searches", 0) / row["policy_frames"])
        append_supervision(self.supervision_log, row)

    def _reset_policy_state(self):
        if self.model is not None:
            self._wframes = 0
            return
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        # 搜索类策略 (P28/exact) 的沙盘 RNG 按局种子派生, 换局必须重设
        if (hasattr(self.policy, "set_round_seed")
                and self._official_seed is not None):
            self.policy.set_round_seed(
                self._official_seed + self._official_rounds)

    def _start_official_round(self):
        seed = None
        if self._official_seed is not None:
            seed = self._official_seed + self._official_rounds
        self.game = Game(seed=seed, ai_enabled=True,
                         self_harm_immune=self.self_harm_immune,
                         invincible=self.invincible,
                         hit_immunity_frames=self.hit_immunity_frames)
        self.game.scores = list(self._official_scores)
        self.game.round_number = self._official_rounds + 1
        self._reset_policy_state()
        self._start_supervision_round()

    def _finish_official_round(self, winner):
        if winner in (0, 1):
            self._official_scores[winner] += 1
        self._official_rounds += 1
        self._start_official_round()

    def _tick(self):
        # App.__init__ starts the first tick before PolicyApp.__init__ returns.
        # Initialize supervision lazily once the base class has created Game.
        if not hasattr(self, "_round_tracker"):
            self._start_supervision_round()
        started = time.perf_counter()
        g = self.game
        if self.model is not None:
            # 模型策略: 用训练观测编码器
            self.model_env.game = g
            self.model_env._build_wall_boxes()
            if not hasattr(self, "_wframes"):
                self._wframes = 0
            self.model_env._frames = self._wframes
            obs = self.model_env._obs()
            action, _ = self.model.predict(obs, deterministic=True)
            t0 = g.tanks[0]
            t0.forward = int(action[0]) == 2
            t0.backup = int(action[0]) == 0
            t0.turn_left = int(action[1]) == 0
            t0.turn_right = int(action[1]) == 2
            t0.fire = int(action[2]) == 1
            self._wframes += 1
        else:
            inp = self.policy.act(g)
            t0 = g.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
        self._round_tracker.pre_step()
        events = g.step()
        self._supervision_frames += 1
        self._round_tracker.post_step(events, 1)
        ended = any(ev[0] == "round_end" for ev in events)
        # The headless evaluator truncates at TRUNCATE_FRAMES because mutual
        # stalemates exist and never emit round_end.  The GUI loop had no such
        # cap, so one stalemate hung the window indefinitely and silently
        # stopped the supervision log -- observed on seed 970805, which ran
        # 8.5 hours against a ~270 frame median.
        truncated = not ended and self._supervision_frames >= self.max_round_frames
        for ev in events:
            if ev[0] == "round_end":
                self._record_supervision_round(ev[1])
                break
        if truncated:
            self._record_supervision_round(None)
        if self.official_reset:
            if ended:
                for ev in events:
                    if ev[0] == "round_end":
                        self._finish_official_round(ev[1])
                        break
            elif truncated:
                # Nobody won: advance the seed without crediting a score.
                self._official_rounds += 1
                self._start_official_round()
        elif truncated:
            self.game = Game(seed=None, ai_enabled=True,
                             self_harm_immune=self.self_harm_immune,
                             invincible=self.invincible,
                             hit_immunity_frames=self.hit_immunity_frames)
            self._reset_policy_state()
            self._start_supervision_round()
        elif any(ev[0] == "new_round" for ev in events):
            self._reset_policy_state()
            self._start_supervision_round()
        self._draw()
        # 25 FPS; 扣掉决策耗时, 慢策略(搜索类)自然退化成尽力播放而不是叠加延迟
        spent = int((time.perf_counter() - started) * 1000)
        self.root.after(max(1, 40 - spent), self._tick)


class SurvivalApp(PolicyApp):
    """P24v3 生存模式回放: 免疫穿透计分板 + 遥测 HUD。

    经济规则复用 survival_mode.Ledger (与采集/评测同一实现)。
    死亡走引擎原生回合循环 (new_round 重开账本); 流干/到时换新局。
    """

    def __init__(self, policy, seed=None):
        from training.survival_mode import Ledger, ECON, FPS
        self._Ledger, self._econ, self._fps = Ledger, ECON, FPS
        self.ledger = None
        self.dead_wait = False
        self.expire_count = -1
        self.settle_msg = ""
        super().__init__(
            policy, seed=seed, invincible={1},
            hit_immunity_frames={1: ECON["hit_immunity"]})
        self.root.title(
            f"Tank Trouble — 生存模式v3: {policy_name(policy)} vs 无敌Laika")

    def _tick(self):
        g = self.game
        if self.ledger is None:
            self.ledger = self._Ledger(g)
        if self.dead_wait:
            for ev in g.step():
                if ev[0] == "new_round":
                    self.ledger = None
                    self.dead_wait = False
                    self.settle_msg = ""
        elif self.expire_count >= 0:
            self.expire_count -= 1
            if self.expire_count == 0:
                from tank_trouble_original.game import Game
                self.game = Game(
                    seed=None, ai_enabled=True, invincible={1},
                    hit_immunity_frames={1: self._econ["hit_immunity"]})
                self.ledger = None
                self.expire_count = -1
                self.settle_msg = ""
        else:
            if hasattr(self.policy, "act_ctx"):
                inp = self.policy.act_ctx(g, self.ledger)
            else:
                inp = self.policy.act(g)
            t0 = g.tanks[0]
            t0.forward = bool(inp.get("forward", False))
            t0.backup = bool(inp.get("backup", False))
            t0.turn_left = bool(inp.get("turn_left", False))
            t0.turn_right = bool(inp.get("turn_right", False))
            t0.fire = bool(inp.get("fire", False))
            events = g.step()
            end = self.ledger.on_frame(g, events)
            if end == "death":
                self.settle_msg = (f"被击杀 — 结算 0 "
                                   f"(身价 {self.ledger.pool:.0f} 没收)")
                self.dead_wait = True
            elif end == "drain":
                self.settle_msg = "分数流干 — 结算 0"
                self.expire_count = 37
            elif end == "cap":
                self.settle_msg = f"30s 到时 — 结算 {self.ledger.pool:.0f}"
                self.expire_count = 50
        self._draw()
        self.root.after(40, self._tick)

    def _draw(self):
        super()._draw()
        led = self.ledger
        if led is None:
            return
        cv = self.canvas
        x0, y0, w, h = 12, 4, 200, 12
        frac = min(1.0, max(0.0, led.pool / 300.0))      # 满条 = 300 分
        cv.create_rectangle(x0, y0, x0 + w, y0 + h,
                            outline="#333333", fill="#DDDDDD")
        color = ("#2E8B57" if led.pool >= self._econ["start"]
                 else "#CC7722" if led.pool >= 40 else "#CC3322")
        cv.create_rectangle(x0, y0, x0 + w * frac, y0 + h,
                            outline="", fill=color)
        cv.create_line(x0 + w / 3.0, y0, x0 + w / 3.0, y0 + h,
                       fill="#333333")                   # 100 分基准线
        stuck_pct = (100.0 * led.stuck_frames / led.frames
                     if led.frames else 0.0)
        empty_pct = (100.0 * led.empty_frames / led.frames
                     if led.frames else 0.0)
        stationary_pct = (
            100.0 * led.stationary_frames / led.stationary_observed_frames
            if led.stationary_observed_frames else 0.0)
        immunity = self.game.hit_immunity_remaining[1] / self._fps
        remain = max(0, self._econ["cap"] - led.frames) / self._fps
        cv.create_text(x0 + w + 10, y0 + h / 2, anchor="w",
                       text=(f"分数 {led.pool:.0f}  命中 {led.hits}  "
                             f"风格 {led.style:+.0f}  卡墙 {stuck_pct:.0f}%"
                             f"  空仓 {empty_pct:.0f}%  静止 {stationary_pct:.0f}%"
                             f"  免疫 {immunity:.1f}s  剩余 {remain:.0f}s"),
                       font=("Helvetica", 12, "bold"), fill="#333333")
        if self.settle_msg:
            cv.create_text(x0, y0 + h + 16, anchor="w",
                           text=self.settle_msg,
                           font=("Helvetica", 13, "bold"), fill="#CC3322")


def policy_name(p):
    return getattr(p, "name", type(p).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="hunter",
                    choices=["idle", "random", "hunter", "model",
                             "hybrid", "mpc", "scorenet", "p26",
                             "p27b", "p27-exact-shield", "exact", "key-hybrid",
                             "topology-hybrid", "temporal-hybrid",
                             "p27-hybrid",
                             "survival"])
    ap.add_argument("--model", default="training/models/best_model.zip")
    ap.add_argument("--net", default=None,
                    help="scorenet 权重路径 (默认: scorenet_best 现任冠军, "
                         "无则回退 p21b_scorenet); p26 时为 P26 权重")
    ap.add_argument("--fire-margin", type=float, default=0.16,
                    help="p26 部署开火门槛")
    ap.add_argument("--fire-assist-line", type=float, default=0.0,
                    help="p26 清晰炮线补射阈值; 0 表示关闭")
    ap.add_argument("--fire-assist-max-risk", type=float, default=0.35,
                    help="p26 补射允许的最大来弹风险")
    ap.add_argument("--fire-assist-min-delta", type=float, default=-0.03,
                    help="p26 补射要求 fire/no-fire 分差下限")
    ap.add_argument("--suppress-blind-fire-line", type=float, default=0.0,
                    help="p26 低炮线盲射抑制阈值; 0 表示关闭")
    ap.add_argument("--p27b-net",
                    default="training/models/p27b_risk_value_iter00.pt",
                    help="p27b/exact 的风险价值头权重")
    ap.add_argument("--top-k", type=int, default=12,
                    help="p27-hybrid/exact 搜索的先验候选数")
    ap.add_argument("--search-horizon", type=int, default=72,
                    help="p27-hybrid/exact 搜索的 rollout 帧数")
    ap.add_argument("--deadline-ms", type=float, default=30.0,
                    help="p27-hybrid 每帧决策截止时间（毫秒）")
    ap.add_argument("--action-hold-frames", type=int, default=6,
                    help="p27-hybrid 搜索动作保持帧数")
    ap.add_argument("--prior-refresh-frames", type=int, default=6,
                    help="p27-hybrid 重新计算P27b候选排序的间隔")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--k", type=int, default=5,
                    help="混合体候选数 (越大越强越慢)")
    ap.add_argument("--immune", action="store_true",
                    help="对手(Laika)对自己的子弹免疫, 不再神风自杀")
    ap.add_argument("--official-reset", action="store_true",
                    help="每回合按正式评测口径新建 Game(seed+i)")
    ap.add_argument(
        "--supervision-log",
        default="training/analysis/live/watch_supervision.jsonl",
        help="每局实时监督与失败归因 JSONL")
    ap.add_argument(
        "--temporal-intent-net",
        default="training/models/temporal_intent_topology_v1.pt")
    ap.add_argument("--temporal-confidence", type=float, default=0.60)
    ap.add_argument(
        "--movement-continuity-epsilon", type=float, default=0.0,
        help="搜索时在同等安全的动作里优先保持当前移动方向; 0.0 关闭")
    ap.add_argument(
        "--max-round-frames", type=int, default=TRUNCATE_FRAMES,
        help="单局最长帧数, 超过按平局截断换局 (双方僵持的死局会永远不结束)")
    ap.add_argument("--hunt", action="store_true",
                    help="survival 用纯狩猎档老师 (无风格项, 作对照)")
    args = ap.parse_args()

    model = model_env = None
    if args.policy == "idle":
        policy = IdlePolicy()
    elif args.policy == "random":
        policy = RandomPolicy(seed=1)
    elif args.policy == "hunter":
        policy = HunterPolicy()
    elif args.policy == "hybrid":
        from training.hybrid_agent import HybridPolicy
        policy = HybridPolicy(k=args.k)
        policy.name = "hybrid"
    elif args.policy == "mpc":
        from training.mpc_agent import MPCPolicy
        policy = MPCPolicy("L2", horizon=48, hold=16, n_samples=1)
        policy.name = "mpc"
    elif args.policy == "survival":
        from training.survival_mode import SurvivalMPC
        policy = SurvivalMPC(style=not args.hunt)
        policy.name = "纯狩猎老师" if args.hunt else "风格老师(MPC)"
        app = SurvivalApp(policy, seed=args.seed)
        app.run()
        return
    elif args.policy == "scorenet":
        from training.score_distill import ScoreNetPolicy
        net_path = args.net
        if net_path is None:
            best = "training/models/scorenet_best.pt"
            net_path = best if os.path.exists(best) else \
                "training/models/p21b_scorenet.pt"
        policy = ScoreNetPolicy(net_path)
        policy.name = f"scorenet ({os.path.basename(net_path)})"
    elif args.policy == "p26":
        from training.p26_amortized_mpc import P26Policy
        net_path = args.net or "training/models/p26_amortized_mpc_iter05.pt"
        policy = P26Policy(
            net_path=net_path,
            fire_margin=args.fire_margin,
            fire_threshold=0.0,
            kill_weight=0.0,
            death_weight=0.0,
            double_death_weight=0.0,
            survive_weight=0.0,
            fire_prob_weight=0.0,
            fire_assist_line=args.fire_assist_line,
            fire_assist_max_risk=args.fire_assist_max_risk,
            fire_assist_min_delta=args.fire_assist_min_delta,
            suppress_blind_fire_line=args.suppress_blind_fire_line,
        )
        policy.name = (
            f"p26 ({os.path.basename(net_path)}, margin={args.fire_margin:g}, "
            f"assist={args.fire_assist_line:g}, "
            f"suppress={args.suppress_blind_fire_line:g})")
    elif args.policy == "p27b":
        # 现任部署冠军: P26 动作评分底座 + P27b 风险/价值头 (纯网络, 零在线搜索)
        # 官方定级 1000@970000 = 88.2% / 500@990000 = 89.6%
        # 其余系数用 P27BRiskValuePolicy 的类默认值, 与官方定级配置一致
        from training.p27_risk_value import P27BRiskValuePolicy
        net_path = args.net or "training/models/p26_amortized_mpc_iter05.pt"
        policy = P27BRiskValuePolicy(
            base_net=net_path,
            value_net=args.p27b_net,
            fire_margin=args.fire_margin,
        )
        policy.name = (
            f"p27b ({os.path.basename(net_path)} + "
            f"{os.path.basename(args.p27b_net)}, "
            f"margin={args.fire_margin:g})")
    elif args.policy == "p27-hybrid":
        from training.p27_guided_search import P27GuidedSearchPolicy
        net_path = args.net or "training/models/p26_amortized_mpc_iter05.pt"
        policy = P27GuidedSearchPolicy(
            base_net=net_path,
            value_net=args.p27b_net,
            fire_margin=args.fire_margin,
            top_k=args.top_k,
            horizon=args.search_horizon,
            hold=min(8, args.search_horizon),
            deadline_ms=args.deadline_ms,
            action_hold_frames=args.action_hold_frames,
            prior_refresh_frames=args.prior_refresh_frames,
            seed=args.seed or 0,
        )
        policy.name = (
            f"P27b Hybrid (top{args.top_k}, h={args.search_horizon}, "
            f"deadline={args.deadline_ms:g}ms)")
    elif args.policy == "p27-exact-shield":
        # Frozen local-product configuration: 119/120 on the registered gate.
        from training.sparse_exact_safety_policy import SparseExactSafetyPolicy
        net_path = args.net or "training/models/p26_amortized_mpc_iter05.pt"
        policy = SparseExactSafetyPolicy(
            base_net=net_path,
            value_net=args.p27b_net,
            fire_margin=args.fire_margin,
            top_k=args.top_k,
            search_horizon=args.search_horizon,
            search_max_death=0.0,
            search_max_dd=0.0,
            successor_shield=True,
            successor_horizon=args.search_horizon,
            successor_shield_max_safe_roots=2,
            suppress_secured_fire=True,
            min_unsecured_fire_gain=2.0,
            audit_interval=1,
            proactive_interval=48,
            behavior_full_search=True,
            search_hold_frames=12,
            search_on_fire=True,
            risk_search_threshold=0.18,
            long_tail_fire_horizon=375,
            deterministic_search_seeds=True,
            unsafe_fallback_mode="strip_fire",
        )
        if args.seed is not None:
            policy.set_round_seed(args.seed)
        policy.name = "P27b + Exact Shield (local product champion)"
    elif args.policy == "exact":
        # 现任搜索冠军: Exact-State Safety-Shielded MPC (特权老师)
        # 固定基准 120/120, 未见种子 297/300; 每帧现场搜索, 回放会明显慢于实时
        from training.exact_state_mpc_teacher import ExactStatePriorGuidedMPC
        net_path = args.net or "training/models/p26_amortized_mpc_iter05.pt"
        policy = ExactStatePriorGuidedMPC(
            base_net=net_path,
            value_net=args.p27b_net,
            fire_margin=args.fire_margin,
            top_k=args.top_k,
            search_horizon=args.search_horizon,
            search_samples=1,
            search_death_penalty=0.18,
            search_dd_penalty=0.45,
            search_kill_bonus=0.05,
            search_max_death=0.0,
            search_max_dd=0.0,
            successor_shield=True,
            successor_horizon=args.search_horizon,
            successor_shield_max_safe_roots=2,
            suppress_secured_fire=True,
            min_unsecured_fire_gain=2.0,
            movement_continuity_epsilon=args.movement_continuity_epsilon,
        )
        if args.seed is not None:
            policy.set_round_seed(args.seed)
        policy.name = (f"exact-state MPC (top{args.top_k}, "
                       f"h={args.search_horizon}, 慢放)")
    elif args.policy in ("key-hybrid", "topology-hybrid", "temporal-hybrid"):
        from training.sparse_exact_safety_policy import SparseExactSafetyPolicy
        net_path = args.net or "training/models/p26_amortized_mpc_iter05.pt"
        topology_enabled = args.policy in ("topology-hybrid", "temporal-hybrid")
        temporal_enabled = args.policy == "temporal-hybrid"
        policy = SparseExactSafetyPolicy(
            base_net=net_path,
            value_net=args.p27b_net,
            fire_margin=args.fire_margin,
            top_k=args.top_k,
            search_horizon=args.search_horizon,
            search_max_death=0.0,
            search_max_dd=0.0,
            successor_shield=True,
            successor_horizon=args.search_horizon,
            successor_shield_max_safe_roots=2,
            suppress_secured_fire=True,
            min_unsecured_fire_gain=2.0,
            audit_interval=6,
            proactive_interval=24,
            behavior_full_search=True,
            search_hold_frames=6,
            search_on_fire=True,
            risk_search_threshold=0.18,
            long_tail_fire_horizon=375,
            topology_assist=topology_enabled,
            topology_intent_max_frames=75,
            topology_cooldown_frames=25,
            topology_pursuit_delay_frames=20,
            network_move_hold_frames=(
                0 if temporal_enabled else 4 if topology_enabled else 0),
            temporal_intent_net=(
                args.temporal_intent_net if temporal_enabled else None),
            temporal_confidence=args.temporal_confidence,
            movement_continuity_epsilon=args.movement_continuity_epsilon,
            deterministic_search_seeds=True,
        )
        if args.seed is not None:
            policy.set_round_seed(args.seed)
        experiment = ("learned-temporal" if temporal_enabled else
                      "topology-temporal" if topology_enabled else "key-event")
        policy.name = (f"{experiment} hybrid EXPERIMENT (top{args.top_k}, "
                       f"h={args.search_horizon})")
    else:
        import gymnasium as _g
        from stable_baselines3 import PPO
        from training.tt_gym_env import TankTroubleGym, obs_dim
        model = PPO.load(args.model, device="cpu")
        # 与 evaluate.ModelPolicy 一致: 按模型观测空间自动识别观测版本
        space = model.observation_space
        obs_map = isinstance(space, _g.spaces.Dict)
        dim = (space["vec"] if obs_map else space).shape[0]
        traj, nav = next(
            (t, v) for t in (True, False) for v in (True, False)
            if obs_dim(t, v) == dim)
        model_env = TankTroubleGym(seed=0, obs_traj=traj, obs_nav=nav,
                                   obs_map=obs_map)
        policy = IdlePolicy()   # 占位, 实际由 model 控制
        policy.name = "model"

    immune = {1} if args.immune else None
    app = PolicyApp(policy, seed=args.seed, model_env=model_env, model=model,
                    self_harm_immune=immune,
                    official_reset=args.official_reset,
                    supervision_log=args.supervision_log,
                    max_round_frames=args.max_round_frames,
                    run_config={
                        "cfg_policy": args.policy,
                        "cfg_movement_continuity_epsilon": (
                            args.movement_continuity_epsilon),
                        "cfg_temporal_intent_net": args.temporal_intent_net,
                        "cfg_top_k": args.top_k,
                        "cfg_search_horizon": args.search_horizon,
                        "cfg_temporal_confidence": args.temporal_confidence,
                    })
    app.run()


if __name__ == "__main__":
    main()
