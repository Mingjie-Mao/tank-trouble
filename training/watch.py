"""
回放器 — 在 tkinter 渲染器里观看策略对战 Laika

用法:
  python training/watch.py --policy hunter          # 观看手写猎杀脚本
  python training/watch.py --policy model           # 观看训练好的模型
  python training/watch.py --policy model --model training/models/best_model.zip
  python training/watch.py --policy hunter --seed 910007   # 复现指定局
  python training/watch.py --policy survival               # P24 生存老师狩猎回放
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from play_tank_trouble import App  # noqa: E402
from tank_trouble_original import constants  # noqa: E402
from training.baselines import IdlePolicy, RandomPolicy, HunterPolicy  # noqa: E402


class PolicyApp(App):
    """让策略接管 tank0 的渲染窗口 (R 键换局仍可用)"""

    def __init__(self, policy, seed=None, model_env=None, model=None,
                 self_harm_immune=None, invincible=None,
                 hit_immunity_frames=None, human_opponent=False):
        self.policy = policy
        self.model_env = model_env    # ModelPolicy 用: 独立观测环境
        self.model = model
        self.human_opponent = bool(human_opponent)
        # Keep this attribute available to the title code and tolerate older
        # local copies that passed the member (rather than the argument) into
        # App.__init__.
        self.self_harm_immune = self_harm_immune
        super().__init__(seed=seed, two_players=self.human_opponent,
                         self_harm_immune=self_harm_immune,
                         invincible=invincible,
                         hit_immunity_frames=hit_immunity_frames)
        tag = " [Laika免疫自伤]" if self_harm_immune else ""
        opponent = "真人 tank1（方向键移动，M开火）" \
            if self.human_opponent else "Laika"
        self.root.title(
            f"Tank Trouble — {policy_name(policy)} vs {opponent}{tag}")

    def _tick(self):
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
        if self.human_opponent:
            self._apply_input(g.tanks[1], self.p2_keys)
        g.step()
        self._draw()
        self.root.after(40, self._tick)   # 25 FPS


class ArenaApp(App):
    """两个策略直接对打的回放（无 Laika）。

    红 tank0 = A，黑 tank1 = B。B 通过 arena.MirrorView 拿到"我是 tank1"
    的世界视图，因为所有老师都写死了 me = tanks[0]。
    HUD 显示双方配置、已用帧数和僵持计时。
    """

    def __init__(self, policy_a, policy_b, label_a, label_b, seed=None,
                 ranked=False, cap_frames=750, score_rays=256):
        from training.arena import SideAdapter

        self.side_a = SideAdapter(policy_a, 0)
        self.side_b = SideAdapter(policy_b, 1)
        self.label_a = label_a
        self.label_b = label_b
        self.frames = 0
        self.verdict = ""
        self.ranked = bool(ranked)
        self.cap_frames = int(cap_frames)
        self.score_rays = int(score_rays)
        self.board = None
        # App.__init__ 里会立刻跑一帧 _tick()。如果让它跑，双方各会消耗
        # 一个动作、游戏推进到 frame 1，而随后的 reset() 又把回放的帧计数
        # 归零、把老师的场缓存和承诺状态清空 —— 动作序列就和游戏错开一帧。
        # 对帧级精确的 exploit 时间线来说，一帧就足以毁掉它。
        # 所以在准备就绪前，_tick 只排下一帧、不做任何事。
        self._ready = False
        super().__init__(seed=seed, two_players=True)
        self._ensure_board()
        self.side_a.reset()
        self.side_b.reset()
        self._ready = True
        self.root.title(f"擂台 — 红 {label_a}  vs  黑 {label_b}")

    def _ensure_board(self):
        """惰性创建计分板：构造期第一次 _tick 时 self.game 尚未就绪。"""
        if not self.ranked or self.board is not None:
            return
        if getattr(self, "game", None) is None:
            return
        from training.arena import ChainScoreboard
        self.board = ChainScoreboard(self.game, rays=self.score_rays)

    def _tick(self):
        g = getattr(self, "game", None)
        if g is None or not getattr(self, "_ready", False):
            if getattr(self, "root", None) is not None:
                self.root.after(40, self._tick)   # 仍要排下一帧
            return
        self._ensure_board()
        if not g.frozen:
            if g.tanks[0].alive:
                a = self.side_a.act(g)
                t = g.tanks[0]
                t.forward = bool(a.get("forward", False))
                t.backup = bool(a.get("backup", False))
                t.turn_left = bool(a.get("turn_left", False))
                t.turn_right = bool(a.get("turn_right", False))
                t.fire = bool(a.get("fire", False))
            if g.tanks[1].alive:
                b = self.side_b.act(g)
                t = g.tanks[1]
                t.forward = bool(b.get("forward", False))
                t.backup = bool(b.get("backup", False))
                t.turn_left = bool(b.get("turn_left", False))
                t.turn_right = bool(b.get("turn_right", False))
                t.fire = bool(b.get("fire", False))
            g.step()
            if self.board is not None:
                self.board.on_frame(g)
            self.frames += 1
            if (self.ranked and self.board is not None
                    and self.frames >= self.cap_frames
                    and g.tanks[0].alive and g.tanks[1].alive
                    and not self.verdict):
                sa, sb = self.board.totals
                if sa > sb:
                    self.verdict = f"到时 — {self.label_a} 分数胜 (0.4)"
                elif sb > sa:
                    self.verdict = f"到时 — {self.label_b} 分数胜 (0.4)"
                else:
                    self.verdict = "到时 — 同分判平"
        elif not self.verdict:
            alive0, alive1 = g.tanks[0].alive, g.tanks[1].alive
            if alive0 and not alive1:
                self.verdict = f"{self.label_a} 击杀获胜 (1.0)"
            elif alive1 and not alive0:
                self.verdict = f"{self.label_b} 击杀获胜 (1.0)"
            elif not alive0 and not alive1:
                self.verdict = "双亡 (0.1)"
            else:
                self.verdict = "超时"
        self._draw()
        self.root.after(40, self._tick)

    def _draw(self):
        super()._draw()
        cv = self.canvas
        seconds = self.frames / 25.0
        cv.create_text(
            12, 8, anchor="nw", fill="#B22222", font=("Helvetica", 11, "bold"),
            text=f"红 {self.label_a}")
        cv.create_text(
            12, 24, anchor="nw", fill="#222222", font=("Helvetica", 11, "bold"),
            text=f"黑 {self.label_b}")
        if self.ranked and self.board is not None:
            remain = max(0, self.cap_frames - self.frames) / 25.0
            sa, sb = self.board.totals
            cv.create_text(
                12, 44, anchor="nw", fill="#444444",
                font=("Helvetica", 11),
                text=f"倒计时 {remain:4.1f}s     链分  红 {sa:.0f} : "
                     f"{sb:.0f} 黑")
        else:
            cv.create_text(
                12, 44, anchor="nw", fill="#444444",
                font=("Helvetica", 11),
                text=f"{self.frames} 帧 / {seconds:.1f} 秒")
        if seconds > 20 and not self.verdict and not self.ranked:
            cv.create_text(
                12, 62, anchor="nw", fill="#CC7722",
                font=("Helvetica", 11, "bold"), text="⚠ 僵持中")
        if self.verdict:
            cv.create_text(
                12, 62, anchor="nw", fill="#2E8B57",
                font=("Helvetica", 13, "bold"), text=self.verdict)


class SurvivalApp(PolicyApp):
    """P24v3 生存模式回放: 免疫穿透计分板 + 遥测 HUD。

    经济规则复用 survival_mode.Ledger (与采集/评测同一实现)。
    死亡走引擎原生回合循环 (new_round 重开账本); 流干/到时换新局。
    """

    def __init__(self, policy, seed=None, econ=None, ledger_class=None):
        from training.survival_mode import Ledger, ECON, FPS
        self._Ledger = Ledger if ledger_class is None else ledger_class
        self._econ = dict(ECON if econ is None else econ)
        self._fps = FPS
        self.ledger = None
        self.dead_wait = False
        self.expire_count = -1
        self.settle_msg = ""
        super().__init__(
            policy, seed=seed, invincible={1},
            hit_immunity_frames={1: self._econ["hit_immunity"]})
        version = "v3" if self._econ["hit_immunity"] else "v2"
        self.root.title(
            f"Tank Trouble — 生存模式{version}: "
            f"{policy_name(policy)} vs 无敌Laika")

    def _tick(self):
        g = self.game
        if self.ledger is None:
            self.ledger = self._Ledger(g, self._econ)
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
            elif end == "wall_death":
                self.settle_msg = "碰墙 — 立即死亡，结算 0"
                self.expire_count = 37
            elif end == "cell_death":
                self.settle_msg = "同格停留满 2 秒 — 立即死亡，结算 0"
                self.expire_count = 37
            elif end == "cap":
                self.settle_msg = f"30s 到时 — 结算 {self.ledger.pool:.0f}"
                self.expire_count = 50
        self._draw()
        self.root.after(40, self._tick)

    def _draw(self):
        from tank_trouble_original import constants as constants

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
        mobility = "" if not hasattr(led, "cell_frames") else \
            f"  同格 {led.cell_frames/self._fps:.1f}/2.0s"
        cv.create_text(x0 + w + 10, y0 + h / 2, anchor="w",
                       text=(f"分数 {led.pool:.0f}  命中 {led.hits}  "
                             f"风格 {led.style:+.0f}  卡墙 {stuck_pct:.0f}%"
                             f"  空仓 {empty_pct:.0f}%  静止 {stationary_pct:.0f}%"
                             f"  免疫 {immunity:.1f}s  剩余 {remain:.0f}s"
                             f"{mobility}"),
                       font=("Helvetica", 12, "bold"), fill="#333333")
        if self.settle_msg:
            cv.create_text(x0, y0 + h + 16, anchor="w",
                           text=self.settle_msg,
                           font=("Helvetica", 13, "bold"), fill="#CC3322")


class CoinPathApp(App):
    """P35 金币课程回放：真实金币结算、MPC 控制和经济 HUD。"""

    def __init__(self, policy, seed=None, cap_seconds=30):
        from training.coin_path_rl import CoinPathEnv

        self.policy = policy
        self.coin_env = CoinPathEnv(
            35_500_001 if seed is None else seed, cap_seconds)
        self.coin_env.reset()
        self.finished = False
        self.finish_info = None
        super().__init__(seed=seed, two_players=False)
        self.root.title(
            f"Tank Trouble — {policy_name(policy)} vs Laika (seed={seed})")

    def _on_key_down(self, event):
        if self._key_name(event) == "r":
            self.coin_env.reset()
            self.policy.reset()
            self.game = self.coin_env.game
            self.finished = False
            self.finish_info = None
            return
        super()._on_key_down(event)

    def _tick(self):
        self.game = self.coin_env.game
        if not self.finished:
            action_index = self.policy.act_index(self.coin_env)
            _, _, self.finished, info = self.coin_env.step(action_index)
            if self.finished:
                self.finish_info = info
        self.game = self.coin_env.game
        self._draw()
        self.root.after(80, self._tick)

    def _draw(self):
        super()._draw()
        game = self.coin_env.game
        canvas = self.canvas
        offset_x, offset_y = 10.0, 10.0
        world_width = len(game.maze) * game.scale
        offset_x += max(0, (constants.MOVIEWIDTH - world_width) / 2)
        for (cell_x, cell_y), value in self.coin_env.coins.items():
            centre_x = offset_x + (cell_x + 0.5) * game.scale
            centre_y = offset_y + (cell_y + 0.5) * game.scale
            radius = 4.0
            canvas.create_oval(
                centre_x - radius, centre_y - radius,
                centre_x + radius, centre_y + radius,
                fill="#F5B700", outline="#755B00")
        remaining = max(
            0.0, (self.coin_env.cap - self.coin_env.ledger.frames) / 25.0)
        chain_seconds = self.coin_env.chain_timers[0] / 25.0
        opponent_chain_seconds = self.coin_env.chain_timers[1] / 25.0
        status = (
            f"金币 {self.coin_env.banks[0]:.0f} : "
            f"{self.coin_env.banks[1]:.0f} Laika   "
            f"连吃 x{2 ** self.coin_env.chain_counts[0]} "
            f"{chain_seconds:.1f}s : "
            f"x{2 ** self.coin_env.chain_counts[1]} "
            f"{opponent_chain_seconds:.1f}s   "
            f"参考 50   剩余 {remaining:.1f}s   "
            f"开火 {self.coin_env.shots}   "
            f"格子 {len(self.coin_env.ledger.visited)}")
        canvas.create_text(
            20, 8, anchor="nw", text=status,
            font=("Helvetica", 13, "bold"), fill="#8A5500")
        if self.finish_info is not None:
            success = self.finish_info["course_success"]
            message = (
                f"{'课程成功' if success else '课程失败'} — "
                f"{self.finish_info['end']} / "
                f"{self.finish_info['bank']:.0f} 金币   (R 下一局)")
            canvas.create_text(
                400, 36, anchor="n", text=message,
                font=("Helvetica", 16, "bold"),
                fill="#188038" if success else "#C5221F")


class KillFieldApp(PolicyApp):
    """P37 回放：在格子中心叠加反演弹道密度层级。"""

    FIELD_COLOURS = (
        "#8EC5FF", "#63B3ED", "#48BB78", "#ECC94B",
        "#ED8936", "#E53E3E", "#9F7AEA",
    )

    def _draw(self):
        super()._draw()
        field = getattr(self.policy, "field", None)
        if field is None:
            return
        game = self.game
        canvas = self.canvas
        offset_x, offset_y = 10.0, 10.0
        world_width = len(game.maze) * game.scale
        offset_x += max(0, (constants.MOVIEWIDTH - world_width) / 2)
        for cell_x in range(field.counts.shape[0]):
            for cell_y in range(field.counts.shape[1]):
                tier = int(field.tiers[cell_x, cell_y])
                if tier <= 0:
                    continue
                centre_x = offset_x + (cell_x + 0.5) * game.scale
                centre_y = offset_y + (cell_y + 0.5) * game.scale
                radius = 2.0 + 1.35 * tier
                colour = self.FIELD_COLOURS[
                    min(tier, len(self.FIELD_COLOURS)) - 1]
                canvas.create_oval(
                    centre_x - radius, centre_y - radius,
                    centre_x + radius, centre_y + radius,
                    outline=colour, width=2)
        me = game.tanks[0]
        cell = (int(me.x // game.scale), int(me.y // game.scale))
        telemetry = self.policy.telemetry()
        canvas.create_text(
            20, 8, anchor="nw",
            text=(f"P37 击杀场  当前层 {field.tier_at(cell)}  "
                  f"指数值 {field.value_at(cell):.0f}  "
                  f"追猎链 x{2 ** min(telemetry['hunt_chain'], 6)} "
                  f"{telemetry['hunt_chain_timer']/25.0:.1f}s  "
                  f"包络 {field.guidance_at(cell):.2f}  "
                  f"射线 {field.count_at(cell)}/{field.ray_count}  "
                  f"动作失效 {telemetry.get('no_effect_frames', 0)}/"
                  f"{telemetry.get('no_effect_events', 0)}  "
                  f"建场 {telemetry['mean_field_build_seconds']*1000:.0f}ms"),
            font=("Helvetica", 12, "bold"), fill="#6B2FA0")


class OriginalSurvivalTeacher:
    """P24v2 风格生存老师使用原版战斗规则回放。"""

    name = "P24v2 风格生存老师（原版规则）"

    def __init__(self):
        from training.survival_mode import ECON, SurvivalMPC
        self.econ = dict(ECON, empty_mag=0.0, hit_immunity=0)
        self.teacher = SurvivalMPC(style=True, econ=self.econ)
        self.game = None
        self.round_number = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.last_action = {}

    def reset(self):
        self.game = None
        self.round_number = None
        self.ledger = None
        self.context_game = None
        self.context_round = None
        self.context_step = 0
        self.last_action = {}
        self.teacher.reset()

    def act_ctx(self, game, ledger):
        if not game.tanks[0].alive:
            return {}
        if (game is not self.context_game
                or game.round_number != self.context_round):
            self.context_game = game
            self.context_round = game.round_number
            self.context_step = 0
            self.last_action = {}
            self.teacher.reset()
        if self.context_step % 2 == 0:
            self.last_action = self.teacher.act_ctx(game, ledger)
        self.context_step += 1
        return self.last_action

    def act(self, game):
        from training.survival_mode import Ledger
        if not game.tanks[0].alive:
            return {}
        if game is not self.game or game.round_number != self.round_number:
            self.game = game
            self.round_number = game.round_number
            self.ledger = Ledger(game, self.econ)
        else:
            end = self.ledger.on_frame(game, game.events)
            if end in ("drain", "cap"):
                self.ledger = Ledger(game, self.econ)
        return self.act_ctx(game, self.ledger)


def policy_name(p):
    return getattr(p, "name", type(p).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="hunter",
                    choices=["idle", "random", "hunter", "model",
                             "hybrid", "mpc", "scorenet", "survival",
                             "best", "p25v3-teacher",
                             "p24-student-original",
                             "p24-teacher-original",
                             "p24-teacher-survival",
                             "p24v4-student-original",
                             "p24v4-student-survival",
                             "p24r530-student-original",
                             "p24r530-student-survival",
                             "p27-frontier-original",
                             "p27-frontier-survival",
                             "p28-dense-frontier-original",
                             "p28-dense-frontier-survival",
                             "p29-opportunity-original",
                             "p29-opportunity-survival",
                             "p30-initiative-original",
                             "p30-initiative-survival",
                             "p31-guide-original",
                             "p31-guide-survival",
                             "p32-decisive-original",
                             "p32-decisive-survival",
                             "p33-mobility-survival",
                             "p35-coin-mpc",
                             "p36-chain-mpc",
                             "p37-killfield-teacher",
                             "p37-killfield-fast",
                             "p37-killfield-realtime",
                             "p37-killfield-student",
                             "p37-killfield-full",
                             "p38-killfield-fast",
                             "p39-killfield",
                             "p40-killfield",
                             "arena",
                             "exploit-replay",
                             "p35-coin-student"])
    ap.add_argument("--model", default="training/models/best_model.zip")
    ap.add_argument("--net", default=None,
                    help="评分网络权重路径；不同回放预设有各自默认模型")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--k", type=int, default=5,
                    help="混合体候选数 (越大越强越慢)")
    ap.add_argument("--field-rays", type=int, default=512,
                    help="P37 反演场射线数；录像默认512，离线老师可用2048")
    ap.add_argument("--field-bounces", type=int, default=2,
                    help="P37 反演场允许的最大反弹次数")
    ap.add_argument("--field-flight-frames", type=int, default=75,
                    help="P37 有效弹道最长飞行帧数；默认75帧=3秒")
    ap.add_argument("--exploit-path", default=None,
                    help="回放指定的击杀路径 json；不给则用找到的第一条")
    ap.add_argument("--ranked", action="store_true",
                    help="擂台用新规则：30 秒结算 + 追猎链分裁决")
    ap.add_argument("--cap-seconds", type=float, default=30.0,
                    help="新规则的结算时间")
    ap.add_argument("--chain-weight", type=float, default=None,
                    help="老师的 HUNT_CHAIN_GAIN_WEIGHT（原值 12）")
    ap.add_argument("--arena-horizon-a", type=int, default=36,
                    help="擂台红方 MPC 视野帧数")
    ap.add_argument("--arena-horizon-b", type=int, default=36,
                    help="擂台黑方 MPC 视野帧数")
    ap.add_argument("--field-workers", type=int, default=4,
                    help="p37-killfield-fast 的并行进程数；本机 4 性能核")
    ap.add_argument("--opponent-profile",
                    choices=["laika", "mixed", "human"], default="laika",
                    help="实时P37对手模型；laika保留原行为，human不调用Laika白盒")
    ap.add_argument("--plan-deadline", type=float, default=2.0,
                    help="实时P37后台搜索硬时限（秒），逾期结果丢弃")
    ap.add_argument("--search-budget-ms", type=float, default=35.0,
                    help="搜索+网络混合学生的单次同步搜索预算")
    ap.add_argument("--immune", action="store_true",
                    help="对手(Laika)对自己的子弹免疫, 不再神风自杀")
    ap.add_argument("--human-opponent", action="store_true",
                    help="关闭Laika，由真人用方向键+M控制tank1")
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
        policy = HybridPolicy(
            k=args.k, opponent_profile=args.opponent_profile,
            budget_ms=args.search_budget_ms)
        policy.name = f"hybrid-{args.opponent_profile}"
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
    elif args.policy == "best":
        from training.opportunity_distill_v2 import OpportunityScoreNetPolicyV2
        net_path = args.net or "training/models/best_model.pt"
        policy = OpportunityScoreNetPolicyV2(net_path)
        policy.name = f"当前最佳 P25v2 ({os.path.basename(net_path)})"
    elif args.policy == "p25v3-teacher":
        from training.opportunity_teacher_v3 import OpportunityMPCInverse
        policy = OpportunityMPCInverse(seed=313)
        policy.name = "P25v3 反演击杀场老师"
    elif args.policy == "p24-student-original":
        from training.survival_distill import SurvivalScoreNetPolicy
        from training.survival_mode import ECON
        net_path = args.net or "training/models/p24_scorenet.pt"
        policy = SurvivalScoreNetPolicy(net_path)
        policy._econ = dict(ECON, empty_mag=0.0, hit_immunity=0)
        policy.name = "P24v2.1 生存课程学生（原版规则）"
    elif args.policy == "p24-teacher-original":
        policy = OriginalSurvivalTeacher()
    elif args.policy == "p24-teacher-survival":
        policy = OriginalSurvivalTeacher()
        app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
        app.run()
        return
    elif args.policy in ("p24v4-student-original",
                         "p24v4-student-survival"):
        from training.survival_distill_v2 import SurvivalTwoHeadPolicy
        net_path = args.net or "training/models/p24v4_survival_best.pt"
        policy = SurvivalTwoHeadPolicy(net_path)
        policy.name = f"P24v4 生存学生（{os.path.basename(net_path)}）"
        if args.policy == "p24v4-student-survival":
            app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
            app.run()
            return
    elif args.policy in ("p24r530-student-original",
                         "p24r530-student-survival"):
        from training.survival_expert_iter_530 import SurvivalReplica530Policy
        net_path = args.net or "training/models/p24r530_best.pt"
        policy = SurvivalReplica530Policy(net_path)
        policy.name = f"P24-P22 Replica-530（{os.path.basename(net_path)}）"
        if args.policy == "p24r530-student-survival":
            app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
            app.run()
            return
    elif args.policy in ("p27-frontier-original",
                         "p27-frontier-survival",
                         "p28-dense-frontier-original",
                         "p28-dense-frontier-survival"):
        from training.survival_frontier_rl import FrontierRLPolicy
        default_net = "training/models/p28_dense_frontier_actor.pt" \
            if args.policy.startswith("p28") else \
            "training/models/p27_frontier_actor.pt"
        net_path = args.net or default_net
        policy = FrontierRLPolicy(net_path)
        policy.name = f"可观测前沿课程（{os.path.basename(net_path)}）"
        if args.policy.endswith("survival"):
            app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
            app.run()
            return
    elif args.policy in ("p29-opportunity-original",
                         "p29-opportunity-survival"):
        from training.survival_opportunity_rl import OpportunityRLPolicy
        net_path = args.net or "training/models/p29_opportunity_actor.pt"
        policy = OpportunityRLPolicy(net_path)
        policy.name = f"P29 生存机会课程（{os.path.basename(net_path)}）"
        if args.policy == "p29-opportunity-survival":
            app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
            app.run()
            return
    elif args.policy in ("p30-initiative-original",
                         "p30-initiative-survival"):
        from training.survival_initiative_rl import InitiativeRLPolicy
        net_path = args.net or "training/models/p30_initiative_actor.pt"
        policy = InitiativeRLPolicy(net_path)
        policy.name = f"P30 反事实主动性课程（{os.path.basename(net_path)}）"
        if args.policy == "p30-initiative-survival":
            app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
            app.run()
            return
    elif args.policy in ("p31-guide-original",
                         "p31-guide-survival"):
        from training.survival_guided_attack import GuidedAttackPolicy
        policy = GuidedAttackPolicy(seed=0)
        if args.policy == "p31-guide-survival":
            app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
            app.run()
            return
    elif args.policy in ("p32-decisive-original",
                         "p32-decisive-survival"):
        from training.survival_decisive_attack import DecisiveAttackPolicy
        policy = DecisiveAttackPolicy(seed=0)
        if args.policy == "p32-decisive-survival":
            app = SurvivalApp(policy, seed=args.seed, econ=policy.econ)
            app.run()
            return
    elif args.policy == "p33-mobility-survival":
        from training.survival_mobility_attack import MobilityAttackPolicy
        from training.survival_mobility_law import MobilityLawLedger
        policy = MobilityAttackPolicy(seed=0)
        app = SurvivalApp(
            policy, seed=args.seed, econ=policy.econ,
            ledger_class=MobilityLawLedger)
        app.run()
        return
    elif args.policy == "p35-coin-mpc":
        from training.coin_path_mpc import CoinPathMPC
        policy = CoinPathMPC(seed=353, horizon=32, hold=12, samples=1)
        app = CoinPathApp(policy, seed=args.seed)
        app.run()
        return
    elif args.policy == "p36-chain-mpc":
        from training.coin_path_mpc import ChainCoinMPC
        policy = ChainCoinMPC(seed=363, horizon=48, hold=12, samples=1)
        app = CoinPathApp(policy, seed=args.seed)
        app.run()
        return
    elif args.policy == "p37-killfield-teacher":
        from training.killfield_teacher import KillFieldTeacher
        policy = KillFieldTeacher(
            seed=373, ray_count=args.field_rays,
            max_bounces=args.field_bounces,
            max_flight_frames=args.field_flight_frames)
        app = KillFieldApp(
            policy, seed=args.seed,
            human_opponent=args.human_opponent)
        app.run()
        return
    elif args.policy == "p37-killfield-fast":
        # 与 p37-killfield-teacher 行为逐帧等价，只是调度不同：
        # 跳过必被屏蔽的 8 个候选 + 推演与建场按射线并行。
        from training.killfield_prebuild import FastKillFieldTeacher
        policy = FastKillFieldTeacher(
            seed=373, ray_count=args.field_rays,
            max_bounces=args.field_bounces,
            max_flight_frames=args.field_flight_frames,
            skip_masked=True,
            parallel_workers=args.field_workers,
            parallel_field=True)
        app = KillFieldApp(
            policy, seed=args.seed,
            human_opponent=args.human_opponent)
        app.run()
        return
    elif args.policy == "p37-killfield-realtime":
        from training.killfield_realtime import RealtimeKillFieldTeacher
        policy = RealtimeKillFieldTeacher(
            seed=373, ray_count=args.field_rays,
            max_bounces=args.field_bounces,
            max_flight_frames=args.field_flight_frames,
            opponent_profile=args.opponent_profile,
            max_plan_seconds=args.plan_deadline)
        policy.name = f"P37 实时老师（{args.opponent_profile}）"
        app = KillFieldApp(
            policy, seed=args.seed,
            human_opponent=args.human_opponent)
        app.run()
        return
    elif args.policy == "p37-killfield-student":
        from training.killfield_distill import KillFieldStudentPolicy
        net_path = args.net or \
            "training/models/p37_killfield_student_short.pt"
        policy = KillFieldStudentPolicy(
            net_path, ray_count=args.field_rays,
            max_bounces=args.field_bounces,
            max_flight_frames=args.field_flight_frames)
        policy.name = f"P37 击杀场蒸馏学生（{os.path.basename(net_path)}）"
        app = KillFieldApp(policy, seed=args.seed)
        app.run()
        return
    elif args.policy == "p37-killfield-full":
        from training.killfield_full_distill import KillFieldFullPolicy
        net_path = args.net or "training/models/p37_killfield_full.pt"
        policy = KillFieldFullPolicy(
            net_path, rays=args.field_rays,
            bounces=args.field_bounces,
            flight_frames=args.field_flight_frames)
        policy.name = f"P37 完整蒸馏单网络（{os.path.basename(net_path)}）"
        app = KillFieldApp(policy, seed=args.seed)
        app.run()
        return
    elif args.policy == "exploit-replay":
        from training.exploit_search import RecordedExploit, list_paths
        from training.killfield_teacher import KillFieldTeacher
        import json as _json
        paths = list_paths()
        chosen = args.exploit_path or (paths[0] if paths else None)
        if not chosen:
            raise SystemExit("还没有搜索出来的击杀路径")
        spec = _json.load(open(chosen))
        print(f"回放 {os.path.basename(chosen)}  地图种子 {spec['seed']}  "
              f"{spec['depth']} 个宏动作")
        target = KillFieldTeacher(
            seed=spec["teacher_seed"], ray_count=spec["rays"],
            max_bounces=2, max_flight_frames=75, horizon=spec["horizon"])
        app = ArenaApp(target, RecordedExploit(chosen),
                       "P37 模型", "搜索出的杀法",
                       seed=spec["seed"], ranked=False)
        app.run()
        return
    elif args.policy == "arena":
        # 用行为等价的加速版跑，两边各 ~15ms/决策，25FPS 下看得动。
        from training.killfield_prebuild import FastKillFieldTeacher

        import training.killfield_teacher as _kt
        if args.chain_weight is not None:
            _kt.HUNT_CHAIN_GAIN_WEIGHT = args.chain_weight

        def build(horizon, seed):
            return FastKillFieldTeacher(
                seed=seed, ray_count=args.field_rays,
                max_bounces=args.field_bounces,
                max_flight_frames=args.field_flight_frames,
                horizon=horizon, skip_masked=True, parallel_workers=0)

        app = ArenaApp(
            build(args.arena_horizon_a, 373),
            build(args.arena_horizon_b, 991),
            f"P37 H{args.arena_horizon_a}",
            f"P37 H{args.arena_horizon_b}",
            seed=args.seed,
            ranked=args.ranked,
            cap_frames=int(args.cap_seconds * 25),
            score_rays=args.field_rays)
        app.run()
        return
    elif args.policy == "p40-killfield":
        from training.killfield_p39_distill import make_p40_policy
        policy = make_p40_policy(
            args.net or "training/models/p40_killfield_fieldin.pt",
            rays=args.field_rays, bounces=args.field_bounces,
            flight=args.field_flight_frames)
        app = KillFieldApp(policy, seed=args.seed,
                           human_opponent=args.human_opponent)
        app.run()
        return
    elif args.policy == "p39-killfield":
        from training.killfield_p39_distill import make_p39_policy
        net_path = args.net or "training/models/p39_killfield.pt"
        policy = make_p39_policy(net_path)
        app = KillFieldApp(
            policy, seed=args.seed,
            human_opponent=args.human_opponent)
        app.run()
        return
    elif args.policy == "p38-killfield-fast":
        from training.killfield_fast_distill import KillFieldFastPolicy
        net_path = args.net or "training/models/p38_killfield_fast.pt"
        policy = KillFieldFastPolicy(net_path)
        policy.name = f"P38 快速特权蒸馏网络（{os.path.basename(net_path)}）"
        app = KillFieldApp(policy, seed=args.seed)
        app.run()
        return
    elif args.policy == "p35-coin-student":
        from training.coin_path_rl import CoinPathRLPolicy
        net_path = args.net or "training/models/p35_coin_actor.pt"
        policy = CoinPathRLPolicy(net_path)
        policy.name = f"P35 金币课程 RL 学生（{os.path.basename(net_path)}）"
        app = CoinPathApp(policy, seed=args.seed)
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
                    self_harm_immune=immune)
    app.run()


if __name__ == "__main__":
    main()
