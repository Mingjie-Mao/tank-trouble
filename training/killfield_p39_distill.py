"""P39：分解目标的 P37 蒸馏。

P38 用一个回归头去拟合 `teacher_score / 12000`。那个标签里同时住着两种
量级完全不同的东西：

* 终局值 ±1.0（死亡 / 主动击杀），占 22.7% 的动作格；
* 塑形值 ~±0.05（导航、对准、场爬升、开火收益）。

单头 smooth_l1 的梯度被终局主导，塑形被挤掉。实测后果：网络逐动作预测
误差中位 0.269，而状态内塑形分差中位只有 0.051——噪声是信号的 5 倍。
在纯塑形状态上 top-1 一致率 6.9%，低于随机。网络学会了"哪些动作会死"，
没学到"该往哪走"，而后者正是 P37 那种果断跑图风格的来源。

P39 把目标拆开：

* `outcome_head`：每个候选属于 {致死, 制胜, 中性} 哪一类（交叉熵）。
  这是干净的大信号。
* `shape_head`：只在中性候选内部做**逐状态归一化**的排序回归，并按该
  状态的真实分差加权——老师本来就无差别的状态不该贡献梯度。
* `margin_head`：终局候选内部的残差（早杀优于晚杀，晚死优于早死）。

部署时不需要反归一化：制胜 > 中性 > 致死 这个序在老师的量纲下恒成立
（制胜 +1.0，中性 ≤ 0.15，致死 -1.0），而归一化是状态内单调的，所以
中性组内部的排序也被保留。

候选从 18 降到 10：`mask_moving_fire_scores` 会把 8 个"移动+开火"列
无条件覆写，它们既不该占推演也不该占网络容量。
"""

import argparse
import os
import sys
import warnings
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.killfield_fast_distill import (  # noqa: E402
    FAST_VECTOR_DIM, FAST_MAP_CHANNELS, MAP_H, MAP_W,
    KILLFIELD_EXTRA_DIM, ACTION_PREVIEW_DIM,
    load_all,
)
from training.killfield_teacher import LIVE_ACTION_INDICES  # noqa: E402

DEFAULT_DATA_DIR = "training/killfield_p39_data"
DEFAULT_MODEL = "training/models/p39_killfield.pt"

# 标签分类阈值。取自实测分布：致死 14.6%、制胜 8.1%、中性 77.3%。
# 中性区间刻意做宽，把"对手自杀"(0.125) 和开火收益/自杀弹道罚
# (+0.15 / -0.21) 都收进来，交给状态内归一化处理。
LETHAL_THRESHOLD = -0.5
WINNING_THRESHOLD = 0.5

CLASS_LETHAL, CLASS_WINNING, CLASS_NEUTRAL = 0, 1, 2
NUM_CLASSES = 3
NUM_ACTIONS = len(LIVE_ACTION_INDICES)

# 状态内分差小于这个值时，老师是真无差别，排序损失不该给梯度。
SPREAD_FLOOR = 1e-3
SPREAD_REFERENCE = 0.05     # 分差加权的归一化基准（实测中位）


def derive_targets(Y):
    """从 P38 格式的 18 维标签导出 P39 的分解目标。

    返回 dict：
      cls      (N,10) int64   类别
      shape    (N,10) float32 中性组内 [0,1] 归一化排序目标
      shape_m  (N,10) float32 掩码（只有中性候选参与）
      margin   (N,10) float32 终局组内残差
      margin_m (N,10) float32 掩码
      weight   (N,)   float32 该状态的排序权重（按真实分差）
    """
    live = np.asarray(LIVE_ACTION_INDICES, dtype=np.int64)
    y = np.asarray(Y, dtype=np.float32)[:, live]

    cls = np.full(y.shape, CLASS_NEUTRAL, dtype=np.int64)
    cls[y < LETHAL_THRESHOLD] = CLASS_LETHAL
    cls[y > WINNING_THRESHOLD] = CLASS_WINNING

    neutral = cls == CLASS_NEUTRAL
    terminal = ~neutral

    # ---- 中性组：逐状态 min-max 归一化 ----
    # 少数状态没有任何中性候选（全是致死/制胜），nanmin 会整行 NaN。
    masked = np.where(neutral, y, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmin(masked, axis=1)
        hi = np.nanmax(masked, axis=1)
    lo = np.nan_to_num(lo, nan=0.0)
    hi = np.nan_to_num(hi, nan=0.0)
    spread = hi - lo
    denominator = np.maximum(spread, 1e-8)[:, None]
    shape = np.clip((y - lo[:, None]) / denominator, 0.0, 1.0)
    shape = np.where(neutral, shape, 0.0).astype(np.float32)

    # ---- 终局组：残差（制胜 y-1，致死 y+1），放大到可见量级 ----
    base = np.where(cls == CLASS_WINNING, 1.0,
                    np.where(cls == CLASS_LETHAL, -1.0, 0.0))
    margin = np.clip((y - base) * 40.0, -3.0, 3.0)
    margin = np.where(terminal, margin, 0.0).astype(np.float32)

    # ---- 状态权重：老师真无差别时不给排序梯度 ----
    usable = (neutral.sum(axis=1) >= 2) & (spread > SPREAD_FLOOR)
    weight = np.where(
        usable, np.clip(spread / SPREAD_REFERENCE, 0.0, 3.0), 0.0)

    return {
        "cls": cls,
        "shape": shape,
        "shape_m": (neutral & usable[:, None]).astype(np.float32),
        "margin": margin,
        "margin_m": terminal.astype(np.float32),
        "weight": weight.astype(np.float32),
    }


# 场作为输入时的向量维度：308 廉价事实 + 134 击杀场事实 + 90 移动预演
# + 1 有效位（击杀后没有对手，场无定义）。
FIELD_INPUT_DIM = (FAST_VECTOR_DIM + KILLFIELD_EXTRA_DIM
                   + ACTION_PREVIEW_DIM + 1)


def augmented_vectors(data):
    """把辅助标签 F/P 挪进输入。浏览器负担得起精确建场，所以不必让
    网络从迷宫位图里自己重建反弹几何——实测它重建不出来
    （MAE 0.1772 对信号 0.2662，只比全输出 0 好 33%）。"""
    valid = np.asarray(data["FM"], dtype=np.float32)[:, None]
    return np.concatenate(
        [data["V"], data["F"] * valid, data["P"] * valid, valid],
        axis=1).astype(np.float32)


def build_p39_network(field_input=False):
    import torch
    import torch.nn as nn

    vector_dim = FIELD_INPUT_DIM if field_input else FAST_VECTOR_DIM

    class P39Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.field_input = bool(field_input)
            self.map_encoder = nn.Sequential(
                nn.Conv2d(FAST_MAP_CHANNELS, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * MAP_H * MAP_W, 256), nn.ReLU(),
            )
            self.vector_encoder = nn.Sequential(
                nn.Linear(vector_dim, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(768, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
            )
            self.outcome_head = nn.Linear(1024, NUM_ACTIONS * NUM_CLASSES)
            self.shape_head = nn.Linear(1024, NUM_ACTIONS)
            self.margin_head = nn.Linear(1024, NUM_ACTIONS)
            # 场已经是输入时，重建头等于预测自己的输入，没有意义
            self.field_head = (None if field_input
                               else nn.Linear(1024, KILLFIELD_EXTRA_DIM))
            self.preview_head = (None if field_input
                                 else nn.Linear(1024, ACTION_PREVIEW_DIM))
            self.survival_head = nn.Linear(1024, 1)

        def forward(self, vector, spatial):
            latent = self.fusion(torch.cat([
                self.vector_encoder(vector), self.map_encoder(spatial),
            ], dim=1))
            outcome = self.outcome_head(latent).view(
                -1, NUM_ACTIONS, NUM_CLASSES)
            return {
                "outcome": outcome,
                "shape": self.shape_head(latent),
                "margin": self.margin_head(latent),
                "field": (None if self.field_head is None
                          else self.field_head(latent)),
                "preview": (None if self.preview_head is None
                            else self.preview_head(latent)),
                "survival": self.survival_head(latent).squeeze(1),
            }

    return P39Net()


def reconstruct_scores(outcome_logits, shape_pred, margin_pred):
    """把三个头合成一个可 argmax 的评分。

    制胜 > 中性 > 致死 在老师量纲下恒成立，而 shape 的状态内单调性保证
    中性组内部序不变，因此不需要反归一化。
    """
    import torch

    cls = outcome_logits.argmax(dim=-1)
    scores = torch.zeros_like(shape_pred)
    winning = cls == CLASS_WINNING
    lethal = cls == CLASS_LETHAL
    neutral = cls == CLASS_NEUTRAL
    # 中性：直接用归一化排序值，落在 [0,1]
    scores = torch.where(neutral, shape_pred.clamp(0.0, 1.0), scores)
    # 制胜：抬到 10 以上，margin 越小（越早杀）越好
    scores = torch.where(
        winning, 10.0 - margin_pred.clamp(-3.0, 3.0), scores)
    # 致死：压到 -10 以下，margin 越大（越晚死）越好
    scores = torch.where(
        lethal, -10.0 + margin_pred.clamp(-3.0, 3.0), scores)
    return scores


def _build_p38_compatible(core):
    """把 P39 的多头包成 P38 的 (18维评分, field, preview, survival) 接口。

    这样 `KillFieldFastPolicy.act()` 的整套状态机——承诺执行、强制开火、
    击杀后屏蔽——可以原样复用，不必复制一遍。
    """
    import torch
    import torch.nn as nn

    live = torch.as_tensor(LIVE_ACTION_INDICES, dtype=torch.long)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core

        def forward(self, vector, spatial):
            out = self.core(vector, spatial)
            ten = reconstruct_scores(
                out["outcome"], out["shape"], out["margin"])
            full = torch.full(
                (ten.shape[0], 18), -1.0e9,
                dtype=ten.dtype, device=ten.device)
            full[:, live.to(ten.device)] = ten
            return (full, out["field"], out["preview"], out["survival"])

    return Adapter()


def make_p39_policy(model_path=DEFAULT_MODEL, cap_frames=500, **kwargs):
    """P39 部署策略：复用 P38 的执行器，只换评分网络。"""
    import torch
    from training.killfield_fast_distill import KillFieldFastPolicy

    policy = KillFieldFastPolicy.__new__(KillFieldFastPolicy)
    from training.tt_gym_env import TankTroubleGym

    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if payload.get("version") != "p39_decomposed_distill":
        raise ValueError(f"{model_path} 不是 P39 权重")
    core = build_p39_network()
    core.load_state_dict(payload["state_dict"])
    core.eval()
    policy.torch = torch
    policy.network = _build_p38_compatible(core)
    policy.network.eval()
    policy.encoder = TankTroubleGym(
        seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
    policy.cap_frames = int(cap_frames)
    policy.name = "P39 分解目标蒸馏网络"
    policy.reset()
    with torch.inference_mode():
        policy.network(
            torch.zeros(1, FAST_VECTOR_DIM),
            torch.zeros(1, FAST_MAP_CHANNELS, MAP_H, MAP_W))
    return policy


def train(args):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    data = load_all(args.data_dir)
    targets = derive_targets(data["Y"])
    n = len(data["V"])
    if args.field_input:
        data["V"] = augmented_vectors(data)
        print(f"[场作为输入] 向量维度 -> {data['V'].shape[1]}", flush=True)

    generator = np.random.default_rng(args.seed)
    groups = generator.permutation(np.unique(data["G"]))
    val_count = max(1, int(len(groups) * args.validation_fraction))
    val_groups = groups[:val_count]
    is_val = np.isin(data["G"], val_groups)
    val_index = np.flatnonzero(is_val)
    train_index = np.flatnonzero(~is_val)

    device = torch.device(args.device)
    tensors = {k: torch.as_tensor(v, device=device)
               for k, v in data.items()}
    tensors.update({k: torch.as_tensor(v, device=device)
                    for k, v in targets.items()})

    net = build_p39_network(field_input=args.field_input).to(device)
    optimiser = torch.optim.AdamW(
        net.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    print(f"===== P39 train: {len(train_index)} train / "
          f"{len(val_index)} val / {len(np.unique(data['G']))} 局 =====",
          flush=True)
    for name, value in (
            ("致死", (targets["cls"] == CLASS_LETHAL).mean()),
            ("制胜", (targets["cls"] == CLASS_WINNING).mean()),
            ("中性", (targets["cls"] == CLASS_NEUTRAL).mean())):
        print(f"   类别占比 {name}: {100 * value:.1f}%", flush=True)
    print(f"   有排序信号的状态: "
          f"{100 * (targets['weight'] > 0).mean():.1f}%", flush=True)

    def batch_loss(index):
        out = net(tensors["V"][index], tensors["M"][index])
        cls_target = tensors["cls"][index]
        outcome_loss = F.cross_entropy(
            out["outcome"].reshape(-1, NUM_CLASSES),
            cls_target.reshape(-1), reduction="none",
        ).view(cls_target.shape).mean(1)

        shape_mask = tensors["shape_m"][index]
        shape_loss = (F.smooth_l1_loss(
            out["shape"], tensors["shape"][index], reduction="none"
        ) * shape_mask).sum(1) / shape_mask.sum(1).clamp_min(1.0)
        shape_loss = shape_loss * tensors["weight"][index]

        margin_mask = tensors["margin_m"][index]
        margin_loss = (F.smooth_l1_loss(
            out["margin"], tensors["margin"][index], reduction="none"
        ) * margin_mask).sum(1) / margin_mask.sum(1).clamp_min(1.0)

        if out["field"] is None:
            field_loss = preview_loss = torch.zeros_like(outcome_loss)
        else:
            privileged = tensors["FM"][index]
            field_loss = F.smooth_l1_loss(
                out["field"], tensors["F"][index], reduction="none"
            ).mean(1) * privileged
            preview_loss = F.smooth_l1_loss(
                out["preview"], tensors["P"][index], reduction="none"
            ).mean(1) * privileged
        survival_loss = F.binary_cross_entropy_with_logits(
            out["survival"], tensors["S"][index], reduction="none")

        total = (outcome_loss
                 + args.shape_coef * shape_loss
                 + args.margin_coef * margin_loss
                 + args.field_coef * field_loss
                 + args.preview_coef * preview_loss
                 + args.survival_coef * survival_loss)
        return total.mean()

    def validate():
        net.eval()
        with torch.inference_mode():
            index = torch.as_tensor(val_index, device=device)
            out = net(tensors["V"][index], tensors["M"][index])
            cls_pred = out["outcome"].argmax(-1)
            cls_true = tensors["cls"][index]
            cls_acc = (cls_pred == cls_true).float().mean().item()
            lethal_true = cls_true == CLASS_LETHAL
            lethal_recall = (
                (cls_pred == CLASS_LETHAL)[lethal_true].float().mean().item()
                if lethal_true.any() else 0.0)
            # 排序质量：只在有信号的状态上比较中性组的 argmax
            weight = tensors["weight"][index]
            mask = tensors["shape_m"][index]
            usable = (weight > 0) & (mask.sum(1) >= 2)
            rank_acc, rank_regret = 0.0, 1.0
            if usable.any():
                pred = torch.where(mask.bool(), out["shape"],
                                   torch.full_like(out["shape"], -1e9))
                true_full = tensors["shape"][index]
                true = torch.where(mask.bool(), true_full,
                                   torch.full_like(out["shape"], -1e9))
                chosen = pred.argmax(1)
                rank_acc = (chosen == true.argmax(1))[
                    usable].float().mean().item()
                # 组内 regret：目标已是逐状态 [0,1] 归一化，所以
                # 1.0 - 被选中动作的真值 = 损失掉的排序质量比例。
                # 0.0 = 选中最优，1.0 = 选中最差。这个数比 argmax
                # 命中率可解释：并列时选错几乎不产生 regret。
                got = true_full.gather(1, chosen[:, None]).squeeze(1)
                rank_regret = (1.0 - got)[usable].mean().item()
        net.train()
        return cls_acc, lethal_recall, rank_acc, rank_regret

    best = (-1.0, None)
    net.train()
    started = time.time()
    for epoch in range(args.epochs):
        order = np.random.permutation(train_index)
        for start in range(0, len(order), args.batch):
            index = torch.as_tensor(
                order[start:start + args.batch], device=device)
            optimiser.zero_grad()
            loss = batch_loss(index)
            loss.backward()
            optimiser.step()
        cls_acc, lethal_recall, rank_acc, rank_regret = validate()
        # 选点标准：路线排序准确率优先，这正是 P38 完全没学到的部分
        key = -rank_regret
        marker = ""
        if key > best[0]:
            best = (key, {k: v.detach().clone()
                          for k, v in net.state_dict().items()})
            marker = "  <= best"
        print(f"  epoch {epoch:2d}  类别 {cls_acc:.3f}  "
              f"致死召回 {lethal_recall:.3f}  路线排序 {rank_acc:.3f}  "
              f"组内regret {rank_regret:.4f}{marker}", flush=True)

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    import torch as _torch
    _torch.save({
        "version": "p39_decomposed_distill",
        "state_dict": best[1] if best[1] is not None else net.state_dict(),
        "vector_dim": (FIELD_INPUT_DIM if args.field_input
                       else FAST_VECTOR_DIM),
        "field_input": bool(args.field_input),
        "map_channels": FAST_MAP_CHANNELS,
        "live_actions": list(LIVE_ACTION_INDICES),
        "samples": n,
        "rank_regret": -best[0],
    }, args.model)
    print(f"===== 保存 {args.model}  组内regret {-best[0]:.4f}  "
          f"用时 {time.time() - started:.0f}s =====", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--shape-coef", type=float, default=3.0)
    parser.add_argument("--margin-coef", type=float, default=0.3)
    parser.add_argument("--field-coef", type=float, default=0.15)
    parser.add_argument("--preview-coef", type=float, default=0.10)
    parser.add_argument("--survival-coef", type=float, default=0.10)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=39_000_000)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--field-input", action="store_true",
                        help="把击杀场和移动预演作为输入而不是重建目标")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()


def make_p40_policy(model_path="training/models/p40_killfield_fieldin.pt",
                    cap_frames=500, rays=512, bounces=2, flight=75,
                    horizon=36, **_ignored):
    """P40 部署策略：运行时自己算击杀场和移动预演，拼进输入。

    与 P39 的区别只在观测——决策器、执行器（承诺/扳机/击杀后屏蔽）
    完全复用 KillFieldFastPolicy。
    """
    import numpy as _np
    import torch
    from training.killfield_fast_distill import KillFieldFastPolicy
    from training.killfield_full_distill import action_preview_features
    from training.killfield_student import KillFieldFeatureState
    from training.tt_gym_env import TankTroubleGym

    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if not payload.get("field_input"):
        raise ValueError(f"{model_path} 不是 field-input 权重")
    core = build_p39_network(field_input=True)
    core.load_state_dict(payload["state_dict"])
    core.eval()

    class P40Policy(KillFieldFastPolicy):
        name = "P40 击杀场作输入蒸馏网络"

        def __init__(self):
            self.torch = torch
            self.network = _build_p38_compatible(core)
            self.network.eval()
            self.encoder = TankTroubleGym(
                seed=0, obs_traj=True, obs_nav=True, terminal_mode="score")
            self.cap_frames = int(cap_frames)
            self.horizon = int(horizon)
            self.field_state = KillFieldFeatureState(rays, bounces, flight)
            self.reset()

        def reset(self):
            super().reset()
            if getattr(self, "field_state", None) is not None:
                self.field_state.reset()

        def _observation(self, game):
            vector, spatial = super()._observation(game)
            blank_f = _np.zeros(KILLFIELD_EXTRA_DIM, dtype=_np.float32)
            blank_p = _np.zeros(ACTION_PREVIEW_DIM, dtype=_np.float32)
            if game.tanks[1].alive:
                # 场以对手格为目标；对手死了就无定义（有效位=0）。
                self.field_state.ensure_field(game)
                self.field_state.advance(game)
                field_facts = self.field_state.features(game)
                preview = action_preview_features(
                    game, self.field_state.field, self.field_state.chain,
                    self.horizon)
                valid = 1.0
            else:
                field_facts, preview, valid = blank_f, blank_p, 0.0
            augmented = _np.concatenate(
                [vector, field_facts, preview,
                 _np.asarray([valid], dtype=_np.float32)]
            ).astype(_np.float32)
            return augmented, spatial

    return P40Policy()
