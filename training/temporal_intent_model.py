"""Lightweight temporal movement residual for the frozen P26/P27 policy.

The model deliberately does not own firing.  It learns movement continuity
from frozen policy outputs while exact search remains responsible for firing
and long-tail safety decisions.
"""

import numpy as np

from training.map_topology_planner import TOPOLOGY_FEATURE_DIM


MOVEMENT_COUNT = 9
ACTION_COUNT = 18
AUX_DIM = 6
HOLD_BINS = (1, 2, 4, 6, 8, 12, 16, 24)
POLICY_FEATURE_DIM = (
    ACTION_COUNT + ACTION_COUNT * AUX_DIM + MOVEMENT_COUNT
    + MOVEMENT_COUNT + 1
)
TEMPORAL_FEATURE_DIM = POLICY_FEATURE_DIM + TOPOLOGY_FEATURE_DIM
STATE_FEATURE_DIM = 440
TEMPORAL_STATE_FEATURE_DIM = TEMPORAL_FEATURE_DIM + STATE_FEATURE_DIM


def _sigmoid(values):
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def build_temporal_features(score, aux_logits, fire_logits, last_movement,
                            frames_since_change, topology_features=None,
                            state_features=None):
    """Build bounded features from a frozen policy forward pass."""
    score = np.asarray(score, dtype=np.float32).reshape(ACTION_COUNT)
    aux_logits = np.asarray(aux_logits, dtype=np.float32).reshape(
        ACTION_COUNT, AUX_DIM)
    fire_logits = np.asarray(fire_logits, dtype=np.float32).reshape(
        MOVEMENT_COUNT)

    centered_score = score - np.max(score)
    # P26 checkpoints emit scores near [-1, 0]. Dividing by the historical
    # SCORE_SCALE=100 would erase almost all movement-ranking information.
    centered_score = np.clip(centered_score, -5.0, 0.0)
    last_one_hot = np.zeros(MOVEMENT_COUNT, dtype=np.float32)
    if last_movement is not None:
        last_one_hot[int(last_movement)] = 1.0
    duration = np.asarray([
        min(max(float(frames_since_change), 0.0), 60.0) / 60.0
    ], dtype=np.float32)
    if topology_features is None:
        topology_features = np.zeros(TOPOLOGY_FEATURE_DIM, dtype=np.float32)
    topology_features = np.asarray(
        topology_features, dtype=np.float32).reshape(TOPOLOGY_FEATURE_DIM)
    parts = [
        centered_score,
        _sigmoid(aux_logits).reshape(-1),
        _sigmoid(fire_logits),
        last_one_hot,
        duration,
        topology_features,
    ]
    if state_features is not None:
        parts.append(np.asarray(
            state_features, dtype=np.float32).reshape(STATE_FEATURE_DIM))
    return np.concatenate(parts).astype(np.float32, copy=False)


def hold_bin_index(remaining_frames):
    remaining_frames = max(1, int(remaining_frames))
    return int(np.argmin(np.abs(
        np.asarray(HOLD_BINS, dtype=np.int32) - remaining_frames)))


def movement_run_targets(movements):
    """Return remaining run length and hold-bin labels for each frame."""
    movements = np.asarray(movements, dtype=np.int64)
    remaining = np.ones(len(movements), dtype=np.int64)
    run_end = len(movements)
    for index in range(len(movements) - 1, -1, -1):
        if index == len(movements) - 1 or movements[index] != movements[index + 1]:
            run_end = index + 1
        remaining[index] = run_end - index
    bins = np.asarray([hold_bin_index(value) for value in remaining],
                      dtype=np.int64)
    return remaining, bins


def build_temporal_intent_net(feature_dim=TEMPORAL_FEATURE_DIM,
                              hidden_dim=192, layers=1):
    import torch.nn as nn

    class _TemporalIntentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gru = nn.GRU(
                hidden_dim, hidden_dim, num_layers=layers,
                batch_first=True)
            self.movement_delta = nn.Linear(hidden_dim, MOVEMENT_COUNT)
            self.hold = nn.Linear(hidden_dim, len(HOLD_BINS))
            self.interrupt = nn.Linear(hidden_dim, 1)
            self.progress = nn.Linear(hidden_dim, 1)
            self.search_needed = nn.Linear(hidden_dim, 1)

        def forward(self, features, hidden=None):
            encoded = self.encoder(features)
            temporal, hidden = self.gru(encoded, hidden)
            return {
                "movement_delta": self.movement_delta(temporal),
                "hold": self.hold(temporal),
                "interrupt": self.interrupt(temporal).squeeze(-1),
                "progress": self.progress(temporal).squeeze(-1),
                "search_needed": self.search_needed(temporal).squeeze(-1),
                "hidden": hidden,
            }

    return _TemporalIntentNet()


class TemporalIntentRuntime:
    """Stateful one-frame inference wrapper for a trained intent network."""

    def __init__(self, path):
        import torch

        payload = torch.load(path, weights_only=False)
        self.torch = torch
        self.feature_dim = int(payload.get(
            "feature_dim", TEMPORAL_FEATURE_DIM))
        self.model = build_temporal_intent_net(
            feature_dim=self.feature_dim,
            hidden_dim=int(payload.get("hidden_dim", 192)),
            layers=int(payload.get("layers", 1)),
        )
        # V1 checkpoints predate the search-needed head. They remain valid for
        # movement inference; the untrained head is ignored unless explicitly
        # enabled by the sparse policy.
        self.model.load_state_dict(payload["state_dict"], strict=False)
        self.model.eval()
        self.hidden = None

    def reset(self):
        self.hidden = None

    def predict(self, features):
        tensor = self.torch.as_tensor(
            np.asarray(features, dtype=np.float32)).reshape(1, 1, -1)
        with self.torch.no_grad():
            output = self.model(tensor, self.hidden)
        self.hidden = output["hidden"].detach()
        movement = self.torch.softmax(
            output["movement_delta"][0, 0], dim=0).numpy()
        hold = self.torch.softmax(output["hold"][0, 0], dim=0).numpy()
        return {
            "movement_prob": movement,
            "hold_prob": hold,
            "interrupt_prob": float(self.torch.sigmoid(
                output["interrupt"][0, 0])),
            "progress": float(output["progress"][0, 0]),
            "search_needed_prob": float(self.torch.sigmoid(
                output["search_needed"][0, 0])),
        }
