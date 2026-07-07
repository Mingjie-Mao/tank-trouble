"""
P18 — CNN 地图头特征提取器

观测为 Dict{"vec": 向量特征, "map": 迷宫栅格}:
  map 形状 (4, 10, 12) float32, 通道:
    0 = 该格下边有墙  1 = 该格左边有墙  (外圈 padding 置 1 = 实心)
    2 = 自车所在格    3 = 敌车所在格
  小 CNN 读地图出 96 维, 与 vec 拼接后进 MLP 主干。
  (迷宫最大 12x10 格, NatureCNN 要求 36x36 起, 故必须自定义)
"""

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

MAP_OUT = 96


class TankMapExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        vec_dim = observation_space["vec"].shape[0]
        c, h, w = observation_space["map"].shape
        super().__init__(observation_space,
                         features_dim=vec_dim + MAP_OUT)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * h * w, MAP_OUT),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        vec = observations["vec"]
        map_feat = self.cnn(observations["map"])
        return torch.cat([vec, map_feat], dim=1)
