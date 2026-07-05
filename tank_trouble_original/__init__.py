"""
Tank Trouble 原版 1:1 复刻 (Python)

移植自反编译的 Flash 源码 (swf_decompiled/scripts/), 逻辑、常量、
帧序与原版一致。核心无第三方依赖, 可无头高速运行, 适合 AI 训练。

  from tank_trouble_original import Game, TankTroubleEnv
"""

from .game import Game, Tank, Bullet, norm_rot
from .laika import LaikaAI
from .env import TankTroubleEnv, TankTroubleGymEnv, discrete_to_input
from . import constants

__all__ = [
    "Game", "Tank", "Bullet", "LaikaAI", "norm_rot",
    "TankTroubleEnv", "TankTroubleGymEnv", "discrete_to_input",
    "constants",
]
