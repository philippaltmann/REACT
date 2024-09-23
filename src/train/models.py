__all___ = [
    "A2C",
    "PPO",
    "DQN",
    "DDPG",
    "SAC"
]

import stable_baselines3
from class_resolver import ClassResolver

model_resolver = ClassResolver(
    classes={
        stable_baselines3.A2C,
        stable_baselines3.DQN,
        stable_baselines3.PPO,
        stable_baselines3.DDPG,
        stable_baselines3.SAC
    },
    base=stable_baselines3.common.base_class.BaseAlgorithm,
    default=stable_baselines3.PPO
)