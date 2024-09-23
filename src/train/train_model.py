import random

import click
import gymnasium as gym
import hyphi_gym
import stable_baselines3.common.base_class
from class_resolver import Hint
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from constants import TRAIN_LOG_PATH, MODEL_PATH
from train.models import model_resolver


@click.command("train")
@click.option("--env-name")
@click.option("--name", default="train")
@model_resolver.get_option("--model", as_string=True)
@click.option("--render", is_flag=True)
@click.option("--steps", default=10000)
@click.option("--save-freq", default=5000)
@click.option("--env-seed", default=42)
def train(env_name: str, name: str, model: Hint[stable_baselines3.common.base_class.BaseAlgorithm], render: bool,
          steps: int, save_freq: int, env_seed: int):
    name = name + "_" + model

    if "FlatGrid" in env_name:
        env = gym.make(**hyphi_gym.named(env_name), render_mode="3D")
        policy = "MlpPolicy"

    elif "HoleyGrid" in env_name:
        env_kwargs = hyphi_gym.named(env_name)
        env = gym.make(env_kwargs["id"], level=env_kwargs["level"], sparse=env_kwargs["sparse"],
                       detailed=env_kwargs["detailed"], explore=env_kwargs["explore"], random=env_kwargs["random"],
                       seed=env_seed)
        policy = "MlpPolicy"

    elif env_name == "FetchReach":

        if render:
            env = gym.make('FetchReach-v2', max_episode_steps=50, render_mode="human")
        else:
            env = gym.make('FetchReach-v2', max_episode_steps=50)

        policy = "MultiInputPolicy"
    else:
        raise ValueError("environment can only be FlatGrid, HoleyGrid or FetchReach")

    eval_env = hyphi_gym.Monitor(env, record_video=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_PATH.joinpath(env_name + "/" + name),
                                log_path=TRAIN_LOG_PATH.joinpath(env_name + "/" + name), eval_freq=save_freq,
                                deterministic=True, render=render, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=MODEL_PATH.joinpath(env_name + "/" + name))



    # seed always creates the same holes
    kwargs = {"policy": policy, "env": env, "seed": env_seed}
    model = model_resolver.make(model, kwargs)
    new_logger = configure(str(TRAIN_LOG_PATH.joinpath(env_name + "/" + name)), ["csv"])
    model.set_logger(new_logger)

    random.seed() # Such that the seed from the environment is not used!

    model.learn(total_timesteps=steps, callback=[checkpoint_callback, eval_callback])

    env.close()
