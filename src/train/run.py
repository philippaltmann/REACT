import os

import click
import gymnasium as gym
import hyphi_gym
from gymnasium.wrappers import AutoResetWrapper
from hyphi_gym import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder

from constants import MODEL_PATH, VIDEO_PATH
from train.models import model_resolver


def run_trained_model(model, env):
    observation = env.reset()
    done = False
    acc_rew = 0
    while not done:
        action, state = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        acc_rew += reward
        env.render()
    print("final reward:", acc_rew)


# click command to view trained model
@click.command("run")
@click.option("--env-name")
@click.option("--saved-model")
@click.option("--nr-episodes", type=int, default=5)
@click.option("--checkpoint", type=int, default=0)
@click.option("--seed", default=42)
def run(env_name: str, saved_model: str, nr_episodes: int, checkpoint: int, seed: int):
    if "FlatGrid" in env_name:
        env_kwargs = hyphi_gym.named(env_name)
        env = AutoResetWrapper(Monitor(gym.make(env_kwargs["id"], size=env_kwargs["size"], sparse=env_kwargs["sparse"],
                       detailed=env_kwargs["detailed"], explore=env_kwargs["explore"], random=env_kwargs["random"], render_mode="blender"), record_video=True))
        policy = "MlpPolicy"
        nr_episodes = 1

    elif "HoleyGrid" in env_name:
        env_kwargs = hyphi_gym.named(env_name)
        env = AutoResetWrapper(Monitor(gym.make(env_kwargs["id"], level=env_kwargs["level"], sparse=env_kwargs["sparse"],
                       detailed=env_kwargs["detailed"], explore=env_kwargs["explore"], random=env_kwargs["random"], render_mode="blender", seed=seed), record_video=True))
        policy = "MlpPolicy"
        nr_episodes = 1

    elif env_name == "FetchReach":
        gym.register(id='CustomFetchReach-v1', entry_point='custom_env.custom_env:CustomFetchReachEnv')
        env = gym.make('CustomFetchReach-v1', max_episode_steps=50, render_mode='rgb_array')
        policy = "MultiInputPolicy"

    else:
        raise ValueError("environment can only be FlatGrid, HoleyGrid or FetchReach")

    _, model_str = saved_model.split("_")
    kwargs = {"policy": policy, "env": env, "seed": seed}
    model = model_resolver.make(model_str, kwargs)
    if checkpoint == 0:
        model = model.load(MODEL_PATH.joinpath(env_name + "/" + saved_model + "/best_model"), env=env)
    else:
        path = MODEL_PATH.joinpath(env_name + "/" + saved_model + "/rl_model_" + str(checkpoint) + "_steps.zip")
        if os.path.exists(path):
            model = model.load(path, env=env)
        else:
            raise ValueError(f"No model at {path}")
    vec_env = model.get_env()

    if env_name == "FetchReach":
        path = VIDEO_PATH.joinpath(env_name + "/train/")
        vec_env = VecVideoRecorder(vec_env, path, False, name_prefix=saved_model + str(checkpoint))

    for _ in range(nr_episodes):
        run_trained_model(model, vec_env)

    if "Grid" in env_name:
        try:
            path = VIDEO_PATH.joinpath(env_name + "/train/")
            if not os.path.exists(path):
                os.makedirs(path)
            path = str(path) + "/" + saved_model + "-" + str(checkpoint) + "-" + str(seed) + ".gif"
            env.get_wrapper_attr('save_video')(path)
        except:
            pass
    elif env_name == "FetchReach":
        vec_env.close_video_recorder()
    env.close()
