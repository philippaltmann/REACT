import os; import math; import click
import numpy as np; import pandas as pd
import gymnasium as gym; import hyphi_gym
from gymnasium.wrappers import Autoreset
from hyphi_gym import Monitor
from config import CONFIG

import evo.utils as utils
from constants import MODEL_PATH, VIDEO_PATH, LOG_PATH
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm


def run_trained_model(model, env):
    env_name = env.envs[0].unwrapped.spec.id
    observation = env.reset()
    states = [utils.clean_observation(observation)]
    done = False; acc_rew = 0
    while not done:
        action, state = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        states.append(utils.clean_observation(observation))
        acc_rew += reward #; env.render()
    # the environment does not return the terminal state, but the reset state we have to get the terminal state from info
    terminal_obs = utils.clean_observation(np.expand_dims(info[0].get("terminal_observation"), axis=0))

    # in HoleyGrid the last state is not available, when a hole is reached
    if "Grid" in env_name and terminal_obs == [None, None]: states = states[:-1]
    else: states[-1] = terminal_obs

    reward = info[0].get("episode")['r']; traj_length = 0
    if "Fetch" in env_name:
        traj_length = sum([math.dist(a,b) for a,b in zip(states[:-1], states[1:])])
    else:
        state_sequence_without_duplicates = utils.remove_duplicate_states(states)
        traj_length = max(len(state_sequence_without_duplicates) - 1, 0)

    return reward, traj_length
  

# click command to view trained model
@click.command("run")
@click.option("--env-name")
@click.option("--saved-model")
@click.option("--nr-episodes", type=int, default=1)
@click.option("--checkpoint", type=int, default=0)
@click.option("--name", default='')
@click.option("--seed", default=42)
@click.option("--render", is_flag=True)
def run_model(env_name: str, saved_model: str, nr_episodes: int, checkpoint: int, name: str, seed: int, render: bool):
    render_mode = "blender" if "Grid" in env_name else "3D"
    checkpoint = f"rl_model_{checkpoint}_steps" if checkpoint > 0 else "best_model"
    model_path = MODEL_PATH.joinpath(f'{env_name}/{saved_model}/{checkpoint}.zip')
    if render: video_path = VIDEO_PATH.joinpath(f'{env_name}/train/'); os.makedirs(video_path, exist_ok=True)
    log_path = LOG_PATH.joinpath(f"Train/{env_name}/"); os.makedirs(log_path, exist_ok=True)
    base_file = f'{env_name if len(name) == 0 else name}-{seed}'
    
    if env_name == "FetchReach": env_name += 'Agents' # Train on random targets / evaluate on random initial pos
    env = Autoreset(Monitor(gym.make(**hyphi_gym.named(env_name), seed=33, render_mode=render_mode), record_video=render))
    policy = "MlpPolicy"

    CONFIG.set_eval_config(
      env_name=env_name, env_seed=seed, map_size=None, 
      saved_model=saved_model, checkpoint=checkpoint, 
      exp_name=name, env=env,  
    )

    _, model_str = saved_model.split("_")
    model:BaseAlgorithm = eval(model_str)(policy=policy, env=env, seed=seed).load(model_path, env=env)

    R, L = (np.array(a) for a in zip(*[run_trained_model(model, model.get_env()) for _ in range(nr_episodes)]))
    F = L / L.sum() * abs(R.mean() - R) 

    pd.DataFrame({"iteration": [0], "reward": R, "trajectory_length": L, "fidelity": F}).to_csv(f'{log_path}/{base_file}.csv')

    if render: env.get_wrapper_attr('save_video')(f'{video_path}/{base_file}.gif')
    env.close()
