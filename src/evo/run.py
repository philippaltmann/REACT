import math
import os
import random
import time

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.vec_env import VecVideoRecorder

from config import CONFIG
from constants import MODEL_PATH, VIDEO_PATH
from train.models import model_resolver
import evo.utils as utils


def get_deterministic_prob(model, obs, deterministic):
    mean_actions, log_std, kwargs = model.policy.actor.get_action_dist_params(obs)
    model.policy.actor.action_dist.proba_distribution(mean_actions, log_std)
    actions = model.policy.actor.action_dist.actions_from_params(mean_actions, log_std, deterministic)
    log_prob = model.policy.actor.action_dist.log_prob(actions)
    prob = torch.tanh(log_prob)
    return actions, prob

def get_behaviour(model, env, render):
    if "Fetch" in CONFIG.env_name: observation = env.reset() #gymnasium video recorder
    else: observation = np.expand_dims(env.envs[0].reset(layout=env.envs[0].layout)[0], 0)

    done = False
    states = [utils.clean_observation(observation)]
    certainties = []
    actions = []

    acc_reward = 0
    while not done:
        action, state = model.predict(observation, deterministic=True)

        if isinstance(model, stable_baselines3.SAC):
            tensored_obs = model.policy.obs_to_tensor(observation)
            _, prob = get_deterministic_prob(model, tensored_obs[0], True)

        elif isinstance(model, stable_baselines3.PPO):
            tensored_obs = model.policy.obs_to_tensor(observation)
            _, log_prob, entropy_of_action_distribution = model.policy.evaluate_actions(tensored_obs[0], torch.tensor([action]))
            prob = torch.exp(log_prob)

        observation, reward, done, info = env.step(action)

        if isinstance(model, stable_baselines3.SAC) and render:
            time.sleep(0.1)

        acc_reward += reward[0]
        states.append(utils.clean_observation(observation))
        certainties.append(prob)
        actions.append(action)

    # the environment does not return the terminal state, but the reset state
    # we have to get the terminal state from info
    if "Grid" in CONFIG.env_name: # final_observation = last state
        terminal_obs = info[0].get("final_observation")
    else:
        terminal_obs = info[0].get("terminal_observation")
    if CONFIG.env_name == "FetchReach":
        terminal_obs = terminal_obs['achieved_goal'].tolist()
    if isinstance(terminal_obs, np.ndarray):
        terminal_obs = utils.clean_observation(np.array([terminal_obs]))
    ## in HoleyGrid the last state is not available, when a hole is reached
    #if terminal_obs != [None, None]:
    states[-1] = terminal_obs

    return states, acc_reward, certainties, actions


def run_hyphi_grid_individual(state, model_str, model_path, render, i):
    done, reward, layout = utils.convert_state_to_custom_map(state, CONFIG.env_name, CONFIG.env_seed)

    states = []
    certainties = []
    actions = []
    random_state = random.getstate()

    if not done:
        env = CONFIG.env
        env.layout = layout
        # print(layout)
        policy = "MlpPolicy"
        kwargs = {"policy": policy, "env": env}
        model = model_resolver.make(model_str, kwargs)
        if os.path.exists(model_path):
            model = model.load(model_path, env=env)

        vec_env = model.get_env()
        vec_env.set_options({"layout": layout})

        states, reward, certainties, actions = get_behaviour(model, vec_env, render=False)

        if render:
            path = VIDEO_PATH.joinpath(CONFIG.env_name + "/eval/"+ CONFIG.saved_model + "-" + str(CONFIG.checkpoint) + CONFIG.name + "/")
            if not os.path.exists(path):
                os.makedirs(path)
            path = str(path) +"/" + str(i) + ".gif"
            env.get_wrapper_attr('save_video')(path)
        # assert False

    random.seed(CONFIG.seed)
    random.setstate(random_state)
    return states, reward, certainties, actions


def run_fetch_reach_individual(state, model_str, model_path, render, i):
    random_state = random.getstate()

    start_difference = state[:3]
    goal_difference = state[3:]

    env = gym.make('CustomFetchReach-v1', max_episode_steps=50, render_mode='rgb_array' if render else None,
                       initial_startup_position=start_difference, goal_difference=goal_difference)

    policy = "MultiInputPolicy"
    kwargs = {"policy": policy, "env": env}
    model = model_resolver.make(model_str, kwargs)
    model = model.load(model_path, env=env, use_sde=False)
    vec_env = model.get_env()

    if render:
        path = VIDEO_PATH.joinpath(CONFIG.env_name + "/eval/")
        if not os.path.exists(path):
            os.makedirs(path)
        path = str(path) + "/" + CONFIG.saved_model + "-" + str(CONFIG.checkpoint) + CONFIG.name
        recording_env = VecVideoRecorder(vec_env, path, lambda step: step == 0, name_prefix="episode" + str(i), video_length=49)

        states, reward, certainties, actions = get_behaviour(model, recording_env, render=render)
        recording_env.close_video_recorder()
    else:
        states, reward, certainties, actions = get_behaviour(model, vec_env, render=render)

    env.close()

    random.seed(CONFIG.seed)
    random.setstate(random_state)

    return states, reward, certainties, actions


def run_individual(state, render, i=None):
    _, model_str = CONFIG.saved_model.split("_")
    if CONFIG.checkpoint == 0:
        model_path = MODEL_PATH.joinpath(CONFIG.env_name + "/" + CONFIG.saved_model + "/best_model")
    else:
        model_path = MODEL_PATH.joinpath(CONFIG.env_name + "/" + CONFIG.saved_model + "/rl_model_" + str(CONFIG.checkpoint) + "_steps.zip")

    if "Grid" in CONFIG.env_name:
        return run_hyphi_grid_individual(state, model_str, model_path, render, i)
    elif CONFIG.env_name == "FetchReach":
        return run_fetch_reach_individual(state, model_str, model_path, render, i)
    else:
        raise ValueError()
