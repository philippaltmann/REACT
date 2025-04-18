import os
import random
import time

import numpy as np
import stable_baselines3
import torch

from config import CONFIG
from constants import MODEL_PATH, VIDEO_PATH
# from train.models import model_resolver
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
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
    if "Grid" in CONFIG.env_name: observation = np.expand_dims(env.envs[0].reset()[0], 0)

    states, certainties, actions, done = [utils.clean_observation(observation)], [], [], False

    acc_reward = 0
    while not done:
        action, _ = model.predict(observation, deterministic=True)

        if isinstance(model, stable_baselines3.SAC):
            tensored_obs = model.policy.obs_to_tensor(observation)
            _, prob = get_deterministic_prob(model, tensored_obs[0], True)

        elif isinstance(model, stable_baselines3.PPO):
            tensored_obs = model.policy.obs_to_tensor(observation)
            _, log_prob, _ = model.policy.evaluate_actions(tensored_obs[0], torch.tensor(np.array([action])))
            prob = torch.exp(log_prob)

        observation, reward, done, info = env.step(action)
        if isinstance(model, stable_baselines3.SAC) and render: time.sleep(0.1)

        acc_reward += reward[0]
        states.append(utils.clean_observation(observation))
        certainties.append(prob)
        actions.append(action)
    # the environment does not return the terminal state, but the reset state
    # we have to get the terminal state from info
    terminal_obs = utils.clean_observation(np.array([info[0].get("terminal_observation")]))    
    # in HoleyGrid the last state is not available, when a hole is reached
    if 'HoleyGrid' in CONFIG.env_name and terminal_obs == [None, None]: states = states[:-1]
    else: states[-1] = terminal_obs
    
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
        model:BaseAlgorithm = eval(model_str)(policy="MlpPolicy", env=env)
        if os.path.exists(model_path): model = model.load(model_path, env=env)
        else: assert False, "Model not found"

        vec_env = model.get_env(); vec_env.envs[0].unwrapped.layout = layout
        states, reward, certainties, actions = get_behaviour(model, vec_env, render=False)

        if render:
            path = VIDEO_PATH.joinpath(CONFIG.env_name + "/eval/"+ CONFIG.saved_model + "-" + str(CONFIG.checkpoint) + CONFIG.exp_name + "/")
            if not os.path.exists(path): os.makedirs(path)
            path = str(path) +"/" + str(i) + ".gif"
            env.get_wrapper_attr('save_video')(path)

    random.seed(CONFIG.seed)
    random.setstate(random_state)
    return states, reward, certainties, actions


def run_individual(state, render, i=None):
    _, model_str = CONFIG.saved_model.split("_")
    model_path = 'best_model' if CONFIG.checkpoint == 0 else 'rl_model_' + str(CONFIG.checkpoint) + '_steps'
    model_path = MODEL_PATH.joinpath(f'{CONFIG.env_name}/{CONFIG.saved_model}/{model_path}.zip')

    random_state = random.getstate(); env = CONFIG.env
    model:BaseAlgorithm = eval(model_str)(policy="MlpPolicy", env=env)
    if os.path.exists(model_path): model = model.load(model_path, env=env)
    else: assert False, f"Model not found at {model_path}"
    vec_env = model.get_env()

    # Setup Layout 
    if 'Grid' in CONFIG.env_name:
        done, reward, layout = utils.convert_state_to_custom_map(state, CONFIG.env_name, CONFIG.env_seed)
        if done: random.seed(CONFIG.seed); random.setstate(random_state); return [], reward, [], []
        vec_env.envs[0].unwrapped.layout = layout

    elif 'Fetch' in CONFIG.env_name:
        vec_env.envs[0].unwrapped.agent, vec_env.envs[0].unwrapped.position_noise = np.array(state[:3]), 0
        vec_env.envs[0].unwrapped.target, vec_env.envs[0].unwrapped.target_noise = np.array(state[3:]), 0

    else: assert False, f'{CONFIG.env_name} not supported'

    states, reward, certainties, actions = get_behaviour(model, vec_env, render=False)

    if render:
        path = VIDEO_PATH.joinpath(f'{CONFIG.env_name}/eval/{CONFIG.saved_model}-{CONFIG.checkpoint}{CONFIG.exp_name}/')
        if not os.path.exists(path): os.makedirs(path)
        env.get_wrapper_attr('save_video')(f'{path}/{i}.gif')

    random.seed(CONFIG.seed); random.setstate(random_state)
    return states, reward, certainties, actions
