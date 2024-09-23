import random
import click
import gymnasium as gym
import hyphi_gym
import numpy as np
from gymnasium.wrappers import AutoResetWrapper
from hyphi_gym import Monitor

from baselines.logger import BaselineLogger
from config import CONFIG
from constants import RANDOM_SEARCH_LOG_PATH
from evo import utils
from evo.run import run_individual
import evo.metrics as metrics
from evo.utils import remove_duplicate_states


def generate_random_population(pop_size: int, id: int, logger: BaselineLogger, iteration: int, exp_name, plot):
    population = []

    bitstring_length = CONFIG.dimensions * CONFIG.state_encoding_length

    for _ in range(pop_size):
        state_encoding = f'{random.getrandbits(bitstring_length):=0{bitstring_length}b}'
        population.append(state_encoding)

    state_matrix = np.zeros((CONFIG.map_size, CONFIG.map_size), dtype=np.int8)
    heatmap = np.zeros((CONFIG.map_size, CONFIG.map_size), dtype=np.int8)

    for state_encoding in population:
        state = utils.get_state(state_encoding, CONFIG.dimensions, CONFIG.is_discrete, CONFIG.min_state,
                                CONFIG.max_state)
        state_sequence, reward, _, _ = run_individual(state, False)

        if CONFIG.env_name != "FetchReach":
            state_sequence_without_duplicates = remove_duplicate_states(state_sequence)
            for s in state_sequence_without_duplicates:
                state_matrix[s[0]][s[1]] = 1
                heatmap[s[0]][s[1]] += 1

        trajectory_length = utils.get_traj_length(CONFIG.env_name, state_sequence)
        if state_sequence != 0:
            episode_length = len(state_sequence)  # - 1
        else:
            episode_length = 0
        logger.log_baseline(iteration, id, state, reward, trajectory_length, episode_length)

        id += 1
    if plot and CONFIG.env_name != "FetchReach":
        metrics.plot_heatmap(heatmap, exp_name, iteration, len(population), "Reds")
        metrics.plot_3d_histogram(heatmap, exp_name, iteration, len(population), "Reds")
    return id


@click.command("baseline1")
@click.option("--env-name")
@click.option("--saved-model")
@click.option("--pop-size", default=4)
@click.option("--iterations", default=5)
@click.option("--name", default="test")
@click.option("--checkpoint", default=0, type=int)
@click.option("--encoding-length", default=8, type=int)
@click.option("--env-seed", default=33, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--plot", is_flag = True)
def random_search(env_name: str, saved_model: str, pop_size: int, iterations: int, name: str,
                  checkpoint: int, encoding_length: int, env_seed: int, seed: int, plot):
    env = None
    id = [42,13,24,18,46,19,28,32,91,12]
    name = f"{env_name}-{id.index(seed)}-{seed}"
    if "Grid" in env_name:  # if is_gridworld

        env_kwargs = hyphi_gym.named(env_name)
        if "Flat" in env_name:
            env = AutoResetWrapper(
                Monitor(gym.make(env_kwargs["id"], size=env_kwargs["size"], sparse=env_kwargs["sparse"],
                                 detailed=env_kwargs["detailed"], explore=env_kwargs["explore"],
                                 random=env_kwargs["random"],
                                 render_mode="3D"), record_video=False))
        else:
            env = AutoResetWrapper(
                Monitor(gym.make(env_kwargs["id"], level=env_kwargs["level"], sparse=env_kwargs["sparse"],
                                 detailed=env_kwargs["detailed"], explore=env_kwargs["explore"],
                                 random=env_kwargs["random"],
                                 render_mode="3D", seed=env_seed), record_video=False))

        dimensions = 2
        is_discrete_env = True
        min_state = 0
        try:  # -2 because we do not want to consider outside walls
            map_size = int(env_name[-2:]) - 2
        except:
            try:
                map_size = int(env_name[-1]) - 2
            except:
                raise ValueError()
        max_state = map_size - 1
    else:  # is continuous env aka FetchReach
        dimensions = 6  # for now, 3 for start, 3 for goal
        is_discrete_env = False
        min_state = -0.15
        max_state = 0.15
        map_size = 0 # there is no map

    # register custom gym environment
    if env_name == "FetchReach":
        gym.register(id='CustomFetchReach-v1', entry_point='custom_env.custom_env:CustomFetchReachEnv')

    logger = BaselineLogger(save_path=RANDOM_SEARCH_LOG_PATH.joinpath(env_name), experiment_name=name)

    # can also be used here
    CONFIG.set_baseline_config(
        env=env,
        env_name=env_name,
        saved_model=saved_model,
        map_size=map_size,
        name=name,
        checkpoint=checkpoint,
        env_seed=env_seed,
        seed=seed,
        state_encoding_length=encoding_length,
        dimensions=dimensions,
        min_state=min_state,
        max_state=max_state,
        is_discrete_env=is_discrete_env
    )
    random.seed(seed)
    id = 1
    for i in range(iterations):
        id = generate_random_population(pop_size, id, logger, i, name, plot)

    print(logger.data)
    logger.close()
