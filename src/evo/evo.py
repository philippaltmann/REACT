import click
import random
from operator import attrgetter
from typing import List, Tuple
import gymnasium as gym
import hyphi_gym
from gymnasium.wrappers import AutoResetWrapper
from hyphi_gym import Monitor

from config import CONFIG
from constants import EVO_LOG_PATH
from evo.crossover import cross_over
from evo.evo_logger import EvoLogger
from evo.individual import Individual
from evo.mutation import mutate
from evo.utils import multisort, get_max_owd
import evo.metrics as metrics


def generate_initial_population(pop_size: int, id: int, weights: List[float]) -> (
        Tuple)[List[Individual], int]:
    population = []

    bitstring_length = CONFIG.dimensions * CONFIG.state_encoding_length
    for _ in range(pop_size):
        state_encoding = f'{random.getrandbits(bitstring_length):=0{bitstring_length}b}'
        population.append(Individual(state_encoding, id, weights))
        id += 1

    return population, id


def reduce_population(pop_size: int, population: List[Individual]):
    if len(population) > pop_size:
        # print("reducing population: ", population[-1].state, " with fitness ", population[-1].fitness)
        population.pop(-1)
        reduce_population(pop_size, population)
    else:
        return


def genetic_algorithm(pop_size: int, max_iter: int, crossover_prob: float, mutation_prob: float, env_name: str,
                      logger: EvoLogger, render: bool, is_elitist: bool, weights: List[float],
                      exp_name: str, plot_frequency: int) -> List[Individual]:
    # generate population
    population, next_id = generate_initial_population(pop_size, 0, weights)

    iteration = 0

    # compute fitness of each individual in the first generation
    states_of_previously_evaluated_individuals, local_diversities, certainties = [], [], []
    for individual in population:
        individual.compute_fitness(render, states_of_previously_evaluated_individuals, local_diversities, certainties)
        if individual.state_sequence:  # avoid owd error because len(traj) could be 0
            states_of_previously_evaluated_individuals.append(individual.state_sequence_without_duplicates)
            local_diversities.append(individual.local_diversity_measure)
            certainties.append(individual.certainty_measure)

    # recompute first individual to change default values
    states_of_previously_evaluated_individuals.pop(0)
    local_diversities.pop(0)
    certainties.pop(0)
    # recompute the first (valid) individual, it is valid if it has a state sequence
    for individual in population:
        if individual.state_sequence:
            individual.compute_fitness(render, states_of_previously_evaluated_individuals, local_diversities,
                                       certainties, True)
            break

    population.sort(key=attrgetter("fitness"), reverse=True)

    # Start logging
    for i in population:
        # compute metric
        traj_length = i.get_traj_length()
        if i.state_sequence != 0:
            episode_length = len(i.state_sequence) - 1
        else:
            episode_length = 0
        logger.log(iteration, i.id, i.state,
                   i.local_diversity_measure, i.global_diversity_measure, i.certainty_measure,
                   i.fitness,
                   i.dist_local_div, i.dist_certainty, i.min_dist_of_other_measures,
                   i.reward, traj_length, episode_length)
    iteration += 1

    while iteration <= max_iter:
        print("Iteration: ", iteration)
        population, next_id = cross_over(population, next_id, crossover_prob, weights)
        population, next_id = mutate(population, next_id, mutation_prob, is_elitist, weights)

        # compute fitness of new individuals
        states_of_population, local_diversities, certainties = [], [], []
        # add all evaluated individuals to the lists, so they can be used in the evaluation of future individuals
        for i in range(len(population)):
            if not population[i].fitness is None:
                if population[i].state_sequence:  # avoid owd error because len(traj) could be 0
                    states_of_population.append(population[i].state_sequence_without_duplicates)
                    local_diversities.append(population[i].local_diversity_measure)
                    certainties.append(population[i].certainty_measure)

        for i in range(len(population)):
            if population[i].fitness is None:

                population[i].compute_fitness(render, states_of_population, local_diversities,
                                              certainties)
                if population[i].state_sequence:  # avoid owd error because len(traj) could be 0
                    states_of_population.append(population[i].state_sequence_without_duplicates)
                    local_diversities.append(population[i].local_diversity_measure)
                    certainties.append(population[i].certainty_measure)

        population.sort(key=attrgetter("fitness"), reverse=True)
        reduce_population(pop_size, population)

        # Logging
        for i in population:
            # compute metric
            traj_length = i.get_traj_length()
            if i.state_sequence != 0:
                episode_length = len(i.state_sequence) - 1
            else:
                episode_length = 0
            logger.log(iteration, i.id, i.state,
                       i.local_diversity_measure, i.global_diversity_measure, i.certainty_measure,
                       i.fitness,
                       i.dist_local_div, i.dist_certainty, i.min_dist_of_other_measures,
                       i.reward, traj_length, episode_length)

        # plot metrics
        if iteration % plot_frequency == 0 or iteration == 1:
            if env_name == "FetchReach":
                metrics.plot_3d_trajectories(population, exp_name, iteration)
            else:
                metrics.compute_coverage(population, exp_name, iteration)
        iteration += 1

    return population


@click.command("evo")
@click.option("--env-name")
@click.option("--saved-model")
@click.option("--render", is_flag=True)
@click.option("--pop-size", default=4)
@click.option("--iterations", default=5)
@click.option("--crossover", default=0.9)
@click.option("--mutation", default=0.4)
@click.option("--name", default="test")
@click.option("--is-elitist", is_flag=True)
@click.option("--plot-frequency", default=10, type=int)
@click.option("--checkpoint", default=0, type=int)
@click.option("--encoding-length", default=8, type=int)
@click.option("--env-seed", default=33, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--w1", default=1.0)
@click.option("--w2", default=1.0)
@click.option("--w3", default=1.0)
@click.option("--w4", default=1.0)
def evo_run(env_name: str, saved_model: str, render: bool, pop_size: int, iterations: int,
            crossover: float, mutation: float, name: str, is_elitist: bool, plot_frequency: int,
            checkpoint: int, encoding_length: int, env_seed: int, seed: int, w1: float, w2: float, w3: float, w4: float):
    env = None
    if "Grid" in env_name:  # if is_gridworld
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
        max_owd = get_max_owd(map_size)
        if "Grid" in env_name:
            env_kwargs = hyphi_gym.named(env_name)
            if "Flat" in env_name:
                env = AutoResetWrapper(
                    Monitor(gym.make(env_kwargs["id"], size=env_kwargs["size"], sparse=env_kwargs["sparse"],
                                     detailed=env_kwargs["detailed"], explore=env_kwargs["explore"],
                                     random=env_kwargs["random"],
                                     render_mode="3D"), record_video=render))
            else:
                env = AutoResetWrapper(
                    Monitor(gym.make(env_kwargs["id"], level=env_kwargs["level"], sparse=env_kwargs["sparse"],
                                     detailed=env_kwargs["detailed"], explore=env_kwargs["explore"],
                                     random=env_kwargs["random"],
                                     render_mode="3D", seed=env_seed), record_video=render))
    else:  # is continuous env aka FetchReach
        dimensions = 6  # for now, 3 for start, 3 for goal
        is_discrete_env = False
        min_state = -0.15
        max_state = 0.15
        max_owd = 0.5196 # two_way_distance([0.15,0.15,0.15], [-0.15,-0.15,-0.15])
        map_size = 0 # there is no map

    # seed for the algorithm has to be initialized after initializing the environment HoleyGrid, otherwise it is going
    # to be reset
    random.seed(seed)

    # set global configuration
    CONFIG.set_evo_config(
        env=env,
        env_name=env_name,
        saved_model=saved_model,
        map_size=map_size,
        pop_size=pop_size,
        name=name,
        checkpoint=checkpoint,
        dimensions=dimensions,
        is_discrete_env=is_discrete_env,
        min_state=min_state,
        max_state=max_state,
        state_encoding_length=encoding_length,
        env_seed=env_seed,
        max_owd=max_owd,
        seed=seed
    )

    # register custom gym environment
    if env_name == "FetchReach":
        gym.register(id='CustomFetchReach-v1', entry_point='custom_env.custom_env:CustomFetchReachEnv')

    logger = EvoLogger(save_path=EVO_LOG_PATH.joinpath(env_name), experiment_name=name)

    genetic_algorithm(
        pop_size=pop_size,
        max_iter=iterations,
        crossover_prob=crossover,
        mutation_prob=mutation,
        env_name=env_name,
        logger=logger,
        render=render,
        is_elitist=is_elitist,
        weights=[w1, w2, w3, w4],
        exp_name=name,
        plot_frequency=plot_frequency
    )
    print(logger.data)
    logger.close()
