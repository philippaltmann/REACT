import os
import click
import random
from operator import attrgetter
from typing import List, Tuple
import gymnasium as gym
import hyphi_gym
from hyphi_gym import Monitor
import numpy as np

from config import CONFIG
from constants import EVO_LOG_PATH, LOG_PATH
from evo.crossover import cross_over
from evo.evo_logger import EvoLogger
from evo.individual import Individual
from evo.mutation import mutate
from evo.utils import multisort, get_max_owd
from evo.one_way_distance import two_way_distance
from plot.plot import plot_movement, ALGS, plot_heatmap, run_trajectory


def vizualize_population(population: List[Individual], path: str, render: bool, iteration: int):
    initial_states = [[str(i.state)] for i in population]
    algorithms, colors, _ = ALGS(path.split("/")[-3]); os.makedirs(path, exist_ok=True);
    save_path = f"{path}/Trajectory_{iteration}"
    if 'Grid' in CONFIG.env_name:
      plot_heatmap(*run_trajectory(initial_states, [CONFIG.seed], algorithms[0]), list(colors.values())[0], save_path)
    if 'Fetch' in CONFIG.env_name:
      plot_movement([initial_states], [[CONFIG.seed]], algorithms, colors, save_path)            


def reduce_population(pop_size: int, population: List[Individual]):
    if len(population) > pop_size: return reduce_population(pop_size, population[:-1])
    else: return population


def genetic_algorithm(pop_size: int, max_iter: int, crossover_prob: float, mutation_prob: float, env_name: str,
                      logger: EvoLogger, render: bool, is_elitist: bool, weights: List[float], fidelity_fitness: bool,
                      exp_name: str, plot_frequency: int) -> List[Individual]:

    bsl = CONFIG.dimensions * CONFIG.state_encoding_length  # generate population with this bit string length
    population = [Individual(f'{random.getrandbits(bsl):=0{bsl}b}', id, weights) for id in range(pop_size)]
    next_id = pop_size; iteration = 0

    # compute fitness of each individual in the first generation
    prev_individuals, local_diversities, certainties = [], [], []
    for individual in population:
        individual.compute_fitness(render, prev_individuals, local_diversities, certainties)
        if individual.state_sequence:  # avoid owd error because len(traj) could be 0
            prev_individuals.append(individual.state_sequence_without_duplicates)
            local_diversities.append(individual.local_diversity_measure)
            certainties.append(individual.certainty_measure)

    # recompute first individual to change default values
    prev_individuals.pop(0); local_diversities.pop(0); certainties.pop(0)
    if population[0].state_sequence: population[0].compute_fitness(render, prev_individuals, local_diversities, certainties, True)
          
    # Calculate fidelity for each individual
    L = np.array([i.get_traj_length() for i in population])
    R = np.array([i.reward for i in population])
    F = L / L.sum() * abs(R.mean() - R) 

    for (individual, f) in zip(population, F): 
        individual.fidelity = f
        if fidelity_fitness: individual.fitness = f

    population.sort(key=attrgetter("fitness"), reverse=True)
    if plot_frequency: vizualize_population(population, f'{logger.save_path}/{CONFIG.exp_name}', render, iteration)

    # Start logging
    for i in population:
        # compute metric
        traj_length = i.get_traj_length()
        episode_length = max(len(i.state_sequence) - 1, 0)
        logger.log(iteration, i.id, i.state,
                   i.local_diversity_measure, i.global_diversity_measure, i.certainty_measure,
                   i.fitness, i.fidelity,
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
        
        # Calculate fidelity for each individual
        L = np.array([i.get_traj_length() for i in population])
        R = np.array([i.reward for i in population])
        F = L / L.sum() * abs(R.mean() - R) 
        for (individual, f) in zip(population, F): 
            individual.fidelity = f
            if fidelity_fitness: individual.fitness = f

        population.sort(key=attrgetter("fitness"), reverse=True)
        population = reduce_population(pop_size, population)

        # Logging
        for i in population:
            traj_length = i.get_traj_length()
            if i.state_sequence != 0: episode_length = len(i.state_sequence) - 1
            else: episode_length = 0
            logger.log(iteration, i.id, i.state,
                       i.local_diversity_measure, i.global_diversity_measure, i.certainty_measure,
                       i.fitness, i.fidelity,
                       i.dist_local_div, i.dist_certainty, i.min_dist_of_other_measures,
                       i.reward, traj_length, episode_length)

        if plot_frequency and iteration % plot_frequency == 0: 
            vizualize_population(population, f'{logger.save_path}/{CONFIG.exp_name}', render, iteration)

        iteration += 1

    return population


@click.command("evo")
@click.option("--env-name")
@click.option("--saved-model")
@click.option("--render", is_flag=True)
@click.option("--pop-size", default=10)
@click.option("--iterations", default=40)
@click.option("--crossover", default=0.75)
@click.option("--mutation", default=0.5)
@click.option("--name", default="")
@click.option("--is-elitist", is_flag=True)
@click.option("--plot-frequency", default=0, type=int)
@click.option("--checkpoint", default=0, type=int)
@click.option("--encoding-length", default=8, type=int)
@click.option("--env-seed", default=33, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--w1", default=1.0, help="Weight for global diversity")
@click.option("--w2", default=1.0, help="Weight for local diversity")
@click.option("--w3", default=1.0, help="Weight for certainty")
@click.option("--w4", default=1.0, help="Weight for local min distance")
def evo_run(env_name: str, saved_model: str, render: bool, pop_size: int, iterations: int,
            crossover: float, mutation: float, name: str, is_elitist: bool, plot_frequency: int,
            checkpoint: int, encoding_length: int, env_seed: int, seed: int, w1: float, w2: float, w3: float, w4: float):
    
    name = env_name if len(name) == 0 else name; exp_name = f'{name}-{seed}'
    env = Monitor(gym.make( 
        **hyphi_gym.named( # Train on random targets / evaluate on random initial pos
            env_name + 'Agents' if "Fetch" in env_name else env_name), 
          render_mode = "blender" if "Grid" in env_name else "3D", seed=env_seed
        ), record_video=render
    )
    # TODO: autoreset needed? 
    is_discrete_env = 'Grid' in env_name
    if is_discrete_env:  # if is_gridworld
        # TODO: check with paper: optimize agent and target?
        # TODO: mby try agent first 
        dimensions = len(env.unwrapped.size); min_state = 0
        # -2 because we do not want to consider outside walls
        map_size = int(sum(env.unwrapped.size)/dimensions) - 2
        max_state = map_size - 1; max_owd = get_max_owd(map_size)
    else:  # is continuous env aka FetchReach
        if env.unwrapped.position_noise != env.unwrapped.target_noise:
            assert False, "Fetch with inconsistent state ranges"
        dimensions = len(env.unwrapped.target) + len(env.unwrapped.agent)
        max_state = env.unwrapped.position_noise; min_state = -max_state
        max_owd = two_way_distance(
            np.full((1, *env.unwrapped.target.shape), -env.unwrapped.target_noise),
            np.full((1, *env.unwrapped.agent.shape), env.unwrapped.position_noise)
        )
        map_size = 0  # there is no map

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
        exp_name=exp_name,
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

    w = f"({w1:.0f},{w2:.0f},{w3:.0f},{w4:.0f})"
    if w == '(1,1,1,1)': alg_name = 'REACT'
    elif w == '(1,0,0,0)': alg_name = 'REACT_G'
    elif w == '(0,1,1,1)': alg_name = 'REACT_D'
    elif w == '(1,1,1,0)': alg_name = 'REACT_P' # No distance
    elif w == '(0,1,0,0)': alg_name = 'REACT_L' # No distance
    elif w == '(0,0,1,0)': alg_name = 'REACT_C' # No distance
    elif w == '(0,0,0,0)': alg_name = 'REACT_F' # Use fidelity as fitness as a baseline
    else: assert False, f"Unsupported weight combination: {w}"

    if iterations == 0: alg_name = 'Random'

    logger = EvoLogger(save_path=LOG_PATH.joinpath(f"{alg_name}/").joinpath(env_name), exp_name=exp_name)

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
        fidelity_fitness=w == '(0,0,0,0)',
        exp_name=exp_name,
        plot_frequency=plot_frequency
    )
    logger.close()
