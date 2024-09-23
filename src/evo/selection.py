import random
from operator import attrgetter
from typing import List, Tuple

from evo.individual import Individual


def best_pair_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    return population[0], population[1]


def roulette_wheel_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    # population should already be sorted by fitness
    sum_of_fitnesses = sum(i.fitness for i in population)
    probability_dist = list(map(lambda x: x.fitness / sum_of_fitnesses, population))
    # probability for choosing twice the best individual is very high -> 2 times choices, but remove first choice from options
    parent1_index = random.choices(range(0, len(population)), weights=probability_dist, k=1)
    probability_dist[parent1_index[0]] = 0
    parent2_index = random.choices(range(0, len(population)), weights=probability_dist, k=1)
    return population[parent1_index[0]], population[parent2_index[0]]


def tournament_selection(population: List[Individual]) -> Tuple[Individual, Individual]:
    k = 3
    parents = []
    for _ in range(2):
        sample = random.sample(population, k)
        sample.sort(key=attrgetter("fitness"), reverse=True)

        parents.append(sample[0])
    return parents[0], parents[1]
