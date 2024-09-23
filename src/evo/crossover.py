import random
from typing import List, Tuple

from evo.individual import Individual
from evo.selection import tournament_selection


# only crossover if two parents are not the same
def cross_over(population: List[Individual], id: int, crossover_prob: float, weights: List[float]
               ) -> Tuple[List[Individual], int]:
    if crossover_prob < random.random():  # 0.0 <= x < 1.0
        return population, id
    parent1, parent2 = tournament_selection(population)

    crossover_point = random.randint(1, len(parent1.state_encoding) - 1)
    child1_state_encoding = parent1.state_encoding[:crossover_point] + parent2.state_encoding[crossover_point:]
    child2_state_encoding = parent2.state_encoding[:crossover_point] + parent1.state_encoding[crossover_point:]
    child1 = Individual(state=child1_state_encoding, id=id, weights=weights)
    id += 1
    child2 = Individual(state=child2_state_encoding, id=id, weights=weights)
    id += 1
    population.append(child1)
    population.append(child2)

    return population, id
