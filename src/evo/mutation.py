import random
from typing import List, Tuple

from evo.individual import Individual


def mutate(population: List[Individual], id: int, mutation_prob: float, is_elitist: bool, weights: List[float]
           ) -> Tuple[List[Individual], int]:
    if mutation_prob < random.random():
        return population, id
    random_individual_position = random.randint(0, len(population) - 1)
    random_individual = population[random_individual_position]
    random_bit_position = random.randint(0, len(random_individual.state_encoding) - 1)
    state_encoding_as_list = list(random_individual.state_encoding)
    if state_encoding_as_list[random_bit_position] == "0":
        state_encoding_as_list[random_bit_position] = "1"
    else:
        state_encoding_as_list[random_bit_position] = "0"
    mutated_state_encoding = "".join(state_encoding_as_list)
    if is_elitist:
        population.append(Individual(mutated_state_encoding, id, weights))
    else:
        population[random_individual_position] = Individual(mutated_state_encoding, id, weights)
    id += 1
    #print("mutated from ", random_individual.state, " to ", mutated_state_encoding)
    return population, id
