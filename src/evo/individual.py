import math
from typing import List

from config import CONFIG
from evo.one_way_distance import two_way_distance
from evo.run import run_individual
import evo.utils as utils


class Individual:
    def __init__(self, state: str, id: int, weights: List[float]):
        self.id = id
        if len(weights) != 4:
            ValueError("not correctly initialized")
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.w3 = weights[2]
        self.w4 = weights[3]

        self.state_encoding = state
        self.state = utils.get_state(self.state_encoding, CONFIG.dimensions, CONFIG.is_discrete, CONFIG.min_state,
                                     CONFIG.max_state)
        print(id, " individual with state", self.state)
        if CONFIG.env_name == "FetchReach":
            self.precision = 3
        else:
            self.precision = None

        self.state_sequence = []
        self.state_sequence_without_duplicates = []
        self.reward = None

        # measures are initialized as None to not compute them twice
        self.local_diversity_measure = None
        self.global_diversity_measure = None
        self.certainty_measure = None

        # to investigate the impact of the different measures on the fitness
        self.dist_local_div = None
        self.dist_certainty = None
        self.min_dist_of_other_measures = None

        self.fitness = None

    def get_traj_length(self):
        traj_length = 0
        if CONFIG.env_name == "FetchReach":
            for s_index in range(len(self.state_sequence)-1):
                traj_length += math.dist(self.state_sequence[s_index], self.state_sequence[s_index + 1])
        else:
            traj_length = len(self.state_sequence_without_duplicates) - 1
            if traj_length == -1:
                traj_length = 0
        return traj_length

    def compute_global_diversity(self, previous: List[List[List[float]]]) -> float:
        # remove repeated states from sequence of states to avoid having a high global diversity on all initial states,
        # that lead to the agent, not going anywhere but standing against the wall. This is only once interesting!
        current = self.state_sequence_without_duplicates
        # previous is also without duplicates

        for i in previous:
            if utils.sublist_exists_in_list(self.state_sequence_without_duplicates, i):
                return 0

        # has to be defined for every distance problem
        max_owd = CONFIG.max_owd
        # one-way-distance for continuous state_sequences with measurable distances
        min_owd = max_owd
        for i in previous:
            owd = two_way_distance(i, current)
            if owd < min_owd:
                min_owd = owd
        print("min owd:", min_owd)
        normalized_min_owd = min_owd / max_owd
        return self.w1 * normalized_min_owd

    def compute_local_diversity(self) -> float:
        """
        Local diversity is defined by the coverage of the space. The more distinct states have been visited, the higher
        this measure is.
        :return: local_diversity
        """
        state_sequence_strings = [str(state) for state in self.state_sequence]
        distinct_state_strings = set(state_sequence_strings)
        distinct_states = [eval(i) for i in distinct_state_strings]

        if self.precision:
            max_states = len(self.state_sequence)
        else:
            max_states = CONFIG.map_size * CONFIG.map_size

        return self.w2 * (len(distinct_states) / max_states)

    def compute_certainty(self, certainties) -> float:
        certainty = sum(certainties) / len(certainties)
        certainty = certainty.item()
        return self.w3 * certainty

    def compute_diversity_measure(self, local_diversities, certainties):
        if local_diversities:
            min_dist = 100000  # initialize with high value that should never actually occur
        else:
            min_dist = 0

        distances = []
        dists_local_div = []
        dists_certainty = []
        for i in range(len(local_diversities)):
            new_dist = math.dist([
                self.local_diversity_measure,
                self.certainty_measure
            ], [
                local_diversities[i],
                certainties[i]
            ])

            distances.append(new_dist)
            dists_local_div.append(math.dist([self.local_diversity_measure], [local_diversities[i]]))
            dists_certainty.append(math.dist([self.certainty_measure], [certainties[i]]))

        if distances:
            min_dist = min(distances)
            min_dist_index = distances.index(min_dist)
            return min_dist, dists_local_div[min_dist_index], dists_certainty[min_dist_index]

        return min_dist, None, None

    def compute_fitness(self, render: bool, states_of_previously_evaluated_individuals,
                        previous_local_diversities, previous_certainties, recompute=False):
        # to avoid rerunning the episode and recomputing everything
        if recompute:
            self.global_diversity_measure = self.compute_global_diversity(states_of_previously_evaluated_individuals)
            self.min_dist_of_other_measures, self.dist_local_div, self.dist_certainty = self.compute_diversity_measure(
                previous_local_diversities, previous_certainties)
            self.fitness = self.w4 * self.min_dist_of_other_measures + self.global_diversity_measure
            return

        # get trajectory
        states, self.reward, certainties, actions = run_individual(self.state, render)

        if self.precision:
            # make states discrete and save in self.state_sequence
            self.state_sequence = utils.reduce_precision_of_states(states, self.precision)
        else:
            self.state_sequence = states
        self.state_sequence_without_duplicates = utils.remove_duplicate_states(self.state_sequence)

        # no need to compute the fitness if the initial state is an immediate failure
        if states:
            self.local_diversity_measure = self.compute_local_diversity()
            self.global_diversity_measure = self.compute_global_diversity(states_of_previously_evaluated_individuals)
            self.certainty_measure = self.compute_certainty(certainties)

            self.min_dist_of_other_measures, self.dist_local_div, self.dist_certainty = self.compute_diversity_measure(previous_local_diversities, previous_certainties)
            self.fitness = self.w4 * self.min_dist_of_other_measures + self.global_diversity_measure
        else:  # move to beginning
            self.local_diversity_measure = 0
            self.global_diversity_measure = 0
            self.certainty_measure = 0
            self.fitness = 0
