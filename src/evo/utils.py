import copy
import math
from collections import OrderedDict
from operator import attrgetter

import numpy as np

from config import CONFIG
from constants import LAYOUTS
from evo.one_way_distance import two_way_distance


############################################
# utility methods for fitness computation ##
############################################
def sublist_exists_in_list(sl, l):
    for i in range(len(l) - len(sl) + 1):
        if l[i: i+len(sl)] == sl:
            return True
    return False


def remove_duplicate_states(states):
    if not states:
        return []
    new_states = [states[0]]
    for i in range(1, len(states)):
        if states[i-1] != states[i]:
            new_states.append(states[i])
    return new_states


def reduce_precision_of_states(states, precision): 
  return [states.round(precision).tolist() for states in states]


def get_traj_length(env_name, state_sequence, state_sequence_without_duplicates=None):
    traj_length = 0
    if env_name == "FetchReach":
        for s_index in range(len(state_sequence) - 1):
            traj_length += math.dist(state_sequence[s_index], state_sequence[s_index + 1])
    else:
        if not state_sequence_without_duplicates:
            state_sequence_without_duplicates = remove_duplicate_states(state_sequence)
        traj_length = len(state_sequence_without_duplicates) - 1
        if traj_length == -1:
            traj_length = 0
    return traj_length


#######################################
# utility methods for state encoding ##
#######################################
def map_state_encoding_to_value(state_encoding):
    # inverse normalization
    if CONFIG.is_discrete:
        normalized_state = int(state_encoding, 2) / (2 ** len(state_encoding))
        state = math.floor(normalized_state * (CONFIG.max_state + 1 - CONFIG.min_state) + CONFIG.min_state)
    else:
        normalized_state = int(state_encoding, 2) / (2 ** len(state_encoding) - 1)
        state = normalized_state * (CONFIG.max_state - CONFIG.min_state) + CONFIG.min_state
    return state


def get_state(state_encoding):
    coordinate_state_encoding_length = len(state_encoding) / CONFIG.dimensions # split state_encoding
    idx = lambda i: int(i * coordinate_state_encoding_length) 
    return [map_state_encoding_to_value(state_encoding[idx(i):idx(i + 1)]) for i in range(CONFIG.dimensions)]


def convert_state_to_custom_map(state, env_name, seed):
    if "Holey" in env_name:
        layout_key = env_name + "-" + str(seed)
    else:
        layout_key = env_name
    row, column = state[0] + 1, state[1] + 1
    layout = copy.deepcopy(LAYOUTS[layout_key])

    # GOAL: 0.5 * max_steps = 50
    # FAIL: -0.5 * max_steps = -50
    # STEP: -1
    if layout[row][column] == 1:
        layout[row][column] = 2
        return False, 0, layout
    elif layout[row][column] == 4:
        return True, -50, layout
    elif layout[row][column] == 3:
        return True, 50, layout
    else:
        raise ValueError()


############################
# utility methods for evo ##
############################
def multisort(xs, specs):
    for key, reverse in reversed(specs):
        xs.sort(key=attrgetter(key), reverse=reverse)
    return xs


def get_max_owd(map_size):
    top_left_corner = [[0, 0]]
    lower_right_corner = [[map_size - 1, map_size - 1]]
    max_owd = two_way_distance(top_left_corner, lower_right_corner)
    return max_owd


def clean_observation(obs):
    # hyphi gym Fetch -> return agent pos, (target: obs[0][-3:])
    if "Fetch" in CONFIG.env_name: return obs[0][:3]
    if "Grid" in CONFIG.env_name: 
        idx = np.where(obs[0] == 2)[0]
        if len(idx) == 0: return [None, None]
        size = int(math.sqrt(len(obs[0])))
        return [int(idx[0]) // size - 1, int(idx[0]) % size - 1]
    return obs[0]
