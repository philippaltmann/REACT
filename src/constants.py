
# constants that are used all over the project
import pathlib

import numpy as np

EXPERIMENTS_PATH = pathlib.Path(__file__).parents[1].joinpath("experiments/")
LOG_PATH = EXPERIMENTS_PATH.joinpath("logs/")
MODEL_PATH = EXPERIMENTS_PATH.joinpath("model/")
EVO_LOG_PATH = LOG_PATH.joinpath("REACT/")
BASELINE_PATH = LOG_PATH.joinpath("baseline/")
TRAIN_LOG_PATH = LOG_PATH.joinpath("train/")
VIDEO_PATH = EXPERIMENTS_PATH.joinpath("videos/")
RANDOM_SEARCH_LOG_PATH = LOG_PATH.joinpath("Random/")

# layouts are defined in training, and have to be copied here manually
# the keys of the dict consist of the env_name, map_size+2, - the seed
# 0 = WALL, 1 = EMPTY, 2 = AGENT, 3 = TARGET, 4 = HOLE
LAYOUTS = {
    "FlatGrid11": np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    "HoleyGrid11-42": np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 4, 1, 4, 1, 1, 1, 1, 3, 0],
                      [0, 4, 1, 1, 1, 4, 1, 1, 1, 4, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 4, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 4, 1, 0],
                      [0, 4, 1, 1, 1, 1, 1, 1, 4, 4, 0],
                      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 4, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    "HoleyGrid11-33": np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 4, 4, 1, 1, 1, 1, 1, 0],
                                [0, 1, 4, 4, 1, 1, 4, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 4, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 1, 4, 1, 1, 1, 1, 0],
                                [0, 1, 1, 1, 4, 4, 1, 1, 1, 4, 0],
                                [0, 1, 1, 1, 1, 1, 1, 4, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
}