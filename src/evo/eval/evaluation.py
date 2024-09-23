import ast
import os

import click
from evo import utils
from evo.metrics import plot_3d_histogram, plot_heatmap
import hyphi_gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gymnasium as gym
from hyphi_gym import Monitor

from config import CONFIG
from evo.run import run_individual
from constants import EVO_LOG_PATH


def plot_3d_trajectories(p1, p2, exp_name, iteration):
    color_discrete_map = {}
    j = 0
    data = []
    for i in p1:
        x,y,z = [], [], []
        for state in i:
            x.append(state[0])
            y.append(state[1])
            z.append(state[2])
            j += 1
            color_discrete_map = color_discrete_map | {j: "#CC173A"}
        line = go.Scatter3d(x=x, y=y, z=z, mode='lines', marker_color="#CC173A")
        data.append(line)

    for i in p2:
        x,y,z = [], [], []
        for state in i:
            x.append(state[0])
            y.append(state[1])
            z.append(state[2])
            j += 1
            color_discrete_map = color_discrete_map | {j: "#2E9961"}
        line = go.Scatter3d(x=x, y=y, z=z, mode='lines', marker_color="#2E9961")
        data.append(line)
    fig = go.Figure(data=data)
    path = EVO_LOG_PATH.joinpath(CONFIG.env_name + "/images/concat/")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.update_layout(font=dict(size=14), paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(xaxis_range=[0.8,1.6], yaxis_range=[0.5,1.0], zaxis_range=[0.2,1.0]), showlegend=False)
    fig.write_image(path.joinpath(exp_name + "-trajectories-" + str(iteration) + ".pdf"))

def generate_heatmap(population):
    heatmap = np.zeros((CONFIG.map_size, CONFIG.map_size), dtype=np.int8)
    for states in population: 
        for state in utils.remove_duplicate_states(states[:-1]): heatmap[state[0]][state[1]] += 1
    return heatmap
    

def evaluate(exp_name, render, plot_trajectories):
    data = pd.read_csv(EVO_LOG_PATH.joinpath(CONFIG.env_name + "/" + exp_name + ".csv"))

    if CONFIG.env_name == "FetchReach":
        gym.register(id='CustomFetchReach-v1', entry_point='custom_env.custom_env:CustomFetchReachEnv')

    # show only last generation
    last_iteration = data["iteration"].max()
    data2 = data[data["iteration"] == last_iteration]
    data2["state"] = data2["state"].apply(ast.literal_eval).copy()
    p2 = [run_individual(data2.iloc[i]["state"], render, i=i)[0] for i in range(len(data2))]


    if plot_trajectories:
        first_iteration = data["iteration"].min()
        data1 = data[data["iteration"] == first_iteration]
        data1["state"] = data1["state"].apply(ast.literal_eval).copy()
        p1 = [run_individual(data1.iloc[i]["state"], render, i=i)[0] for i in range(len(data2))]

        if "Grid" in CONFIG.env_name:
            plot_3d_histogram(generate_heatmap(p2), exp_name, 'REACT', len(p2), "Greens")
            plot_3d_histogram(generate_heatmap(p1), exp_name, 'Random', len(p1), "Reds")
        else: # Fetch Trajectories
            plot_3d_trajectories(p1, p2, exp_name, last_iteration)


@click.command("eval")
@click.option("--env-name")
@click.option("--saved-model")
@click.option("--exp-name")
@click.option("--checkpoint", default=0)
@click.option("--seed", default=42)
@click.option("--render", is_flag=True)
@click.option("--plot", is_flag=True)
def evo_eval(env_name: str, saved_model: str, exp_name: str, checkpoint: int, seed: int, render: bool, plot: bool):
    map_size = 8
    env = None
    if "Grid" in env_name:
        env_kwargs = hyphi_gym.named(env_name)
        if "Flat" in env_name:
            env = Monitor(gym.make(env_kwargs["id"], size=env_kwargs["size"], sparse=env_kwargs["sparse"],
                                 detailed=env_kwargs["detailed"], explore=env_kwargs["explore"],
                                 random=env_kwargs["random"], seed=seed,
                                 render_mode="blender"), record_video=render)
        else: 
          env = Monitor(gym.make(env_kwargs["id"], level=env_kwargs["level"], sparse=env_kwargs["sparse"],
                                 detailed=env_kwargs["detailed"], explore=env_kwargs["explore"],
                                 random=env_kwargs["random"],
                                 render_mode="blender", seed=seed), record_video=render)

        try:  # -2 because we do not want to consider outside walls
            map_size = int(env_name[-2:]) - 2
        except:
            try: map_size = int(env_name[-1]) - 2
            except: pass

    CONFIG.set_eval_config(
        env=env,
        env_name=env_name,
        saved_model=saved_model,
        map_size=map_size,
        name=exp_name,
        checkpoint=checkpoint,
        env_seed=seed
    )

    evaluate(exp_name, render, plot)
