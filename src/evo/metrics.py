import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from config import CONFIG
from constants import EVO_LOG_PATH


def plot_heatmap(heatmap, exp_name, iteration, pop_size, colors):
    # heatmap does not count duplicate states from the same trajectory

    fig = px.imshow(heatmap, range_color=[0, pop_size], color_continuous_scale=colors,
                    labels=dict(x="x", y="y", color="cell count"))
    path = EVO_LOG_PATH.joinpath(CONFIG.env_name + "/images/" + exp_name + "/")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.update_layout(font=dict(size=14), paper_bgcolor="rgba(0,0,0,0)")
    fig.write_image(path.joinpath(exp_name + "-heatmap-" + str(iteration) + ".pdf"))


def create_mesh(x, y, h, color):
    mesh = go.Mesh3d(
        # 8 vertices of a cube
        x=[x, x, x+1, x+1, x, x, x+1, x+1],
        y=[y, y+1, y+1, y, y, y+1, y+1, y],
        z=[0, 0, 0, 0, h, h, h, h],
        # i, j and k give the vertices of triangles
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        flatshading=True,
        color=color
    )
    return mesh


def plot_3d_histogram(heatmap, exp_name, iteration, pop_size, colors):
    print(heatmap.max())
    print(pop_size)

    x = np.linspace(0, 1, heatmap.max() + 1)
    # x = np.linspace(0, 1, pop_size + 1)
    c = sample_colorscale(colors, list(x))

    meshes = []
    for i in range(len(heatmap)):
        for j in range(len(heatmap)):
            height = heatmap[i][j]
            meshes.append(create_mesh(i, j, height, c[height]))
    fig = go.Figure(data=meshes)
    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='cell count', zaxis_range=[0,10]),
        font=dict(size=14),
        paper_bgcolor="rgba(0,0,0,0)"
        )
    path = EVO_LOG_PATH.joinpath(CONFIG.env_name + "/images/")
    if not os.path.exists(path): os.makedirs(path)
    fig.write_image(path.joinpath(exp_name + "-histogram-" + str(iteration) + ".pdf"))


def compute_coverage(population, exp_name, iteration):
    if "Grid" in CONFIG.env_name:
        state_matrix = np.zeros((CONFIG.map_size, CONFIG.map_size), dtype=np.int8)
        heatmap = np.zeros((CONFIG.map_size, CONFIG.map_size), dtype=np.int8)
        for individual in population:
            for state in individual.state_sequence_without_duplicates:
                state_matrix[state[0]][state[1]] = 1
                heatmap[state[0]][state[1]] += 1
        plot_heatmap(heatmap, exp_name, iteration, len(population), "Greens")
        plot_3d_histogram(heatmap, exp_name, iteration, len(population), "Greens")
    else:
        return
    num_ones = np.sum(state_matrix)
    return num_ones


##########################
# plot 3d trajectories ###
##########################
def plot_3d_trajectories(population, exp_name, iteration):
    #exp_name = f"trajectories in iteration {iteration}"
    df = pd.DataFrame(columns=["index", "id", "x", "y", "z"])

    j = 0
    for i in population:
        for state in i.state_sequence:
            df = pd.concat([df, pd.DataFrame({"id": i.id, "x": state[0], "y": state[1], "z": state[2]}, index=[j])])
            j += 1

    fig = px.line_3d(df, x="x", y="y", z="z", color='id')

    path = EVO_LOG_PATH.joinpath(CONFIG.env_name + "/images/" + exp_name + "/")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.update_layout(
        font=dict(size=14),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(xaxis_range=[0.8, 1.6], yaxis_range=[0.5, 1.0], zaxis_range=[0.2, 1.0]), showlegend=False)
    fig.write_image(path.joinpath("trajectories-" + str(iteration) + ".pdf"))
