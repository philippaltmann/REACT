import ast 
import os
import statistics

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from _plotly_utils.colors import sample_colorscale

from src.constants import EVO_LOG_PATH, BASELINE_PATH


# TODO: move to matplot 

def save_fig(fig, plot_name, path=None):
    if path is None:
        path = EVO_LOG_PATH.joinpath(env_name + "/images/plots/")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.write_image(path.joinpath(plot_name + ".pdf"))

def get_random_search_heatmap(df, name):
    df["row"] = df["state"].map(lambda x: ast.literal_eval(x)[0])
    df["col"] = df["state"].map(lambda x: ast.literal_eval(x)[1])

    subplot = go.Histogram2d(
        x=df["row"],
        y=df["col"],
        xbins={'start': 0, 'size': 1},
        ybins={'start': 0, 'size': 1},
        coloraxis="coloraxis",
        name=name)
    return subplot


def get_random_data(nr_runs, popsize=10, compute_avg=False, model =""):
    df = pd.read_csv(BASELINE_PATH.joinpath("random_search" + "/" + env_name + "/" + "exp" + model +"-random-0.csv"))
    df = df.head(popsize)
    if compute_avg:
        dfs = [df]
        for i in range(1, nr_runs):
            df2 = pd.read_csv(BASELINE_PATH.joinpath("random_search" + "/" + env_name + "/" + "exp" + model + "-random-" + str(i) + ".csv"))
            df2 = df2.head(popsize)
            dfs.append(df2)
    else:
        dfs = df
        for i in range(1, nr_runs):
            df2 = pd.read_csv(BASELINE_PATH.joinpath("random_search" + "/" + env_name + "/" + "exp" + model + "-random-" + str(i) + ".csv"))
            df2 = df2.head(popsize)
            dfs = pd.concat([dfs, df2])
    return [dfs, "Random", "#CC173A"]



def get_probabilities_data(exp_prefix, nr_runs: int = 0, top_x: int = 0, compute_avg=False, compare_with_random=False):
    x = np.linspace(0, 1, 8)
    colors = sample_colorscale('YlOrBr', samplepoints=list(x))

    exps = [[exp_prefix + "-1", "cp 0.9, mp 0.25", colors[3]],
            [exp_prefix + "-2", "cp 0.9, mp 0.4", colors[4]],
            [exp_prefix + "-3", "cp 0.75, mp 0.5", colors[5]],
            [exp_prefix + "-4", "cp 0.5, mp 0.75", colors[6]],
            [exp_prefix + "-5", "cp 0.25, mp 0.9", colors[7]]]
    data = get_data(exps, nr_runs, top_x, compute_avg)
    if compare_with_random:
        random_data = get_random_data(nr_runs, popsize, compute_avg)
        data.insert(0, random_data)
    return data



def comparison_of_encoding_length():
    exps = [["exp_enc4", "encoding length = 4"],
            ["exp_enc5", "encoding length = 5"],
            ["exp_enc6", "encoding length = 6"],
            ["exp_enc7", "encoding length = 7"],
            ["exp_enc8", "encoding length = 8"]]
    data = []
    for exp, name in exps:
        data.append([pd.read_csv(BASELINE_PATH.joinpath("random_search" + "/" + env_name + "/" + exp + ".csv")), name])

    fig = make_subplots(1, 5, subplot_titles=(data[0][1], data[1][1], data[2][1], data[3][1], data[4][1]))

    for i in range(len(data)):
        fig.add_trace(get_random_search_heatmap(data[i][0], data[i][1]), row=1, col=i + 1)

    kwargs = {"height": 500, "width": 2500}
    fig.update_layout(kwargs)
    fig.update_layout(font=dict(size=20), paper_bgcolor="rgba(0,0,0,0)")
    fig.update_annotations(font_size=20)
    save_fig(fig, "random_states_heatmap")




def get_top_x(df, top_x) -> pd.DataFrame:
    if top_x == 0:
        return df
    pop_size = len(df[df["iteration"] == 1])
    indices = []
    max_iter = df["iteration"].max()
    for i in range(max_iter):
        for j in range(top_x, pop_size):
            indices.append(i * pop_size + j)
    df = df.drop(indices)
    return df


def get_data(exps, nr_runs, top_x, compute_avg):
    data = []
    if compute_avg:
        for exp, name, color in exps:
            df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp + "-0.csv"))
            dfs = [get_top_x(df, top_x)]
            for i in range(1, nr_runs):
                df2 = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp + "-" + str(i) + ".csv"))
                dfs.append(get_top_x(df2, top_x))
            data.append([dfs, name, color])
    else:
        for exp, name, color in exps:
            df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp + "-0.csv"))
            df = get_top_x(df, top_x)
            for i in range(1, nr_runs):
                df2 = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp + "-" + str(i) + ".csv"))
                df2 = get_top_x(df2, top_x)
                df = pd.concat([df, df2], ignore_index=True)
            data.append([df, name, color])
    return data


def go_box_iteration_new_individuals(df, name, self_compute_box=False):
    iter_points = []
    max_id = df["id"].max()
    for id in range(max_id):
        df2 = df[df["id"] == id]
        if df2.size == 0:
            continue
        min_iter = df2["iteration"].min()
        iter_points.append(min_iter)
    if self_compute_box:
        df3 = pd.DataFrame({"first_iteration": iter_points})
        min = df3["first_iteration"].min()
        q1 = df3["first_iteration"].quantile(0.25)
        median = df3["first_iteration"].median()
        q3 = df3["first_iteration"].quantile(0.75)
        max = df3["first_iteration"].max()
        return min, q1, median, q3, max
    return go.Box(y=iter_points, name=name, boxpoints="all")


def go_violin_new_individuals_per_iteration(df, name, color):
    iter_points = []
    max_id = df["id"].max()
    for id in range(max_id):
        df2 = df[df["id"] == id]
        if df2.size == 0:
            continue
        min_iter = df2["iteration"].min()
        iter_points.append(min_iter)
    return go.Violin(y=iter_points, name=name, marker_color=color, spanmode="hard")


def go_box_by_key(df, key, name, self_compute_box, color=None):
    max_iteration = df["iteration"].max()
    df2 = df[(df["iteration"] == max_iteration)]

    if self_compute_box:
        min = df2[key].min()
        q1 = df2[key].quantile(0.25)
        median = df2[key].median()
        q3 = df2[key].quantile(0.75)
        max = df2[key].max()
        return min, q1, median, q3, max

    return go.Box(y=df2[key], name=name, boxpoints="all", marker_color = color)


def boxplots_results(key, data, title, xaxis_name, yaxis_name):
    xaxis = dict(title=xaxis_name)
    yaxis = dict(title=yaxis_name)
    kwargs = {"yaxis": yaxis, "xaxis": xaxis, "showlegend": False}
    fig = go.Figure(layout=kwargs)
    for df, name, color in data:
        fig.add_trace(go_box_by_key(df, key, name, False, color))
    fig.update_traces()
    fig.update_layout(font=dict(size=18), paper_bgcolor="rgba(0,0,0,0)")
    save_fig(fig, title)


def avg_boxplots_results(key, data, title, xaxis_name, yaxis_name, with_policy):
    xaxis = dict(title=xaxis_name)
    yaxis = dict(title=yaxis_name)
    kwargs = {"yaxis": yaxis, "xaxis": xaxis, "showlegend": False}
    fig = go.Figure(layout=kwargs)
    for dfs, name, color in data:
        q1s, q3s, ms, lfs, ufs = [], [], [], [], []
        for df in dfs:
            v1, v2, v3, v4, v5 = go_box_by_key(df, key, name, True, None)
            lfs.append(float(v1))
            q1s.append(v2)
            ms.append(v3)
            q3s.append(v4)
            ufs.append(float(v5))
        q1 = statistics.mean(q1s)
        q3 = statistics.mean(q3s)
        median = statistics.mean(ms)
        lf = statistics.mean(lfs)
        uf = statistics.mean(ufs)
        fig.add_trace(go.Box(q1=[q1], median=[median], q3=[q3], lowerfence=[lf], upperfence=[uf], x=[name], name=name,
                             marker_color=color))
    if with_policy:
        line_value = policy_return if key == "reward" else policy_traj_length
        fig.add_hline(y=line_value, annotation_text="Policy")
    fig.update_layout(font=dict(size=18), paper_bgcolor="rgba(0,0,0,0)")
    save_fig(fig, title)


def plot_popsize_avg_new_individuals(data, title):
    xaxis = dict(title="population size")
    yaxis = dict(title="number of iterations")
    kwargs = {"yaxis": yaxis, "xaxis": xaxis, "showlegend": False}
    fig = go.Figure(layout=kwargs)
    for dfs, name in data:
        q1s, q3s, ms, lfs, ufs = [], [], [], [], []
        for df in dfs:
            v1, v2, v3, v4, v5 = go_box_iteration_new_individuals(df, name, True)
            lfs.append(v1)
            q1s.append(v2)
            ms.append(v3)
            q3s.append(v4)
            ufs.append(v5)
        q1 = statistics.mean(q1s)
        q3 = statistics.mean(q3s)
        median = statistics.mean(ms)
        lf = statistics.mean(lfs)
        uf = statistics.mean(ufs)
        fig.add_trace(
            go.Box(q1=[q1], median=[median], q3=[q3], lowerfence=[lf], upperfence=[uf], x=[name], name=name))
    save_fig(fig, title)


def get_popsize_data(exp_prefix, nr_runs: int = 0, top_x: int = 0, compute_avg=False):
    x = np.linspace(0, 1, 8)
    colors = sample_colorscale('Greens', samplepoints=list(x))

    exps = [[exp_prefix + "-1", "5", colors[3]],
            [exp_prefix + "-2", "10", colors[4]],
            [exp_prefix + "-3", "20", colors[5]],
            [exp_prefix + "-4", "30", colors[6]],
            [exp_prefix + "-5", "40", colors[7]]]
    if top_x == 10:
        exps.pop(0)

    data = get_data(exps, nr_runs, top_x, compute_avg)
    return data


def plot_popsize_new_individuals(data, title):
    xaxis = dict(title="population size")
    yaxis = dict(title="number of iterations")
    kwargs = {"yaxis": yaxis, "xaxis": xaxis, "showlegend": False}
    fig = go.Figure(layout=kwargs)
    for df, name, color in data:
        fig.add_trace(go_violin_new_individuals_per_iteration(df, name, color))
    fig.update_layout(font=dict(size=18), paper_bgcolor="rgba(0,0,0,0)")
    save_fig(fig, title)


def get_popsize_data_of_several_runs_with_id_change(exp_prefix, nr_runs, top_x):
    x = np.linspace(0, 1, 8)
    colors = sample_colorscale('Greens', samplepoints=list(x))

    exps = [[exp_prefix + "-1", 5, colors[3]],
            [exp_prefix + "-2", 10, colors[4]],
            [exp_prefix + "-3", 20, colors[5]],
            [exp_prefix + "-4", 30, colors[6]],
            [exp_prefix + "-5", 40, colors[7]]]

    if top_x > 5:
        exps.pop(0)
    data = []
    for exp, pop_size, color in exps:
        df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp + "-0.csv"))
        df = get_top_x(df, top_x)
        max_id = df["id"].max() + 1
        for i in range(1, nr_runs):
            df2 = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp + "-" + str(i) + ".csv"))
            df2 = get_top_x(df2, top_x)
            df2["id"] = df2["id"].map(lambda x: x + max_id)
            max_id = df2["id"].max() + 1
            df = pd.concat([df, df2], ignore_index=True)
        data.append([df, str(pop_size), color])
    return data



def comparison_of_population_sizes(exp_prefix, nr_runs, top_x, compute_avg=False):
    if not compute_avg:
        data = get_popsize_data_of_several_runs_with_id_change(exp_prefix, nr_runs, top_x)
        plot_popsize_new_individuals(data, "iter_popsizes_" + str(top_x) + "_" + str(nr_runs))
    else:
        data = get_popsize_data(exp_prefix, nr_runs, top_x, compute_avg)
        plot_popsize_avg_new_individuals(data, "avg_iter_popsizes_" + str(top_x) + "_" + str(nr_runs))


def comparison_of_returns_and_trajectory_lengths_for_different_probabilities(exp_prefix, nr_runs, top_x, compute_avg=False, compare_with_random=False):
    data = get_probabilities_data(exp_prefix, nr_runs, top_x, compute_avg, compare_with_random)
    if not compute_avg:
        boxplots_results("reward", data, "prob_returns_top" + str(top_x) + "_" + str(nr_runs), "", "return")
        boxplots_results("trajectory_length", data, "prob_trajectory_lengths_top" +  str(top_x) + "_" + str(nr_runs), "", "trajectory length")
    else:
        avg_boxplots_results("reward", data, "prob_avg_returns_top" + str(top_x) + "_" + str(nr_runs), "", "return", compare_with_random)
        avg_boxplots_results("trajectory_length", data, "prob_avg_trajectory_lengths_top" + str(top_x) + "_" + str(nr_runs), "", "trajectory length", compare_with_random)


if __name__ == "__main__":
    # this plot is created using random search for FlatGrid11 for 81000 iterations
    comparison_of_encoding_length()

    ### settings for FlatGrid11
    env_name = "FlatGrid11"
    popsize = 10
    policy_return = 34
    policy_traj_length = 16
    # experiment names should end with "-1-x", "-2-x", "-3-x", "-4-x", "-5-x" for population sizes 5, 10, 20, 30, 40
    # where x are numbers between 0 and 3 for each of the runs
    exp_prefix = "exp-pop-"
    comparison_of_population_sizes(exp_prefix, 4, 0, False) # all individuals
    comparison_of_population_sizes(exp_prefix, 4, 10, False) # best 10
    
    comparison_of_returns_and_trajectory_lengths_for_different_probabilities("exp-prob-", 4, 0, True, True)
    