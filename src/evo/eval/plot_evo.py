import ast
import os
import statistics
import plotly.io as pio

pio.kaleido.scope.mathjax = None

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from _plotly_utils.colors import sample_colorscale
from plotly.subplots import make_subplots

from src.constants import EVO_LOG_PATH, BASELINE_PATH


def save_fig(fig, plot_name, path=None):
    if path is None:
        path = EVO_LOG_PATH.joinpath(env_name + "/images/plots/")
    if not os.path.exists(path):
        os.makedirs(path)
    fig.write_image(path.joinpath(plot_name + ".pdf"))


#######################################################################################################################
# load data ###########################################################################################################
#######################################################################################################################
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


def get_weights_data(exp_prefix, weight, nr_runs: int = 0, top_x:int = 0, compute_avg=False, compare_with_random=False):
    x = np.linspace(0, 1, 12)
    colors = sample_colorscale('Burg', samplepoints=list(x))

    exps = [
        [exp_prefix + weight + "-1", "0.0", colors[3]],
        [exp_prefix + weight + "-2", "0.25", colors[4]],
        [exp_prefix + weight + "-3", "0.5", colors[5]],
        [exp_prefix + weight + "-4", "0.75", colors[6]],
        [exp_prefix + weight + "-5", "1.0", colors[7]],
        [exp_prefix + weight + "-6", "1.25", colors[8]],
        [exp_prefix + weight + "-7", "1.5", colors[9]],
        [exp_prefix + weight + "-8", "1.75", colors[10]],
        [exp_prefix + weight + "-9", "2.0", colors[11]],
    ]
    data = get_data(exps, nr_runs, top_x, compute_avg)
    if compare_with_random:
        random_data = get_random_data(nr_runs, 10, compute_avg)
        data.insert(0,random_data)
    return data


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


def get_single_exp_data(exp_name, nr_runs, compute_avg, compare_with_random, model=""):
    exp = [exp_name, "REACT", "#2E9961"]

    data = get_data([exp], nr_runs, False, compute_avg)
    if compare_with_random:
        random_data = get_random_data(nr_runs, popsize, compute_avg, model)
        data.insert(0, random_data)
    return data

def get_exp_data(nr_runs, compute_avg, compare_with_random):
    #x = np.linspace(0, 1, 4)
    #colors = sample_colorscale('RdBu', samplepoints=list(x))
    #exps = [["exp-weights2-1", "w = 1.0", colors[0]],
    #        ["exp-weights-1", "w1 = 0.75, w4 = 1.25", colors[1]],
    #        ["exp-weights3-1", "w2 = 0.75, w3 = 1.25", colors[2]],
    #        ["exp-weights4-1", "w1 = w2 = 0.75, w3 = w4 = 1.25", colors[3]]]

    x = np.linspace(0, 1, 4)
    colors = sample_colorscale('BlueRed', samplepoints=list(x))
    exps = [["exp1-1", "100000", colors[1]],
            ["exp2-1", "3000000", colors[2]],
            ["exp3-1", "5000000", colors[3]]]
    data = get_data(exps, nr_runs, False, compute_avg)
    if compare_with_random:
        random_data = get_random_data(nr_runs, popsize, compute_avg)
        data.insert(0, random_data)
    return data


########################################################################################################################
# plots fitness interpretation #########################################################################################
########################################################################################################################
def plot_fitness_ranges(exp_name, nr_runs):
    df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp_name + ".csv"))
    for i in range(1, nr_runs):
        df2 = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp_name + "-" + str(i) + ".csv"))
        df = pd.concat([df, df2], ignore_index=True)
    kwargs = {"showlegend": False}
    fig = go.Figure(layout=kwargs)

    columns_to_display = [["global_diversity_measure", "global diversity"], ["local_diversity_measure", "local diversity"],
                          ["certainty_measure", "certainty"], ["min_dist_of_measures", "minimum distance"], ["fitness", "fitness"]]

    for col, name in columns_to_display:
        fig.add_trace(go.Box(y=df[col].values, name=name))
    fig.update_layout(font=dict(size=18), paper_bgcolor="rgba(0,0,0,0)")
    save_fig(fig, "fitness_ranges")


def plot_fitness_range_first_vs_last_iteration(df):
    title = "Range of all fitness measures comparing first and last generation"
    # df keep only first and last iteration
    max_iteration = df["iteration"].max()
    df2 = df[((df["iteration"] == 1) | (df["iteration"] == max_iteration))]

    columns_to_display = ["local_diversity_measure", "certainty_measure", "min_dist_of_measures",
                          "global_diversity_measure",  "fitness"]

    new_df = pd.DataFrame(columns=["index", "iteration", "fitness_measure", "fitness_value"])
    i = 0
    for row in df2.iterrows():
        for col in columns_to_display:
            new_df = pd.concat([new_df, pd.DataFrame({"iteration": row[1]["iteration"], "fitness_measure": col,
                                                      "fitness_value": row[1][col]}, index=[i])])
            i += 1

    fig = px.box(new_df, x="fitness_measure", y="fitness_value", color="iteration", labels={"fitness_measure": "fitness measure", "fitness_value": "value"})
    save_fig(fig, "fitness_range_comparison")


def plot_population_3d(df):
    # x and fitness during evolution of population
    fig = px.line_3d(df, x="global_diversity_measure", y="fitness", z="iteration", color='id')
    save_fig(fig, "population_3d_global_diversity")
    #fig = px.line_3d(df, x="local_diversity_measure", y="fitness", z="iteration", color='id')
    #save_fig(fig, "population_3d_local_diversity")
    #fig = px.line_3d(df, x="certainty_measure", y="fitness", z="iteration", color='id')
    #save_fig(fig, "population_3d_certainty")


def plot_fitness_impact(df, exp_name):
    max_iteration = df["iteration"].max()
    df2 = df[(df["iteration"] == max_iteration)]
    ids = [i for i in range(1,len(df2) + 1)]

    xaxis = dict(title="individual")
    yaxis = dict(title="value")
    kwargs = {"yaxis": yaxis, "xaxis": xaxis}
    fig = go.Figure(layout=kwargs)

    fig.add_trace(go.Scatter(x=ids, y=df2.global_diversity_measure,
                             mode='lines+markers',
                             name='Global Diversity',
                             marker_color='#173ACC'))
    fig.add_trace(go.Scatter(x=ids, y=df2.dist_local_diversity,
                             mode='lines+markers',
                             name='Local Diversity',
                             marker_color='#1795CC'))
    fig.add_trace(go.Scatter(x=ids, y=df2.dist_certainty,
                             mode='lines+markers',
                             name='Certainty',
                             marker_color='#CC9017'))
    fig.add_trace(go.Scatter(x=ids, y=df2.min_dist_of_measures,
                             mode='lines+markers',
                             name='Local Distance',
                             marker_color='#B2CC17'))
    fig.add_trace(go.Scatter(x=ids, y=df2.fitness,
                             mode='lines+markers',
                             name='Joint Fitness',
                             marker_color='#2E9961'))
    fig.update_layout(font=dict(size=10), paper_bgcolor="rgba(0,0,0,0)")
    save_fig(fig, exp_name + "fitness_impact")


def plot_fitness_over_time(df, exp_name):
    max_iteration = df["iteration"].max()
    df2 = pd.DataFrame({"iteration": [], "measure": [], "value": []})
    measures = [["global_diversity_measure", "Global Diversity"], ["dist_local_diversity", "Local Diversity"],
                ["dist_certainty", "Certainty"], ["min_dist_of_measures", "Local Distance"], ["fitness", "Joint Fitness"]]
    color_discrete_map = {"Global Diversity": "#173ACC", "Local Diversity": "#1795CC", "Certainty": "#CC9017", "Local Distance": "#B2CC17", "Joint Fitness": "#2E9961"}
    for i in range(1, max_iteration + 1):
        for m, n in measures:
            v = df.loc[df['iteration'] == i, m].sum()
            df3 = pd.DataFrame({"iteration": [i], "measure": [n], "value": [v]})
            df2 = pd.concat([df2, df3], ignore_index=True)

    fig = px.line(df2, x="iteration", y="value", color="measure",
                  labels={"value": "sum of values", "iteration": "iteration"},
                  color_discrete_map=color_discrete_map)
    fig.update_layout(font=dict(size=18), paper_bgcolor="rgba(0,0,0,0)")

    save_fig(fig, exp_name + "fitness_over_time")


def plot_sum_of_fitnesses(df):
    max_iteration = df["iteration"].max()
    x = []
    y = []
    for i in range(1, max_iteration + 1):
        x.append(i)
        y.append(df.loc[df['iteration'] == i, 'fitness'].sum())
    fig = px.line(x=x, y=y)
    save_fig(fig, "fitness_sum_3")


def plot_individual_survival(df):
    xaxis = dict(title="iteration")
    yaxis = dict(title="n-th individual")
    kwargs = {"yaxis": yaxis, "xaxis": xaxis, "showlegend": False}

    fig = go.Figure(layout=kwargs)

    max_id = df["id"].max()
    print(max_id)
    for id in range(max_id + 1):
        df2 = df[df["id"] == id]
        print("id", id)
        if df2.size == 0:
            continue
        min_iter = df2["iteration"].min()
        max_iter = df2["iteration"].max()
        print(min_iter, max_iter)

        fig.add_trace(go.Scatter(
            x=[*range(min_iter, max_iter + 1)],
            y=[id]*(max_iter + 1 - min_iter),
            name=id
        ))
    save_fig(fig, "survival")


##########################################################################################################
# population sizes #######################################################################################
##########################################################################################################
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


def plot_popsize_new_individuals(data, title):
    xaxis = dict(title="population size")
    yaxis = dict(title="number of iterations")
    kwargs = {"yaxis": yaxis, "xaxis": xaxis, "showlegend": False}
    fig = go.Figure(layout=kwargs)
    for df, name, color in data:
        fig.add_trace(go_violin_new_individuals_per_iteration(df, name, color))
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


def comparison_of_population_sizes(exp_prefix, nr_runs, top_x, compute_avg=False):
    if not compute_avg:
        data = get_popsize_data_of_several_runs_with_id_change(exp_prefix, nr_runs, top_x)
        plot_popsize_new_individuals(data, "iter_popsizes_" + str(top_x) + "_" + str(nr_runs))
    else:
        data = get_popsize_data(exp_prefix, nr_runs, top_x, compute_avg)
        plot_popsize_avg_new_individuals(data, "avg_iter_popsizes_" + str(top_x) + "_" + str(nr_runs))


##########################################################################################################
# comparison of return and trajectory length distribution #################################################
##########################################################################################################
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


def avg_boxplots_grouped_by_models(data, key, y_axis, path):
    x = ["100000", "3000000", "5000000"]
    policy_values = policy_returns if key == "reward" else policy_traj_lengths
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top', 'visible': False})
    for algo in data:  # random search, ga
        model_data = []
        for model in algo:
            q1s, q3s, ms, lfs, ufs = [], [], [], [], []
            for df in model[0]:
                v1, v2, v3, v4, v5 = go_box_by_key(df, key, model[1], True, None)
                lfs.append(float(v1))
                q1s.append(v2)
                ms.append(v3)
                q3s.append(v4)
                ufs.append(float(v5))
            lf = statistics.mean(lfs)
            q1 = statistics.mean(q1s)
            median = statistics.mean(ms)
            q3 = statistics.mean(q3s)
            uf = statistics.mean(ufs)
            model_data.append([lf, q1, median, q3, uf])
            name = model[1]
            color = model[2]
        fig.add_trace(go.Box(
            x=x,
            name=name,
            marker_color=color,
            lowerfence=[model_data[0][0], model_data[1][0], model_data[2][0]],
            q1=[model_data[0][1], model_data[1][1], model_data[2][1]],
            median=[model_data[0][2], model_data[1][2], model_data[2][2]],
            q3=[model_data[0][3], model_data[1][3], model_data[2][3]],
            upperfence=[model_data[0][4], model_data[1][4], model_data[2][4]], yaxis='y1'
        ), secondary_y=False)
    # add policy lines
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[policy_values[0], policy_values[0]], name="Policy", mode='lines', xaxis='x2', yaxis='y1', line=dict(color="black")),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=[1, 2], y=[policy_values[1], policy_values[1]], mode='lines', xaxis='x2', yaxis='y1', line=dict(color="black"), showlegend=False),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=[2, 3], y=[policy_values[2], policy_values[2]], mode='lines', xaxis='x2', yaxis='y1', line=dict(color="black"), showlegend=False),
        secondary_y=True
    )
    fig.update_layout(
        yaxis_title=y_axis,
        xaxis_title='trainingsteps',
        boxmode='group',
        font=dict(size=18),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.data[2].update(xaxis='x2', yaxis='y1')
    fig.data[3].update(xaxis='x2', yaxis='y1')
    fig.data[4].update(xaxis='x2', yaxis='y1')
    save_fig(fig, path)


def comparison_of_returns_and_trajectory_lengths_for_different_weights(exp_prefix, nr_runs, top_x, compute_avg=False, compare_with_random=False):
    weights = ["w1", "w2", "w3", "w4"]
    for w in weights:
        data = get_weights_data(exp_prefix, w, nr_runs, top_x, compute_avg, compare_with_random)
        if not compute_avg:
            boxplots_results("reward", data, "w" + w + "_returns_top" + str(top_x) + "_" + str(nr_runs), w, "return")
            boxplots_results("trajectory_length", data, "w" + w + "_trajectory_lengths_top" + str(top_x) + "_" + str(nr_runs), w, "trajectory length")
        else:
            avg_boxplots_results("reward", data, "w" + w +"_avg_returns_top" + str(top_x) + "_" + str(nr_runs), w, "return", compare_with_random)
            avg_boxplots_results("trajectory_length", data, "w" + w + "_avg_trajectory_lengths_top" + str(top_x) + "_" + str(nr_runs), w, "trajectory length", compare_with_random)


def comparison_of_returns_and_trajectory_lengths_for_different_probabilities(exp_prefix, nr_runs, top_x, compute_avg=False, compare_with_random=False):
    data = get_probabilities_data(exp_prefix, nr_runs, top_x, compute_avg, compare_with_random)
    if not compute_avg:
        boxplots_results("reward", data, "prob_returns_top" + str(top_x) + "_" + str(nr_runs), "", "return")
        boxplots_results("trajectory_length", data, "prob_trajectory_lengths_top" +  str(top_x) + "_" + str(nr_runs), "", "trajectory length")
    else:
        avg_boxplots_results("reward", data, "prob_avg_returns_top" + str(top_x) + "_" + str(nr_runs), "", "return", compare_with_random)
        avg_boxplots_results("trajectory_length", data, "prob_avg_trajectory_lengths_top" + str(top_x) + "_" + str(nr_runs), "", "trajectory length", compare_with_random)


def comparison_of_returns_and_trajectory_lengths_for_different_popsizes(exp_prefix, nr_runs, top_x, compute_avg=False):
    data = get_popsize_data(exp_prefix, nr_runs, top_x, compute_avg)
    if not compute_avg:
        boxplots_results("reward", data, "popsize_returns_top" + str(top_x) + "_" + str(nr_runs), "population size", "return")
        boxplots_results("trajectory_length", data, "popsize_trajectory_lengths_top" + str(top_x) + "_" + str(nr_runs), "population size", "trajectory_length")
    else:
        avg_boxplots_results("reward", data, "popsize_avg_returns_top" + str(top_x) + "_" + str(nr_runs), "population size", "return", False)
        avg_boxplots_results("trajectory_length", data, "popsize_avg_trajectory_lengths_top" + str(top_x) + "_" + str(nr_runs), "population size", "trajectory length", False)


def comparison_of_returns_and_trajectory_lengths_for_single_experiment(exp_name, nr_runs, compute_avg, compare_with_random,model=""):
    data = get_single_exp_data(exp_name, nr_runs, compute_avg, compare_with_random, model)
    if not compute_avg:
        boxplots_results("reward", data, exp_name + "_returns_top" + str(0) + "_" + str(nr_runs),
                         "", "return")
        boxplots_results("trajectory_length", data,
                         exp_name + "_trajectory_lengths_top" + str(0) + "_" + str(nr_runs),
                         "", "trajectory_length")
    else:
        avg_boxplots_results("reward", data, exp_name + "_avg_returns_top" + str(0) + "_" + str(nr_runs),
                             "", "return", compare_with_random)
        avg_boxplots_results("trajectory_length", data,
                             exp_name + "_avg_trajectory_lengths_top" + str(0) + "_" + str(nr_runs),
                             "", "trajectory length", compare_with_random)


def comparison_of_returns_and_trajectory_lengths_for_different_experiments(nr_runs, compute_avg, compare_with_random):
    data = get_exp_data(nr_runs, compute_avg, compare_with_random)
    if not compute_avg:
        boxplots_results("reward", data, "exp_returns_top" + str(0) + "_" + str(nr_runs),
                         "", "return")
        boxplots_results("trajectory_length", data,
                         "exp_trajectory_lengths_top" + str(0) + "_" + str(nr_runs),
                         "", "trajectory_length")
    else:
        avg_boxplots_results("reward", data, "exp_avg_returns_top" + str(0) + "_" + str(nr_runs),
                             "", "return", compare_with_random)
        avg_boxplots_results("trajectory_length", data,
                             "exp_avg_trajectory_lengths_top" + str(0) + "_" + str(nr_runs),
                             "", "trajectory length", compare_with_random)


def comparison_of_returns_and_trajectory_lengths_for_different_models(exp_prefix, nr_runs, model1, model2, model3):
    exps = [[exp_prefix + model1, "REACT", "#2E9961"],
            [exp_prefix + model2, "REACT", "#2E9961"],
            [exp_prefix + model3, "REACT", "#2E9961"]]
    exp_data = get_data(exps, nr_runs, False, True)
    random_data = [get_random_data(nr_runs, popsize, True, model1),
                   get_random_data(nr_runs, popsize, True, model2),
                   get_random_data(nr_runs, popsize, True, model3)]
    data = [random_data, exp_data]
    avg_boxplots_grouped_by_models(data, "trajectory_length", "trajectory length", "models_avg_trajectory_length")
    avg_boxplots_grouped_by_models(data, "reward", "return", "models_avg_return")


##########################################################################################################
# comparison of encoding length ##########################################################################
##########################################################################################################
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


if __name__ == "__main__":
    # If there are any problems with file namings, please refer to the get_data methods in the top of this file
    # in order to be able to average over several runs, the experiment names should end with -x where x is a number
    # between 0 and 9

    # experiments from random search should be called "exp-random-x" where x is a number between 0 and 9 for each of the runs
    # for FetchReach: "expM-random-x", where M is a string to represent the policy stage

    # Gridworlds likeliness of states
    # experiments are named "exp_encX" where X is 4,5,6,7,8 to represent the encoding length
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
    comparison_of_returns_and_trajectory_lengths_for_different_popsizes(exp_prefix, 4, 10, True)
    exp_prefix = "exp-prob-"
    comparison_of_returns_and_trajectory_lengths_for_different_probabilities(exp_prefix,4, 0, True, True)
    exp_name = "exp" # "actual experiment log files should end with "-x" where x is a number between 0 and 9 for each of the runs
    comparison_of_returns_and_trajectory_lengths_for_single_experiment(exp_name, 10, True, True, "")
    exp_name = "exp-x" # "-x" represents the run to plot
    df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp_name + ".csv"))
    plot_fitness_impact(df, exp_name)
    plot_fitness_over_time(df, exp_name)

    ### settings for HoleyGrid11
    env_name = "HoleyGrid11"
    popsize = 10
    policy_return = 34
    policy_traj_length = 16
    exp_name = "exp" # "actual experiment log files should end with "-x" where x is a number between 0 and 9 for each of the runs
    comparison_of_returns_and_trajectory_lengths_for_single_experiment(exp_name, 10, True, True, "")
    # only one run
    exp_name = "exp-x" # "-x" represents the run to plot
    df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp_name + ".csv"))
    plot_fitness_impact(df, exp_name)
    plot_fitness_over_time(df, exp_name)

    ### settings for FetchReach:
    env_name = "FetchReach"
    popsize = 30
    policy_returns = [-1.808, -1.683, -1.615]
    policy_traj_lengths = [0.7371, 0.1200, 0.1166]
    exp_prefix = "exp" # exp_name = exp_prefix + model1 + -x where x is a number between 0 and 9 for each of the runs
    model1 = "1"
    model2 = "2"
    model3 = "3"
    comparison_of_returns_and_trajectory_lengths_for_different_models(exp_prefix, 10, model1, model2, model3)
    exp_name = "exp1"
    df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp_name + ".csv"))
    plot_fitness_impact(df, exp_name)
    plot_fitness_over_time(df, exp_name)
    exp_name = "exp2"
    df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp_name + ".csv"))
    plot_fitness_impact(df, exp_name)
    plot_fitness_over_time(df, exp_name)
    exp_name = "exp3"
    df = pd.read_csv(EVO_LOG_PATH.joinpath(env_name + "/" + exp_name + ".csv"))
    plot_fitness_impact(df, exp_name)
    plot_fitness_over_time(df, exp_name)
