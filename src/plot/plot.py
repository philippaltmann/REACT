import ast
import os

import click
from evo import utils
# from evo.metrics import plot_3d_histogram, plot_heatmap
import hyphi_gym
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gymnasium as gym
from hyphi_gym import Monitor
hyphi_gym.register_envs() 

from config import CONFIG
from evo.run import run_individual
from train.run_model import run_trained_model
from constants import EXPERIMENTS_PATH, LOG_PATH, VIDEO_PATH, MODEL_PATH

from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO, SAC

# !pip3 install git+https://github.com/google-research/rliable
# Adapted from https://github.com/google-research/rliable

import numpy as np
import pandas as pd
import os

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

#@title Plotting: Seaborn style and matplotlib params
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns

sns.set_style("white")

# Matplotlib params
from matplotlib import rcParams
from matplotlib import rc

# ALL = ['Train', 'Random', 'REACT_F', 'REACT', 'REACT_P', 'REACT_D', 'REACT_G', 'REACT_L', 'REACT_C'] 
ALL = ['Train', 'Random', 'REACT_F', 'REACT', 'REACT_P', 'REACT_G','REACT_L','REACT_C']

def ALGS(*keys):
  algorithms, v = zip(*{ key: {
    'Train':   ('#C0C0C0', "Training"),          # unaltered training state (light grey)
    'Random':  ('#CC173A', "Random"),            # Random (red)
    'REACT_F': ('#3F3F3F', "Fidelity"),          # Fidelity (grey)
    'REACT':   ('#15CC70', "REACT (ours)"),      # REACT (green)
    'REACT_P': ('#0F7F3F', "Simple Sum"),        # No Distance (dark green)
    'REACT_D': ('#B2CC17', "Min Distance"),      # Local Distance (yellow)
    'REACT_G': ('#173ACC', "Global Diversity"),  # Global Diversity (blue)
    'REACT_L': ('#1795CC', "Local Diversity"),   # Local Diversity (light blue)
    'REACT_C': ('#CC9017', "Certainty"),         # Certainty (orange)
  }.get(key) for key in keys }.items()); 
  c, names = zip(*v); colors = dict(zip(names, c))
  return algorithms, colors, names



ENVS = lambda env_name: { 
  'FlatGrid11':  ('train_PPO',  35000, 'FlatGrid11'), 
  'HoleyGrid11': ('train_PPO', 150000, 'HoleyGrid11'),
  'Fetch50k':    ('train_SAC',  50000, 'FetchReach'),
  'Fetch100k':   ('train_SAC', 100000, 'FetchReach'),
  'Fetch100k':   ('train_SAC', 100000, 'FetchReach'),
  'Fetch200k':   ('train_SAC', 200000, 'FetchReach'),
  'Fetch300k':   ('train_SAC', 300000, 'FetchReach'),
}[env_name]



def fetch(alg, env, metric, get_seeds=False):
  exp = env
  # norm = 10 if ('Fetch' in exp and normalize) else 1
  
  if 'Fetch' in exp: env = 'FetchReach' 
  dir = f"{LOG_PATH}/{alg}/{env}"
  load = lambda d: {'seed': int(d.name[:-4].split('-')[1]), 'run': pd.read_csv(d.path)}
  exp = [load(dir) for dir in os.scandir(dir) if '.csv' in dir.name if exp in dir.name]
  data = np.rollaxis(np.array([[i[metric] for _, i in e['run'].groupby('iteration')] for e in exp]), -1)
  if get_seeds: return data, [e['seed'] for e in exp]
  # print(data.shape)
  # assert False
  return data



def write_figure(fig, save_path, overwrite=True):
  # print(f"Saving figure to {save_path}")
  # if os.path.isfile(save_path) and overwrite: os.system(f"rm {save_path}")
  fig.savefig(save_path, format='pdf', bbox_inches='tight')



def plot_training(env_name, save_path=None):
  metrics = [{"rollout/ep_rew_mean": "Mean reward"},
    {"rollout/ep_len_mean": "Mean episode length" } ]
  saved_model, checkpoint, env = ENVS(env_name)
  fig, axs = plt.subplots(1, len(metrics), figsize=(len(metrics)*5, 5))

  def plot_progress(df, ax, metric, x_label='Step', x_value="time/total_timesteps"):
    y_value, y_label = list(metric.items())[0]
    ax.plot(df[y_value]); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
  
  df = pd.read_csv(f'{MODEL_PATH}/{env}/{saved_model}/progress.csv').set_index('time/total_timesteps').loc[:checkpoint]
  [ plot_progress(df, ax, metric) for ax, metric in zip(axs, metrics) ]
  if save_path: write_figure(fig, save_path)



def plot_ablation(env, save_path=None):
  algorithms, colors, names = ALGS('Train', 'Random', 'REACT_F', 'REACT', 'REACT_P', 'REACT_G','REACT_L','REACT_C')
  # algorithms, colors, names = ALGS('Train', 'Random', 'REACT')

  mtrcs = {
    'Fidelity IQM': lambda r,f: metrics.aggregate_iqm(f[:,:,-1]),
    'Optimality Gap': lambda r,f: metrics.aggregate_optimality_gap(r[:,:,-1]),  
  }

  data = { name: [fetch(alg, env, metric) for metric in ['reward', 'fidelity']] for alg, name in zip(algorithms,names)}

  scores, interval_estimates = rly.get_interval_estimates(data, lambda r,f: np.array([m(r,f) for m in mtrcs.values()]), reps=50000)
  fig, axes = plot_utils.plot_interval_estimates(scores, interval_estimates, metric_names=list(mtrcs.keys()), algorithms=names, colors=colors, xlabel='')
  if save_path: write_figure(fig, save_path)
  else: plt.show()
  # # fig.savefig('Fidelity.pdf', format='pdf', bbox_inches='tight')



def plot_fitness(env, save_path=None):
  """Create a stacked plot of the fitness components' progress."""   
  # ['global_diversity_measure', 'certainty_measure', 'fitness', 'min_dist_of_measures']
  m = ['iteration', 'global_diversity_measure', 'dist_local_diversity', 'dist_certainty']
  d = dict(zip(m, [fetch('REACT', env, metric) for metric in m]))

  # plt.figure(figsize=)
  _, ax = plt.subplots(figsize=(7,5))

  # TODO: fill nan with value after
  ax.stackplot(
      d['iteration'].mean(axis=(0,1)), 
      d['global_diversity_measure'].sum(axis=1).mean(axis=0),
      d['dist_local_diversity'].sum(axis=1).mean(axis=0),
      d['dist_certainty'].sum(axis=1).mean(axis=0),
      labels=["Global Diversity", "Local Diversity Distance", "Certainty Distance"],
      colors=[(23/255, 58/255, 204/255), (23/255, 149/255, 204/255), (204/255, 144/255, 23/255)],
      alpha=0.8
  )

  plt.xlabel("Generation")
  plt.ylabel("Cumulative Fitness")
  plt.legend(loc="upper left")
  if save_path:  write_figure(plt, save_path)
  else: plt.show()



def plot_fidelity(env, save_path=None, window=10):
  """Create a stacked plot of the fidelity components' progress."""
  algorithms, colors, names = ALGS('REACT','REACT_F', 'Random', 'Train')
  # algorithms, colors, names = ALGS('REACT','REACT_F', 'Random')
  # algorithms, colors, names = ALGS('REACT','Random')
  global steps, frames; steps, frames = None, None

  def frame(d):
    global steps, frames
    # if steps is None: steps = d.shape[-1]; frames = np.arange(0, steps, window)
    if steps is None: 
      steps = d.shape[-1]; frames = np.arange(0, steps, window)
      # frames = np.arange(0, steps, window * (5 if 'Fetch' in env else 1))
    d = d[:, :, frames] if d.shape[-1] == steps else np.repeat(d[:, :, -1:], frames.shape[0], axis=-1) 
    return d
  
  data = {name: frame(fetch(alg, env, 'fidelity')) for alg, name in zip(algorithms,names)}
  mtrc = lambda d: np.array([metrics.aggregate_iqm(d[:, :, frame]) for frame in range(d.shape[-1])])

  iqm_scores, iqm_cis = rly.get_interval_estimates(data, mtrc, reps=2000)

  # fig, ax = plt.subplots(figsize=(7, 4.5))
  plot_utils.plot_sample_efficiency_curve(
    frames+1, iqm_scores, iqm_cis, algorithms=names, colors=colors,
    xlabel='Generation', ylabel='Fidelity IQM', marker=None
  )
  plt.legend(loc="upper left")

  if save_path: write_figure(plt, save_path)
  else: plt.show()



def run_trajectory(states, seeds, name):
  exp = CONFIG.exp_name; CONFIG.exp_name = name
  state_matrix = [] if 'Fetch' in CONFIG.env_name else np.zeros((CONFIG.map_size, CONFIG.map_size), dtype=np.int8)
  def _add_T(t): 
    if 'Fetch' in CONFIG.env_name: state_matrix.extend(t)
    else: state_matrix[tuple(np.array(utils.remove_duplicate_states(t)).T)] += 1

  for initial_states in states:
    for seed, state in zip(seeds, initial_states):
      state = np.fromstring(state[1:-1], dtype=int if 'Grid' in CONFIG.env_name else float, sep=', ')
      _add_T(run_individual(state, False, name)[0])
  CONFIG.exp_name = exp
  return state_matrix, name




def plot_heatmap(state_matrix, name, color, save_path, f=np.log1p):
  state_matrix = f(state_matrix); plt.figure(figsize=(5,5))
  cmap = clr.LinearSegmentedColormap.from_list("", ["white", color])
  plt.imshow(state_matrix, cmap=cmap, interpolation='nearest')
  plt.xticks([]); plt.yticks([])
  if save_path: plt.savefig(f'{save_path}-{name}.pdf', format='pdf', bbox_inches='tight')
  else: plt.show()



def plot_movement(initial_states, seeds, algorithms, colors, save_path):
  fig = plt.figure(figsize=(5,20)); 
  ax1 = fig.add_subplot(1,4,1, projection='3d') # Gripper
  ax2 = fig.add_subplot(1,4,2, projection='3d') # Target
  ax3 = fig.add_subplot(1,4,3, projection='3d') # Gripper + Target
  ax4 = fig.add_subplot(1,4,4, projection='3d') # Full trajectory
  for ax in [ax1, ax2, ax3, ax4]: ax.xaxis.set_ticklabels([]); ax.yaxis.set_ticklabels([]); ax.zaxis.set_ticklabels([])

  # gripper = np.array([1.34201875, 0.74913933, 0.54138954])
  for states, color, name, seed in zip(initial_states, colors.values(), algorithms, seeds):
    d = np.array([np.fromstring(state[1:-1], dtype=float, sep=', ') for s in states for state in s])
    # start_difference = state[:3] goal_difference = state[3:]
    # start = gripper + d[:, :3]; goal = start + d[:, 3:]
    start = d[:, :3]; goal = d[:, 3:]
    ax1.scatter( *start.T, color=color, marker='.', alpha=0.5)
    ax2.scatter( *goal.T, color=color, marker='.', alpha=0.5)
    ax3.scatter( *start.T, color=color, marker='.', alpha=0.5)
    ax3.scatter( *goal.T, color=color, marker='.', alpha=0.1)
    ax4.scatter( *np.array(run_trajectory(states, seed, name)[0]).T, color=color, marker='.', alpha=0.5)
  if save_path: plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight'); print(f'{save_path}.pdf')


def plot_trajectory(env_name, save_path=None):
  saved_model, checkpoint, env = ENVS(env_name)
  env_config = { 'env_name': env, 'env_seed': 33, 'map_size': 0,
    'saved_model': saved_model, 'checkpoint': checkpoint, 'exp_name': None }
  env = Monitor(gym.make(**hyphi_gym.named(env + 'Agents' if "Fetch" in env else env), seed=env_config['env_seed']))
  if 'Grid' in env_name: env_config['map_size'] = env.unwrapped.size[0] - 2
  CONFIG.set_eval_config(env=env, **env_config)

  algorithms, colors, names = ALGS('Random', 'REACT_F', 'REACT')
  # algorithms, colors, names = ALGS('REACT', 'Random')
  _f = lambda st,se: (st[:,:,-1],se) # Get the final entry
  initial_states, seeds = zip(*[_f(*fetch(a, env_name, 'state', get_seeds=True)) for a in algorithms])

  if 'Grid' in env_name:
    for states, s, name, c in zip(initial_states, seeds, algorithms, colors.values()):
      plot_heatmap(*run_trajectory(states, s, name), c, save_path)
  else: plot_movement(initial_states, seeds, algorithms, colors, save_path)



def render_videos(env_name, record_seeds=[42]):
  saved_model, checkpoint, env_base = ENVS(env_name);
  env_config = { 'env_name': env_base, 'env_seed': 33, 'map_size': 0,
    'saved_model': saved_model, 'checkpoint': checkpoint, 'exp_name': None }
  render_mode = 'blender' if 'Grid' in env_name else '3D'
  env = Monitor(gym.make(**hyphi_gym.named(env_base + 'Agents' if "Fetch" in env_base else env_base), 
    seed=env_config['env_seed'], render_mode=render_mode), record_video=True)
  if 'Grid' in env_name: env_config['map_size'] = env.unwrapped.size[0] - 2
  CONFIG.set_eval_config(env=env, **env_config)
  video_path = VIDEO_PATH.joinpath(f"{CONFIG.env_name}/"); os.makedirs(video_path, exist_ok=True);

  model_path = 'best_model' if checkpoint == 0 else 'rl_model_' + str(checkpoint) + '_steps'
  model_path = MODEL_PATH.joinpath(f'{env_base}/{saved_model}/{model_path}')
  for seed in record_seeds:
    model = eval(saved_model.split("_")[1])(policy="MlpPolicy", env=env, seed=seed).load(model_path, env=env)
    run_trained_model(model, model.get_env())
  env.save_video(f"{video_path}/Train.gif")

  algorithms, colors, names = ALGS('REACT', 'Random', 'REACT_F')
  _f = lambda st,se: (st[:,:,-1],se) # Get the final entry
  initial_states, seeds = zip(*[_f(*fetch(a, env_name, 'state', get_seeds=True)) for a in algorithms])
  for states, seed, name, c in zip(initial_states, seeds, algorithms, colors.values()):
    [run_trajectory(states, [s], name) for s in seed if s in record_seeds]
    env.save_video(f"{video_path}/{name}.gif")




@click.command("plot")
@click.option("--env-name")
@click.option("--render", is_flag=True)
def plot(env_name: str, render: bool):

    rcParams['legend.loc'] = 'best'; rcParams['pdf.fonttype'] = 42; rcParams['ps.fonttype'] = 42
    path = EXPERIMENTS_PATH.joinpath(f"plots/"); os.makedirs(path, exist_ok=True); rc('text', usetex=False)

    # plot_training(env_name, save_path=path.joinpath(f"{env_name}-Training.pdf"))
    # plot_ablation(env_name, save_path=path.joinpath(f"{env_name}-Ablation.pdf"))
    # plot_fitness(env_name, save_path=path.joinpath(f"{env_name}-Fitness.pdf"))
    # plot_fidelity(env_name, save_path=path.joinpath(f"{env_name}-Fidelity.pdf"))
    # plot_trajectory(env_name, save_path=path.joinpath(f"{env_name}-Heatmap"))
    if render: render_videos(env_name, record_seeds=[42])
