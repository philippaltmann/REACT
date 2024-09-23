import os 
import sys 
import click
import pandas as pd
import numpy as np
import scipy.stats as st
import plotly.graph_objects as go

iterate = lambda run, f: [f(i['trajectory_length'], i['reward']) for _, i in run.groupby('iteration')]
S = lambda l, r: sum(l / sum(l) * abs(r.mean() - r))


def load(dir):
  env, seed = dir.name[:-4].split('-')
  return {'seed': int(seed), 'run': pd.read_csv(dir.path)}


def prepare_ci(data):
  # Helper to claculate confidence interval
  ci = lambda d, confidence=0.95: st.t.ppf((1+confidence)/2, len(d)-1) * st.sem(d)
  mean, h = data.mean(axis=0), data.apply(ci, axis=0)
  ci = pd.concat([mean-h, (mean+h).iloc[::-1]])
  return (mean, ci)


def plot_ci(mean, ci, name):
  smooth = {'shape':  'spline',  'smoothing': 0.4}
  scatter = lambda data, **kwargs: go.Scatter(x=data.index, y=data, **kwargs)

  # 150, 200, 230, 40, 350, 70
  hue = {'REACT': 150, 'Random': 350}[name]
  color = lambda dim=0: 'hsva({},{}%,{}%,{:.2f})'.format(hue, 90-dim*20, 80+dim*20, 1.0-dim*0.8)
  
  # go.Figure(data=
  return [ scatter(mean, name=name, mode='lines', line={'color': color(), **smooth}),
    scatter(ci, fillcolor=color(1), fill='toself', line={'color': 'rgba(255,255,255,0)', **smooth}, showlegend=False),
  ]
  

def plot_fidelity(alg, env, xmax=40):
  exp = env; env = 'FetchReach' if 'Fetch' in exp else env
  dir = f"experiments/logs/{alg}/{env}"
  exp = [load(dir) for dir in os.scandir(dir) if '.csv' in dir.name if exp in dir.name]

  fidelities = pd.DataFrame([iterate(e['run'], S) for e in exp])
  if alg == 'Random': fidelities[xmax] = fidelities[0]
  mean, ci = prepare_ci(fidelities)
  print(f"{env} | {alg}: {mean[xmax]:.2f}±{mean[xmax]-ci.loc[xmax].iloc[0]:.2f}")
  return plot_ci(mean, ci, alg)


@click.command("fidelity")
@click.argument("env")
def fidelity(env: str):
  xmax = 40 if "Grid" in env else 1000
  
  d,m,l = 'rgb(64, 64, 64)', 'rgba(64, 64, 64, 0.32)', 'rgba(64,64,64,0.04)'
  axis = lambda title: {'gridcolor': m, 'linecolor': d, 'title': title, 'mirror':True, 'ticks':'outside', 'showline':True, 'zeroline': True, 'zerolinecolor': m} 

  fig = go.Figure(
    data=[g for alg in ['Random','REACT'] for g in plot_fidelity(alg, env, xmax)],
    layout=go.Layout(
      width=640, height=480,showlegend=False, margin=dict(l=8, r=8, t=8, b=8),
      font=dict(size=20), xaxis=axis('Generation'), yaxis=axis('Fidelity'), plot_bgcolor=l
    )) 
  fig.update_xaxes(range=[0,xmax], dtick=xmax/4, tickmode='linear') 
  fig.write_image(f"experiments/plots/{env}-Fidelity.pdf")
