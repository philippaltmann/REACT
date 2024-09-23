import pandas
import plotly.express as px
import tensorflow as tf
import numpy as np

from numpy import load
from tensorboardX import SummaryWriter

from src.constants import TRAIN_LOG_PATH


def print_eval_data(experiment_name):
    data = load(TRAIN_LOG_PATH + experiment_name + "/evaluations.npz")
    lst = data.files
    for item in lst:
        print(item)
        print(len(data[item]))
        print(data[item])


def plot_loss(df):
    df = df[["train/value_loss", "time/total_timesteps"]]
    df = df.rename(columns={"train/value_loss": "loss", "time/total_timesteps": "timestep"})
    #df = df[["train/loss", "time/total_timesteps"]]
    #df = df.rename(columns={"train/loss": "loss", "time/total_timesteps": "timestep"})
    df = df.dropna()
    fig = px.line(df, x="timestep", y="loss", title='Loss')
    fig.show()


def plot_reward(df):
    df = df[["rollout/ep_rew_mean", "time/total_timesteps"]]
    df = df.rename(columns={"rollout/ep_rew_mean": "mean reward", "time/total_timesteps": "timestep"})
    df = df.dropna()
    fig = px.line(df, x="timestep", y="mean reward", title='Mean reward averaged over 100 episodes')
    fig.show()


def plot_episode_length(df):
    df = df[["rollout/ep_len_mean", "time/total_timesteps"]]
    df = df.rename(columns={"rollout/ep_len_mean": "mean episode length", "time/total_timesteps": "timestep"})
    df = df.dropna()
    fig = px.line(df, x="timestep", y="mean episode length", title='Mean episode length averaged over 100 episodes')
    fig.show()


def plot_progress_data(experiment_name):
    df = pandas.read_csv(TRAIN_LOG_PATH + experiment_name + "/progress.csv")
    plot_loss(df)
    plot_reward(df)
    plot_episode_length(df)


def plot_training_using_tensorboard(env_name, exp_name):
    df = pandas.read_csv(TRAIN_LOG_PATH.joinpath(env_name + "/" + exp_name + "/progress.csv"))

    # dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    print(df)
    assert False
    writer = SummaryWriter()

    i = 0
    for feature_batch in dataset:
        for key, value in feature_batch.items():
            print("  {!r:20s}: {}".format(key, value))
            x = value.numpy()
            if x is not np.nan:
                writer.add_scalar(key, x, i)
        i += 1

    writer.close()


if __name__ == "__main__":
    #plot_progress_data(experiment_name= "non-slippery_a2c")
    plot_training_using_tensorboard("FetchReach", "train1_sac")