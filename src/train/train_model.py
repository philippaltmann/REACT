import click
import gymnasium as gym
from hyphi_gym import named, register_envs, Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor as SB3M
from constants import MODEL_PATH
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm


class Evaluate(EvalCallback):      
    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and self.render:
          self.eval_env.envs[0].save_video(f'{self.log_path[:-11]}eval-{self.n_calls}.gif')
        return continue_training


@click.command("train")
@click.option("--env-name")
@click.option("--name", default="train")
@click.option("--model", default="PPO", type=click.Choice(["PPO", "SAC"]))
@click.option("--render", is_flag=True)
@click.option("--steps", default=10000)
@click.option("--save-freq", default=50000)
@click.option("--env-seed", default=42)
def train(env_name: str, name: str, model, render: bool, steps: int, save_freq: int, env_seed: int):
    register_envs(); save_freq = min(save_freq, steps)
    log_dir = f'{env_name}/{name}_{model}'
    save_path = MODEL_PATH.joinpath(log_dir)

    if env_name == "FetchReach": env_name += 'Targets' # Train on random targets / evaluate on random initial pos
    render_mode = "blender" if "Grid" in env_name else "3D"
    env = gym.make(**named(env_name), seed=env_seed, render_mode=render_mode) # , continue_task=True

    eval_callback = Evaluate(
        Monitor(SB3M(env), record_video=render), best_model_save_path=save_path,
        log_path=save_path, eval_freq=save_freq, deterministic=True, render=render, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path)

    model:BaseAlgorithm = eval(model)(policy="MlpPolicy", env=env, seed=env_seed)

    new_logger = configure(str(save_path), ["csv"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=steps, callback=[checkpoint_callback, eval_callback])
    env.close()
