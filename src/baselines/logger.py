import pandas as pd


class BaselineLogger():
    def __init__(self, save_path=None, experiment_name=None):
        self.save_path = save_path
        self.experiment_name = experiment_name + ".csv"
        self.data = pd.DataFrame()

    def log_baseline(self, iteration, id, state, reward, trajectory_length, episode_length):  # coverage?
        df = pd.DataFrame({"iteration": [iteration],
                           "id": [id],
                           "state": [state],
                           "reward": [reward],
                           "trajectory_length": [trajectory_length],
                           "episode_length": [episode_length]})
        self.data = pd.concat([self.data, df], ignore_index=True)

    def close(self):
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(self.save_path / self.experiment_name)
