import pandas as pd


class EvoLogger:
    def __init__(self, save_path=None, experiment_name=None):
        self.save_path = save_path
        self.experiment_name = experiment_name + ".csv"
        self.data = pd.DataFrame()

    def log(self, iteration, id, state,
            m1, m2, m3, f,
            d3, d4, md, r, tl, el):
        df = pd.DataFrame({"iteration": [iteration],
                           "id": [id],
                           "state": [state],
                           "local_diversity_measure": [m1],
                           "global_diversity_measure": [m2],
                           "certainty_measure": [m3],
                           "fitness": [f],
                           "dist_local_diversity": [d3],
                           "dist_certainty": [d4],
                           "min_dist_of_measures": [md],
                           "reward": [r],
                           "trajectory_length": [tl],
                           "episode_length": [el]})
        self.data = pd.concat([self.data, df], ignore_index=True)

    def close(self):
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.data.to_csv(self.save_path / self.experiment_name)
