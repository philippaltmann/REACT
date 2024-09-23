
class Configuration:

    def __init__(self):
        # evo config
        self.env = None
        self.name = None
        self.env_name = None
        self.saved_model = None
        self.map_size = None
        self.checkpoint = 0
        self.dimensions = None
        self.is_discrete = None
        self.min_state = None
        self.max_state = None
        self.state_encoding_length = None
        self.env_seed = None
        self.max_owd = None
        self.seed = None

    def set_evo_config(self, env, env_name, saved_model, map_size, pop_size, name, checkpoint, dimensions, is_discrete_env,
                       min_state: float, max_state: float, state_encoding_length: int, env_seed: int, max_owd: float, seed: int):
        self.env = env
        self.env_name = env_name
        self.saved_model = saved_model
        self.map_size = map_size
        #self.population_size = pop_size
        self.name = name
        self.checkpoint = checkpoint
        self.dimensions = dimensions
        self.is_discrete = is_discrete_env
        self.min_state = min_state
        self.max_state = max_state
        self.state_encoding_length = state_encoding_length
        self.env_seed = env_seed
        self.max_owd = max_owd
        self.seed = seed

    def set_eval_config(self, env, env_name, saved_model, map_size, name, checkpoint, env_seed):
        self.env = env
        self.env_name = env_name
        self.saved_model = saved_model
        self.map_size = map_size
        self.name = name
        self.checkpoint = checkpoint
        self.env_seed = env_seed

    def set_baseline_config(self, env, env_name, saved_model, map_size, name, checkpoint, env_seed, seed, state_encoding_length,
                            dimensions, is_discrete_env, min_state, max_state):
        self.env = env
        self.env_name = env_name
        self.saved_model = saved_model
        self.map_size = map_size
        self.name = name
        self.checkpoint = checkpoint
        self.env_seed = env_seed
        self.seed = seed
        self.state_encoding_length = state_encoding_length
        self.dimensions = dimensions
        self.is_discrete = is_discrete_env
        self.min_state = min_state
        self.max_state = max_state

# make config global since it is only set in the beginning of the program
CONFIG = Configuration()
