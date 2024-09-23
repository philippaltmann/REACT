import numpy as np
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 3.2,
    "azimuth": -125.0,
    "elevation": -20.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


class CustomFetchReachEnv(MujocoFetchReachEnv):
    def __init__(self, initial_startup_position=[0,0,0], goal_difference=None, **kwargs):
        self.start_difference = np.array(initial_startup_position)
        self.goal_difference = goal_difference
        self.reward_threshold = None
        super().__init__(default_camera_config=DEFAULT_CAMERA_CONFIG, **kwargs)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # Move end effector into position.
        gripper_target = self.start_difference + np.array(
           [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", gripper_rotation
        )
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # Extract information for sampling goals.
        self.initial_gripper_xpos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()

    def _sample_goal(self):
        goal_difference = self.np_random.uniform(-self.target_range, self.target_range, size=3) if self.goal_difference is None else np.array(self.goal_difference, dtype=float)
        goal = self.initial_gripper_xpos[:3] + goal_difference
        return goal.copy()
