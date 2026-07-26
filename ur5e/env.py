"""UR5e reach environment (Gymnasium).

The task: drive the end-effector to a randomly placed target. Same shape as any
Gymnasium env: reset() starts an episode, step(action) advances it one control
step and returns (observation, reward, terminated, truncated, info).

Observation (21 numbers): 6 joint positions, 6 joint velocities, the 3D
end-effector position, the 3D target, and the 3D vector from hand to target, all
scaled to roughly unit size (see OBS_SCALE).
Action (6 numbers in [-1, 1]): a small position change for each joint.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE = Path(__file__).parent / "assets" / "universal_robots_ur5e" / "reach_scene.xml"


class UR5eReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    # Task constants. These are the knobs to tune once it trains.
    N_SUBSTEPS = 10            # physics steps per env step
    EPISODE_STEPS = 200        # env steps before the episode is truncated
    ACTION_SCALE = 0.2         # max joint move per step, radians
    SUCCESS_THRESHOLD = 0.05   # end-effector within 5 cm counts as a reach
    TARGET_LOW = np.array([0.25, -0.35, 0.15])   # target sampling box, min corner
    TARGET_HIGH = np.array([0.60, 0.35, 0.55])   # target sampling box, max corner

    # Rough per-component scale to normalize the observation: joint angles ~pi,
    # velocities ~a few rad/s, positions ~1 m. Dividing by this keeps every input
    # near unit size, which SAC trains far more stably on than raw mixed scales.
    OBS_SCALE = np.array([np.pi] * 6 + [5.0] * 6 + [1.0] * 9, dtype=np.float32)

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None

        self._home_id = self.model.keyframe("home").id
        self._ee_site = self.model.site("attachment_site").id
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        obs_dim = 6 + 6 + 3 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)

        self.target = np.zeros(3)
        self._elapsed = 0

    # --- helpers ---

    def _ee_pos(self):
        """3D position of the end-effector site in world coordinates."""
        return self.data.site_xpos[self._ee_site].copy()

    def _get_obs(self):
        ee = self._ee_pos()
        raw = np.concatenate([
            self.data.qpos[:6],   # joint positions
            self.data.qvel[:6],   # joint velocities
            ee,                   # where the hand is
            self.target,          # where it needs to go
            self.target - ee,     # vector from hand to target (direction + distance)
        ])
        return (raw / self.OBS_SCALE).astype(np.float32)   # normalize to ~unit scale

    def _sample_target(self):
        return self.np_random.uniform(self.TARGET_LOW, self.TARGET_HIGH)

    # --- Gymnasium API ---

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_id)
        self.target = self._sample_target()
        self.data.mocap_pos[0] = self.target      # move the red marker to the goal
        mujoco.mj_forward(self.model, self.data)
        self._elapsed = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # Position-delta control: nudge each joint's target from where it is now,
        # clipped to that joint's limits, then let the servos run for a few
        # physics steps.
        ctrl = self.data.qpos[:6] + action * self.ACTION_SCALE
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_step(self.model, self.data, nstep=self.N_SUBSTEPS)

        obs = self._get_obs()
        distance = float(np.linalg.norm(self._ee_pos() - self.target))
        reward = self._reward(distance)

        self._elapsed += 1
        terminated = False                          # reach task has no failure state
        truncated = self._elapsed >= self.EPISODE_STEPS
        info = {"is_success": float(distance < self.SUCCESS_THRESHOLD),
                "distance": distance}
        return obs, reward, terminated, truncated, info

    def _reward(self, distance):
        """Dense reward: negative distance to the target, closer is better.

        A dense signal every step is what lets the agent learn to reach.
        """
        return -distance

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
