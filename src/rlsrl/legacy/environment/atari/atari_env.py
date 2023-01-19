"""Simple wrapper around the Atari environments provided by gym.
"""
from typing import List, Union
import cv2
import gym
import logging
import numpy as np
import os
import time

import rlsrl.api.environment as env_base
import rlsrl.api.env_utils as env_utils

logger = logging.getLogger("env-atari")

_HAS_DISPLAY = len(os.environ.get("DISPLAY", "").strip()) > 0


class AtariEnvironment(env_base.Environment):

    def __init__(self,
                 game_name,
                 render: bool = False,
                 pause: bool = False,
                 noop_max: int = 0,
                 episode_life: bool = False,
                 clip_reward: bool = False,
                 frame_skip: int = 1,
                 stacked_observations: Union[int, None] = None,
                 max_episode_steps: int = 108000,
                 gray_scale: bool = False,
                 obs_shape: Union[List[int], None] = None):
        """Atari environment
        Parameters
        ----------
        noop_max: int
            upon reset, do no-op action for a number of steps in [1, noop_max]
        episode_life: bool
            terminal upon loss of life
        clip_reward: bool
            reward -> sign(reward)
        frame_skip: int
            repeat the action for `frame_skip` steps and return max of last two frames
        max_episode_steps: int
            episode length
        gray_scale: bool
            use gray image observation
        obs_shape: list
            resize observation to `obs_shape`
        """
        self.game_name = game_name
        self.__render = render
        self.__pause = pause
        self.__env = gym.make(game_name)
        self.__space = env_utils.DiscreteActionSpace(self.__env.action_space)

        self.__noop_max = noop_max
        self.__episode_life = episode_life
        self.__clip_reward = clip_reward
        self.__frame_skip = frame_skip
        self.__stacked_observations = stacked_observations
        self.__max_episode_steps = max_episode_steps
        self.__gray_scale = gray_scale
        self.__ori_obs_shape = self.__env.observation_space.shape
        self.__obs_shape = obs_shape

        self.lives = 0
        self.was_real_done = True

        if self.__stacked_observations:
            # We would delay creating stack_obs
            self.stack_obs = None

        self.__step_count = np.zeros(1, dtype=np.int32)
        self.__episode_return = np.zeros(1, dtype=np.float32)

    @property
    def agent_count(self) -> int:
        return 1  # We are a simple Atari environment here.

    @property
    def observation_spaces(self):
        return [{"obs": (3, 160, 210)}]

    @property
    def action_spaces(self):
        return [self.__space]

    def reset(self) -> List[env_base.StepResult]:
        if self.__episode_life:
            if self.was_real_done:
                frame = self.noop_reset()
            else:
                frame, _, _, _ = self.frame_skip_step(0)
        else:
            frame = self.noop_reset()
        if isinstance(frame, tuple):
            frame, *_ = frame
        frame = self.transform_frame(frame)
        self.lives = self.__env.unwrapped.ale.lives()
        self.__step_count[:] = 0
        self.__episode_return[:] = 0
        if self.__stacked_observations:
            self.stack_obs = None
            frame = self.update_stack_observation(frame)
        return [
            env_base.StepResult(obs=dict(obs=np.swapaxes(frame, 0, 2)),
                                reward=np.array([0.0], dtype=np.float32),
                                done=np.array([False], dtype=np.uint8),
                                info=dict(episode_length=self.__step_count.copy(),
                                          episode_return=self.__episode_return.copy()))
        ]

    def step(self, actions: List[env_utils.DiscreteAction]) -> List[env_base.StepResult]:
        assert len(actions) == 1, len(actions)
        frame, reward, done, *truncated, info = self.frame_skip_step(actions[0].x.item())
        if len(truncated) > 0:
            done = done or truncated[0]
        frame = self.transform_frame(frame)
        if self.__stacked_observations:
            frame = self.update_stack_observation(frame)
        if self.__step_count > self.__max_episode_steps:
            done = True
        if self.__episode_life:
            self.was_real_done = done
            lives = self.__env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True
            self.lives = lives
        if self.__clip_reward:
            reward = np.sign(reward)
        if self.__render:
            logger.info("Step %d: reward=%.2f, done=%d", self.__step_count, reward, done)
            if _HAS_DISPLAY:
                self.render()
                if self.__pause:
                    input()
                else:
                    time.sleep(0.05)
        return [
            env_base.StepResult(obs=dict(obs=np.swapaxes(frame, 0, 2)),
                                reward=np.array([reward], dtype=np.float32),
                                done=np.array([done], dtype=np.uint8),
                                info=dict(episode_length=self.__step_count.copy(),
                                          episode_return=self.__episode_return.copy()))
        ]

    def render(self) -> None:
        self.__env.render()

    def seed(self, seed=None):
        self.__env.seed(seed)
        return seed

    def noop_reset(self):
        """Act no-op for `noop_max` steps upon reset
        """
        if self.__noop_max == 0:
            return self.__env.reset()
        self.__env.reset()
        noops = self.__env.unwrapped.np_random.integers(1, self.__noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.__env.step(0)
            if done:
                obs = self.__env.reset()
        return obs

    def frame_skip_step(self, action):
        """Skip `frame_skip` frames by repeating the same action, take max over the last two frames
        """
        total_reward = 0.0
        done = None
        max_frame = np.zeros(self.__ori_obs_shape, dtype=np.uint8)
        for i in range(self.__frame_skip):
            obs, reward, done, *truncated, info = self.__env.step(action)
            if len(truncated) > 0:
                done = done or truncated[0]

            self.__step_count += 1
            self.__episode_return += reward
            if i >= self.__frame_skip - 2:
                max_frame = np.max([max_frame, obs], axis=0)
            total_reward += reward
            if done:
                break
        return max_frame, total_reward, done, info

    def update_stack_observation(self, frame):
        """Stack last `stacked_observations` frames
        
        Warning: This might cause wasted memory usage
        """
        if self.stack_obs is None:
            self.stack_obs = np.zeros((self.__stacked_observations, *frame.shape), dtype=np.uint8)
        self.stack_obs[:-1] = self.stack_obs[1:]
        self.stack_obs[-1] = frame
        return self.stack_obs.copy()

    def transform_frame(self, frame):
        """Transform frame shape and RGB format
        """
        if self.__gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.__obs_shape:
            frame = cv2.resize(frame, self.__obs_shape, interpolation=cv2.INTER_AREA)
        if self.__gray_scale:
            frame = np.expand_dims(frame, -1)
        return frame

