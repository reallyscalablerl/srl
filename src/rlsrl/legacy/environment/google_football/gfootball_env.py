from typing import List
from gfootball.env import create_environment
import copy
import getpass
import numpy as np
import os
import shutil

import rlsrl.api.environment
import rlsrl.api.env_utils
import rlsrl.base.user

_HAS_DISPLAY = len(os.environ.get("DISPLAY", "").strip()) > 0


class FootballEnvironment(rlsrl.api.environment.Environment):
    """A wrapper of google football environment
    """

    def __copy_videos(self):
        for file in os.listdir(self.__tmp_log_dir):
            shutil.move(os.path.join(self.__tmp_log_dir, file), os.path.join(self.__log_dir, file))

    def __del__(self):
        self.__env.close()
        self.__copy_videos()

    def set_curriculum_stage(self, stage_name: str):
        self.__copy_videos()
        if stage_name is None:
            raise ValueError()
        if self.__env_name == stage_name:
            return
        self.__env_name = stage_name
        self.__log_dir = os.path.join(os.path.dirname(self.__log_dir), self.__env_name)
        os.makedirs(self.__log_dir, exist_ok=True)

        self.__tmp_log_dir = rlsrl.base.user.get_random_tmp()
        kwargs = dict(env_name=stage_name,
                      representation=self.__representation,
                      number_of_left_players_agent_controls=self.control_left,
                      number_of_right_players_agent_controls=self.control_right,
                      write_video=self.__write_video,
                      write_full_episode_dumps=self.__write_full_episode_dumps,
                      logdir=self.__tmp_log_dir,
                      dump_frequency=self.__dump_frequency,
                      render=False)
        self.__env.close()
        self.__env = create_environment(**kwargs)

    def __init__(self, seed=None, share_reward=False, **kwargs):
        self.__env_name = kwargs["env_name"]
        self.__representation = kwargs["representation"]
        self.control_left = kwargs.get("number_of_left_players_agent_controls", 1)
        self.control_right = kwargs.get("number_of_right_players_agent_controls", 0)
        self.__render = kwargs.get("render", False)
        self.__write_video = kwargs.get("write_video", False)
        self.__write_full_episode_dumps = kwargs.get("write_full_episode_dumps", False)
        self.__dump_frequency = kwargs.get("dump_frequency", 1)
        self.__log_dir = os.path.join(kwargs.get("logdir", f"/home/{getpass.getuser()}/fb_replay"),
                                      self.__env_name)
        os.makedirs(self.__log_dir, exist_ok=True)
        if "logdir" in kwargs:
            kwargs.pop("logdir")
        self.__tmp_log_dir = rlsrl.base.user.get_random_tmp()
        self.__env = create_environment(**kwargs, logdir=self.__tmp_log_dir)
        self.__share_reward = share_reward
        self.__space = rlsrl.api.env_utils.DiscreteActionSpace(self.__env.action_space[0])
        self.__step_count = np.zeros(1, dtype=np.int32)
        self.__episode_return = np.zeros((self.agent_count, 1), dtype=np.float32)

    @property
    def agent_count(self) -> int:
        return self.control_left + self.control_right

    @property
    def observation_spaces(self) -> List[dict]:
        return [{"minimap": (72, 96, 4)}]

    @property
    def action_spaces(self) -> List[rlsrl.api.environment.ActionSpace]:
        return [self.__space for _ in range(self.agent_count)]

    def reset(self):
        obs = self.__env.reset()
        self.__step_count[:] = self.__episode_return[:] = 0
        obs, _ = self.__post_process_obs_and_rew(obs, np.zeros(self.agent_count))
        return [
            rlsrl.api.environment.StepResult(obs=dict(obs=obs[i]),
                                       reward=np.array([0.0], dtype=np.float64),
                                       done=np.array([False], dtype=np.uint8),
                                       info=dict()) for i in range(self.agent_count)
        ]

    def __post_process_obs_and_rew(self, obs, reward):
        if self.agent_count == 1:
            obs = obs[np.newaxis, :]
            reward = [reward]
        if self.__representation == "extracted":
            obs = np.swapaxes(obs, 1, 3)
        if self.__representation in ("simple115", "simple115v2"):
            obs[obs == -1] = 0
        if self.__share_reward:
            left_reward = np.mean(reward[:self.control_left])
            right_reward = np.mean(reward[self.control_left:])
            reward = np.array([left_reward] * self.control_left + [right_reward] * self.control_right)
        return obs, reward

    def step(self, actions: List[rlsrl.api.env_utils.DiscreteAction]) -> List[rlsrl.api.environment.StepResult]:
        assert len(actions) == self.agent_count, len(actions)
        obs, reward, done, info = self.__env.step([a.x.item() for a in actions])
        obs, reward = self.__post_process_obs_and_rew(obs, reward)
        self.__step_count += 1
        self.__episode_return += reward[:, np.newaxis]
        return [
            rlsrl.api.environment.StepResult(obs=dict(obs=obs[i]),
                                       reward=np.array([reward[i]], dtype=np.float64),
                                       done=np.array([done], dtype=np.uint8),
                                       info=dict(episode_length=copy.deepcopy(self.__step_count),
                                                 episode_return=copy.deepcopy(self.__episode_return[i])))
            for i in range(self.agent_count)
        ]

    def render(self) -> None:
        self.__env.render()


