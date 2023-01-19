try:
    import mujoco_py
except Exception as e:
    raise ModuleNotFoundError('Run python with "MUJOCO_PY_MUJOCO_PATH=/local/pkgs/mujoco210 '
                              'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/local/pkgs/mujoco210/bin:/usr/lib/nvidia" '
                              'to locate mujoco_py.')
from typing import List
import gym
import logging
import numpy as np

import rlsrl.api.environment as env_base
import rlsrl.legacy.environment.hide_and_seek.hns_utils as hns_utils
import rlsrl.legacy.environment.hide_and_seek.scenarios as scenarios

logger = logging.getLogger("env-hns")


class HideAndSeekEnvironment(env_base.Environment):

    def __init__(self, scenario_name, **kwargs):
        self.__scenario = scenario_name
        self.__n_agents = kwargs.get('n_agents', kwargs['n_seekers'] + kwargs['n_hiders'])

        if scenario_name == "box_locking":
            self.__env = scenarios.box_locking.make_env(**kwargs)
        elif scenario_name == "blueprint_construction":
            self.__env = scenarios.blueprint_construction.make_env(**kwargs)
        elif scenario_name == "hide_and_seek":
            self.n_seekers = kwargs['n_seekers']
            self.n_hiders = kwargs['n_hiders']
            self.__env = scenarios.hide_and_seek.make_env(**kwargs)
        else:
            raise NotImplementedError

        self.__hider_return = self.__seeker_return = 0
        self.__episode_return = 0

        self.__obs_space = {}
        for key in self.__env.observation_space.spaces.keys():
            space = tuple(self.__env.observation_space[key].shape)
            if 'lidar' in key:
                self.__obs_space[key] = (space[1], space[0])
            else:
                self.__obs_space[key] = space

        self.__act_space = {
            'move_x': gym.spaces.Discrete(11),
            'move_y': gym.spaces.Discrete(11),
            'move_z': gym.spaces.Discrete(11),
            'lock': gym.spaces.Discrete(2),
            'grab': gym.spaces.Discrete(2)
        }
        self.seed()

    @property
    def agent_count(self) -> int:
        return self.__n_agents

    @property
    def observation_spaces(self) -> List[dict]:
        return [self.__obs_space for _ in range(self.agent_count)]

    @property
    def action_spaces(self) -> List[dict]:
        return [hns_utils.HNSActionSpace(self.__act_space) for _ in range(self.agent_count)]

    def reset(self) -> List[env_base.StepResult]:
        self.__hider_return = self.__seeker_return = 0
        self.__episode_return = 0
        obs = self.__env.reset()
        if 'lidar' in obs.keys():
            # transpose lidar signal such that channel is the first dim
            obs['lidar'] = np.transpose(obs['lidar'], (0, 2, 1))
        return [
            env_base.StepResult(obs={k: v[i]
                                     for k, v in obs.items()},
                                reward=np.array([0.0], dtype=np.float32),
                                done=np.array([False], dtype=np.uint8),
                                info=dict()) for i in range(self.agent_count)
        ]

    def step(self, actions: List[hns_utils.HNSAction]) -> List[env_base.StepResult]:
        action_movement = np.array(
            [[int(action.move_x.item()),
              int(action.move_y.item()),
              int(action.move_z.item())] for action in actions])
        action_pull = np.array([int(action.grab.item()) for action in actions])
        action_glueall = np.array([int(action.lock.item()) for action in actions])
        actions_env = {
            'action_movement': action_movement,
            'action_pull': action_pull,
            'action_glueall': action_glueall
        }

        obs, rewards, done, info = self.__env.step(actions_env)
        dones = [np.array([done], dtype=np.uint8) for _ in range(self.agent_count)]
        rewards = [np.array([reward], dtype=np.float32) for reward in rewards]
        if 'lidar' in obs.keys():
            # transpose lidar signal such that channel is the first dim
            obs['lidar'] = np.transpose(obs['lidar'], (0, 2, 1))
        episode_infos = [{
            k: np.array([float(info[k])], dtype=np.float32) if k in info else np.zeros(1, dtype=np.float32)
            for k in hns_utils.EPISODE_INFO_FIELDS
        } for _ in range(self.agent_count)]
        if self.__scenario == 'hide_and_seek':
            self.__hider_return += rewards[0]
            self.__seeker_return += rewards[self.agent_count - 1]
            for info in episode_infos:
                info['hider_return'][:] = self.__hider_return
                info['seeker_return'][:] = self.__seeker_return
        else:
            self.__episode_return += rewards[0]
            for info in episode_infos:
                info['episode_return'][:] = self.__episode_return

        agent_obs = [{k: v[i] for k, v in obs.items()} for i in range(self.agent_count)]
        return [env_base.StepResult(*x) for x in zip(agent_obs, rewards, dones, episode_infos)]

    def render(self):
        self.__env.render()

    def seed(self, seed=None):
        if seed is None:
            self.__env.seed(1)
        else:
            self.__env.seed(seed)
        return seed

    def close(self):
        self.__env.close()

