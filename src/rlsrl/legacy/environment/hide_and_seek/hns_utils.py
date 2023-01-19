from typing import Dict
import gym
import numpy as np

from rlsrl.base.namedarray import namedarray
from rlsrl.api import environment

EPISODE_INFO_FIELDS = [
    "max_box_move_prep", "max_box_move", "num_box_lock_prep", "num_box_lock", "max_ramp_move_prep",
    "max_ramp_move", "num_ramp_lock_prep", "num_ramp_lock", "episode_return", "hider_return", "seeker_return"
]


def hns_action(data_cls):

    def __eq__(self, other):
        assert isinstance(other, HNSAction), \
            "Cannot compare HNSAction to object of class{}".format(other.__class__.__name__)
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    @property
    def key(self):
        return f"{self.move_x}-{self.move_y}-{self.move_z}-{self.lock}-{self.grab}"

    data_cls.__eq__ = __eq__
    data_cls.__hash__ = __hash__
    data_cls.key = key
    return data_cls


@hns_action
@namedarray
class HNSAction(environment.Action):
    move_x: np.ndarray  # Discrete(11)
    move_y: np.ndarray  # Discrete(11)
    move_z: np.ndarray  # Discrete(11)
    lock: np.ndarray  # Discrete(2)
    grab: np.ndarray  # Discrete(2)


class HNSActionSpace(environment.ActionSpace):

    def __init__(self, spaces: Dict[str, gym.Space]):
        self.__spaces = spaces

    @property
    def n(self):
        return {k: x.n for k, x in self.__spaces.items()}

    def sample(self) -> HNSAction:
        return HNSAction(*[np.array([x.sample()]) for x in self.__spaces.values()])


#
# Below are deprecated definitions of HnS observation/episode_info. Kept here as reminder.
#
# @namedarray
# class HNSObservation(env_base.Observation):
#     observation_self: np.ndarray
#     # other agents
#     agent_qpos_qvel: np.ndarray
#     mask_aa_obs: np.ndarray
#     # boxes
#     box_obs: np.ndarray
#     mask_ab_obs: np.ndarray
#     # ramps
#     ramp_obs: np.ndarray = None
#     mask_ar_obs: np.ndarray = None
#     # food
#     food_obs: np.ndarray = None
#     mask_af_obs: np.ndarray = None
#     # lidar
#     lidar: np.ndarray = None
#     # spoof masks
#     mask_aa_obs_spoof: np.ndarray = None
#     mask_ab_obs_spoof: np.ndarray = None
#     mask_ar_obs_spoof: np.ndarray = None
#     mask_af_obs_spoof: np.ndarray = None

# @namedarray
# class HNSEpisodeInfo(env_base.EpisodeInfo):
#     max_box_move_prep: np.ndarray = np.zeros(1, dtype=np.float32)
#     max_box_move: np.ndarray = np.zeros(1, dtype=np.float32)
#     num_box_lock_prep: np.ndarray = np.zeros(1, dtype=np.float32)
#     num_box_lock: np.ndarray = np.zeros(1, dtype=np.float32)
#     max_ramp_move_prep: np.ndarray = np.zeros(1, dtype=np.float32)
#     max_ramp_move: np.ndarray = np.zeros(1, dtype=np.float32)
#     num_ramp_lock_prep: np.ndarray = np.zeros(1, dtype=np.float32)
#     num_ramp_lock: np.ndarray = np.zeros(1, dtype=np.float32)
#
#     episode_return: np.ndarray = np.zeros(1, dtype=np.float32)
#     hider_return: np.ndarray = np.zeros(1, dtype=np.float32)
#     seeker_return: np.ndarray = np.zeros(1, dtype=np.float32)
