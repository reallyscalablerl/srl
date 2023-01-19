"""Abstraction of the RL environment and related concepts.

This is basically a clone of the gym interface. The reasons of replicating are:
- Allow easy changing of APIs when necessary.
- Avoid hard dependency on gym.
"""
from typing import List, Union, Dict
import dataclasses
import numpy as np

from rlsrl.api import config as config


class Action:
    pass


class ActionSpace:

    def sample(self, *args, **kwargs) -> Action:
        raise NotImplementedError()


@dataclasses.dataclass
class StepResult:
    """Step result for a single agent. In multi-agent scenario, env.step() essentially returns
    List[StepResult].
    """
    obs: Dict
    reward: np.ndarray
    done: np.ndarray
    info: Dict


class Environment:

    @property
    def agent_count(self) -> int:
        raise NotImplementedError()

    @property
    def observation_spaces(self) -> List[dict]:
        """Return a list of observation spaces for all agents.

        Each element in self.observation_spaces is a Dict, which contains
        shapes of observation entries specified by the key.
        Example:
        -------------------------------------------------------------
        self.observation_spaces = [{
            'observation_self': (10, ),
            'box_obs': (9, 15),
        }, {
            'observation_self': (20, ),
            'box_obs': (9, 15),
        }]
        -------------------------------------------------------------
        Observation spaces of different agents can be different.
        In this case, policies *MUST* be *DIFFERENT*
        among agents with different observation dimension.
        """
        raise NotImplementedError()

    @property
    def action_spaces(self) -> List[ActionSpace]:
        """Return a list of action spaces for all agents.

        Each element in self.action_spaces is an instance of
        env_base.ActionSpace, which is basically a wrapped Dict.
        The Dict contains shapes of action entries specified by the key.
        **We force each action entry to be either gym.spaces.Discrete
        or gym.spaces.Box.**
        Example:
        -------------------------------------------------------------
        self.action_spaces = [
            SomeEnvActionSpace(dict(move_x=Discrete(10), move_y=Discrete(10), cursur=Box(2))),
            SomeEnvActionSpace(dict(cursur=Box(2)))
        ]
        -------------------------------------------------------------
        Action spaces of different agents can be different.
        In this case, policies *MUST* be *DIFFERENT*
        among agents with different action output.
        """
        raise NotImplementedError()

    def reset(self) -> List[StepResult]:
        """Reset the environment, and returns a list of step results for all agents.

        Returns:
            List[StepResult]: StepResult with valid Observations only.
        """
        raise NotImplementedError()

    def step(self, actions: List[Action]) -> List[StepResult]:
        """ Consume actions and advance one env step.

        Args:
            actions (List[Action]): Actions of all agents.

        Returns:
            step result (StepResult): An object with 4 members:
            - obs (namedarray): It contains observations, available actions, masks, etc.
            - reward (numpy.ndarray): A numpy array with shape [1].
            - done (numpy.ndarray): A numpy array with shape [1],
                indicating whether an episode is done or an agent is dead.
            - info (namedarray): Customized namedarray recording required summary infos.
        """
        raise NotImplementedError()

    def render(self) -> None:
        pass

    def seed(self, seed):
        """Set a random seed for the environment.

        Args:
            seed (Any): The seed to be set. It could be int,
            str or any other types depending on the implementation of
            the specific environment. Defaults to None.

        Returns:
            Any: The new seed.
        """
        raise NotImplementedError()

    def set_curriculum_stage(self, stage_name: str):
        """Set the environment to be in a certain stage.
        Args:
            stage_name: name of the stage to be set.
        """
        raise NotImplementedError()


ALL_ENVIRONMENT_CLASSES = {}
ALL_ENVIRONMENT_MODULES = {}
ALL_AUGMENTER_CLASSES = {}


def register(name, env_class, module_name=None):
    ALL_ENVIRONMENT_CLASSES[name] = env_class
    if module_name:
        ALL_ENVIRONMENT_MODULES[name] = module_name


def register_relabler(name, relabeler_class):
    ALL_AUGMENTER_CLASSES[name] = relabeler_class


def make(cfg: Union[str, config.Environment]) -> Environment:
    env_type_ = cfg if isinstance(cfg, str) else cfg.type_
    if isinstance(cfg, str):
        cfg = config.Environment(type_=cfg)

    if not env_type_ in ALL_ENVIRONMENT_MODULES:
        cls = ALL_ENVIRONMENT_CLASSES[env_type_]
        assert type(cls) == type, "If module name is not provided, registered env_class should be a class, not str."
    else:
        import importlib
        module_name = ALL_ENVIRONMENT_MODULES[env_type_]
        class_name = ALL_ENVIRONMENT_CLASSES[env_type_]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    
    return cls(**cfg.args)
