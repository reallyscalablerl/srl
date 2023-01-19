from abc import ABC
from typing import Union, Dict, Optional
import dataclasses
import numpy as np
import torch
import torch.distributed as dist

from rlsrl.base.namedarray import NamedArray
from rlsrl.api.environment import Action
import rlsrl.api.policy
import rlsrl.api.config


class SampleBatch(NamedArray):
    # `SampleBatch` is the general data structure and will be used for ALL the algorithms we implement.
    # There could be some entries that may not be used by a specific algorithm,
    # e.g. log_probs and value are not used by DQN, which will be left as None.

    # `obs`, `on_reset`, `action`, `reward`, and `info` are environment-related data entries.
    # `obs` and `on_reset` can be obtained once environment step is preformed.

    def __init__(
            self,
            obs: NamedArray,
            on_reset: np.ndarray = None,
            truncated: np.ndarray = None,

            # `action` and `reward` can be obtained when the inference is done.
            action: Action = None,
            reward: np.ndarray = None,

            # Currently we assume info contains all the information we want to gather in an environment.
            # It is NOT agent-specific and should include summary information of ALL agents.
            info: NamedArray = None,

            # `info_mask` is recorded for correctly recording summary info when there are
            # multiple agents and some agents may die before an episode is done.
            info_mask: np.ndarray = None,

            # In some cases we may need Policy State. e.g. Partial Trajectory, Mu-Zero w/o reanalyze.
            policy_state: rlsrl.api.policy.PolicyState = None,

            # The followings are algorithm-related data entries.
            # PPO
            log_probs: np.ndarray = None,
            value: np.ndarray = None,
            adv: np.
        ndarray = None,  # Actor-Critic algorithm will use this field to avoid recomputing the advantage.

            # TODO: move PPO-related entries into `analyzed_result` field.
            # `analyzed_result` records algorithm-related analyzed results
        analyzed_result: rlsrl.api.policy.AnalyzedResult = None,

            # Metadata
            policy_name: np.ndarray = None,
            policy_version_steps: np.ndarray = None,
            **kwargs):
        super(SampleBatch, self).__init__(
            obs=obs,
            on_reset=on_reset,
            truncated=truncated,
            action=action,
            reward=reward,
            info=info,
            info_mask=info_mask,
            policy_state=policy_state,
            log_probs=log_probs,
            value=value,
            adv=adv,
            analyzed_result=analyzed_result,
            policy_name=policy_name,
            policy_version_steps=policy_version_steps,
        )


@dataclasses.dataclass
class TrainerStepResult:
    stats: Dict  # Stats to be logged.
    step: int  # current step count of trainer.
    agree_pushing: Optional[bool] = True  # whether agree to push parameters


class Trainer:

    @property
    def policy(self) -> rlsrl.api.policy.Policy:
        """Running policy of the trainer.
        """
        raise NotImplementedError()

    def step(self, samples: SampleBatch) -> TrainerStepResult:
        """Advances one training step given samples collected by actor workers.

        Example code:
          ...
          some_data = self.policy.analyze(sample)
          loss = loss_fn(some_data, sample)
          self.optimizer.zero_grad()
          loss.backward()
          ...
          self.optimizer.step()
          ...

        Args:
            samples (SampleBatch): A batch of data required for training.

        Returns:
            TrainerStepResult: Entry to be logged by trainer worker.
        """
        raise NotImplementedError()

    def distributed(self, **kwargs):
        """Make the trainer distributed.
        """
        raise NotImplementedError()

    def get_checkpoint(self, *args, **kwargs):
        """Get checkpoint of the model, which typically includes:
        1. Policy state (e.g. neural network parameter).
        2. Optimizer state.
        Return:
            checkpoint to be saved.
        """
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint, **kwargs):
        """Load a saved checkpoint.
        Args:
            checkpoint: checkpoint to be loaded.
        """
        raise NotImplementedError()


class PytorchTrainer(Trainer, ABC):

    @property
    def policy(self) -> rlsrl.api.policy.Policy:
        return self._policy

    def __init__(self, policy: rlsrl.api.policy.Policy):
        """Initialization method of Pytorch Trainer.
        Args:
            policy: Policy to be trained.

        Note:
            After initialization, access policy from property trainer.policy
        """
        if policy.device != "cpu":
            torch.cuda.set_device(policy.device)
            torch.cuda.empty_cache()
        self._policy = policy

    def distributed(self, rank, world_size, init_method, **kwargs):
        is_gpu_process = all([
            torch.distributed.is_nccl_available(),
            torch.cuda.is_available(),
            self.policy.device != "cpu",
        ])
        dist.init_process_group(backend="nccl" if is_gpu_process else "gloo",
                                init_method=init_method,
                                rank=rank,
                                world_size=world_size)
        self.policy.distributed()

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()


ALL_TRAINER_CLASSES = {}


def register(name, trainer_class):
    ALL_TRAINER_CLASSES[name] = trainer_class


def make(cfg: Union[str, rlsrl.api.config.Trainer], policy_cfg: Union[str, rlsrl.api.config.Policy]) -> Trainer:
    if isinstance(cfg, str):
        cfg = rlsrl.api.config.Trainer(type_=cfg)
    if isinstance(policy_cfg, str):
        policy_cfg = rlsrl.api.config.Policy(type_=policy_cfg)
    cls = ALL_TRAINER_CLASSES[cfg.type_]
    policy = rlsrl.api.policy.make(policy_cfg)
    policy.train_mode()  # To be explicit.
    return cls(policy=policy, **cfg.args)
