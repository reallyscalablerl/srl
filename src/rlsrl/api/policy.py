from abc import ABC
from pydoc import visiblename
from typing import List, Union
import logging
import numpy as np
import torch.distributed as dist
import torch.nn

from rlsrl.base.namedarray import NamedArray
from rlsrl.base.gpu_utils import get_gpu_device
import rlsrl.api.config as config
import rlsrl.api.environment as environment


class PolicyState:
    pass


class AnalyzedResult:
    pass


logger = logging.getLogger("Policy")


class RolloutResult(NamedArray):

    def __init__(self,
                 action: environment.Action,
                 policy_state: PolicyState = None,
                 log_probs: np.ndarray = None,
                 value: np.ndarray = None,
                 analyzed_result: AnalyzedResult = None,
                 client_id: np.ndarray = None,
                 request_id: np.ndarray = None,
                 received_time: np.ndarray = None,
                 policy_name: np.ndarray = None,
                 policy_version_steps: np.ndarray = None,
                 **kwargs):
        super(RolloutResult, self).__init__(action=action,
                                            policy_state=policy_state,
                                            log_probs=log_probs,
                                            value=value,
                                            analyzed_result=analyzed_result,
                                            client_id=client_id,
                                            request_id=request_id,
                                            received_time=received_time,
                                            policy_name=policy_name,
                                            policy_version_steps=policy_version_steps,
                                            **kwargs)


class RolloutRequest(NamedArray):

    def __init__(self,
                 obs: NamedArray,
                 policy_state: PolicyState = None,
                 is_evaluation: np.ndarray = np.array([False], dtype=np.uint8),
                 on_reset: np.ndarray = np.array([False], dtype=np.uint8),
                 step_count: np.ndarray = np.array([-1], dtype=np.int32),
                 client_id: np.ndarray = np.array([-1], dtype=np.int32),
                 request_id: np.ndarray = np.array([-1], dtype=np.int64),
                 received_time: np.ndarray = np.array([-1], dtype=np.int64),
                 **kwargs):
        super(RolloutRequest, self).__init__(obs=obs,
                                             policy_state=policy_state,
                                             is_evaluation=is_evaluation,
                                             on_reset=on_reset,
                                             step_count=step_count,
                                             client_id=client_id,
                                             request_id=request_id,
                                             received_time=received_time,
                                             **kwargs)


class Policy:

    def __init__(self):
        devices = get_gpu_device()
        logger.info(f"{devices}")
        if len(devices) == 1:
            [self.device] = devices
        elif dist.is_initialized():
            self.device = devices[dist.get_rank()]
        else:
            raise RuntimeError(f'Policy cannot resolve a specific device! Available devices are {devices}.')

    @property
    def default_policy_state(self):
        """Default value of policy state.
        """
        raise NotImplementedError()

    @property
    def version(self) -> int:
        """Current version of the policy.
        """
        raise NotImplementedError()

    @property
    def net(self) -> Union[List[torch.nn.Module], torch.nn.Module]:
        """Neural Network of the policy.
        """
        raise NotImplementedError()

    def analyze(self, sample, target, **kwargs):
        """ Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies.
        Args:
            sample (namedarraytuple): Customized namedarraytuple containing
                all data required for loss computation.
            target (string): purpose of the analysis, specified by trainers..
        Returns:
            training_seg (namedarraytuple): Data generated for loss computation.
        """
        raise NotImplementedError()

    def reanalyze(self, sample, target, **kwargs):
        """Reanalyze the sample with the current parameters.
        Args:
            sample (namedarraytuple): sample to be reanalyzed.
            target (string): purpose of the analysis, usually an algorithm.

        Returns:
            Reanalyzed sample (algorithm.trainer.SampleBatch).
        """
        raise NotImplementedError()

    def rollout(self, requests: RolloutRequest, **kwargs) -> RolloutResult:
        """ Generate actions (and rnn hidden states) during evaluation.
        Args:
            requests: All request received from actor generated by env.step.
        Returns:
            RolloutResult: Rollout results to be distributed (namedarray).
        """
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def get_checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def train_mode(self):
        raise NotImplementedError()

    def eval_mode(self):
        raise NotImplementedError()

    def inc_version(self):
        """Increase the policy version.
        """
        raise NotImplementedError()

    def distributed(self):
        """Make the policy distributed.
        """
        raise NotImplementedError()


class SingleModelPytorchPolicy(Policy, ABC):

    def __init__(self, neural_network: torch.nn.Module):
        """Initialization method of SingleModelPytorchPolicy
        Args:
            neural_network: nn.module.

        Note:
            After initialization, access the neural network from property policy.net
        """
        super(SingleModelPytorchPolicy, self).__init__()
        self._net: torch.nn.Module = neural_network.to(self.device)
        self._version = -1

    def distributed(self, **kwargs):
        """Wrapper of Pytorch DDP method.
        Ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
        """
        from torch.nn.parallel import DistributedDataParallel as DDP
        # import DDP globally will cause incompatible issue between CUDA and multiprocessing.
        if dist.is_initialized():
            if self.device == 'cpu':
                self._net = DDP(self._net, **kwargs)
            else:
                self._net = DDP(self._net, device_ids=[self.device], output_device=self.device, **kwargs)

    @property
    def version(self) -> int:
        """In single model policy, version tells the the number of trainer steps have been performed on the mode.
        Specially, -1 means the parameters are from arbitrary initialization.
        0 means the first version that is pushed by the trainer
        """
        return self._version

    @property
    def net(self):
        return self._net

    def inc_version(self):
        self._version += 1

    def parameters(self):
        return self._net.parameters(recurse=True)

    def train_mode(self):
        self._net.train()

    def eval_mode(self):
        self._net.eval()

    def load_checkpoint(self, checkpoint):
        """Load a checkpoint.
        If "steps" is missing in the checkpoint. We assume that the checkpoint is from a pretrained model. And
        set version to 0. So that the trainer side won't ignore the sample generated by this version.
        """
        self._version = checkpoint.get("steps", 0)
        self._net.load_state_dict(checkpoint["state_dict"])

    def get_checkpoint(self):
        if dist.is_initialized():
            return {
                "steps": self._version,
                "state_dict": {k.replace("module.", ""): v.cpu()
                               for k, v in self._net.state_dict().items()}
            }
        else:
            return {
                "steps": self._version,
                "state_dict": {k: v.cpu()
                               for k, v in self._net.state_dict().items()}
            }


ALL_POLICY_CLASSES = {}


def register(name, policy_class):
    ALL_POLICY_CLASSES[name] = policy_class


def make(cfg: Union[str, config.Policy]) -> Policy:
    if isinstance(cfg, str):
        cfg = config.Policy(type_=cfg)
    cls = ALL_POLICY_CLASSES[cfg.type_]
    return cls(**cfg.args)
