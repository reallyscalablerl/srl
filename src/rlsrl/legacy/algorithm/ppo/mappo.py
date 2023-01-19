from typing import Optional
from torch.nn import functional as F
import dataclasses
import torch.nn as nn
import torch

from rlsrl.legacy.algorithm import modules
from rlsrl.legacy.algorithm.modules import init_optimizer
from rlsrl.api.trainer import PytorchTrainer, register, TrainerStepResult
from rlsrl.api.policy import Policy
from rlsrl.base.namedarray import recursive_apply


@dataclasses.dataclass
class SampleAnalyzedResult:
    """PPO loss computation requires:
    REMEMBER that you drop the last action.
    1. Old / New action probabilities
    2. State Values
    3. Rewards
    4. Entropy
    """
    old_action_log_probs: torch.Tensor  # For chosen actions only, flattened
    new_action_log_probs: torch.Tensor  # for chosen actions only, flattened
    state_values: torch.Tensor  # len(state_values) = len(new_action_probs) + 1
    entropy: Optional[torch.Tensor] = None


@dataclasses.dataclass
class PPOStepResult:
    advantage: torch.Tensor
    entropy: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    done: torch.Tensor
    clip_ratio: torch.Tensor
    value_targets: torch.Tensor
    denorm_value: Optional[torch.Tensor]


class MultiAgentPPO(PytorchTrainer):
    """Multi Agent Proximal Policy Optimization

    Ref:
        The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
        Available at (https://arxiv.org/pdf/2103.01955.pdf)
    """

    def get_checkpoint(self):
        checkpoint = self.policy.get_checkpoint()
        checkpoint.update({"optimizer_state_dict": self.optimizer.state_dict()})
        return checkpoint

    def load_checkpoint(self, checkpoint, **kwargs):
        if "optimizer_state_dict" in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.load_checkpoint(checkpoint)

    def __init__(self, policy: Policy, **kwargs):
        super().__init__(policy)
        # discount & clip
        self.discount_rate = kwargs.get("discount_rate", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.97)
        self.eps_clip = kwargs.get("eps_clip", 0.2)
        self.clip_value = kwargs.get("clip_value", False)
        self.dual_clip = kwargs.get("dual_clip", True)
        self.c_clip = kwargs.get("c_clip", 3)
        # value tracer
        self.vtrace = kwargs.get("vtrace", False)
        # value loss
        self.value_eps_clip = kwargs.get("value_eps_clip", self.eps_clip)
        self.value_loss_weight = kwargs.get('value_loss_weight', 0.5)
        # entropy
        self.entropy_bonus_weight = kwargs.get('entropy_bonus_weight', 0.01)
        self.entropy_decay_per_steps = kwargs.get("entropy_decay_per_steps", None)
        self.entropy_bonus_decay = kwargs.get("entropy_bonus_decay", 0.99)
        # gradient norm
        self.max_grad_norm = kwargs.get('max_grad_norm')
        # value normalization
        self.popart = kwargs.get('popart', False)
        self.bootstrap_steps = kwargs.get("bootstrap_steps", 1)
        # ppo_epochs: How many updates per sample.
        self.ppo_epochs = kwargs.get("ppo_epochs", 1)

        optimizer_name = kwargs.get('optimizer', 'adam')
        self.optimizer = init_optimizer(self.policy.parameters(), optimizer_name,
                                        kwargs.get('optimizer_config', {}))

        value_loss_name = kwargs.get('value_loss', 'mse')
        self.value_loss_fn = self._init_value_loss_fn(value_loss_name, kwargs.get('value_loss_config', {}))

        self.frames = 0

    def _init_value_loss_fn(self, value_loss_name, value_loss_config):
        value_loss_collection = ['mse', 'huber']
        assert value_loss_name in value_loss_collection, (
            f'Value loss name {value_loss_name}'
            f'does not match any implemented loss functions ({value_loss_collection})')

        if value_loss_name == 'mse':
            value_loss_cls = torch.nn.MSELoss
        elif value_loss_name == 'huber':
            value_loss_cls = torch.nn.HuberLoss
        else:
            raise ValueError(f"Unknown loss function {value_loss_name}")

        value_loss_fn = value_loss_cls(reduction='none', **value_loss_config)

        if self.clip_value:
            value_loss_fn = modules.get_clip_value_loss_fn(value_loss_fn, self.value_eps_clip)

        return value_loss_fn

    def _compute_adv(self, sample, analyzed_result):
        # If Vtrace, importance_ratio.size()[0] == sample_length - 1, but will be cliped to sample_length - bootstrap
        # If Gae, importance_ratio.size()[0] == sample_length - bootstrap
        importance_ratio = (analyzed_result.new_action_log_probs - analyzed_result.old_action_log_probs).exp()
        if self.popart:
            trace_target_value = self.policy.denormalize_value(sample.value)
        else:
            trace_target_value = sample.value

        if self.vtrace:
            adv = modules.vtrace(importance_ratio,
                                 sample.reward[:-1],
                                 trace_target_value,
                                 sample.on_reset[1:],
                                 gamma=torch.tensor(self.discount_rate),
                                 lmbda=torch.tensor(self.gae_lambda))
        else:
            adv = modules.gae_trace(sample.reward[:-1],
                                    trace_target_value,
                                    sample.on_reset[1:],
                                    gamma=torch.tensor(self.discount_rate),
                                    gae_lambda=torch.tensor(self.gae_lambda))
        return adv

    def _compute_loss(self, sample, adv, analyzed_result, done):
        importance_ratio = (analyzed_result.new_action_log_probs - analyzed_result.old_action_log_probs).exp()
        if self.popart:
            trace_target_value = self.policy.denormalize_value(sample.value)
        else:
            trace_target_value = sample.value
        # 2. Remove bootstrapped steps.
        train_episode_length = sample.on_reset.shape[0]

        state_values = analyzed_result.state_values[:train_episode_length]
        target_values = (adv[:train_episode_length] + trace_target_value[:train_episode_length]).detach()
        old_value = sample.value[:train_episode_length]

        importance_ratio = importance_ratio[:train_episode_length]
        adv = adv[:train_episode_length]
        entropy = analyzed_result.entropy[:train_episode_length]
        done = done[:train_episode_length]

        # 3. critic loss
        denorm_target_values = None
        if self.popart:
            self.policy.update_popart(target_values)
            denorm_target_values = target_values
            target_values = self.policy.normalize_value(target_values)

        if self.clip_value:
            assert sample.value is not None
            value_loss = self.value_loss_fn(state_values, old_value, target_values)
        else:
            value_loss = self.value_loss_fn(state_values, target_values)
        value_loss = value_loss.mean()

        # 4. actor loss
        norm_adv = modules.masked_normalization(adv, 1 - done)
        surrogate1: torch.Tensor = importance_ratio * norm_adv
        surrogate2: torch.Tensor = torch.clamp(importance_ratio, 1 - self.eps_clip,
                                               1 + self.eps_clip) * norm_adv
        if self.dual_clip:
            surrogate3: torch.Tensor = -torch.sign(norm_adv) * self.c_clip * norm_adv
            policy_loss = -(torch.max(torch.min(surrogate1, surrogate2), surrogate3) *
                            (1 - done)).sum() / (1 - done).sum()
        else:
            policy_loss = -(torch.min(surrogate1, surrogate2) * (1 - done)).sum() / (1 - done).sum()

        entropy_loss = -(entropy * (1 - done)).sum() / (1 - done).sum()

        # final loss of clipped objective PPO
        loss = policy_loss + self.value_loss_weight * value_loss + self.entropy_bonus_weight * entropy_loss

        return loss, PPOStepResult(advantage=adv,
                                   entropy=entropy,
                                   policy_loss=policy_loss,
                                   value_loss=value_loss,
                                   done=done,
                                   clip_ratio=(surrogate2 < surrogate1).to(dtype=torch.float32).mean(),
                                   value_targets=target_values,
                                   denorm_value=denorm_target_values)

    def step(self, sample):
        tensor_sample = recursive_apply(
            sample, lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.policy.device))
        for _ in range(self.ppo_epochs):
            tail_len = 1 if (self.vtrace and sample.adv is None) else self.bootstrap_steps
            analyzed_result = self.policy.analyze(tensor_sample[:-tail_len], target="ppo")
            if sample.adv is None:
                dims = len(tensor_sample.value.shape)
                tensor_sample.adv = F.pad(self._compute_adv(tensor_sample, analyzed_result),
                                          (0,) * (dims * 2 - 1) + (1,))
                sample.adv = tensor_sample.adv.cpu().numpy()
            done = tensor_sample.on_reset[1:]
            loss, step_result = self._compute_loss(tensor_sample[:-self.bootstrap_steps], tensor_sample.adv,
                                                   analyzed_result, done)
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.inc_version()

        self.frames += len(sample) - self.bootstrap_steps

        # entropy bonus coefficient decay
        if self.entropy_decay_per_steps and self.policy.version % self.entropy_decay_per_steps == 0:
            self.entropy_bonus_weight *= self.entropy_bonus_decay

        # Logging
        elapsed_episodes = sample.info_mask.sum()
        if elapsed_episodes == 0:
            info = {}
        else:
            info = recursive_apply(sample.info * sample.info_mask, lambda x: x.sum()) / elapsed_episodes
        # TODO: len(sample) equals sample_steps + bootstrap_steps, which is incorrect for frames logging
        # Reference for the following workaround:
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
        stats = dict(
            **{
                f.name: getattr(step_result, f.name).detach().mean().item()
                for f in dataclasses.fields(step_result) if getattr(step_result, f.name) is not None
            }, **info)
        return TrainerStepResult(stats=stats, step=self.policy.version)


register('mappo', MultiAgentPPO)
