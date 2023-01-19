import dataclasses
import torch.nn as nn
import torch

from rlsrl.api.trainer import PytorchTrainer, register, TrainerStepResult
from rlsrl.api.policy import Policy
from rlsrl.base.namedarray import recursive_apply


@dataclasses.dataclass
class SampleAnalyzedResult:
    """QMix loss computation requires:
    REMEMBER that you drop the last action.
    1. Current and target Qtot
    2. Rewards
    """
    q_tot: torch.Tensor  # (T, B, 1) q_tot value of training policy
    target_q_tot: torch.Tensor  # (T, B, 1) q_tot value of target policy
    reward: torch.Tensor  # (T, B, 1)


class QMix(PytorchTrainer):
    """QMix

    Ref: 
        Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
        Available at (http://arxiv.org/abs/1803.11485)
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

        super(QMix, self).__init__(policy)
        # discount
        self.gamma = kwargs.get("gamma", 0.99)
        # soft update
        self.use_soft_update = kwargs.get("use_soft_update", True)
        self.tau = kwargs.get("tau", 0.005)
        # hard update
        self.hard_update_interval = kwargs.get("hard_update_interval", 200)
        # grad norm
        self.use_max_grad_norm = kwargs.get("use_max_grad_norm", True)
        self.max_grad_norm = kwargs.get("max_grad_norm", 10.0)
        # value normalization
        self.use_popart = kwargs.get('use_popart', True)

        optimizer_name = kwargs.get('optimizer', 'adam')
        self.optimizer = self._init_optimizer(
            optimizer_name, kwargs.get('optimizer_config', dict(lr=5e-4, eps=1e-5, weight_decay=0.)))

        value_loss_name = kwargs.get('value_loss', 'mse')
        self.value_loss_fn = self._init_value_loss_fn(value_loss_name, kwargs.get('value_loss_config', {}))

        self.last_hard_update_step = 0
        self.hard_update_count = 0
        self.steps = 0
        self.frames = 0
        self.total_timesteps = 0

    def _init_optimizer(self, optimizer_name, optimizer_config):
        optimizer_collection = ['adam', 'rmsprop', 'sgd', 'adamw']
        assert optimizer_name in optimizer_collection, (
            f'Optimizer name {optimizer_name} '
            f'does not match any implemented optimizers ({optimizer_collection}).')

        if optimizer_name == 'adam':
            optimizer_fn = torch.optim.Adam
        elif optimizer_name == 'rmsprop':
            optimizer_fn = torch.optim.RMSprop
        elif optimizer_name == 'sgd':
            optimizer_fn = torch.optim.SGD
        elif optimizer_name == 'adamw':
            optimizer_fn = torch.optim.AdamW
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")

        optim = optimizer_fn(self.policy.parameters()["train"], **optimizer_config)
        return optim

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

        return value_loss_fn

    def step(self, sample):
        stats = {}
        analyzed_result: SampleAnalyzedResult = self.policy.analyze(sample)

        done = torch.from_numpy(sample.on_reset[1:]).to(analyzed_result.reward).mean(dim=2)

        train_q_tot = analyzed_result.q_tot[:-1]
        target_q_tot = analyzed_result.target_q_tot[1:]

        if self.use_popart:
            target_q_tot = self.policy.denormalize_target_value(target_q_tot)

        target_q_tot = analyzed_result.reward[:-1] + self.gamma * (1. - done[1:]) * target_q_tot

        if self.use_popart:
            target_q_tot = self.policy.normalize_value(target_q_tot)

        value_loss = self.value_loss_fn(train_q_tot, target_q_tot)
        value_loss = value_loss * (1. - done[:-1])
        value_loss = value_loss.sum() / (1. - done[:-1]).sum()

        if self.use_popart:
            train_q_tot = self.policy.denormalize_value(train_q_tot)
            target_q_tot = self.policy.denormalize_value(target_q_tot)
            self.policy.update_popart(target_q_tot)

        self.optimizer.zero_grad()
        value_loss.backward()
        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters()["train"], self.max_grad_norm)
        self.optimizer.step()

        self.total_timesteps += (1. - done[:-1]).sum().item()

        if self.use_soft_update:
            self.policy.soft_target_update(self.tau)
        else:
            if (self.steps - self.last_hard_update_step) / self.hard_update_interval >= 1.:
                self.policy.hard_target_update()
                self.last_hard_update_step = self.steps
                self.hard_update_count += 1
        self.steps += 1
        self.frames += len(sample)
        self.policy.inc_version()
        stats.update(
            dict(value_loss=value_loss.item(),
                 grad_norm=grad_norm.item(),
                 train_q_tot=train_q_tot.mean().item(),
                 target_q_tot=target_q_tot.mean().item(),
                 epsilon=sample.policy_state.epsilon.mean(),
                 frames=self.frames,
                 hard_update_count=self.hard_update_count,
                 total_timesteps=self.total_timesteps))

        # Logging
        elapsed_episodes = sample.info_mask.sum()
        if elapsed_episodes == 0:
            info = {}
        else:
            info = recursive_apply(sample.info * sample.info_mask, lambda x: x.sum()) / elapsed_episodes
        stats.update(
            dict(
                **{
                    f.name: getattr(analyzed_result, f.name).detach().mean().item()
                    for f in dataclasses.fields(analyzed_result)
                    if getattr(analyzed_result, f.name) is not None
                }, **info))
        return TrainerStepResult(stats=stats, step=self.steps)


register('qmix', QMix)
