from typing import Union, Any

from torch.distributions import Categorical
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


@torch.no_grad()
def masked_normalization(x, mask=None, dim=None, inplace=False, unbiased=True, eps=torch.tensor(1e-5)):
    """Normalize x with a mask. Typically used in advantage normalization.

    Args:
        x (torch.Tensor):
            Tensor to be normalized.
        mask (torch.Tensor, optional):
            A mask with the same shape as x. Defaults to None.
        dim (int or tuple of ints, optional):
            Dimensions to be normalized. Defaults to None.
        inplace (bool, optional):
            Whether to perform in-place operation. Defaults to False.
        eps (torch.Tensor, optional):
            Minimal denominator. Defaults to torch.Tensor(1e-5).

    Returns:
        torch.Tensor:
            Normalized x, with the same shape as x.
    """
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        mask = torch.ones_like(x)
    x = x * mask
    factor = mask.sum(dim=dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized():
        dist.all_reduce(factor, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM)
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return (x - mean) / (var.sqrt() + eps)


class RunningMeanStd(nn.Module):

    def __init__(self, input_shape, beta=0.999, epsilon=1e-5):
        super().__init__()
        self.__beta = beta
        self.__eps = epsilon
        self.__input_shape = input_shape

        self.__mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.__mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.__debiasing_term = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__debiasing_term.zero_()

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError

    def __check(self, x):
        assert isinstance(x, torch.Tensor)
        trailing_shape = x.shape[-len(self.__input_shape):]
        assert trailing_shape == self.__input_shape, (
            'Trailing shape of input tensor'
            f'{x.shape} does not equal to configured input shape {self.__input_shape}')

    @torch.no_grad()
    def update(self, x):
        self.__check(x)
        norm_dims = tuple(range(len(x.shape) - len(self.__input_shape)))

        batch_mean = x.mean(dim=norm_dims)
        batch_sq_mean = x.square().mean(dim=norm_dims)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(batch_mean)
            dist.all_reduce(batch_sq_mean)
            batch_mean /= world_size
            batch_sq_mean /= world_size

        self.__mean.mul_(self.__beta).add_(batch_mean * (1.0 - self.__beta))
        self.__mean_sq.mul_(self.__beta).add_(batch_sq_mean * (1.0 - self.__beta))
        self.__debiasing_term.mul_(self.__beta).add_(1.0 * (1.0 - self.__beta))

    @torch.no_grad()
    def mean_std(self):
        debiased_mean = self.__mean / self.__debiasing_term.clamp(min=self.__eps)
        debiased_mean_sq = self.__mean_sq / self.__debiasing_term.clamp(min=self.__eps)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var.sqrt()

    @torch.no_grad()
    def normalize(self, x):
        self.__check(x)
        mean, std = self.mean_std()
        return (x - mean) / std

    @torch.no_grad()
    def denormalize(self, x):
        self.__check(x)
        mean, std = self.mean_std()
        return x * std + mean


def mlp(sizes, activation=nn.ReLU, layernorm=True):
    # refer to https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L15
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
        if layernorm:
            layers += [nn.LayerNorm([sizes[j + 1]])]
    return nn.Sequential(*layers)


def to_chunk(x, num_chunks):
    """Split sample batch into chunks along the time dimension.
    Typically with input shape [T, B, *D].

    Args:
        x (torch.Tensor): The array to be chunked.
        num_chunks (int): Number of chunks. Note as C.

    Returns:
        torch.Tensor: The chunked array.
            Typically with shape [T//C, B*C, *D].
    """
    return torch.cat(torch.split(x, x.shape[0] // num_chunks, dim=0), dim=1)


def back_to_trajectory(x, num_chunks):
    """Inverse operation of to_chunk.
    Typically with input shape [T//C, B*C, *D].

    Args:
        x (torch.Tensor): The array to be inverted.
        num_chunks (int): Number of chunks. Note as C.

    Returns:
        torch.Tensor: Inverted array.
            Typically with shape [T, B, *D]
    """
    return torch.cat(torch.split(x, x.shape[1] // num_chunks, dim=1), dim=0)


def distribution_back_to_trajctory(distribution: Union[Categorical, torch.distributions.Normal], num_chunks):
    if isinstance(distribution, Categorical):
        return Categorical(logits=back_to_trajectory(distribution.logits, num_chunks))
    elif isinstance(distribution, torch.distributions.Normal):
        return torch.distributions.Normal(loc=back_to_trajectory(distribution.loc, num_chunks),
                                          scale=back_to_trajectory(distribution.scale, num_chunks))
    else:
        raise NotImplementedError(f"Don't know how to process {distribution.__class__.__name__}")


def distribution_detach_to_cpu(distribution: Union[torch.distributions.Distribution]):
    if isinstance(distribution, Categorical):
        return Categorical(logits=distribution.logits.cpu().detach())
    elif isinstance(distribution, torch.distributions.Normal):
        return torch.distributions.Normal(loc=distribution.loc.cpu().detach(),
                                          scale=distribution.scale.cpu.detach())
    else:
        raise NotImplementedError(f"Don't know how to process {distribution.__class__.__name__}")


def distribution_to(distribution: torch.distributions.Distribution, to_able: Any):
    if isinstance(distribution, Categorical):
        return Categorical(logits=distribution.logits.to(to_able))
    elif isinstance(distribution, torch.distributions.Normal):
        return torch.distributions.Normal(loc=distribution.loc.to(to_able),
                                          scale=distribution.scale.to(to_able))
    else:
        raise NotImplementedError(f"Don't know how to process {distribution.__class__.__name__}")


def get_clip_value_loss_fn(value_loss_fn, value_eps_clip):

    def foo(value, old_value, target_value):
        value_loss_original = value_loss_fn(value, target_value)

        value_clipped = old_value + (value - old_value).clamp(-value_eps_clip, value_eps_clip)
        value_loss_clipped = value_loss_fn(value_clipped, target_value)

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        return value_loss

    return foo


def init_optimizer(parameters, optimizer_name, optimizer_config):
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

    optim = optimizer_fn(parameters, **optimizer_config)
    return optim
