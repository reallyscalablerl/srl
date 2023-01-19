import torch


@torch.no_grad()
def vtrace(imp_ratio,
           reward,
           value,
           done,
           gamma=torch.tensor(0.99),
           lmbda=torch.tensor(0.97),
           rho=1.0,
           c=1.0):
    """Compute V-trace given a trajectory.

    We assume the trajectory is not auto-reseted, i.e., when done is True,
    the observation should be bootstrapped instead of a new one at the beginning
    of the next episode (which affects value computation), and the reward should be 0.

    Note T as the trajectory length, B as the batch size,
    and Nc as the critic dim (e.g. with intrinsic rewards).

    Args:
        imp_ratio (torch.Tensor):
            Importance sampling ratio of shape [T, B, 1].
        reward (torch.Tensor):
            Reward of shape [T, B, Nc].
        value (torch.Tensor):
            Bootstrapped value of shape [T + 1, B, Nc].
        done (torch.Tensor):
            Done of shape [T, B, 1].
        gamma (torch.Tensor, optional):
            Discount factor. It can be a single number or
            a torch.Tensor of shape [T, B, 1]. Defaults to torch.Tensor(0.99).
        lmbda (torch.Tensor, optional):
            V-trace discount factor. It can be a single number or
            a torch.Tensor of shape [T, B, 1]. Defaults to torch.Tensor(0.97).
        rho (float, optional):
            Clipping hyperparameter rho as described in the paper.
            Defaults to 1.0.
        c (float, optional):
            Clipping hyperparameter c as described in the paper.
            Defaults to 1.0.

    Returns:
        torch.Tensor:
            V-trace advantage, with shape [T, B, Nc].
    """
    episode_length = int(reward.shape[0])

    rho_ = imp_ratio.clip(max=rho)
    delta = rho_ * (reward + gamma * value[1:] * (1 - done) - value[:-1])
    gae = torch.zeros_like(reward[0])
    adv = torch.zeros_like(reward)
    m = gamma * lmbda * (1 - done) * imp_ratio.clip(max=c)
    step = episode_length - 1
    while step >= 0:
        gae = delta[step] + m[step] * gae
        adv[step] = gae
        step -= 1
    return adv
