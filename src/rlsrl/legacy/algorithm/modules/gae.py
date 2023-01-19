import torch


@torch.no_grad()
def gae_trace(reward, value, done, gamma=torch.tensor(0.99), gae_lambda=torch.tensor(0.97)):
    """Compute Generalized Advantage Estimation (GAE) given a trajectory.

    We assume the trajectory is not auto-reseted, i.e., when done is True,
    the observation should be bootstrapped instead of a new one at the beginning
    of the next episode (which affects value computation), and the reward should be 0.

    Note T as the trajectory length, B as the batch size,
    and Nc as the critic dim (e.g. with intrinsic rewards).

    Args:
        reward (torch.Tensor):
            Reward of shape [T, B, Nc].
        value (torch.Tensor):
            Bootstrapped value of shape [T + 1, B, Nc].
        done (torch.Tensor):
            Done of shape [T, B, 1].
        gamma (torch.Tensor, optional):
            Discount factor. It can be a single number or
            a torch.Tensor of shape [T, B, 1]. Defaults to torch.Tensor(0.99).
        gae_lambda (torch.Tensor, optional):
            GAE discount factor. It can be a single number or
            a torch.Tensor of shape [T, B, 1]. Defaults to torch.Tensor(0.97).

    Returns:
        torch.Tensor:
            GAE, with shape [T, B, Nc].
    """
    episode_length = int(reward.shape[0])

    delta = reward + gamma * value[1:] * (1 - done) - value[:-1]
    gae = torch.zeros_like(reward[0])
    adv = torch.zeros_like(reward)
    m = gamma * gae_lambda * (1 - done)
    step = episode_length - 1
    while step >= 0:
        gae = delta[step] + m[step] * gae
        adv[step] = gae
        step -= 1
    return adv
