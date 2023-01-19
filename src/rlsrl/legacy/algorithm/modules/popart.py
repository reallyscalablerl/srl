from rlsrl.legacy.algorithm.modules.utils import RunningMeanStd
import torch
import torch.nn as nn
import torch.nn.functional as F


class PopArtValueHead(nn.Module):

    def __init__(self, input_dim, critic_dim, beta=0.999, epsilon=1e-5, burn_in_updates=torch.inf):
        super().__init__()
        self.__rms = RunningMeanStd((critic_dim,), beta, epsilon)

        self.__weight = nn.Parameter(torch.zeros(critic_dim, input_dim))
        nn.init.orthogonal_(self.__weight)
        self.__bias = nn.Parameter(torch.zeros(critic_dim))

        self.__burn_in_updates = burn_in_updates
        self.__update_cnt = 0

    def forward(self, feature):
        return F.linear(feature, self.__weight, self.__bias)

    @torch.no_grad()
    def update(self, x):
        old_mean, old_std = self.__rms.mean_std()
        self.__rms.update(x)
        new_mean, new_std = self.__rms.mean_std()
        self.__update_cnt += 1

        if self.__update_cnt > self.__burn_in_updates:
            self.__weight.data[:] = self.__weight * (old_std / new_std).unsqueeze(-1)
            self.__bias.data[:] = (old_std * self.__bias + old_mean - new_mean) / new_std

    @torch.no_grad()
    def normalize(self, x):
        return self.__rms.normalize(x)

    @torch.no_grad()
    def denormalize(self, x):
        return self.__rms.denormalize(x)
