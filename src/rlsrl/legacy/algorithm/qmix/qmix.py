from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import numpy as np
import torch.distributed as dist

from rlsrl.api.policy import Policy, RolloutRequest, RolloutResult, register
from rlsrl.api.trainer import SampleBatch
from rlsrl.legacy.algorithm.qmix.qmix_trainer import SampleAnalyzedResult as QMIXAnalyzeResult
from rlsrl.base.namedarray import recursive_apply, NamedArray
from .mixer import *


def merge_BN(x):
    if x is None:
        return x
    return x.reshape(x.shape[0], -1, *x.shape[3:])


def split_BN(x, batch_size, num_agents):
    if x is None:
        return x
    return x.reshape(x.shape[0], batch_size, num_agents, *x.shape[2:])


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DecayThenFlatSchedule():

    def __init__(self, start, finish, time_length, decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / \
                np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(-T / self.exp_scaling)))


class PolicyState(NamedArray):

    def __init__(self, hx: np.ndarray, epsilon: np.ndarray):
        super(PolicyState, self).__init__(hx=hx, epsilon=epsilon)


class AgentQFunction(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim, num_dense_layers, rnn_type, num_rnn_layers):
        super(AgentQFunction, self).__init__()
        self.obs_norm = nn.LayerNorm([obs_dim])
        self.backbone = RecurrentBackbone(obs_dim=obs_dim,
                                          hidden_dim=hidden_dim,
                                          dense_layers=num_dense_layers,
                                          num_rnn_layers=num_rnn_layers,
                                          rnn_type=rnn_type)
        self.head = nn.Linear(self.backbone.feature_dim, action_dim)
        nn.init.orthogonal_(self.head.weight.data, gain=0.01)

    def forward(self, obs, hx, on_reset=None, available_action=None):
        obs = self.obs_norm(obs)
        features, hx = self.backbone(obs, hx, on_reset)
        q = self.head(features)
        if available_action is not None:
            q[available_action == 0.] = -1e10
        return q, hx


class Qtot(nn.Module):

    def __init__(self, num_agents, obs_dim, action_dim, state_dim, q_i, q_i_config, mixer, mixer_config):
        super(Qtot, self).__init__()
        self.__num_agents = num_agents
        self.__obs_dim = obs_dim
        self.__action_dim = action_dim
        self.__state_dim = state_dim

        self.q_i = q_i(obs_dim, action_dim, **q_i_config)
        self.mixer = mixer(num_agents, state_dim, **mixer_config)

    def forward(self, obs, hx, on_reset=None, available_action=None, action=None, state=None, mode="rollout"):
        batch_size, num_agents = obs.shape[1], self.__num_agents

        obs = merge_BN(obs)
        hx = merge_BN(hx)
        on_reset = merge_BN(on_reset)
        available_action = merge_BN(available_action)

        q_i_full, hx = self.q_i(obs, hx, available_action=available_action)
        greedy_q_i, greedy_action = q_i_full.max(dim=-1, keepdim=True)

        hx = split_BN(hx, batch_size, num_agents)
        q_i_full = split_BN(q_i_full, batch_size, num_agents)
        greedy_q_i = split_BN(greedy_q_i, batch_size, num_agents)
        greedy_action = split_BN(greedy_action, batch_size, num_agents)

        if mode == "rollout":
            return q_i_full, greedy_q_i, greedy_action, hx
        elif mode == "analyze":
            if action is None:
                action = greedy_action
            q_i = torch.gather(q_i_full, -1, action.to(dtype=torch.int64))
            q_tot = self.mixer(q_i, state)
            return q_i_full, q_i, greedy_q_i, greedy_action, q_tot.reshape(-1, batch_size, 1)
        else:
            raise NotImplementedError


class QtotPolicy(Policy):

    @property
    def default_policy_state(self):
        return PolicyState(self.__rnn_default_hidden[:, 0, :, :], np.ones((1,)))

    @property
    def version(self):
        return self._version

    @property
    def net(self):
        return [self._net, self._target_net]

    def __init__(self,
                 num_agents,
                 obs_dim,
                 action_dim,
                 state_dim,
                 chunk_len=100,
                 use_double_q=True,
                 epsilon_start=1.0,
                 epsilon_finish=0.05,
                 epsilon_anneal_time=5000,
                 q_i_config=dict(hidden_dim=128, num_dense_layers=2, rnn_type="gru", num_rnn_layers=1),
                 mixer="qmix",
                 mixer_config=dict(hidden_dim=64, num_hypernet_layers=2, hypernet_hidden_dim=64,
                                   popart=False),
                 state_use_all_local_obs=False,
                 state_concate_all_local_obs=False,
                 seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        super(QtotPolicy, self).__init__()
        self.__num_agents = num_agents
        self.__obs_dim = obs_dim
        self.__action_dim = action_dim
        self.__state_dim = state_dim
        self.__chunk_len = chunk_len
        self.__use_double_q = use_double_q
        self.exploration = DecayThenFlatSchedule(epsilon_start,
                                                 epsilon_finish,
                                                 epsilon_anneal_time,
                                                 decay="linear")

        self.__state_use_all_local_obs = state_use_all_local_obs
        self.__state_concate_all_local_obs = state_concate_all_local_obs

        if self.__state_use_all_local_obs:
            self.__state_dim = self.__obs_dim * self.__num_agents
        if self.__state_concate_all_local_obs:
            self.__state_dim += self.__obs_dim * self.__num_agents

        self.__rnn_default_hidden = np.zeros(
            (q_i_config["num_rnn_layers"], 1, num_agents, q_i_config["hidden_dim"]))

        self._net = Qtot(self.__num_agents, self.__obs_dim, self.__action_dim, self.__state_dim,
                         AgentQFunction, q_i_config, Mixer[mixer], mixer_config).to(self.device)
        self._target_net = copy.deepcopy(self._net).to(self.device)
        self._version = -1

    def analyze(self, sample: SampleBatch, target="qmix", **kwargs):
        """ Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies. Typically,
            data has a shape of [T, B, *D] and RNN states have a shape of
            [num_layers, B, hidden_size].
        Args:
            sample (SampleBatch): Arrays of (obs, action ...) containing
                all data required for loss computation.
            target (str): style by which the algorithm should be analyzed.
        Returns:
            analyzed_result[SampleAnalyzedResult]: Data generated for loss computation.
        """
        sample = recursive_apply(sample,
                                 lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))
        if target == "qmix":
            return self._qmix_analyze(sample)
        else:
            raise ValueError(
                f"Analyze method for algorithm {target} not implemented for {self.__class__.__name__}")

    def _qmix_analyze(self, sample):
        num_chunks = (sample.on_reset.shape[0] - 1) // self.__chunk_len
        sample = recursive_apply(sample, lambda x: modules.to_chunk(x[:-1], num_chunks))
        hx = sample.policy_state.hx[0].transpose(0, 1)

        if hasattr(sample.obs, "obs"):
            obs = sample.obs.obs
        elif hasattr(sample.obs, "local_obs"):
            obs = sample.obs.local_obs
        else:
            raise RuntimeError("sample obs doesn't have local_obs or obs.")

        on_reset = sample.on_reset
        action = sample.action.x
        available_action = sample.obs.available_action if hasattr(sample.obs, "available_action") else None

        state = None
        if hasattr(sample.obs, "state"):
            state = sample.obs.state.mean(dim=2)  # average over agent dimension
        if self.__state_use_all_local_obs:
            state = obs.reshape(obs.shape[0], obs.shape[1], -1)
            assert state.shape[2] == self.__state_dim
        if self.__state_concate_all_local_obs:
            state = torch.cat([state, obs.reshape(obs.shape[0], obs.shape[1], -1)], dim=-1)

        q_i_full, q_i, greedy_q_i, greedy_action, q_tot = self._net(obs,
                                                                    hx,
                                                                    on_reset=on_reset,
                                                                    available_action=available_action,
                                                                    action=action,
                                                                    state=state,
                                                                    mode="analyze")

        with torch.no_grad():
            if self.__use_double_q:
                _, _, _, _, target_q_tot = self._target_net(obs,
                                                            hx,
                                                            on_reset=on_reset,
                                                            action=greedy_action,
                                                            available_action=available_action,
                                                            state=state,
                                                            mode="analyze")
            else:
                _, _, _, _, target_q_tot = self._target_net(obs,
                                                            hx,
                                                            on_reset=on_reset,
                                                            action=None,
                                                            available_action=available_action,
                                                            state=state,
                                                            mode="analyze")

        analyzed_result = QMIXAnalyzeResult(q_tot=modules.back_to_trajectory(q_tot, num_chunks),
                                            target_q_tot=modules.back_to_trajectory(target_q_tot, num_chunks),
                                            reward=modules.back_to_trajectory(
                                                sample.reward.sum(dim=2), num_chunks))
        return analyzed_result

    def rollout(self, requests: RolloutRequest, **kwargs):
        """ Provide inference results for actor workers. Typically,
            data and masks have a shape of [batch_size, *D], and RNN states
            have a shape of [batch_size, num_layers, hidden_size].
        Args:
            requests (RolloutRequest): Observations, policy states,
                evaluation masks and reset masks. 
        Returns:
            RolloutResult: Actions and new policy states, optionally
                with other entries depending on the algorithm.
        """
        eval_mask = torch.from_numpy(requests.is_evaluation).to(dtype=torch.int32, device=self.device)
        requests = recursive_apply(requests,
                                   lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))

        bs = requests.length(0)
        if hasattr(requests.obs, "obs"):
            obs = requests.obs.obs.unqueeze(0)
        elif hasattr(requests.obs, "local_obs"):
            obs = requests.obs.local_obs.unsqueeze(0)
        else:
            raise RuntimeError("requests obs doesn't have local_obs or obs")

        available_action = None
        if hasattr(requests.obs, "available_action"):
            available_action = requests.obs.available_action.unsqueeze(0)
        else:
            available_action = torch.ones(1,
                                          bs,
                                          self.__num_agents,
                                          self.__action_dim,
                                          dtype=torch.float32,
                                          device=self.device)

        hx = requests.policy_state.hx.transpose(0, 1)
        on_reset = requests.on_reset.unsqueeze(0)
        epsilon = self.exploration.eval(self._version)

        with torch.no_grad():
            q_i_full, greedy_q_i, greedy_action, hx = self._net(obs,
                                                                hx,
                                                                on_reset=on_reset,
                                                                available_action=available_action,
                                                                mode="rollout")
            q_i_full = q_i_full.squeeze(0)
            greedy_q_i = greedy_q_i.squeeze(0)
            greedy_action = greedy_action.squeeze(0)

            onehot_greedy_action = torch.scatter(torch.zeros_like(q_i_full),
                                                 dim=-1,
                                                 src=torch.ones_like(greedy_action).to(dtype=torch.float32),
                                                 index=greedy_action)
            available_action = available_action.squeeze(0)
            random_prob = available_action / available_action.sum(dim=-1, keepdim=True)

            prob = eval_mask * onehot_greedy_action + (1. -
                                                       eval_mask) * (epsilon * random_prob +
                                                                     (1. - epsilon) * onehot_greedy_action)
            prob = Categorical(prob)
            action = prob.sample().unsqueeze(-1)

            hx = hx.transpose(0, 1).cpu().numpy()

        return action.cpu().numpy(), PolicyState(hx, epsilon * np.ones((bs, 1)))

    def parameters(self):
        return {
            "train": self._net.parameters(recurse=True),
            "target": self._target_net.parameters(recurse=True)
        }

    def get_checkpoint(self):
        if dist.is_initialized():
            return {
                "steps": self._version,
                "state_dict": {
                    "train": {k.replace("module.", ""): v
                              for k, v in self._net.state_dict().items()},
                    "target": {k.replace("module.", ""): v
                               for k, v in self._target_net.state_dict().items()}
                },
            }
        else:
            return {
                "steps": self._version,
                "state_dict": {
                    "train": self._net.state_dict(),
                    "target": self._target_net.state_dict()
                }
            }

    def load_checkpoint(self, checkpoint):
        self._version = checkpoint.get("steps", 0)
        self._net.load_state_dict(checkpoint["state_dict"]["train"])
        self._target_net.load_state_dict(checkpoint["state_dict"]["target"])

    def train_mode(self):
        self._net.train()
        self._target_net.eval()

    def eval_mode(self):
        self._net.eval()
        self._target_net.eval()

    def inc_version(self):
        self._version += 1

    def distributed(self, **kwargs):
        if dist.is_initialized():
            if self.device == "cpu":
                self._net = DDP(self._net, **kwargs)
                self._target_net = DDP(self._target_net, **kwargs)
            else:
                self._net = DDP(self._net,
                                device_ids=[self.device],
                                output_device=self.device,
                                find_unused_parameters=True,
                                **kwargs)
                self._target_net = DDP(self._target_net,
                                       device_ids=[self.device],
                                       output_device=self.device,
                                       find_unused_parameters=True,
                                       **kwargs)

    def normalize_value(self, x):
        return self._net.module.mixer.popart_head.normalize(x)

    def normalize_target_value(self, x):
        return self._target_net.module.mixer.popart_head.normalize(x)

    def denormalize_value(self, x):
        return self._net.module.mixer.popart_head.denormalize(x)

    def denormalize_target_value(self, x):
        return self._target_net.module.mixer.popart_head.denormalize(x)

    def update_popart(self, x):
        return self._net.module.mixer.popart_head.update(x)

    def soft_target_update(self, tau):
        soft_update(self._target_net, self._net, self.tau)

    def hard_target_update(self):
        hard_update(self._target_net, self._net)


register("qtot-policy", QtotPolicy)
