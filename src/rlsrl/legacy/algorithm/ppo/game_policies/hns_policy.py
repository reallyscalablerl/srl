from typing import Dict
from copy import deepcopy
from torch.distributions import Categorical
import itertools
import gym
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint as cp

from rlsrl.api.trainer import SampleBatch
from rlsrl.legacy.algorithm.ppo.mappo import SampleAnalyzedResult
from rlsrl.base.namedarray import NamedArray, recursive_apply
import rlsrl.api.policy
import rlsrl.legacy.algorithm.modules as modules

_USE_TORCH_CKPT = False


class HNSPolicyState(NamedArray):

    def __init__(self, actor_hx: np.ndarray, critic_hx: np.ndarray):
        super(HNSPolicyState, self).__init__(actor_hx=actor_hx, critic_hx=critic_hx)


class HNSAction(NamedArray):

    def __init__(
            self,
            move_x: np.ndarray,  # Discrete(11)
            move_y: np.ndarray,  # Discrete(11)
            move_z: np.ndarray,  # Discrete(11)
            lock: np.ndarray,  # Discrete(2)
            grab: np.ndarray,  # Discrete(2)
    ):
        super(HNSAction, self).__init__(
            move_x=move_x,
            move_y=move_y,
            move_z=move_z,
            lock=lock,
            grab=grab,
        )


class HNSActionSpace:

    def __init__(self, spaces: Dict[str, gym.Space]):
        self.__spaces = spaces

    @property
    def n(self):
        return {k: x.n for k, x in sorted(self.__spaces.items())}

    def sample(self) -> HNSAction:
        return HNSAction(**{k: np.array([x.sample()]) for k, x in sorted(self.__spaces.items())})


class HNSEncoder(nn.Module):
    # special design for reproducing hide-and-seek paper
    def __init__(self, obs_space, omniscient):
        super(HNSEncoder, self).__init__()
        self.omniscient = omniscient
        self.self_obs_keys = ['observation_self', 'lidar']
        self.ordered_other_obs_keys = ['agent_qpos_qvel', 'box_obs', 'ramp_obs']
        self.ordered_obs_mask_keys = ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs']

        self_dim = obs_space['observation_self'][-1] + obs_space['lidar'][-1] * 9
        others_shape_dict = deepcopy(obs_space)
        others_shape_dict.pop('observation_self')
        others_shape_dict.pop('lidar')

        conv = nn.Conv1d(1, 9, 3, padding=1, padding_mode='circular')
        nn.init.xavier_uniform_(conv.weight.data)

        self.lidar_conv = nn.Sequential(conv, nn.ReLU(inplace=True))
        self.embedding_layer = modules.CatSelfEmbedding(self_dim, others_shape_dict, 128)
        self.attn = modules.ResidualMultiHeadSelfAttention(128, 128, 4)
        self.dense = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(inplace=True),
                                   nn.LayerNorm(256))

    def forward(self, obs):
        lidar = obs.lidar
        if len(lidar.shape) == 4:
            lidar = lidar.view(-1, *lidar.shape[2:])
        x_lidar = self.lidar_conv(lidar).reshape(*obs.lidar.shape[:-2], -1)
        x_self = torch.cat([obs.observation_self, x_lidar], dim=-1)

        x_other = {k: obs[k] for k in self.ordered_other_obs_keys}
        x_self, x_other = self.embedding_layer(x_self, **x_other)

        if self.omniscient:
            if all((obs[k + '_spoof'] is None) for k in self.ordered_obs_mask_keys
                   if (k + "_spoof") in obs.keys()):
                mask = None
            else:
                mask = torch.cat(
                    [obs[k + '_spoof'] for k in self.ordered_obs_mask_keys if (k + "_spoof") in obs.keys()],
                    -1)
        else:
            mask = torch.cat([obs[k] for k in self.ordered_obs_mask_keys], -1)
        if _USE_TORCH_CKPT:
            pooled_attn_other = cp.checkpoint(lambda x, y: modules.masked_avg_pooling(self.attn(x, y), y),
                                              x_other, mask)
        else:
            attn_other = self.attn(x_other, mask)
            pooled_attn_other = modules.masked_avg_pooling(attn_other, mask)
        x = torch.cat([x_self, pooled_attn_other], dim=-1)
        return self.dense(x)


class HNSActorCritic(nn.Module):

    def __init__(self, obs_space, act_space):
        super().__init__()

        self.normalize_obs_keys = [
            'observation_self', 'lidar', 'agent_qpos_qvel', 'box_obs', 'ramp_obs', 'food_obs'
        ]
        for key in self.normalize_obs_keys:
            if key in obs_space:
                setattr(self, key + '_normalization',
                        modules.RunningMeanStd((obs_space[key][-1],), beta=0.99999))

        self.actor_base = HNSEncoder(obs_space, omniscient=False)
        self.critic_base = HNSEncoder(obs_space, omniscient=True)

        self.actor_rnn = modules.AutoResetRNN(256, 256, num_layers=1, rnn_type='lstm')
        self.critic_rnn = modules.AutoResetRNN(256, 256, num_layers=1, rnn_type='lstm')

        self.actor_rnn_norm = nn.LayerNorm([256])
        self.critic_rnn_norm = nn.LayerNorm([256])

        self.policy_outs = nn.ModuleList()
        for act_dim in act_space.values():
            layer = nn.Linear(256, act_dim)
            nn.init.orthogonal_(layer.weight.data, gain=0.01)
            nn.init.zeros_(layer.bias.data)
            self.policy_outs.append(layer)

        self.value_head = modules.PopArtValueHead(256, 1, beta=0.99999)

        for k, p in itertools.chain(self.actor_base.named_parameters(), self.critic_base.named_parameters()):
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data, gain=math.sqrt(2))
            if 'bias' in k:
                nn.init.zeros_(p.data)

        for k, p in itertools.chain(self.actor_rnn.named_parameters(), self.critic_rnn.named_parameters()):
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data)
            if 'bias' in k:
                nn.init.zeros_(p.data)

    def forward(self, obs, actor_hx, critic_hx, on_reset=None):
        for k in self.normalize_obs_keys:
            if k in obs.keys() and obs[k] is not None:
                obs[k] = getattr(self, k + '_normalization').normalize(obs[k])

        actor_features = self.actor_base(obs)
        critic_features = self.critic_base(obs)

        actor_features, actor_hx = self.actor_rnn(actor_features, actor_hx, on_reset)
        critic_features, critic_hx = self.critic_rnn(critic_features, critic_hx, on_reset)
        actor_features = self.actor_rnn_norm(actor_features)
        critic_features = self.critic_rnn_norm(critic_features)

        dists = [Categorical(logits=act(actor_features)) for act in self.policy_outs]
        value = self.value_head(critic_features)

        return dists, value, actor_hx, critic_hx


class HNSPolicy(rlsrl.api.policy.SingleModelPytorchPolicy):

    @property
    def default_policy_state(self):
        return HNSPolicyState(self.__rnn_default_hidden[:, 0, :], self.__rnn_default_hidden[:, 0, :])

    def __init__(self, obs_space: dict, act_space: dict, chunk_len: int, seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.__chunk_len = chunk_len
        self.__rnn_default_hidden = np.zeros((1, 1, 256 * 2), dtype=np.float32)
        neural_network = HNSActorCritic(obs_space, act_space)
        super(HNSPolicy, self).__init__(neural_network)

    def __get_module(self):
        return self.net.module if dist.is_initialized() else self.net

    def normalize_value(self, x):
        return self.__get_module().value_head.normalize(x)

    def denormalize_value(self, x):
        return self.__get_module().value_head.denormalize(x)

    def update_popart(self, x):
        return self.__get_module().value_head.update(x)

    def __update_obs_rms(self, x):
        module = self.__get_module()
        for k in module.normalize_obs_keys:
            if k in x.keys() and x[k] is not None:
                getattr(module, k + '_normalization').update(x[k])

    def analyze(self, sample: SampleBatch, **kwargs) -> SampleAnalyzedResult:
        """ Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies. Typically,
            data has a shape of [T, B, *D] and RNN states have a shape of
            [num_layers, B, hidden_size].
        Args:
            sample (SampleBatch): Arrays of (obs, action ...) containing
                all data required for loss computation.
        Returns:
            analyzed_result[SampleAnalyzedResult]: Data generated for loss computation.
        """
        num_chunks = sample.on_reset.shape[0] // self.__chunk_len
        observation = recursive_apply(sample.obs, lambda x: modules.to_chunk(x, num_chunks))
        # when one step is done, rnn states of the NEXT step should be reset
        on_reset = recursive_apply(sample.on_reset, lambda x: modules.to_chunk(x, num_chunks))
        action = recursive_apply(sample.action, lambda x: modules.to_chunk(x, num_chunks))

        bs = on_reset.shape[1]
        if sample.policy_state is not None:
            policy_state = recursive_apply(sample.policy_state, lambda x: modules.to_chunk(x, num_chunks))
            actor_hx = policy_state.actor_hx[0].transpose(0, 1)
            critic_hx = policy_state.critic_hx[0].transpose(0, 1)
        else:
            actor_hx = critic_hx = torch.from_numpy(np.tile(self.__rnn_default_hidden,
                                                            (1, bs, 1))).to(dtype=torch.float32,
                                                                            device=self.device)

        action_distributions, state_values, _, _ = self.net(observation, actor_hx, critic_hx, on_reset)
        self.__update_obs_rms(sample.obs)

        old_log_probs = sample.log_probs

        # for d in action_distributions:
        for d, a in zip(action_distributions, action):
            print(d, a)

            # print(d.log_prob)
            # print(d.log_prob(a.unsqueeze(-1)))
        #     print(d.log_prob(a.squeeze(-1)))

        new_log_probs = torch.stack([d.log_prob(a.squeeze(-1)) for d, a in zip(action_distributions, action)],
                                    -1)
        new_log_probs = modules.back_to_trajectory(new_log_probs.sum(-1, keepdim=True), num_chunks)

        entropy = modules.back_to_trajectory(
            torch.stack([dist.entropy() for dist in action_distributions], -1).sum(-1, keepdim=True),
            num_chunks)

        analyzed_result = SampleAnalyzedResult(old_action_log_probs=old_log_probs,
                                               new_action_log_probs=new_log_probs,
                                               state_values=modules.back_to_trajectory(
                                                   state_values, num_chunks),
                                               entropy=entropy)

        return analyzed_result

    def rollout(self, requests: rlsrl.api.policy.RolloutRequest, **kwargs):
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
        requests = recursive_apply(requests,
                                   lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))
        bs = requests.length(0)
        actor_hx = critic_hx = torch.from_numpy(np.tile(self.__rnn_default_hidden,
                                                        (1, bs, 1))).to(dtype=torch.float32,
                                                                        device=self.device)
        if requests.policy_state is not None:
            actor_hx = requests.on_reset * actor_hx + (
                (1 - requests.on_reset).unsqueeze(1) * requests.policy_state.actor_hx).transpose(0, 1)
            critic_hx = requests.on_reset * critic_hx + (
                (1 - requests.on_reset).unsqueeze(1) * requests.policy_state.critic_hx).transpose(0, 1)

        with torch.no_grad():
            observation = recursive_apply(requests.obs, lambda x: x.unsqueeze(0))
            action_distributions, value, actor_hx, critic_hx = self.net(observation, actor_hx, critic_hx)
            # .squeeze(0) removes the time dimension
            value = value.squeeze(0)
            deterministic_actions = [dist.probs.argmax(dim=-1).squeeze(0) for dist in action_distributions]
            stochastic_actions = [dist.sample().squeeze(0) for dist in action_distributions]
            # now deterministic/stochastic actions have shape [batch_size]
            eval_mask = requests.is_evaluation.squeeze(-1)
            actions = [
                eval_mask * d_a + (1 - eval_mask) * s_a
                for d_a, s_a in zip(deterministic_actions, stochastic_actions)
            ]
            log_probs = torch.stack(
                [dist.log_prob(a).squeeze(0) for a, dist in zip(actions, action_distributions)],
                -1).sum(-1, keepdim=True)

        # .unsqueeze(-1) adds a trailing dimension 1
        return rlsrl.api.policy.RolloutResult(
            action=HNSAction(*[a.unsqueeze(-1).cpu().numpy() for a in actions]),
            log_probs=log_probs.cpu().numpy(),
            value=value.cpu().numpy(),
            policy_state=HNSPolicyState(
                actor_hx.transpose(0, 1).cpu().numpy(),
                critic_hx.transpose(0, 1).cpu().numpy()),
        )


rlsrl.api.policy.register("hns", HNSPolicy)
