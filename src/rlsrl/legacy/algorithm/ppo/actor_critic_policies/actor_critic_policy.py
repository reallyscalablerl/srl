from typing import Optional, Union, List, Dict, Tuple
from torch.distributions import Categorical
import dataclasses
import functools
import numpy as np
import torch
import torch.nn as nn

from rlsrl.legacy.algorithm import modules
from rlsrl.api.env_utils import DiscreteAction
from rlsrl.api.policy import SingleModelPytorchPolicy, RolloutRequest, RolloutResult, register
from rlsrl.api.trainer import SampleBatch
from rlsrl.legacy.algorithm.ppo.mappo import SampleAnalyzedResult as PPOAnalyzeResult
from rlsrl.legacy.algorithm.ppo.phasic_policy_gradient import PPGPhase1AnalyzedResult, PPGPhase2AnalyzedResult
from rlsrl.base.namedarray import recursive_apply, NamedArray
from rlsrl.legacy.algorithm.modules.recurrent_backbone import RecurrentBackbone
from .utils import get_action_indices, make_models_for_obs


class DualPolicyState(NamedArray):

    def __init__(
        self,
        actor_hx: Union[np.ndarray, torch.Tensor],
        critic_hx: Union[np.ndarray, torch.Tensor],
    ):
        super(DualPolicyState, self).__init__(actor_hx=actor_hx, critic_hx=critic_hx)


class SinglePolicyState(NamedArray):

    def __init__(self, hx: Union[np.ndarray, torch.Tensor]):
        super(SinglePolicyState, self).__init__(hx=hx)


@dataclasses.dataclass
class DiscreteActorResult:
    logits: Union[np.ndarray, torch.Tensor]


@dataclasses.dataclass
class ContinuousActorResult:
    mean: Union[np.ndarray, torch.Tensor]
    std: Union[np.ndarray, torch.Tensor]


@dataclasses.dataclass
class ModuleForwardResult:
    actor: Union[DiscreteActorResult, ContinuousActorResult]
    critic: torch.Tensor
    policy_state: Union[DualPolicyState, SinglePolicyState]
    auxiliary_critic: Optional[torch.Tensor] = None


class ActorCriticSeparate(nn.Module):
    """Classic Actor Critic model with separate backbone.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, state_dim: Optional, value_dim, dense_layers,
                 rnn_type, num_rnn_layers, cnn_layers, use_maxpool, popart, activation, layernorm,
                 shared_backbone, auxiliary_head, continuous_action, **kwargs):
        super(ActorCriticSeparate, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh
        else:
            raise NotImplementedError(f"Activation function {activation} not implemented.")

        self.shared_backbone = shared_backbone
        self.auxiliary_head = auxiliary_head
        self.continuous_action = continuous_action

        if isinstance(obs_dim, int):
            obs_dim = {"obs": obs_dim}

        self.obs_modules_dict = make_models_for_obs(obs_dim=obs_dim,
                                                    hidden_dim=hidden_dim,
                                                    cnn_layers=cnn_layers,
                                                    use_maxpool=use_maxpool,
                                                    activation=self.activation)
        self.actor_backbone = RecurrentBackbone(obs_dim=hidden_dim * len(obs_dim),
                                                hidden_dim=hidden_dim,
                                                dense_layers=dense_layers,
                                                num_rnn_layers=num_rnn_layers,
                                                rnn_type=rnn_type,
                                                activation=activation,
                                                layernorm=layernorm)
        if not shared_backbone:
            self.__state_dim = state_dim
            if self.__state_dim is not None:
                if isinstance(state_dim, int):
                    state_dim = {"state": state_dim}
            self.state_modules_dict = make_models_for_obs(obs_dim=state_dim or obs_dim,
                                                          hidden_dim=hidden_dim,
                                                          cnn_layers=cnn_layers,
                                                          use_maxpool=use_maxpool,
                                                          activation=self.activation)
            self.critic_backbone = RecurrentBackbone(obs_dim=hidden_dim * len(state_dim or obs_dim),
                                                     hidden_dim=hidden_dim,
                                                     dense_layers=dense_layers,
                                                     num_rnn_layers=num_rnn_layers,
                                                     rnn_type=rnn_type,
                                                     activation=activation,
                                                     layernorm=layernorm)
        else:
            self.critic_backbone = self.state_modules_dict = None

        self.actor_head = self._init_layer(nn.Linear(self.actor_backbone.feature_dim, action_dim))

        if self.continuous_action:
            self.std_type = kwargs.get("std_type", "fixed")
            init_log_std = kwargs.get("init_log_std", -0.5)
            if self.std_type == 'fixed':
                self.log_std = nn.Parameter(float(init_log_std) * torch.ones(action_dim), requires_grad=False)
            elif self.std_type == 'separate_learnable':
                self.log_std = nn.Parameter(float(init_log_std) * torch.ones(action_dim), requires_grad=True)
            elif self.std_type == 'shared_learnable':
                self.log_std = self._init_layer(nn.Linear(self.actor_backbone.feature_dim, action_dim))
            else:
                raise NotImplementedError(f"Standard deviation type {self.std_type} not implemented.")

        self.__popart = popart
        f_ = self.critic_backbone.feature_dim if self.critic_backbone is not None else self.actor_backbone.feature_dim
        if self.__popart:
            self.critic_head = modules.PopArtValueHead(f_, value_dim)
        else:
            self.critic_head = self._init_layer(nn.Linear(f_, value_dim))

        if self.auxiliary_head:
            self.auxiliary_value_head = self._init_layer(nn.Linear(self.actor_backbone.feature_dim,
                                                                   value_dim))

    def _init_layer(self, model: nn.Module):
        nn.init.orthogonal_(model.weight.data, gain=0.01)
        nn.init.zeros_(model.bias.data)
        return model

    def forward(self, obs_, policy_state, on_reset=None):
        obs = torch.cat([m(obs_[k]) for k, m in self.obs_modules_dict.items()], dim=-1)
        if self.shared_backbone:
            actor_features, hx = self.actor_backbone(obs, policy_state.hx, on_reset)
            critic_features = actor_features
            policy_state = SinglePolicyState(hx=hx)
        else:
            state = torch.cat([m(obs_[k]) for k, m in self.state_modules_dict.items()], dim=-1)
            actor_features, actor_hx = self.actor_backbone(obs, policy_state.actor_hx, on_reset)
            critic_features, critic_hx = self.critic_backbone(state, policy_state.critic_hx, on_reset)
            policy_state = DualPolicyState(actor_hx=actor_hx, critic_hx=critic_hx)

        a = self.actor_head(actor_features)

        if self.continuous_action:
            if self.std_type == 'fixed' or self.std_type == 'separate_learnable':
                std = self.log_std.exp() * torch.ones_like(a)
            elif self.std_type == 'shared_learnable':
                std = self.log_std(actor_features).exp()
            actor_result = ContinuousActorResult(mean=a, std=std)
        else:
            if hasattr(obs_, "available_action"):
                a[obs_.available_action == 0] = -1e10
            actor_result = DiscreteActorResult(logits=a)

        if self.auxiliary_head:
            return ModuleForwardResult(actor=actor_result,
                                       critic=self.critic_head(critic_features),
                                       auxiliary_critic=self.auxiliary_value_head(actor_features),
                                       policy_state=policy_state)
        else:
            return ModuleForwardResult(actor=actor_result,
                                       critic=self.critic_head(critic_features),
                                       policy_state=policy_state)


class ActorCriticPolicy(SingleModelPytorchPolicy):

    def __init__(self,
                 obs_dim: Union[int, Dict[str, Union[int, Tuple[int]]]],
                 action_dim: Union[int, List[int]],
                 hidden_dim: int = 128,
                 state_dim: Optional[Union[int, Dict[str, Union[int, Tuple[int]]]]] = None,
                 value_dim: int = 1,
                 chunk_len: int = 10,
                 num_dense_layers: int = 2,
                 rnn_type: str = "gru",
                 cnn_layers: Dict[str, Tuple] = None,
                 use_maxpool: Dict[str, Tuple] = None,
                 num_rnn_layers: int = 1,
                 popart: bool = True,
                 activation: str = "relu",
                 layernorm: bool = True,
                 shared_backbone: bool = False,
                 continuous_action: bool = False,
                 auxiliary_head: bool = False,
                 seed=0,
                 **kwargs):
        """Actor critic style policy.
        Args:
            obs_dim: key-value pair of observation-shape. Passing int value is equivalent to {"obs": int}. Currently,
            Supported operations shapes are: int(MLP), Tuple[int, int](Conv1d), Tuple[int, int, int](Conv2d),
            Tuple[int, int, int, int](Conv3d).
            action_dim: action dimension. For discrete action, accepted types are int and list[int](Mulit-Discrete).
            For continuous action, accepted type is int.
            hidden_dim: hidden size of neural network. Observation/States are first mapped to this size, concatenated
            together, then passed through a mlp and possible a rnn.
            state_dim: Similar to obs_dim. If shared_backbone, state_dim has not effect. Overlaps are allowed between
            obse_dim and state_dim.
            value_dim: Size of state_value (same as the size of reward).
            chunk_len: RNN unroll length when training.
            num_dense_layers: number of dense layers between observation concatenation and rnn.
            rnn_type: "lstm" or "gru"
            cnn_layers: Key-value of user-specified convolution layers.
            Format is {"obs_key": [(output_channel, kernel_size, stride, padding, padding_mode), ...]}
            num_rnn_layers: Number of rnn layers.
            popart: Whether to use a popart head.
            activation: Supported are "relu" or "tanh".
            layernorm: Whether to use layer-norm.
            shared_backbone: Whether to use a separate backbone for critic.
            continuous_action: Whether to action space is continuous.
            auxiliary_head: Whether to use a auxiliary_head.
            seed: Seed of initial state.
            kwargs: Additional configuration passed to pytorch model. Currently supports "std_type" and "init_log_std",
            both used for continuous action.
        """
        if auxiliary_head and shared_backbone:
            raise AttributeError("Cannot use shared backbone when requiring auxiliary value head.")
        if continuous_action and not isinstance(action_dim, int):
            raise AttributeError("Cannot support use action head when using continuous action.")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if isinstance(action_dim, int):
            act_dims = [action_dim]
            action_total_dim = action_dim
        else:
            act_dims = action_dim
            action_total_dim = sum(action_dim)
        self.__action_indices = get_action_indices(act_dims)
        self.__rnn_hidden_dim = hidden_dim
        self.__chunk_len = chunk_len
        self.__popart = popart
        self.__shared_backbone = shared_backbone
        self.__continuous_action = continuous_action
        self.__auxiliary_head = auxiliary_head
        self.__log_softmax = torch.nn.LogSoftmax(dim=-1)
        if rnn_type == "gru":
            self.__rnn_default_hidden = np.zeros((num_rnn_layers, 1, self.__rnn_hidden_dim), dtype=np.float32)
        elif rnn_type == "lstm":
            self.__rnn_default_hidden = np.zeros((num_rnn_layers, 1, self.__rnn_hidden_dim * 2),
                                                 dtype=np.float32)
        else:
            raise ValueError(f"Unknown rnn_type {rnn_type} for ActorCriticPolicy.")

        if self.__shared_backbone:
            self.__default_policy_state = SinglePolicyState(hx=self.__rnn_default_hidden[:, 0, :])
        else:
            self.__default_policy_state = DualPolicyState(actor_hx=self.__rnn_default_hidden[:, 0, :],
                                                          critic_hx=self.__rnn_default_hidden[:, 0, :])

        neural_network = ActorCriticSeparate(obs_dim=obs_dim,
                                             hidden_dim=hidden_dim,
                                             action_dim=action_total_dim,
                                             state_dim=state_dim,
                                             value_dim=value_dim,
                                             dense_layers=num_dense_layers,
                                             rnn_type=rnn_type,
                                             num_rnn_layers=num_rnn_layers,
                                             cnn_layers=cnn_layers or {},
                                             use_maxpool=use_maxpool or {},
                                             popart=popart,
                                             activation=activation,
                                             layernorm=layernorm,
                                             shared_backbone=shared_backbone,
                                             continuous_action=continuous_action,
                                             auxiliary_head=auxiliary_head,
                                             **kwargs)
        super(ActorCriticPolicy, self).__init__(neural_network)

    @property
    def popart_head(self):
        if not self.__popart:
            raise ValueError("Set popart=True in policy config to activate popart value head.")
        return self.net.module.critic_head

    def normalize_value(self, x):
        return self.popart_head.normalize(x)

    def denormalize_value(self, x):
        return self.popart_head.denormalize(x)

    def update_popart(self, x):
        return self.popart_head.update(x)

    @property
    def default_policy_state(self):
        return self.__default_policy_state

    def analyze(self, sample: SampleBatch, target="ppo", **kwargs):
        """ Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies. Typically,
            data has a shape of [T, B, *D] and RNN states have a shape of
            [num_layers, B, hidden_size].
        Args:
            sample (SampleBatch): Arrays of (obs, action ...) containing
                all data required for loss computation.
            target (str): style by which the algorithm should be analyzed.
            kwargs: addition keyword arguments.
        Returns:
            analyzed_result[SampleAnalyzedResult]: Data generated for loss computation.
        """
        if target == "ppo":
            return self._ppo_analyze(sample, **kwargs)
        if target == "ppg_ppo_phase":
            return self._ppg_phase1_analyze(sample, **kwargs)
        elif target == "ppg_aux_phase":
            return self._ppg_phase2_analyze(sample, **kwargs)
        else:
            raise ValueError(
                f"Analyze method for algorithm {target} not implemented for {self.__class__.__name__}")

    def get_action_distribution(self, actor_result):
        if not self.__continuous_action:
            return [
                Categorical(logits=actor_result.logits[..., idx.start:idx.end])
                for idx in self.__action_indices
            ]
        else:
            return [torch.distributions.Normal(actor_result.mean, actor_result.std)]

    def __get_log_prob_and_entropy(self, action_dists, actions):
        if not self.__continuous_action:
            new_log_probs = torch.sum(torch.stack(
                [dist.log_prob(actions.x[..., i]) for i, dist in enumerate(action_dists)], dim=-1),
                                      dim=-1,
                                      keepdim=True)
            entropy = torch.sum(torch.stack([dist.entropy() for dist in action_dists], dim=-1),
                                dim=-1,
                                keepdim=True)
        else:
            [action_dist] = action_dists
            new_log_probs = action_dist.log_prob(actions.x).sum(-1, keepdim=True)
            entropy = action_dist.entropy().sum(-1, keepdim=True)
        return new_log_probs, entropy

    def _ppo_analyze(self, sample, **kwargs):
        """Perform PPO style analysis on sample.
        Args:
            bootstrap_steps: the last {bootstrap_steps} steps will not be analyzed. For example, if the
                training algorithm uses GAE only, then `bootstrap_steps` should be align with the config of
                actor workers. Or, if the training algorithm uses Vtrace, then `bootstrap_steps` should be 1
                because all log-probs will be used except for the last step.
        """
        if self.__auxiliary_head:
            raise RuntimeError("When using PPO algorithm, you should disable auxiliary head.")
        num_chunks = sample.on_reset.shape[0] // self.__chunk_len
        sample = recursive_apply(sample, lambda x: modules.to_chunk(x, num_chunks))
        policy_state = recursive_apply(sample.policy_state, lambda x: x[0].transpose(0, 1))
        forward_result = self.net(sample.obs, policy_state, sample.on_reset)
        new_log_probs, entropy = self.__get_log_prob_and_entropy(
            self.get_action_distribution(forward_result.actor), sample.action)

        analyzed_result = PPOAnalyzeResult(
            old_action_log_probs=modules.back_to_trajectory(sample.log_probs, num_chunks),
            new_action_log_probs=modules.back_to_trajectory(new_log_probs, num_chunks),
            state_values=modules.back_to_trajectory(forward_result.critic, num_chunks),
            entropy=modules.back_to_trajectory(entropy, num_chunks))

        return analyzed_result

    def _ppg_phase1_analyze(self, sample, **kwargs):
        if not self.__auxiliary_head:
            raise RuntimeError(
                "Cannot run ppg analysis without auxiliary_head. Try setting auxiliary_head=True.")
        num_chunks = sample.on_reset.shape[0] // self.__chunk_len
        sample = recursive_apply(sample, lambda x: modules.to_chunk(x, num_chunks))
        policy_state = recursive_apply(sample.policy_state, lambda x: x[0].transpose(0, 1))
        forward_result: ModuleForwardResult = self.net(sample.obs, policy_state, sample.on_reset)
        new_log_probs, entropy = self.__get_log_prob_and_entropy(
            self.get_action_distribution(forward_result.actor), sample.action)

        analyzed_result = PPGPhase1AnalyzedResult(
            old_action_log_probs=modules.back_to_trajectory(sample.log_probs, num_chunks),
            new_action_log_probs=modules.back_to_trajectory(new_log_probs, num_chunks),
            aux_values=modules.back_to_trajectory(forward_result.auxiliary_critic, num_chunks),
            state_values=modules.back_to_trajectory(forward_result.critic, num_chunks),
            reward=modules.back_to_trajectory(sample.reward, num_chunks),
            entropy=modules.back_to_trajectory(entropy, num_chunks))

        return analyzed_result

    def _ppg_phase2_analyze(self, sample, **kwargs):
        num_chunks = sample.on_reset.shape[0] // self.__chunk_len
        sample = recursive_apply(sample, lambda x: modules.to_chunk(x, num_chunks))
        # when one step is done, rnn states of the NEXT step should be reset
        policy_state = recursive_apply(sample.policy_state, lambda x: x[0].transpose(0, 1))
        forward_result: ModuleForwardResult = self.net(sample.obs, policy_state, sample.on_reset)

        dists = self.get_action_distribution(forward_result.actor)
        analyzed_result = PPGPhase2AnalyzedResult(
            action_dists=[modules.distribution_back_to_trajctory(d, num_chunks) for d in dists],
            auxiliary_value=modules.back_to_trajectory(forward_result.auxiliary_critic, num_chunks),
            predicted_value=modules.back_to_trajectory(forward_result.critic, num_chunks),
        )

        return analyzed_result

    def rollout(self, requests: RolloutRequest, **kwargs):
        """ Provide inference results for actor workers. Typically,
            data and masks have a shape of [batch_size, *D], and RNN states
            have a shape of [batch_size, num_layers, hidden_size].
        Returns:
            RolloutResult: Actions and new policy states, optionally
                with other entries depending on the algorithm.
        """
        eval_mask = torch.from_numpy(requests.is_evaluation).to(dtype=torch.int32, device=self.device)
        requests = recursive_apply(requests,
                                   lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))
        requests.obs = recursive_apply(requests.obs, lambda x: x.unsqueeze(0))
        policy_state = recursive_apply(requests.policy_state, lambda x: x.transpose(0, 1))

        with torch.no_grad():
            forward_result = self.net(requests.obs, policy_state)
            value = forward_result.critic.squeeze(0)
            if not self.__continuous_action:
                action_logits = forward_result.actor.logits.squeeze(0)
                action_dists = [
                    Categorical(logits=action_logits[..., idx.start:idx.end]) for idx in self.__action_indices
                ]
                # .squeeze(0) removes the time dimension
                deterministic_actions = torch.stack([dist.probs.argmax(dim=-1) for dist in action_dists],
                                                    dim=-1)
                # dist.sample adds an additional dimension
                stochastic_actions = torch.stack([dist.sample().squeeze(0) for dist in action_dists], dim=-1)
                # now deterministic/stochastic actions have shape [batch_size]
                actions = eval_mask * deterministic_actions + (1 - eval_mask) * stochastic_actions
                log_probs = torch.sum(torch.stack(
                    [dist.log_prob(actions[..., i]) for i, dist in enumerate(action_dists)], dim=-1),
                                      dim=-1,
                                      keepdim=True)
            else:
                value = forward_result.critic.squeeze(0)
                mean = forward_result.actor.mean.squeeze(0)
                std = forward_result.actor.std.squeeze(0)
                action_dist = torch.distributions.Normal(mean, std)
                stochastic_actions = action_dist.sample()
                # now deterministic/stochastic actions have shape [batch_size]
                actions = eval_mask * mean + (1 - eval_mask) * stochastic_actions
                log_probs = action_dist.log_prob(actions).sum(-1, keepdim=True)

            # .unsqueeze(-1) adds a trailing dimension 1
            policy_state = recursive_apply(forward_result.policy_state,
                                           lambda x: x.transpose(0, 1).cpu().numpy())
        return RolloutResult(action=DiscreteAction(actions.cpu().numpy()),
                             log_probs=log_probs.cpu().numpy(),
                             value=value.cpu().numpy(),
                             policy_state=policy_state)


register("actor-critic", ActorCriticPolicy)
register("actor-critic-separate",
         functools.partial(ActorCriticPolicy, shared_backbone=False, auxiliary_head=False))
register("actor-critic-shared",
         functools.partial(ActorCriticPolicy, shared_backbone=True, auxiliary_head=False))
register("actor-critic-auxiliary",
         functools.partial(ActorCriticPolicy, shared_backbone=False, auxiliary_head=True))
register("gym_mujoco", functools.partial(ActorCriticPolicy, continuous_action=True))
register("actor-critic-separate-continuous-action",
         functools.partial(ActorCriticPolicy, continuous_action=True))
