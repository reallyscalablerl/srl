from rlsrl.api.config import *


class AtariBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws = 8
        self.pws = 1
        self.tws = 1

    def initial_setup(self):
        policy_name = "default"
        sample_stream = policy_name
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)
        inference_stream = policy_name
        policy = Policy(type_="atari_naive_rnn",
                        args=dict(
                            action_dim=6,
                            obs_dim={"obs": (3, 96, 96)},
                            rnn_hidden_dim=32,
                        ))
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.98,
                              gae_lambda=0.97,
                              eps_clip=0.2,
                              value_loss='huber',
                              value_loss_weight=0.5,
                              value_loss_config=dict(delta=10.0,),
                              entropy_bonus_weight=0.02,
                              optimizer='adam',
                              optimizer_config=dict(lr=1e-4),
                              max_grad_norm=10.0,
                              entropy_decay_per_steps=1000,
                              entropy_bonus_decay=0.99,
                              bootstrap_steps=20,
                          ))
        return ExperimentConfig(
            actor_workers=[
                ActorWorker(
                    env=Environment(type_="atari",
                                    args=dict(game_name="PongNoFrameskip-v4", obs_shape=(96, 96))),
                    inference_streams=[
                        inference_stream,
                    ],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=200,
                            bootstrap_steps=20,
                        )
                    ],
                    max_num_steps=2000,
                    inference_splits=4,
                    ring_size=40,
                    worker_info=WorkerInformation(log_terminal=(i==0),
                                                  log_wandb=(i==0))) for i in range(self.aws)
            ],
            policy_workers=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    policy=policy,
                    worker_info=WorkerInformation(log_terminal=True,
                                                  log_wandb=True,
                                                  device="0")
                ) for _ in range(self.pws)
            ],
            trainer_workers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=20,
                        reuses=1,
                        batch_size=1,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                    worker_info=WorkerInformation(log_terminal=True,
                                                  log_wandb=True,
                                                  device="0")
                ) for _ in range(self.tws)
            ],
        )


class AtariInlineBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws = 1
        self.tws = 1

    def initial_setup(self):
        policy_name = "default"
        sample_stream = policy_name
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)
        policy = Policy(type_="atari_naive_rnn",
                        args=dict(
                            action_dim=6,
                            obs_dim={"obs": (3, 96, 96)},
                            rnn_hidden_dim=32,
                        ))
        inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                           policy_name=policy_name,
                                           policy=policy,
                                           stream_name="")
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.98,
                              gae_lambda=0.97,
                              eps_clip=0.2,
                              value_loss='huber',
                              value_loss_weight=0.5,
                              value_loss_config=dict(delta=10.0,),
                              entropy_bonus_weight=0.02,
                              optimizer='adam',
                              optimizer_config=dict(lr=1e-4),
                              max_grad_norm=10.0,
                              entropy_decay_per_steps=1000,
                              entropy_bonus_decay=0.99,
                              bootstrap_steps=20,
                          ))
        return ExperimentConfig(
            actor_workers=[
                ActorWorker(
                    env=Environment(type_="atari",
                                    args=dict(game_name="PongNoFrameskip-v4", obs_shape=(96, 96))),
                    inference_streams=[
                        inference_stream,
                    ],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=200,
                            bootstrap_steps=20,
                        )
                    ],
                    max_num_steps=2000,
                    inference_splits=1,
                    ring_size=1,
                    worker_info=WorkerInformation(log_terminal=(i==0),
                                                  log_wandb=(i==0))) for i in range(self.aws)
            ],
            trainer_workers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=20,
                        reuses=1,
                        batch_size=10,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                    worker_info=WorkerInformation(log_terminal=True,
                                                  log_wandb=True,
                                                  device="0")
                ) for _ in range(self.tws)
            ],
        )


register_experiment("atari-benchmark", AtariBenchmarkExperiment)
register_experiment("atari-inline-benchmark", AtariInlineBenchmarkExperiment)