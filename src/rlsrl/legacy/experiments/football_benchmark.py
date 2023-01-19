from rlsrl.api.config import *


map_agent_registry = {
    # evn_name: (left, right, game_length)
    "11_vs_11_competition": (11, 11, 3000),
    "11_vs_11_easy_stochastic": (11, 11, 3000),
    "11_vs_11_hard_stochastic": (11, 11, 3000),
    "11_vs_11_kaggle": (11, 11, 3000),
    "11_vs_11_stochastic": (11, 11, 3000),
    "1_vs_1_easy": (1, 1, 500),
    "5_vs_5": (4, 4, 3000),
    "academy_3_vs_1_with_keeper": (3, 1, 400),
    "academy_corner": (11, 11, 400),
    "academy_counterattack_easy": (11, 11, 400),
    "academy_counterattack_hard": (11, 11, 400),
}

class FootballBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws, self.pws, self.tws = 8, 1, 1

    def initial_setup(self):
        policy_name = "default"
        policy = Policy(type_="football-simple115-separate")
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)
        return ExperimentConfig(
            actor_workers=[
                ActorWorker(
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    env=Environment(type_="football",
                                    args=dict(
                                        env_name="11_vs_11_stochastic",
                                        number_of_left_players_agent_controls=11,
                                        number_of_right_players_agent_controls=11,
                                        representation="simple115v2",
                                    )),
                    agent_specs=[AgentSpec(
                        index_regex=".*",
                        inference_stream_idx=0,
                        sample_stream_idx=0,
                    )],
                    worker_info=WorkerInformation(log_terminal=(i==0), log_wandb=(i==0))
                ) for i in range(self.aws)
            ],
            policy_workers=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=policy_name,
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
                        max_size=128,
                        reuses=1,
                        batch_size=10,
                    ),
                    policy_name=policy_name,
                    trainer="mappo",
                    policy=policy,
                    sample_stream=policy_name,
                    parameter_db=parameter_db,
                    worker_info=WorkerInformation(log_terminal=True, log_wandb=True, device="0")
                ) for _ in range(self.tws)
            ],
        )


register_experiment("football-benchmark", FootballBenchmarkExperiment)
