from rlsrl.api.config import *


gym_mujoco_registry = {
    # env_name: (obs_dim, act_dim)
    "Humanoid-v4": (376, 17),
    "Humanoid-v3": (376, 17),
    "HumanoidStandup-v2": (376, 17),
    "HalfCheetah-v3": (17, 6),
    "Ant-v3": (111, 8),
    "Walker2d-v3": (17, 6),
    "Hopper-v3": (11, 3),
    "Swimmer-v3": (8, 2),
    "InvertedPendulum-v2": (4, 1),
    "InvertedDoublePendulum-v2": (11, 1),
    "Reacher-v2": (11, 2),
}


class GymMuJoCoBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws = 8
        self.pws = 1
        self.tws = 1

    def initial_setup(self):
        scenario = 'Humanoid-v4'
        policy_name = "default"
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)
        policy = Policy(type_="gym_mujoco",
                        args=dict(obs_dim=gym_mujoco_registry[scenario][0],
                                  action_dim=gym_mujoco_registry[scenario][1]))
        return ExperimentConfig(
            actor_workers=[
                ActorWorker(
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    env=Environment(type_="gym_mujoco", args=dict(scenario=scenario)),
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


register_experiment("gym-mujoco-benchmark", GymMuJoCoBenchmarkExperiment)
