from rlsrl.api.config import *


class SMACBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws = 4
        self.pws = 1
        self.tws = 1

    def initial_setup(self):
        policy_name = "default"
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)

        map_name = "3m"
        seed = 1

        inference_stream = policy_name
        sample_stream = policy_name
            

        actor_list = [
                ActorWorker(
                    env=Environment(type_="smac", args=dict(map_name=map_name)),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=100,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=8,
                    inference_splits=2,
                    worker_info=WorkerInformation(log_terminal=(i==0), log_wandb=(i==0))
                ) for i in range(self.aws)
            ]

        policy_list = [
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    policy=Policy(type_="smac_rnn",
                              args=dict(
                                  map_name=map_name,
                                  hidden_dim=32,
                                  chunk_len=5,
                                  seed=seed,
                              ),
                           ),
                    worker_info=WorkerInformation(log_terminal=True,
                                                  log_wandb=True,
                                                  device="0")
                ) for _ in range(self.pws)
            ]

        trainer_list = [ 
            TrainerWorker(
                buffer_name='priority_queue',
                buffer_args=dict(
                    max_size=32,
                    reuses=1,
                    batch_size=10,
                ),
                policy_name=policy_name,
                trainer="mappo",
                policy=Policy(type_="smac_rnn",
                              args=dict(
                                  map_name=map_name,
                                  hidden_dim=32,
                                  chunk_len=5,
                                  seed=seed,
                              )),
                sample_stream=sample_stream,
                parameter_db=parameter_db,
                worker_info=WorkerInformation(log_terminal=True, log_wandb=True, device="0")
            ) for _ in range(self.tws)]

        return ExperimentConfig(actor_workers=actor_list, policy_workers=policy_list, trainer_workers=trainer_list)


register_experiment("smac-benchmark", SMACBenchmarkExperiment)
