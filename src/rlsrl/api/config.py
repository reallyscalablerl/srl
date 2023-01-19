from typing import Dict, List, Optional, Any, Union
import enum
import dataclasses


@dataclasses.dataclass
class Environment:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Policy:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Trainer:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ParameterDB:

    class Type(enum.Enum):
        FILESYSTEM = 1  # Saves parameters to shared filesystem.

    type_: Type


@dataclasses.dataclass
class ForeignPolicy:
    """A Policy is foreign if any of the configurations below differs from the worker's original configuration.
    Workers behave differently when receiving a foreign parameter_db.
    1. Trainer will read this policy when initializing, except when it is resuming a previous trial. Trained Parameters
    will be pushed to its domestic parameter db.
    2. Policy Worker/ InlineInference: foreign policy will overwrite domestic policy. i.e. worker will always load
    foreign policy.

    NOTE:
        -1. If absolute_path is not None, All others configurations will be ignored.
        0. When absent(default to None), the absent fields will be replaced by domestic values.
        1. Foreign policies are taken as `static`. No workers should try to update a foreign policy. And foreign policy
        name should not appear in the domestic experiment.
        2. Currently only Trainer Worker has foreign policy implemented. If you need to use foreign policy for
        inference only, a workaround is to use a dummy trainer, with no gpu assigned. to transfer the policy to
        domestic parameter db.
    """
    foreign_experiment_name: Optional[str] = None
    foreign_trial_name: Optional[str] = None
    foreign_policy_name: Optional[str] = None
    foreign_policy_identifier: Optional[str] = None
    absolute_path: Optional[str] = None

    param_db: ParameterDB = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)


@dataclasses.dataclass
class InferenceStream:

    class Type(enum.Enum):
        LOCAL = 1 
        INLINE = 2

    type_: Type
    stream_name: str  # Must be filled. But only NAMED wil use this.
    policy: Policy = None  # Only INLINE will use this.
    policy_name: Optional[str] = None  # Only INLINE will use this.
    foreign_policy: Optional[ForeignPolicy] = None  # Only INLINE will use this.
    accept_update_call: bool = True  # Only INLINE will use this.
    # If None, policy name will be sampled uniformly from available policies.
    policy_identifier: Union[str, Dict, List] = "latest"  # Only INLINE will use this.
    param_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)  # Only INLINE will use this.
    pull_interval_seconds: Optional[int] = None  # Only INLINE will use this.
    worker_info = None


@dataclasses.dataclass
class SampleStream:

    class Type(enum.Enum):
        NULL = 0  # Only producer side is implemented. NULL Producer discards all samples.
        LOCAL = 1

    type_: Type
    stream_name: str = ""

@dataclasses.dataclass
class WorkerInformation:
    """The basic information of an worker. To improve config readability, the experiment starter will fill the
    fields, instead of letting the users do so in experiment configs.
    """
    experiment_name: str = ""
    trial_name: str = ""  # Name of the trial of the experiment; e.g. "{USER}-0".
    worker_type: str = ""  # E.g. "policy", "actor", or "trainer".
    worker_index: int = -1  # The index of the worker of the specific type, starting from 0.
    worker_count: int = 0  # Total number of workers; hence, 0 <= worker_index < worker_count.
    worker_tag: Optional[str] = None  # For actor and policy worker, can be "training" or "evaluation".
    policy_name: Optional[str] = None  # For trainer and policy worker, the name of the policy.
    wandb_entity: Optional[
        str] = None  # wandb_{config} are optional. They overwrite system wandb_configuration.
    wandb_project: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    log_wandb: Optional[bool] = None
    log_terminal: Optional[bool] = None
    device: str = 'cpu'

    def system_setup(self, experiment_name, trial_name, worker_type, worker_index, worker_count, policy_name):
        """Setup system related worker information, while leaving the rest untouched.
        """
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.policy_name = policy_name


@dataclasses.dataclass
class AgentSpec:
    """The configuration of agents of each actor worker consists of a list of AgentSpec-s. Each AgentSpec
    matches some of the agents, as well as provides inference stream and sample stream configs.
    """
    index_regex: str
    inference_stream_idx: int  # Multiple inference stream is not ready yet.
    sample_stream_idx: Union[int, List[int]]
    sample_steps: int = 200
    bootstrap_steps: int = 1
    deterministic_action: bool = False
    send_after_done: bool = False
    send_full_trajectory: bool = False
    pad_trajectory: bool = False  # only used if send_full_trajectory
    send_concise_info: bool = False
    stack_frames: int = 0  # 0: raw stacking; 1: add new axis on 0; >=2: stack n frames on axis 0


@dataclasses.dataclass
class ActorWorker:
    """Provides the full configuration for an actor worker.
    """
    env: Union[str, Environment]  # If str, equivalent to Environment(type_=str).
    sample_streams: List[Union[str, SampleStream]]
    inference_streams: List[Union[str, InferenceStream]]
    agent_specs: List[AgentSpec]
    max_num_steps: int = 100000
    ring_size: int = 2
    # Actor worker will split the ring into `inference_splits` parts # and flush the inference clients for each part.
    inference_splits: int = 2
    worker_info: Optional[WorkerInformation] = WorkerInformation()  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class PolicyWorker:
    """Provides the full configuration for a policy worker.
    """
    policy_name: str
    inference_stream: Union[str, InferenceStream]
    policy: Union[str, Policy]  # If str, equivalent to Policy(type_=str).
    parameter_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)
    batch_size: int = 10240  # Batch size is an upper bound. Set larger unless you experience OOM issue.
    policy_identifier: Union[str, Dict, List] = "latest"
    pull_frequency_seconds: float = 1
    foreign_policy: Optional[ForeignPolicy] = None
    worker_info: Optional[WorkerInformation] = WorkerInformation()  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class TrainerWorker:
    """Provides the full configuration for a trainer worker.
    """
    policy_name: str
    trainer: Union[str, Trainer]  # If str, equivalent to Trainer(type_=str).
    policy: Union[str, Policy]  # If str, equivalent to Policy(type_=str).
    sample_stream: Union[str, SampleStream]
    foreign_policy: Optional[ForeignPolicy] = None
    parameter_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)
    # cudnn benchmark is related to the speed of CNN
    cudnn_benchmark: bool = False
    # cudnn deterministic is related to reproducibility
    cudnn_determinisitc: bool = True
    buffer_name: str = "priority_queue"
    buffer_args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    log_frequency_seconds: int = 5
    log_frequency_steps: int = None
    push_frequency_seconds: Optional[float] = 1.
    push_frequency_steps: Optional[int] = 1
    push_tag_frequency_minutes: Optional[int] = None  # Adds a tagged policy version regularly.
    worker_info: Optional[WorkerInformation] = WorkerInformation()  # Specify your wandb_config here, or leave None.
    world_size: int = 0 # If world_size > 0, DDP will find exact world_size number of DDP peers, 
                        # otherwise raise exception. 
                        # If world_size = 0, DDP will find as much peers as possible. DDP peer did not 
                        # connect to master will fail without terminating experiment. 


@dataclasses.dataclass
class EvaluationManager:
    policy_name: str
    eval_sample_stream: Union[str, SampleStream]
    parameter_db: ParameterDB = ParameterDB(ParameterDB.Type.FILESYSTEM)
    eval_target_tag: str = "latest"
    eval_tag: str = "eval"
    eval_games_per_version: Optional[int] = 100
    eval_time_per_version_seconds: Optional[float] = None
    unique_policy_version: Optional[bool] = True
    log_evaluation: bool = True
    update_metadata: bool = True
    worker_info: Optional[WorkerInformation] = WorkerInformation()  # Specify your wandb_config here, or leave None.


@dataclasses.dataclass
class ExperimentConfig:
    actor_workers: List[ActorWorker] = dataclasses.field(default_factory=list)
    policy_workers: List[PolicyWorker] = dataclasses.field(default_factory=list)
    trainer_workers: List[TrainerWorker] = dataclasses.field(default_factory=list)
    eval_managers: Optional[List[EvaluationManager]] = dataclasses.field(default_factory=list)

    def set_worker_information(self, experiment_name, trial_name):
        for worker_type, workers in [
            ("actor", self.actor_workers),
            ("policy", self.policy_workers),
            ("trainer", self.trainer_workers),
            ("eval_manager", self.eval_managers),
        ]:
            for i, worker in enumerate(workers):
                if worker_type in ("policy", "trainer", "buffer", "eval_manager"):
                    policy_name = worker.policy_name
                else:
                    policy_name = None

                system_worker_info = dict(experiment_name=experiment_name,
                                          trial_name=trial_name,
                                          worker_type=worker_type,
                                          worker_index=i,
                                          worker_count=len(workers),
                                          policy_name=policy_name)
                if worker.worker_info is not None:
                    worker.worker_info.system_setup(**system_worker_info)
                else:
                    worker.worker_info = WorkerInformation(**system_worker_info)


@dataclasses.dataclass
class PolicyInfo:
    """ Information required to pull a policy model from database.
    """
    worker_info: Optional[WorkerInformation] = None
    policy_name: Optional[str] = None
    policy_identifier: Optional[str] = None
    absolute_path: Optional[str] = None
    
    param_db: ParameterDB = ParameterDB(type_=ParameterDB.Type.FILESYSTEM)


class Experiment:
    """Base class for defining the procedure of an experiment.
    """

    def initial_setup(self) -> ExperimentConfig:
        """Returns a list of workers to create when a trial of the experiment is initialized."""
        raise NotImplementedError()


ALL_EXPERIMENT_CLASSES = {}


def register_experiment(name, *cls):
    assert name not in ALL_EXPERIMENT_CLASSES
    ALL_EXPERIMENT_CLASSES[name] = cls


def make_experiment(name) -> List[Experiment]:
    classes = ALL_EXPERIMENT_CLASSES[name]
    return [cls() for cls in classes]
