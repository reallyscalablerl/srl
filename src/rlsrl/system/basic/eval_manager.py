import numpy as np
import logging

from rlsrl.base.timeutil import FrequencyControl
from rlsrl.base.namedarray import recursive_aggregate, recursive_apply
from rlsrl.system import worker_base
import rlsrl.system.parameter_db
import rlsrl.api.config


class EvalManager(worker_base.Worker):
    """Currently, Evaluation Manager can do two jobs:
    1. Run online evaluation for a policy.
    2. (following 1), submit the evaluation result to some curriculum.
    3. Update metadata based on the the evaluation results.

    EvalManager exposes a sample consumer that accepts episodes evaluation results. The eval manager
    expects the first dimension(time dimension) is of size 1. If that is not the case, the eval manager
    only keeps the last time step of the sample.
    The eval manager uses a tag to tell policies/actors
    which version should be evaluated. If the received sample doesn't match the specified version, the
    evaluation result is ignored.
    The eval manager also updates the metadata of parameter versions based on the samples it receives.
    If its parameter db does not support metadata updating, or the update fails due to version in-consistency
    with the parameter-db, nothing will be updated.
    """

    def __init__(self, server=None):
        super().__init__(server)
        self.config = None
        self.__eval_stream = None
        self.__param_db = None
        self.__eval_target_tag = None
        self.__eval_tag = None
        self.__eval_games_per_version = None
        self.__eval_time_per_version_seconds = None
        self.__current_eval_version = None
        self.__unique_policy_version = True

        self.__eval_frames = 0
        self.__eval_cache = None
        self.__eval_results_dict = {}

        self.logger = logging.getLogger("Eval Manager")

    def _configure(self, cfg: rlsrl.api.config.EvaluationManager) -> rlsrl.api.config.WorkerInformation:
        self.config: rlsrl.api.config.EvaluationManager = cfg
        self.__eval_stream = rlsrl.system.sample_stream.make_consumer(cfg.eval_sample_stream,
                                                                      worker_info=cfg.worker_info)
        self.__eval_target_tag = cfg.eval_target_tag
        self.__eval_tag = cfg.eval_tag
        self.__param_db = rlsrl.rlsrl.system.parameter_db.make_db(cfg.parameter_db, worker_info=cfg.worker_info)
        self.__policy_name = cfg.policy_name
        self.__eval_games_per_version = cfg.eval_games_per_version
        self.__eval_time_per_version_seconds = cfg.eval_time_per_version_seconds
        self.__eval_tag_control = FrequencyControl(frequency_steps=cfg.eval_games_per_version,
                                                   frequency_seconds=cfg.eval_time_per_version_seconds)
        self.__unique_policy_version = cfg.unique_policy_version

        self.__log_evaluation = cfg.log_evaluation
        self.__update_metadata = cfg.update_metadata

        r = self.config.worker_info
        r.policy_name = f"{self.__policy_name}_{self.__eval_tag}" or str(r.worker_index)
        return r

    def __tag_new_version_for_evaluation(self):
        self.__param_db.tag(self.__policy_name, self.__eval_target_tag, self.__eval_tag)
        target_version = self.__param_db.version_of(self.__policy_name, self.__eval_tag)
        self.logger.info(f"Setting evaluation version to {target_version}")
        self.__current_eval_version = target_version
        self.__eval_cache = []

    def _poll(self) -> worker_base.PollResult:
        """Dead loop method of evaluation manager.
        Returns:
            PollResults: samples ( how many episodes are received. ) Note that not all
        """
        sample_count = 0
        batch_count = 0
        if self.__current_eval_version is None and self.__log_evaluation:
            if self.__param_db.has_tag(self.__policy_name, self.__eval_target_tag):
                self.__tag_new_version_for_evaluation()

        try:
            sample = self.__eval_stream.consume()
            sample_policy_name = sample.unique_of("policy_name")
            sample_version = sample.unique_of("policy_version_steps")
            elapsed_episodes = sample.info_mask.sum()
            if elapsed_episodes == 0:
                sample_info = None
            else:
                sample_info = recursive_apply(sample.info * sample.info_mask,
                                              lambda x: x.sum()) / elapsed_episodes
            sample_count += 1
            self.__eval_frames += len(sample)
        except rlsrl.system.local_sample.NothingToConsume:
            return worker_base.PollResult(sample_count=sample_count, batch_count=0)

        if sample_info is None:
            return worker_base.PollResult(sample_count=sample_count, batch_count=0)

        if sample_policy_name != self.__policy_name or (self.__unique_policy_version
                                                        and sample_version is None):
            self.logger.info(f"Sample of policy name {sample_policy_name} (expected {self.__policy_name}) "
                             f"version {sample_version} is ignored.")
            return worker_base.PollResult(sample_count=sample_count, batch_count=0)

        if self.__log_evaluation:
            if not self.__unique_policy_version or sample_version == self.__current_eval_version:
                self.__eval_cache.append(sample_info)
                if self.__eval_tag_control.check() and len(self.__eval_cache) > 0:
                    batch_count = 1
                    agg_info = self.__unpack_info(
                        recursive_aggregate(self.__eval_cache, lambda x: np.mean(np.stack(x))))
                    agg_info.update(dict(version=sample_version, frames=self.__eval_frames))
                    self.__eval_results_dict[self.__current_eval_version] = agg_info
                    self.wandb_run.log(agg_info, step=self.__current_eval_version)
                    self.__tag_new_version_for_evaluation()

        if self.__update_metadata:
            metadata = self.__unpack_info(sample_info)
            try:
                self.__param_db.update_metadata(self.__policy_name,
                                                version=str(sample_version),
                                                metadata=metadata)
                batch_count = 1
            except NotImplementedError:
                self.logger.debug("Parameter DB doesn't support metadata updating.")
            except FileNotFoundError:
                self.logger.debug("Parameter version now found in filesystem, possibly due to parameter gc.")
            except KeyError:
                self.logger.debug("Parameter version is not recorded in metadata-db. This usually means that "
                                  "This version has no tag attached to it.")

        return worker_base.PollResult(sample_count=sample_count, batch_count=batch_count)

    def __unpack_info(self, info):
        return {k: info[k].item() for k in info.keys()}

    def pause(self):
        super(EvalManager, self).pause()
        self.__current_eval_version = None
