from typing import List
import logging
import numpy as np
import queue
import time

from rlsrl.base.namedarray import recursive_aggregate
from rlsrl.base.timeutil import FrequencyControl
from rlsrl.system import worker_base, inference_stream
import rlsrl.api.config
import rlsrl.api.policy
import rlsrl.system.parameter_db as db

def get_policy_info(cfg: rlsrl.api.config.PolicyWorker):
    if cfg.foreign_policy:
        p = cfg.foreign_policy
        if p.absolute_path:
            return rlsrl.api.config.PolicyInfo(absolute_path=p.absolute_path)
        worker_info = rlsrl.api.config.WorkerInformation(experiment_name=p.foreign_experiment_name,
                                                   trial_name=p.foreign_trial_name)
        policy_name = p.foreign_policy_name
        policy_identifier = p.foreign_policy_identifier
        param_db = p.param_db
    else:
        worker_info = cfg.worker_info
        policy_name = cfg.policy_name
        policy_identifier = cfg.policy_identifier
        param_db = cfg.parameter_db
    return rlsrl.api.config.PolicyInfo(worker_info=worker_info,
                                 policy_name=policy_name,
                                 policy_identifier=policy_identifier,
                                 param_db=param_db)


class PolicyWorker(worker_base.Worker):

    def __init__(self):
        super().__init__()

        self.config = None
        self.__policy_info = None
        self.__stream = None
        self.__policy = None
        self.__param_db = None
        self.__pull_frequency_control = None
        self.__requests_buffer = []
        self.__is_paused = False

        # Queues in between the data pipeline.
        self.__inference_queue = queue.Queue(1)
        self.__respond_queue = queue.Queue(1)
        self.__param_queue = queue.Queue(1)

        # The mapping threads.
        self._threads: List[worker_base.MappingThread] = []

        self.__log_frequency_control = FrequencyControl(frequency_seconds=10, initial_value=True)

        # logging stats
        self.__samples = 0
        self.__batches = 0

    def _configure(self, cfg: rlsrl.api.config.PolicyWorker):
        self.logger = logging.getLogger(f"PW{cfg.worker_info.worker_index}")
        self.config = cfg
        self.config.worker_info.policy_name = cfg.policy_name
        self.__bs = self.config.batch_size
        self.__policy_info = get_policy_info(cfg)
        self.__pull_frequency_control = FrequencyControl(
            frequency_seconds=cfg.pull_frequency_seconds,
            # If policy has a specified initial state, do not pull the
            # saved version immediately.
            initial_value=(self.__policy_info.absolute_path is None))

        self.__policy: rlsrl.api.policy.Policy = rlsrl.api.policy.make(cfg.policy)
        self.__policy.eval_mode()

        self.__param_db = db.make_db(self.__policy_info.param_db, 
                                     worker_info=self.__policy_info.worker_info)

        # Initialize inference stream server.
        self.__stream = inference_stream.make_server(cfg.inference_stream,
                                                     worker_info=self.config.worker_info)
        self.__stream.set_constant("default_policy_state", self.__policy.default_policy_state)

        # Start inference and respond threads.
        self._threads.append(
            worker_base.MappingThread(self._inference,
                                      self.__inference_queue,
                                      self.__respond_queue,
                                      cuda_device=self.__policy.device))
        self._threads.append(
            worker_base.MappingThread(self._respond,
                                      self.__respond_queue,
                                      downstream_queue=None))
        [t.start() for t in self._threads]

        return self.config.worker_info

    def _inference(self, agg_requests: rlsrl.api.policy.RolloutRequest):
        """Run inference for batched aggregated_request.
        """
        try:
            checkpoint = self.__param_queue.get_nowait()
            self.__policy.load_checkpoint(checkpoint)
            self.logger.debug(f"Loaded checkpoint version: {self.__policy.version}")
        except queue.Empty:
            pass
        responses = self.__policy.rollout(agg_requests)
        responses.client_id = agg_requests.client_id
        responses.request_id = agg_requests.request_id
        responses.received_time = agg_requests.received_time
        responses.policy_name = np.full(shape=agg_requests.client_id.shape, fill_value=self.__policy_info.policy_name)
        responses.policy_version_steps = np.full(shape=agg_requests.client_id.shape,
                                                 fill_value=self.__policy.version)
        return responses

    def _respond(self, responses: rlsrl.api.policy.RolloutResult):
        """Send rollout results.
        """
        self.__stream.respond(responses)

    def _stats(self):
        stats = dict(total_steps = self.__samples,
                     total_episodes = self.__batches)
        return [worker_base.LogEntry(stats=stats)]

    def _poll(self):
        for t in self._threads:
            if not t.is_alive():
                raise RuntimeError("Exception in policy thread.")

        # Pull parameters from server
        if self.__pull_frequency_control.check():
            self.logger.debug("Active pull.")
            while not self.__param_queue.empty():
                self.__param_queue.get()
            is_first_pull = self.__policy.version < 0
            self.__param_queue.put(self.__get_checkpoint_from_db(block=is_first_pull))

        samples = 0
        batches = 0
        # buffer requests, and record when the oldest requests is received.
        request_batch = self.__stream.poll_requests()
        self.__requests_buffer.extend(request_batch)

        if len(self.__requests_buffer) > 0:
            try:
                # If the inference has not started on the queued batch, make the batch larger instead of
                # initiating another batch.
                queued_requests = [self.__inference_queue.get_nowait()]
                samples -= queued_requests[0].length(dim=0)
                batches -= 1
            except queue.Empty:
                queued_requests = []
            agg_request = recursive_aggregate(queued_requests + self.__requests_buffer,
                                              lambda x: np.concatenate(x, axis=0))
            bs = min(self.__bs, agg_request.length(dim=0))
            self.__inference_queue.put_nowait(agg_request[:bs])
            samples += bs
            batches += 1
            if bs == agg_request.length(dim=0):
                self.__requests_buffer = []
            else:
                self.__requests_buffer = [agg_request[bs:]]

        if self.__log_frequency_control.check():
            self.logger.debug(f"Policy version: {self.__policy.version}")

        self.__samples += samples
        self.__batches += batches
        valid = not (samples == 0 and batches == 0)

        # avoid locking thread
        time.sleep(0.005)
        return worker_base.PollResult(valid=valid)

    def __get_checkpoint_from_db(self, block=False):
        if self.__policy_info.absolute_path is not None:
            return self.__param_db.get_file(self.__policy_info.absolute_path)
        else:
            return self.__param_db.get(name=self.__policy_info.policy_name,
                                       identifier=self.__policy_info.policy_identifier,
                                       block=block)

    def __stop_threads(self):
        self.logger.debug(f"Stopping {len(self._threads)} local threads.")
        for t in self._threads:
            t.stop()

    def pause(self):
        super(PolicyWorker, self).pause()
        self.__is_paused = True

    def start(self):
        if self.__is_paused:
            # Set block=True to wait for trainers to push the first model when starting a new PSRO iteration.
            while not self.__param_queue.empty():
                self.__param_queue.get()
            self.__param_queue.put(self.__param_db.get(self.__get_checkpoint_from_db(block=True)))
            self.__is_paused = False

        super(PolicyWorker, self).start()

    def exit(self):
        self.__stop_threads()
        super(PolicyWorker, self).exit()
