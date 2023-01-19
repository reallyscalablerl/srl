"""This module defines the data flow between policy workers and actor workers.

In our design, actor workers are in charge of executing env.step() (typically simulation), while
policy workers running policy.rollout_step() (typically neural network inference). The inference
stream is the abstraction of the data flow between them: the actor workers send environment
observations as requests, and the policy workers return actions as responses, both plus other
additional information.
"""

import multiprocessing
from typing import List, Any
import logging
import numpy as np
import queue
import time
import psutil

from rlsrl.base.namedarray import recursive_aggregate
from rlsrl.system.inference_stream import InferenceClient, InferenceServer, register_client
import rlsrl.api.policy
import rlsrl.api.config
import rlsrl.base.network
import rlsrl.base.timeutil
import rlsrl.system.parameter_db as db

_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS = 2
_INLINE_PULL_PARAMETER_ON_START = True

logger = logging.getLogger("InferenceStream")

def get_policy_info(spec: rlsrl.api.config.InferenceStream):
    """ Only used in inline inference stream
    """
    if spec.foreign_policy:
        p = spec.foreign_policy
        if p.absolute_path:
            return rlsrl.api.config.PolicyInfo(absolute_path=p.absolute_path)
        worker_info = rlsrl.api.config.WorkerInformation(experiment_name=p.foreign_experiment_name,
                                                   trial_name=p.foreign_trial_name)
        policy_name = p.foreign_policy_name
        policy_identifier = p.foreign_policy_identifier
        param_db = p.param_db
    else:
        worker_info = spec.worker_info
        policy_name = spec.policy_name
        policy_identifier = spec.policy_identifier
        param_db = spec.param_db
    return rlsrl.api.config.PolicyInfo(worker_info=worker_info,
                                 policy_name=policy_name,
                                 policy_identifier=policy_identifier,
                                 param_db=param_db)


class InlineInferenceClient(InferenceClient):

    @property
    def type(self):
        return rlsrl.api.config.InferenceStream.Type.INLINE

    def __init__(self, spec: rlsrl.api.config.InferenceStream):
        self.policy_name = spec.policy_name
        import os
        os.environ["MARL_CUDA_DEVICES"] = "cpu"
        self.policy = rlsrl.api.policy.make(spec.policy)
        self.policy.eval_mode()
        self.__logger = logging.getLogger("Inline Inference")
        self._request_count = 0
        self.__request_buffer = []
        self._response_cache = {}
        self.__pull_freq_control = rlsrl.base.timeutil.FrequencyControl(
            frequency_seconds=spec.pull_interval_seconds, initial_value=_INLINE_PULL_PARAMETER_ON_START)
        self.__passive_pull_freq_control = rlsrl.base.timeutil.FrequencyControl(
            frequency_seconds=_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS,
            initial_value=_INLINE_PULL_PARAMETER_ON_START,
        )
        self.__accept_update_call = spec.accept_update_call

        # Parameter DB / Policy name related.
        self.__policy_info = get_policy_info(spec)
        self.__param_db = db.make_db(self.__policy_info.param_db, worker_info=self.__policy_info.worker_info,)

        self.__log_frequency_control = rlsrl.base.timeutil.FrequencyControl(frequency_seconds=300)

        # monitoring cpu usage for inline inference 
        self.__monitor_frequency_control = rlsrl.base.timeutil.FrequencyControl(frequency_seconds=10)
        import threading
        self.__monitor_thread = threading.Thread(target=self.monitor_run)
        self.__monitor_thread.start()

    def post_request(self, request: rlsrl.api.policy.RolloutRequest, flush=True) -> int:
        request.request_id = np.array([self._request_count], dtype=np.int64)
        req_id = self._request_count
        self.__request_buffer.append(request)
        self._request_count += 1
        if flush:
            self.flush()
        return req_id

    def is_ready(self, inference_ids: List[int]) -> bool:
        for req_id in inference_ids:
            if req_id not in self._response_cache:
                return False
        return True

    def consume_result(self, inference_ids: List[int]):
        return [self._response_cache.pop(req_id) for req_id in inference_ids]

    def load_parameter(self):
        """Method exposed to Actor worker so we can reload parameter when env is done.
        """
        if self.__passive_pull_freq_control.check() and self.__accept_update_call:
            # This reduces the unnecessary workload of mongodb.
            self.__load_parameter()

    def __get_checkpoint_from_db(self, block=False):
        if self.__policy_info.absolute_path is not None:
            return self.__param_db.get_file(self.__policy_info.absolute_path)
        else:
            return self.__param_db.get(name=self.__policy_info.policy_name,
                                       identifier=self.__policy_info.policy_identifier,
                                       block=block)

    def __load_parameter(self):
        policy_name = self.policy_name
        checkpoint = self.__get_checkpoint_from_db(block=self.policy.version < 0)
        self.policy.load_checkpoint(checkpoint)
        self.policy_name = policy_name
        self.__logger.debug(f"Loaded {self.policy_name}'s parameter of version {self.policy.version}")

    def flush(self):
        if self.__pull_freq_control.check():
            self.__load_parameter()

        if self.__log_frequency_control.check():
            self.__logger.info(f"Policy Version: {self.policy.version}")

        if len(self.__request_buffer) > 0:
            self.__logger.debug("Inferencing")
            agg_req = recursive_aggregate(self.__request_buffer, np.stack)
            rollout_results = self.policy.rollout(agg_req)
            rollout_results.request_id = agg_req.request_id
            rollout_results.policy_version_steps = np.full(shape=agg_req.client_id.shape,
                                                           fill_value=self.policy.version)
            rollout_results.policy_name = np.full(shape=agg_req.client_id.shape, fill_value=self.policy_name)
            self.__request_buffer = []
            for i in range(rollout_results.length(dim=0)):
                self._response_cache[rollout_results.request_id[i, 0]] = rollout_results[i]

    def get_constant(self, name: str) -> Any:
        if name == "default_policy_state":
            return self.policy.default_policy_state
        else:
            raise NotImplementedError(name)
    
    def monitor_run(self):
        pid = multiprocessing.current_process().pid
        process = psutil.Process(pid=pid)
        while True:
            time.sleep(0.5)
            if self.__monitor_frequency_control.check():
                self.__logger.info(f"inline inference pid {pid}, cpu percent {process.cpu_percent()}")


class LocalInferenceClient(InferenceClient):

    @property
    def type(self):
        return None

    def __init__(self, req_q, resp_q):
        self.req_q = req_q
        self.resp_q = resp_q
        self.client_id = np.random.randint(0, 2147483647)
        self._request_count = 0

        self.__request_buffer = []
        self._response_cache = {}
        self._pending_requests = {}

    def post_request(self, request: rlsrl.api.policy.RolloutRequest, flush=True) -> int:
        request.client_id = np.array([self.client_id], dtype=np.int32)
        request.request_id = np.array([self._request_count], dtype=np.int64)

        req_id = self._request_count
        self.__request_buffer.append(request)

        self._pending_requests[req_id] = request
        self._request_count += 1

        if flush:
            self.flush()
        return req_id

    def __poll_responses(self):
        """Get all action messages from inference servers."""
        try:
            responses = rlsrl.base.namedarray.loads(self.resp_q.get_nowait())
            for i in range(responses.length(dim=0)):
                req_id = responses.request_id[i, 0]
                if req_id in self._response_cache:
                    raise ValueError(
                        "receiving multiple result with request id {}."
                        "Have you specified different inferencer client_ids for actors?".format(req_id))
                else:
                    self._pending_requests.pop(req_id)
                    self._response_cache[req_id] = responses[i]
        except queue.Empty:
            pass
        except Exception as e:
            raise e

    def is_ready(self, inference_ids) -> bool:
        self.__poll_responses()
        for req_id in inference_ids:
            if req_id not in self._response_cache:
                return False
        return True

    def consume_result(self, inference_ids):
        return [self._response_cache.pop(req_id) for req_id in inference_ids]

    def flush(self):
        if len(self.__request_buffer) > 0:
            agg_request = rlsrl.base.namedarray.dumps(recursive_aggregate(self.__request_buffer, np.stack))
            self.req_q.put(agg_request)
            self.__request_buffer = []

    def get_constant(self, name: str) -> Any:
        name_, value = self.resp_q.get(timeout=30)
        if name_ != name:
            raise ValueError(f"Unexpected constant name: {name_} != {name}")
        return value


class LocalInferenceServer(InferenceServer):

    def __init__(self, req_qs, resp_qs):
        self.req_qs = req_qs
        self.resp_qs = resp_qs
        self.client_id_queue_map = {}

    def poll_requests(self):
        request_batches = []
        for q1, q2 in zip(self.req_qs, self.resp_qs):
            try:
                requests: rlsrl.api.policy.RolloutRequest = rlsrl.base.namedarray.loads(q1.get_nowait())
                requests.received_time[:] = time.monotonic_ns()
                client_id = requests.client_id[0, 0].item()
                self.client_id_queue_map[client_id] = q2
                request_batches.append(requests)
            except queue.Empty:
                break
        return request_batches

    def respond(self, responses: rlsrl.api.policy.RolloutResult):
        idx = np.concatenate([[0],
                              np.where(np.diff(responses.client_id[:, 0]))[0] + 1, [responses.length(dim=0)]])
        for i in range(len(idx) - 1):
            client_id = responses.client_id[idx[i], 0].item()
            self.client_id_queue_map[client_id].put(rlsrl.base.namedarray.dumps(responses[idx[i]:idx[i + 1]]))

    def set_constant(self, name, value):
        for q in self.resp_qs:
            q.put((name, value))


register_client(rlsrl.api.config.InferenceStream.Type.INLINE, InlineInferenceClient)


def make_local_server(req_qs, resp_qs) -> InferenceServer:
    return LocalInferenceServer(req_qs, resp_qs)


def make_local_client(req_q, resp_q) -> InferenceClient:
    return LocalInferenceClient(req_q, resp_q)
