from typing import Optional, Dict, Any, List, Union
import dataclasses
import logging
import queue
import threading
import time
import wandb
import os

from rlsrl.api import config as config_pkg
from rlsrl.base.gpu_utils import set_cuda_device
import rlsrl.base.user
import rlsrl.base.names
import rlsrl.base.network

logger = logging.getLogger("worker")
_WANDB_LOG_FREQUENCY_SECONDS = 10
_TERMINAL_LOG_FREQUENCY_SECONDS = 10

@dataclasses.dataclass
class PollResult:
    # Number of total samples and batches processed by the worker. Specifically:
    # - For an actor worker, sample_count = batch_count = number of env.step()-s being executed.
    # - For a policy worker, number of inference requests being handled, versus how many batches were made.
    # - For a trainer worker, number of samples & batches fed into the trainer (typically GPU).
    # sample_count: int
    # batch_count: int
    valid: bool # whether the poll is valid

@dataclasses.dataclass
class LogEntry:
    """ One entry to be logged
    """
    stats: dict # if logged to wandb, wandb.log(stats, step=step) if step >= 0
    step: int = -1


class Worker:
    """The worker base class that provides general methods and entry point.

    For simplicity, we use a single-threaded pattern in implementing the worker RPC server. Logic
    of every worker are executed via periodical calls to the poll() method, instead of inside
    another thread or process (e.g. the gRPC implementation). A subclass only needs to implement
    poll() without duplicating the main loop.

    The typical code on the worker side is:
        worker = make_worker()  # Returns instance of Worker.
        worker.run()
    and the later is standardized here as:
        while exit command is not received:
            if worker is started:
                worker.poll()
    """

    def __init__(self):
        """Initializes a worker.
        """

        self.__running = False
        self.__exiting = False
        self.config = None
        self.__is_configured = False

        self.logger = logging.getLogger("worker")
        self.__worker_type = None
        self.__worker_index = None
        self.__last_successful_poll_time = None

        # Monitoring related.
        self._start_time_ns = None
        self.__wandb_run = None
        self.__wandb_args = None
        self.__log_wandb = None
        self.__wandb_last_log_time_ns = None
        self.__terminal_last_log_time_ns = None

        self.__wait_time_seconds = 0

    def __del__(self):
        if self.__wandb_run is not None:
            self.__wandb_run.finish()

    @property
    def is_configured(self):
        return self.__is_configured

    @property
    def wandb_run(self):
        if self.__wandb_run is None:
            wandb.login()
            for _ in range(10):
                try:
                    self.__wandb_run = wandb.init(dir=rlsrl.base.user.get_user_tmp(),
                                                  config=self.config,
                                                  resume="allow",
                                                  **self.__wandb_args)
                    break
                except wandb.errors.UsageError as e:
                    time.sleep(5)
            else:
                raise e
        return self.__wandb_run

    def _configure(self, config) -> config_pkg.WorkerInformation:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _poll(self) -> PollResult:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _stats(self) -> List[LogEntry]:
        """Implemented by sub-classes."""
        return []

    def configure(self, config):
        assert not self.__running
        self.logger.info("Configuring with: %s", config)
        self.__worker_device = config.worker_info.device
        if self.__worker_device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.__worker_device
        

        r = self._configure(config)
        self.__worker_type = r.worker_type
        self.__worker_index = r.worker_index
        self.logger = logging.getLogger(r.worker_type + "-worker")

        # config wandb logging
        self.__wandb_run = None  # This will be lazy created by self.wandb_run().
        self.__log_wandb = (self.__worker_index == 0
                            and self.__worker_type == "trainer") if r.log_wandb is None else r.log_wandb
        self.__wandb_args = dict(
            entity=r.wandb_entity,
            project=r.wandb_project or f"{r.experiment_name}",
            group=r.wandb_group or r.trial_name,
            job_type=r.wandb_job_type or f"{r.worker_type}",
            name=r.wandb_name or f"{r.policy_name or r.worker_index}",
            id=
            f"{r.experiment_name}_{r.trial_name}_{r.policy_name or 'unnamed'}_{r.worker_type}_{r.worker_index}",
            settings=wandb.Settings(start_method="fork"),
        )
        # config std output logging
        self.__log_terminal = r.log_terminal

        self.__is_configured = True
        self.logger.info(f"Configured {self.__worker_type} {self.__worker_index} successfully, worker device {self.__worker_device}.")

    def start(self):
        self.logger.info("Starting worker")
        self.__running = True

    def pause(self):
        self.logger.info("Pausing worker")
        self.__running = False

    def exit(self):
        self.logger.info("Exiting worker")
        self.__exiting = True

    def run(self):
        self._start_time_ns = time.monotonic_ns()
        self.logger.info("Running worker now")
        try:
            while not self.__exiting:
                
                if not self.__running:
                    time.sleep(0.05)
                if not self.__is_configured:
                    raise RuntimeError("Worker is not configured")
                if self.__last_successful_poll_time:
                    wait_seconds = (time.monotonic_ns() - self.__last_successful_poll_time) / 1e9
                else:
                    wait_seconds = 0.0

                r = self._poll()
                if self.__last_successful_poll_time:
                    one_poll_elapsed_time = (time.monotonic_ns() - self.__last_successful_poll_time) / 1e9
                else:
                    one_poll_elapsed_time = 0.1

                if r.valid:
                    # Record metrics.
                    self.__wait_time_seconds += wait_seconds
                    total_elapsed_time = (time.monotonic_ns() - self._start_time_ns) / 1e9
                    self.__last_successful_poll_time = time.monotonic_ns()
                    basic_stats = dict(
                                    this_poll_elapsed_time = one_poll_elapsed_time,
                                    total_elapsed_time = total_elapsed_time
                                  )
                    basic_log_entry = LogEntry(stats=basic_stats)
                    other_stats = self._stats()
                    if len(other_stats) > 0:
                        self.__log_poll_result(other_stats + [basic_log_entry])

        except KeyboardInterrupt:
            self.exit()
        except Exception as e:
            raise e
    
    def __log_poll_result(self, stats):
        self.__maybe_log_wandb(stats)
        self.__maybe_log_terminal(stats)

    def __maybe_log_wandb(self, stats: List[LogEntry]):
        if not self.__log_wandb:
            return
        now = time.monotonic_ns()
        if self.__wandb_last_log_time_ns is not None:  # Log with a frequency.
            if (now - self.__wandb_last_log_time_ns) / 1e9 < _WANDB_LOG_FREQUENCY_SECONDS:
                return
        self.__wandb_last_log_time_ns = now
        
        for e in stats:
            if e.step >= 0:
                self.wandb_run.log(e.stats, step=e.step)
            else:
                self.wandb_run.log(e.stats)
        
    def __maybe_log_terminal(self, stats):
        if not self.__log_terminal:
            return

        now = time.monotonic_ns()
        if self.__terminal_last_log_time_ns is not None:  # Log with a frequency.
            if (now - self.__terminal_last_log_time_ns) / 1e9 < _TERMINAL_LOG_FREQUENCY_SECONDS:
                return
        self.__terminal_last_log_time_ns = now

        self.logger.info(f"{self.__worker_type} {self.__worker_index} logging stats: ")
        self.__pretty_info_log(stats)
    
    def __pretty_info_log(self, stats: List[LogEntry]):
        res = {}
        for e in stats:
            if e.step >= 0:
                d = dict(**e.stats, step=e.step)
            else:
                d = e.stats
            for k, v in d.items():
                if k in res:
                    res[k].append(v)
                else:
                    res[k] = [v]
                
        for k, v in res.items():
            values = v[0] if len(v) == 1 else v
            self.logger.info(f"{k}: {values} ;")
    
        
        

class MappingThread:
    """Wrapped of a mapping thread.
    A mapping thread gets from up_stream_queue, process data, and puts to down_stream_queue.
    """

    def __init__(self,
                 map_fn,
                 upstream_queue,
                 downstream_queue: queue.Queue = None,
                 cuda_device=None):
        """Init method of MappingThread

        Args:
            map_fn: mapping function.
            upstream_queue: the queue to get data from.
            downstream_queue: the queue to put data after processing. If None, data will be discarded after processing.
        """
        self.__map_fn = map_fn
        self.__interrupt = False
        self.__upstream_queue = upstream_queue
        self.__downstream_queue = downstream_queue
        self.__thread = threading.Thread(target=self._run, daemon=True)
        self.__cuda_device = cuda_device

    def is_alive(self) -> bool:
        """Check whether the thread is alive.

        Returns:
            alive: True if the wrapped thread is alive, False otherwise.
        """
        return self.__interrupt or self.__thread.is_alive()

    def start(self):
        """Start the wrapped thread.
        """
        self.__thread.start()

    def join(self):
        """Join the wrapped thread.
        """
        self.__thread.join()

    def _run(self):
        if self.__cuda_device is not None:
            set_cuda_device(self.__cuda_device)
        while not self.__interrupt:
            self._run_step()

    def _run_step(self):
        try:
            data = self.__upstream_queue.get(timeout=1)
            data = self.__map_fn(data)
            if self.__downstream_queue is not None:
                self.__downstream_queue.put(data)
        except queue.Empty:
            pass

    def stop(self):
        """Stop the wrapped thread.
        """
        self.__interrupt = True
        if self.__thread.is_alive():
            self.__thread.join()
