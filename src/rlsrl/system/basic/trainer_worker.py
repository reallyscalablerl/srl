from itertools import accumulate
from typing import Optional, Tuple, Any
import datetime
import logging
import numpy as np
import queue
import time
import threading
import torch

from rlsrl.base.gpu_utils import set_cuda_device
from rlsrl.base.network import find_free_port
from rlsrl.system.sample_stream import make_consumer
from rlsrl.system import worker_base
import rlsrl.api.config as config
import rlsrl.api.trainer
import rlsrl.base.buffer
import rlsrl.base.timeutil
import rlsrl.base.names as names
import rlsrl.system.parameter_db as parameter_db
import rlsrl.base.name_resolve as name_resolve

DDP_SETUP_WAIT_SECONDS = 1
DDP_MAX_TRAINER_STEP_DIFFERENCE = 3
PARAMETER_DB_GC_FREQUENCY_SECONDS = 300


class GPUThread:

    def __init__(self, buffer, trainer, is_master, log_frequency_seconds, log_frequency_steps,
                 push_frequency_seconds, push_frequency_steps, dist_kwargs):
        self.logger = logging.getLogger("gpu-thread")
        self.dist_kwargs = dist_kwargs
        self.__logging_queue = queue.Queue(8)
        self.__checkpoint_push_queue = queue.Queue(8)
        self.__replay_entry = None

        self.__buffer = buffer
        self.__is_master = is_master
        self.__trainer: rlsrl.api.trainer.Trainer = trainer

        self.__interrupting = False
        self.__interrupt_at_step = 1e10
        self.__steps = 0
        self.__samples = 0
        self.__thread = threading.Thread(target=self._run, daemon=True)

        # See rlsrl.base.timeutil for usage on FrequencyControl
        self.__logging_control = rlsrl.base.timeutil.FrequencyControl(frequency_seconds=log_frequency_seconds,
                                                                frequency_steps=log_frequency_steps)
        self.__push_control = rlsrl.base.timeutil.FrequencyControl(frequency_seconds=push_frequency_seconds,
                                                             frequency_steps=push_frequency_steps)
        
        # logging 
        self.__start_time = None
        self.__accumulated_buffer_wait_time = 0
        self.__last_buffer_wait_time = 0
        self.__last_buffer_get_time = 0
        self.__average_training_step_time = 0

    @property
    def distributed_steps(self):
        return self.__steps

    def stats(self):
        logged_stats = []
        while self.__logging_queue.qsize() > 0:
            e = self.__logging_queue.get_nowait()
            logged_stats.append(e)
        return logged_stats

    def is_alive(self):
        return self.__thread.is_alive()

    def start(self):
        self.__thread.start()

    def _run(self):
        """ Before running main loop of training in GPU thread, setup cuda device and pytorch distributed.
        """
        set_cuda_device(self.__trainer.policy.device)
        self.logger.info(f"Setting cuda device: {self.__trainer.policy.device}")
        self.__trainer.distributed(**self.dist_kwargs)
        self.__start_time = self.__last_buffer_get_time = time.time()
        while True:
            if self.__interrupting:
                self.__interrupt_loop()
                break
            self._run_step()

    def __interrupt_loop(self):
        """ Stop the thread at step: `self.__interrupt_at_step` on last replay entry, which is determined by 
        calling `stop_at_step(step)` in TrainerWorker.  
        """
        self.logger.info("Entering stopping loop.")
        while self.__steps < self.__interrupt_at_step:
            if self.__replay_entry is None:
                break
            self._run_step_on_entry(self.__replay_entry)
        self.logger.info(f"Stopping at {self.__steps}!")

    def _run_step(self):
        if not self.__buffer.empty():
            self.__replay_entry = self.__buffer.get()
            self.__last_buffer_wait_time = time.time() - self.__last_buffer_get_time
            self.__accumulated_buffer_wait_time += self.__last_buffer_wait_time
            self._run_step_on_entry(self.__replay_entry)
            self.__last_buffer_get_time = time.time()
        else:
            time.sleep(
                0.005
            )  # to avoid locking the buffer. We should remove this line when our buffer is thread-safe.

    def _run_step_on_entry(self, replay_entry):

        sample_policy_version = replay_entry.sample.average_of("policy_version_steps", ignore_negative=True)
        sample_policy_version_min = replay_entry.sample.min_of("policy_version_steps", ignore_negative=True)
        if sample_policy_version is None or np.isnan(
                sample_policy_version_min) or sample_policy_version_min < 0:
            self.logger.debug(
                f"Ignored sample with version: avg {sample_policy_version}, min {sample_policy_version_min}.")
            return

        # staleness = self.__trainer.policy.version - sample_policy_version
        # TODO: Temporary workaround to overwrite non-numerical field `policy_name`.
        replay_entry.sample.policy_name = None

        step_start = time.time()
        train_result = self.__trainer.step(replay_entry.sample)
        step_time = time.time() - step_start

        self.__average_training_step_time = (self.__average_training_step_time * self.__steps + step_time)/(self.__steps + 1) 
        self.__steps += 1
        
        samples = replay_entry.sample.length(0) * replay_entry.sample.length(1)
        self.__samples += samples

        if self.__logging_control.check(steps=samples):
            start = time.time()
            while True:
                try:
                    _ = self.__logging_queue.get_nowait()
                except queue.Empty:
                    break
            log_entry = worker_base.LogEntry(stats=dict(**train_result.stats,
                                                        samples=self.__samples, 
                                                        steps=self.__steps,
                                                        frames_per_second = self.__samples/(time.time()-self.__start_time),
                                                        last_buffer_wait = self.__last_buffer_wait_time,
                                                        accumulated_buffer_wait_percent = self.__accumulated_buffer_wait_time/(time.time()-self.__start_time),
                                                        avg_step_time = self.__average_training_step_time),
                                             step=train_result.step)  
            self.__logging_queue.put(log_entry, block=False)
            self.logger.debug("Logged stats, took time: %.2fs", time.time() - start)
        
        if self.__is_master and train_result.agree_pushing and self.__push_control.check():
            # clear checkpoint_push_queue and get checkpoint from trainer
            start = time.time()
            while True:
                try:
                    _ = self.__checkpoint_push_queue.get_nowait()
                except queue.Empty:
                    break
            self.__checkpoint_push_queue.put(self.__trainer.get_checkpoint(), block=False)
            self.logger.debug("Pushed params, took time: %.2fs", time.time() - start)

    def get_checkpoint(self) -> Any:
        """Get checkpoint published by the trainer.
        Returns:
            trainer_checkpoint: checkpoint to be saved/published.
        """
        try:
            return self.__checkpoint_push_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_at_step(self, stop_at_step):
        self.__interrupting = True
        self.__interrupt_at_step = stop_at_step
        self.__thread.join(timeout=60)
        if self.__thread.is_alive():
            raise RuntimeError("Failed to join GPU thread. (timeout=15s)")


class TrainerWorker(worker_base.Worker):

    def __init__(self):
        super().__init__()
        self.config = None
        self.policy_name = None
        self.gpu_thread = None
        self.__stream = None
        self.__buffer = None
        self.__param_db: Optional[parameter_db.ParameterDBClient] = None
        self.__is_master = False
        self.__ddp_init_address = None
        self.__ddp_rank = None
        self.__push_tagged_control = None
        self.__gc_frequency_control = rlsrl.base.timeutil.FrequencyControl(
            frequency_seconds=PARAMETER_DB_GC_FREQUENCY_SECONDS)
        self.__world_size = 0

    def _stats(self):
        if self.__is_master:
            return self.gpu_thread.stats()
        else:
            return []

    def _configure(self, cfg: config.TrainerWorker):
        self.config = cfg
        self.policy_name = cfg.policy_name
        self.__foreign_policy = cfg.foreign_policy
        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name
        self.__worker_index = str(cfg.worker_info.worker_index)
        self.__world_size = cfg.world_size
        r = self.config.worker_info
        r.policy_name = self.policy_name
        if self.config.push_tag_frequency_minutes is not None:
            self.__push_tagged_control = rlsrl.base.timeutil.FrequencyControl(
                frequency_seconds=self.config.push_tag_frequency_minutes * 60, initial_value=True)

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_determinisitc

        self.__buffer = rlsrl.base.buffer.make_buffer(cfg.buffer_name, **cfg.buffer_args)
        self.__stream = make_consumer(cfg.sample_stream, worker_info=cfg.worker_info)
        self.__param_db = parameter_db.make_db(cfg.parameter_db, worker_info=cfg.worker_info)

        # Reveal DDP identity of this worker to world.
        self.__reveal_ddp_identity()

        # Wait DDP_SETUP_WAIT_SECONDS seconds for other DDP peer to reveal
        time.sleep(DDP_SETUP_WAIT_SECONDS)
        self.__setup_ddp()

        trainer = rlsrl.api.trainer.make(self.config.trainer, self.config.policy)
        self.__load_initial_policy(trainer)
        dist_kwargs = dict(world_size=self.__world_size,
                           rank=self.__ddp_rank,
                           init_method=self.__ddp_init_address) 
        self.__make_gpu_thread(trainer, dist_kwargs=dist_kwargs)
        return r

    def __reveal_ddp_identity(self):
        """ Reveal DDP identity, put DDP peer name into name resolve.
        """
        name_resolve.add_subentry(names.trainer_ddp_peer(self.__experiment_name, self.__trial_name,
                                                         self.policy_name),
                                  self.__worker_index,
                                  keepalive_ttl=5)
    
    def __setup_ddp(self):
        """Setup pytorch DDP processes.
        """
        self.logger.info(f"Setup trainer worker {self.__worker_index} for policy {self.policy_name}")

        peers = list(
            sorted(
                name_resolve.get_subtree(
                    names.trainer_ddp_peer(self.__experiment_name, self.__trial_name, self.policy_name))))
        ddp_name_resolve = names.trainer_ddp_master(self.__experiment_name, self.__trial_name,
                                                    self.policy_name)

        assert len(peers) == len(set(peers)), f"Duplicated trainer worker index."
        if self.__world_size > 0 and not len(peers) == self.__world_size:
            raise Exception(f"World size = {self.__world_size}, "
                            f"but not all DDP peers are revealed in {DDP_SETUP_WAIT_SECONDS} second(s).")
        self.__world_size = len(peers)
        
        self.__ddp_rank = peers.index(self.__worker_index)
        # In local version, host_ip is localhost. In distributed version this IP should be socket.gethostbyname(socket.gethostname()).
        if self.__ddp_rank == 0:
            # DDP rank 0 puts init address into name resolve.
            host_ip = "localhost"
            port = find_free_port()
            self.__ddp_init_address = f"tcp://{host_ip}:{port}"
            name_resolve.add(ddp_name_resolve, self.__ddp_init_address, keepalive_ttl=15)
        else:
            try:
                # DDP rank other than 0 get address from name resolve.
                self.__ddp_init_address = name_resolve.wait(ddp_name_resolve, timeout=5)
            except TimeoutError:
                raise TimeoutError(
                    f"DDP trainer(index:{self.__worker_index}), rank {self.__ddp_rank} for policy "
                    f"{self.policy_name} wait for ddp_init_method timeout.")

        if self.__ddp_rank == 0:
            self.__is_master = True
 
    def __load_initial_policy(self, trainer: rlsrl.api.trainer.Trainer):
        try:
            # Loading parameters for master in sufficient for pytorch DDP.
            # Things might be different in other cases.
            checkpoint = self.__param_db.get(self.policy_name, identifier="latest")
            trainer.load_checkpoint(checkpoint)
            self.logger.info(f"Loaded model with tag latest. You can re-run your "
                             f"experiment by deleting your saved model parameters from parameter DB.")
        except FileNotFoundError:
            self.__maybe_read_foreign_policy(trainer)
            if self.__is_master:
                self.logger.warning("No saved model found. This must be the first time you run this trial."
                                    "DDP master is pushing the first version.")
                trainer.policy.inc_version()  # Increase policy version from -1 to 0. We start training now.
                self.__param_db.push(self.policy_name, trainer.get_checkpoint(), str(trainer.policy.version))
    
    def __maybe_read_foreign_policy(self, trainer: rlsrl.api.trainer.Trainer):
        if self.__foreign_policy is not None:
            p = self.__foreign_policy
            spec = p.param_db
            e = p.foreign_experiment_name or self.__experiment_name
            f = p.foreign_trial_name or self.__trial_name
            pn = p.foreign_policy_name or self.policy_name
            i = p.foreign_policy_identifier or "latest"

            foreign_db = parameter_db.make_db(spec=spec,
                                              worker_info=config.WorkerInformation(experiment_name=e,
                                                                                   trial_name=f))
            if self.__foreign_policy.absolute_path is not None:
                checkpoint = foreign_db.get_file(self.__foreign_policy.absolute_path)
                self.logger.info(f"Loaded checkpoint: {self.__foreign_policy.absolute_path}")
            else:
                checkpoint = foreign_db.get(name=pn, identifier=i)
                self.logger.info(f"Loaded foreign parameter: {e} -> {f} -> {pn} -> {i}")
            trainer.policy.load_checkpoint(checkpoint)
   
    
    def __make_gpu_thread(self, trainer: rlsrl.api.trainer.Trainer, dist_kwargs):
        self.gpu_thread = GPUThread(buffer=self.__buffer,
                                    trainer=trainer,
                                    is_master=self.__is_master,
                                    log_frequency_seconds=self.config.log_frequency_seconds,
                                    log_frequency_steps=self.config.log_frequency_steps,
                                    push_frequency_seconds=self.config.push_frequency_seconds,
                                    push_frequency_steps=self.config.push_frequency_steps,
                                    dist_kwargs=dist_kwargs)
    
    def __start_gpu_thread(self):
        if not self.gpu_thread:
            raise RuntimeError("Started empty GPU threads.")
        self.gpu_thread.start()

    def __stop_gpu_thread(self):
        """This method tells gpu thread when to stop running.
        """

        def find_safe_interrupt_step(my_step, assume_max_difference=DDP_MAX_TRAINER_STEP_DIFFERENCE):
            for i in range(my_step - assume_max_difference, my_step + assume_max_difference + 1):
                if i % (assume_max_difference * 2 + 1) == 0:
                    return i + assume_max_difference + 3  # +1 should be enough, +3 is just in case.
            else:
                raise RuntimeError("This is not possible.")

        if self.gpu_thread is not None:
            curr_step = self.gpu_thread.distributed_steps
            self.logger.info(
                f"I am at step {curr_step}. "
                f"I think step difference should be no-larger than {DDP_MAX_TRAINER_STEP_DIFFERENCE}.")
            stop_at_step = find_safe_interrupt_step(curr_step)
            self.logger.info(f"I think we could stop at step {stop_at_step}.")
            self.gpu_thread.stop_at_step(stop_at_step)
            self.gpu_thread = None

    def _poll(self):
        if not self.gpu_thread.is_alive():
            raise RuntimeError("Exception in trainer worker gpu thread.")

        # With a bounded iteration count, logging and checkpoint can be processed with controlled delay.
        self.__stream.consume_to(self.__buffer, max_iter=1024)

        # Checkpoint.
        for _ in range(8):
            checkpoint = self.gpu_thread.get_checkpoint()
            if checkpoint is None:
                break
        
            ckpt = checkpoint
            tags = []
            if self.__push_tagged_control is not None and self.__push_tagged_control.check():
                # If policy tagged, store in param db permanently. 
                tags.append("latest_tagged")
                tags.append(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
                self.logger.info("Saving a tagged policy version: %s", tags[-1])
            # Push every policy.
            step_version = str(ckpt["steps"])
            self.logger.info(f"Saving a policy version {step_version}")
            self.__param_db.push(self.policy_name, ckpt, version=step_version, tags=tags)

            if self.__gc_frequency_control.check():
                # Only keep 10 untagged policies.
                self.__param_db.gc(self.policy_name, max_untagged_version_count=10)
            
        # avoid locking thread
        time.sleep(0.005)
        return worker_base.PollResult(valid=True)

    def exit(self):
        self.__stop_gpu_thread()
        super(TrainerWorker, self).exit()

    def start(self):
        self.__start_gpu_thread()
        super(TrainerWorker, self).start()