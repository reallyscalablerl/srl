from dataclasses import dataclass, field
from typing import Any
import bisect
import numpy as np
import queue
import time

from rlsrl.base.namedarray import recursive_aggregate, recursive_apply


@dataclass(order=True)
class ReplayEntry:
    reuses_left: int
    receive_time: float
    sample: Any = field(compare=False)
    reuses: int = field(default=0, compare=False)

    def __len__(self):
        return len(self.sample)


class Buffer:

    def put(self, x):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def empty(self):
        raise NotImplementedError()


class SimpleQueueBuffer(Buffer):

    def __init__(self, batch_size=None, max_size=1e9):
        self.__queue = queue.SimpleQueue()
        self.__max_size = max_size
        self.__tmp_storage = []
        self.batch_size = batch_size

    def put(self, x):
        if self.batch_size:
            self.__tmp_storage.append(x)
            if len(self.__tmp_storage) >= self.batch_size:
                data = recursive_aggregate(self.__tmp_storage[:self.batch_size],
                                           lambda x: np.stack(x, axis=1))
                self.__tmp_storage = self.__tmp_storage[self.batch_size:]
                self.__queue.put_nowait(data)
        else:
            self.__queue.put_nowait(x)
        if self.__queue.qsize() > self.__max_size:
            self.__queue.get_nowait()

    def get(self) -> ReplayEntry:
        return ReplayEntry(reuses_left=0,
                           reuses=0,
                           receive_time=time.time(),
                           sample=self.__queue.get_nowait())

    def empty(self):
        return self.__queue.empty()


class PriorityBuffer(Buffer):

    def __init__(self, max_size=16, reuses=1, batch_size=1):
        self.__buffer = []
        self.__tmp_storage = []
        self.__max_size = max_size
        self.reuses = reuses
        self.batch_size = batch_size

    @property
    def overflow(self):
        return len(self.__buffer) > self.__max_size

    def full(self):
        return len(self.__buffer) == self.__max_size

    def empty(self):
        return len(self.__buffer) == 0

    def qsize(self):
        return len(self.__buffer)

    def put(self, x):
        if self.batch_size:
            self.__tmp_storage.append(x)
            if len(self.__tmp_storage) >= self.batch_size:
                data = recursive_aggregate(self.__tmp_storage[:self.batch_size],
                                           lambda x: np.stack(x, axis=1))
                self.__tmp_storage = self.__tmp_storage[self.batch_size:]
                self.__put(ReplayEntry(reuses_left=self.reuses, sample=data, receive_time=time.time()))
        else:
            self.__put(ReplayEntry(reuses_left=self.reuses, sample=x, receive_time=time.time()))

    def __put(self, r):
        bisect.insort(self.__buffer, r)
        while self.overflow:
            self.__drop()

    def get(self) -> ReplayEntry:
        assert not self.empty(), "attempting to get from empty buffer."
        r = self.__buffer.pop(-1)
        r.reuses_left -= 1
        r.reuses += 1

        if not self.full() and r.reuses_left > 0:
            self.__put(r)

        return r

    def __drop(self):
        self.__buffer.pop(0)


class SimpleReplayBuffer(Buffer):
    """A simple experience replay buffer that uniformly samples a sample batch of size `batch_size` and of length `batch_length`.
    """

    def __init__(self, max_size, batch_length, batch_size, warmup_size, seed=0):
        self.__max_size = max_size
        self.__batch_length = batch_length
        self.__batch_size = batch_size
        self.__warmup_size = warmup_size
        self.__buffer = []
        self.__total_transitions = 0

        np.random.seed(seed)

    @property
    def overflow(self):
        return len(self.__buffer) > self.__max_size

    def put(self, x):
        self.__put(x)

    def full(self):
        return len(self.__buffer) == self.__max_size

    def empty(self):
        return self.__total_transitions < self.__warmup_size

    def qsize(self):
        return self.__total_transitions

    def __put(self, r):
        self.__buffer.append(r)
        self.__total_transitions += r.on_reset.shape[0] - self.__batch_length + 1
        while self.overflow:
            self.__drop()

    def get(self):
        sample_indicies = np.random.randint(0, len(self.__buffer), (self.__batch_size,))
        state_indicies = [
            np.random.randint(0, self.__buffer[sample_idx].on_reset.shape[0] - self.__batch_length + 1)
            for sample_idx in sample_indicies
        ]
        data = recursive_aggregate([
            self.__buffer[sample_idx][state_idx:state_idx + self.__batch_length]
            for sample_idx, state_idx in zip(sample_indicies, state_indicies)
        ], lambda x: np.stack(x, axis=1))
        return ReplayEntry(reuses_left=-1, sample=data, receive_time=-1)

    def __drop(self):
        r = self.__buffer.pop(0)
        self.__total_transitions -= r.on_reset.shape[0] - self.__batch_length + 1


def make_buffer(name, **buffer_args):
    if name == "simple_queue":
        return SimpleQueueBuffer(**buffer_args)
    elif name == "priority_queue":
        return PriorityBuffer(**buffer_args)
    elif name == "simple_replay_buffer":
        return SimpleReplayBuffer(**buffer_args)
