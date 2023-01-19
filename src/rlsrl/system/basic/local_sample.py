import queue
from typing import Any, Tuple

from rlsrl.system.sample_stream import SampleProducer, SampleConsumer, NothingToConsume, register_producer
import rlsrl.base.namedarray
import rlsrl.api.config


class LocalSampleProducer(SampleProducer):

    @property
    def type(self):
        return None

    def __init__(self, q):
        self.q = q

    def post(self, sample):
        self.q.put(rlsrl.base.namedarray.dumps(sample))


class LocalSampleConsumer(SampleConsumer):

    def __init__(self, q):
        self.q = q
        self.consume_timeout = 0.2

    def consume_to(self, buffer, max_iter=64):
        count = 0
        for _ in range(max_iter):
            try:
                sample = rlsrl.base.namedarray.loads(self.q.get_nowait())
            except queue.Empty:
                break
            buffer.put(sample)
            count += 1
        return count

    def consume(self) -> Any:
        """Note that this method blocks for 0.2 seconds if no sample can be consumed. Therefore, it is safe to make
        a no-sleeping loop on this method. For example:
        while not interrupted:
            try:
                data = consumer.consume()
            except NothingToConsume:
                continue
            process(data)
        """
        try:
            return rlsrl.base.namedarray.loads(self.q.get(timeout=self.consume_timeout))
        except queue.Empty:
            raise NothingToConsume()


class NullSampleProducer(SampleProducer):
    """NullSampleProducer discards all samples.
    """

    @property
    def type(self):
        return rlsrl.api.config.SampleStream.Type.NULL

    def __init__(self, spec):
        pass

    def post(self, sample):
        pass


def make_local_pair(q) -> Tuple[SampleProducer, SampleConsumer]:
    return LocalSampleProducer(q), LocalSampleConsumer(q)


register_producer(rlsrl.api.config.SampleStream.Type.NULL, NullSampleProducer)
