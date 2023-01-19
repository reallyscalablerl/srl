import time
import threading

INFINITE_DURATION = 60 * 60 * 24 * 365 * 1000


class FrequencyControl:
    """An utility to control the execution of code with a time or/and step frequency.
    """

    def __init__(self, frequency_seconds=None, frequency_steps=None, initial_value=False):
        """Initialization method of FrequencyControl.
        Args:
            frequency_seconds: Minimal interval between two trigger.
            frequency_steps: Minimal number of steps between two triggers.
            initial_value: In true, the first call of check() returns True.

        NOTE:
            If both frequency_seconds and frequency_steps are None, the checking will always return False except
            for the specified initial value. If passed both, both frequency and steps conditions have to be met for
             check() to return True. If one is passed, checking on the other condition will be ignored.
        """
        self.frequency_seconds = frequency_seconds
        self.frequency_steps = frequency_steps
        self.__start_time = time.monotonic()
        self.__steps = 0
        self.__last_time = time.monotonic()
        self.__last_steps = 0
        self.__interval_seconds = self.__interval_steps = None
        self.__initial_value = initial_value
        self.__lock = threading.Lock()

    @property
    def total_seconds(self):
        return time.monotonic() - self.__start_time

    @property
    def total_steps(self):
        return self.__steps

    @property
    def interval_seconds(self):
        return self.__interval_seconds

    @property
    def interval_steps(self):
        return self.__interval_steps

    def check(self, steps=1):
        """Check whether frequency condition is met.
        Args:
            steps: number of step between this and the last call of check()

        Returns:
            flag: True if condition is met, False other wise
        """
        with self.__lock:
            now = time.monotonic()
            self.__steps += steps

            if self.__initial_value:
                self.__last_time = now
                self.__last_steps = self.__steps
                self.__initial_value = False
                return True

            self.__interval_seconds = now - self.__last_time
            self.__interval_steps = self.__steps - self.__last_steps
            if self.frequency_steps is None and self.frequency_seconds is None:
                return False
            if self.frequency_seconds is not None and self.__interval_seconds < self.frequency_seconds:
                return False
            if self.frequency_steps is not None and self.__interval_steps < self.frequency_steps:
                return False
            self.__last_time = now
            self.__last_steps = self.__steps

            return True

    def reset_time(self):
        self.__last_time = time.monotonic()

