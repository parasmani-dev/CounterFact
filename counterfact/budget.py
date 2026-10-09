"""Thread-safe accounting of outbound inference attempts."""

import threading


class BudgetExceeded(RuntimeError):
    pass


class DispatchStopped(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class Budget:
    def __init__(self, cap: int):
        if type(cap) is not int or cap < 0:
            raise ValueError("Budget cap must be a nonnegative integer")
        self.cap = cap
        self._used = 0
        self._lock = threading.Lock()

    def reserve(self, n: int = 1) -> None:
        if type(n) is not int or n < 0:
            raise ValueError("Reservation must be a nonnegative integer")
        with self._lock:
            if self._used + n > self.cap:
                raise BudgetExceeded("Inference attempt budget exhausted")
            self._used += n

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def report(self) -> dict:
        return {"used": self.used, "cap": self.cap}
