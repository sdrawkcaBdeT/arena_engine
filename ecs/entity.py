"""Monotonic entity-ID generator."""
class EntityIDGenerator:
    __slots__ = ("_next",)

    def __init__(self) -> None:
        self._next: int = 0

    def next_id(self) -> int:
        eid = self._next
        self._next += 1
        return eid

    def reset(self) -> None:
        self._next = 0
