"""Type-safe component store (stdlib-only)."""
from __future__ import annotations
from typing import TypeVar, Generic, Dict, Iterator, Tuple

T = TypeVar("T")

class ComponentStore(Generic[T]):
    """Maps ``EntityID`` → component instance of type *T*."""
    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[int, T] = {}

    def add(self, eid: int, comp: T) -> None:
        self._data[eid] = comp

    def get(self, eid: int) -> T | None:
        return self._data.get(eid)

    def remove(self, eid: int) -> None:
        self._data.pop(eid, None)

    def items(self) -> Iterator[Tuple[int, T]]:
        return self._data.items()
