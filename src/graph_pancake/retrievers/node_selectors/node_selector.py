import abc
from typing import Iterable

from ..node import Node


class NodeSelector(abc.ABC):
    """Interface for configuring node selection during the traversal."""

    @abc.abstractmethod
    def add_nodes(self, nodes: dict[str, Iterable[Node]]) -> None:
        """Add nodes to the set of available nodes."""
        ...

    @abc.abstractmethod
    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        """Return the nodes to select at the next iteration."""
        ...
