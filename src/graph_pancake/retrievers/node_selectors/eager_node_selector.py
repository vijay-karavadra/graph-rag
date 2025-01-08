from typing import Callable, Iterable

from ..node import Node
from .node_selector import NodeSelector


class EagerNodeSelector(NodeSelector):
    """Node selection that selects all nodes at each step."""

    def __init__(self) -> None:
        self._nodes = []

    @staticmethod
    def factory() -> Callable[[list[float]], NodeSelector]:
        return lambda _k, _query_embedding: EagerNodeSelector()

    def add_nodes(self, nodes: dict[str, Node]) -> None:
        self._nodes.extend(nodes.values())

    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        nodes = self._nodes[:limit]
        self._nodes = []
        return nodes
