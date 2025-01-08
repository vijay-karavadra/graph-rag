import heapq
from typing import Callable, Iterable

from ..node import Node
from .node_selector import NodeSelector


class EagerScoringNodeSelector(NodeSelector):
    """Node selection based on an eager scoring function."""

    @staticmethod
    def factory(
        scorer: Callable[[Node], int], *, select_k: int = 1
    ) -> Callable[[list[float]], NodeSelector]:
        return lambda _k, _query_embedding: EagerScoringNodeSelector(
            scorer,
            select_k=select_k,
        )

    def __init__(self, scorer: Callable[[Node], int], *, select_k: int = 1) -> None:
        self._scorer = scorer
        self._nodes = []
        self._select_k = select_k

    def add_nodes(self, nodes: dict[str, Node]) -> None:
        for node in nodes.values():
            heapq.heappush(self._nodes, (self._scorer(node), node))

    def select_nodes(self, *, limit: int) -> Iterable[Node]:
        selected = []
        for _ in range(0, min(limit, self._select_k)):
            if len(self._nodes) == 0:
                break
            selected.append(heapq.heappop(self._nodes)[1])
        return selected
