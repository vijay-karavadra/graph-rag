import abc
from typing import Any, Generic, Iterable, Protocol, Type, TypeVar

from ..node import Node

T = TypeVar("T", bound="NodeSelector")


class NodeSelectorFactory(Protocol, Generic[T]):
    def __call__(
        self, *, k: int, embedding: list[float], **kwargs: dict[str, Any]
    ) -> T: ...


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

    @classmethod
    def factory(cls: Type[T], **kwargs1: dict[str, Any]) -> NodeSelectorFactory[T]:
        return lambda **kwargs2: cls(**{**kwargs2, **kwargs1})

    def finalize_nodes(self, nodes: Iterable[Node]) -> Iterable[Node]:
        """Finalize the selected nodes."""
        return nodes
