from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """Information about a node during the traversal."""

    id: str
    """The document ID of this node."""

    depth: int
    """The depth (number of edges) through which this node was discovered.
    This may be larger than the *actual* depth of the node in the complete
    graph. If only a subset of edges are retrieved (such as when using
    similarity search to select only the most relevant edges), then this will
    correspond to the depth in the retrieved set of edges, which may be more
    than the true depth, if edges necessary for the shorter path are not used.
    """

    embedding: list[float]

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata from the original document."""

    extra_metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata to add to the original document for the results."""
