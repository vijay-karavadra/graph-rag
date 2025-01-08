import asyncio
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Sequence,
    Tuple,
    Union,
)

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from .document_cache import DocumentCache
from .edge import Edge
from .node import Node
from .node_selectors.node_selector import NodeSelector
from .traversal_adapters.generic.graph_traversal_adapter import GraphTraversalAdapter

BASIC_TYPES = (str, bool, int, float, complex, bytes)


# this class uses pydantic, so store and edges
# must be provided at init time.
class GenericGraphTraversalRetriever(BaseRetriever):
    store: GraphTraversalAdapter
    edges: List[Union[str, Tuple[str, str]]]
    _edges: List[Tuple[str, str]] = PrivateAttr(default=[])
    node_selector_factory: Callable[[int, list[float]], NodeSelector]

    k: int = Field(default=4)
    depth: int = Field(default=2)
    fetch_k: int = Field(default=100)
    adjacent_k: int = Field(default=10)
    lambda_mult: float = Field(default=0.5)
    score_threshold: float = Field(default=float("-inf"))
    use_denormalized_metadata: bool = Field(default=False)
    denormalized_path_delimiter: str = Field(default=".")
    denormalized_static_value: Any = Field(default=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for edge in self.edges:
            if isinstance(edge, str):
                self._edges.append((edge, edge))
            elif (
                isinstance(edge, tuple)
                and len(edge) == 2
                and all(isinstance(item, str) for item in edge)
            ):
                self._edges.append((edge[0], edge[1]))
            else:
                raise ValueError(
                    "Invalid type for edge. must be 'str' or 'tuple[str,str]'"
                )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int | None = None,
        depth: int | None = None,
        fetch_k: int | None = None,
        adjacent_k: int | None = None,
        lambda_mult: float | None = None,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve document nodes from this graph vector store using MMR-traversal.
        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.
        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").
        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        k = self.k if k is None else k
        depth = self.depth if depth is None else depth
        fetch_k = self.fetch_k if fetch_k is None else fetch_k
        adjacent_k = self.adjacent_k if adjacent_k is None else adjacent_k
        lambda_mult = self.lambda_mult if lambda_mult is None else lambda_mult
        score_threshold = (
            self.score_threshold if score_threshold is None else score_threshold
        )

        visited_edges: set[Edge] = set()

        def fetch_initial_candidates() -> tuple[list[float], dict[str, Node]]:
            """Gets the embedded query and the set of initial candidates.
            If fetch_k is zero, there will be no initial candidates.
            """
            query_embedding, initial_nodes = self._get_initial(
                query=query,
                fetch_k=fetch_k,
                filter=filter,
                **kwargs,
            )
            return query_embedding, initial_nodes

        def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, Node]:
            nonlocal visited_edges

            # Initialize the visited_edges with the set of outgoing edges from the
            # neighborhood. This prevents re-visiting them.
            for doc in self.store.get(neighborhood):
                visited_edges.update(self._get_outgoing_edges(doc.metadata))

            # Fetch the candidates.
            adjacent_nodes = self._get_adjacent(
                edges=visited_edges,
                query_embedding=query_embedding,
                k_per_edge=adjacent_k,
                filter=filter,
                **kwargs,
            )

            return adjacent_nodes

        # Tracks the depth of each candidate.
        depths = {}

        # Retrieve initial candidates.
        query_embedding, initial_candidates = fetch_initial_candidates()
        node_selector = self.node_selector_factory(k, query_embedding)
        for node_id in initial_candidates.keys():
            depths[node_id] = 0
        node_selector.add_nodes(initial_candidates)

        if initial_roots:
            neighborhood_candidates = fetch_neighborhood_candidates(initial_roots)
            for node_id in initial_roots.keys():
                depths[node_id] = 0
            node_selector.add_nodes(neighborhood_candidates)

        cache = DocumentCache()
        selected_nodes: list[Node] = []
        while len(selected_nodes) < k:
            selected_nodes_iter = node_selector.select_nodes(k - len(selected_nodes))
            if len(selected_nodes_iter) == 0:
                break

            selected_nodes.extend(selected_nodes_iter)

            selected_outgoing_edges = {}
            for selected_node in selected_nodes:
                next_depth = depths[selected_node.id] + 1
                if next_depth < depth:
                    # If the next nodes would not exceed the depth limit, find
                    # the adjacent nodes.

                    # Find the edges from the selected node.
                    outgoing_edges = self._get_outgoing_edges(selected_node.metadata)
                    selected_outgoing_edges.update(outgoing_edges)

            # Don't re-visit already visited edges.
            selected_outgoing_edges.difference_update(visited_edges)

            # Find the document nodes with incoming edges from those edges.
            adjacent_nodes = self._get_adjacent(
                outgoing_edges=selected_outgoing_edges,
                query_embedding=query_embedding,
                k_per_edge=adjacent_k,
                filter=filter,
                **kwargs,
            )

            # Record the selected_outgoing_edges as visited.
            visited_edges.update(selected_outgoing_edges)

            for adjacent_node in adjacent_nodes:
                if next_depth < depths.get(adjacent_node.id, depth + 1):
                    # If this is a new shortest depth, or there was no
                    # previous depth, update the depths. This ensures that
                    # when we discover a node we will have the shortest
                    # depth available.
                    #
                    # NOTE: No effort is made to traverse from nodes that
                    # were previously selected if they become reachable via
                    # a shorter path via nodes selected later. This is
                    # currently "intended", but may be worth experimenting
                    # with.
                    depths[adjacent_node.id] = next_depth

            node_selector.add_nodes(adjacent_nodes)

        return cache.get_documents(selected_nodes)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int | None = None,
        depth: int | None = None,
        fetch_k: int | None = None,
        adjacent_k: int | None = None,
        lambda_mult: float | None = None,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for.
            run_manager: The callback handler to use.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
        Returns:
            List of relevant documents
        """
        """Retrieve documents from this graph store using MMR-traversal.
        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.
        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").
        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        k = self.k if k is None else k
        depth = self.depth if depth is None else depth
        fetch_k = self.fetch_k if fetch_k is None else fetch_k
        adjacent_k = self.adjacent_k if adjacent_k is None else adjacent_k
        lambda_mult = self.lambda_mult if lambda_mult is None else lambda_mult
        score_threshold = (
            self.score_threshold if score_threshold is None else score_threshold
        )

        visited_edges: set[Edge] = set()

        async def fetch_initial_candidates() -> tuple[list[float], dict[str, Node]]:
            """Gets the embedded query and the set of initial candidates.
            If fetch_k is zero, there will be no initial candidates.
            """

            query_embedding, initial_nodes = await self._aget_initial(
                query=query,
                fetch_k=fetch_k,
                filter=filter,
                **kwargs,
            )

            return query_embedding, initial_nodes

        async def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, Node]:
            nonlocal visited_edges

            # Initialize the visited_edges with the set of outgoing edges from the
            # neighborhood. This prevents re-visiting them.
            for doc in await self.store.aget(neighborhood):
                visited_edges.update(self._get_outgoing_edges(doc.metadata))

            # Fetch the candidates.
            adjacent_nodes = await self._aget_adjacent(
                outgoing_edges=visited_edges,
                query_embedding=query_embedding,
                k_per_edge=adjacent_k,
                filter=filter,
                **kwargs,
            )

            return adjacent_nodes

        query_embedding, initial_candidates = await fetch_initial_candidates()
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        helper.add_candidates(candidates=initial_candidates)

        if initial_roots:
            neighborhood_candidates = await fetch_neighborhood_candidates(initial_roots)
            helper.add_candidates(candidates=neighborhood_candidates)

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        selected_docs: list[Document] = []
        for _ in range(k):
            selected_doc = helper.pop_best()

            if selected_doc is None or selected_doc.id is None:
                break

            selected_docs.append(selected_doc)

            next_depth = depths[selected_doc.id] + 1
            if next_depth < depth:
                # If the next document nodes would not exceed the depth limit, find the
                # adjacent document nodes.

                # Find the edges edgeed to from the selected id.
                selected_outgoing_edges = self._get_outgoing_edges(
                    selected_doc.metadata
                )

                # Don't re-visit already visited edges.
                selected_outgoing_edges.difference_update(visited_edges)

                # Find the document nodes with incoming edges from those edges.
                adjacent_nodes = await self._aget_adjacent(
                    outgoing_edges=selected_outgoing_edges,
                    query_embedding=query_embedding,
                    k_per_edge=adjacent_k,
                    filter=filter,
                    **kwargs,
                )

                # Record the selected_outgoing_edges as visited.
                visited_edges.update(selected_outgoing_edges)

                for adjacent_node in adjacent_nodes:
                    if next_depth < depths.get(adjacent_node.id, depth + 1):
                        # If this is a new shortest depth, or there was no
                        # previous depth, update the depths. This ensures that
                        # when we discover a node we will have the shortest
                        # depth available.
                        #
                        # NOTE: No effort is made to traverse from nodes that
                        # were previously selected if they become reachable via
                        # a shorter path via nodes selected later. This is
                        # currently "intended", but may be worth experimenting
                        # with.
                        depths[adjacent_node.id] = next_depth

                helper.add_candidates(adjacent_nodes)

        return selected_docs

    def _get_initial(
        self,
        query: str,
        fetch_k: int,
        cache: DocumentCache,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> tuple[list[float], dict[str, Node]]:
        query_embedding, docs = self.store.similarity_search_with_embedding(
            query=query,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, cache.add_documents(docs, depth=0)

    async def _aget_initial(
        self,
        query: str,
        fetch_k: int,
        cache: DocumentCache,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> tuple[list[float], dict[str, Node]]:
        (
            query_embedding,
            docs,
        ) = await self.store.asimilarity_search_with_embedding(
            query=query,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )
        return query_embedding, cache.add_documents(docs)

    def _get_adjacent(
        self,
        cache: DocumentCache,
        outgoing_edges: set[Edge],
        query_embedding: list[float],
        k_per_edge: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict[str, Node]:
        """Return the target docs with incoming edges from any of the given edges.
        Args:
            edges: The edges to look for.
            query_embedding: The query embedding. Used to rank target docs.
            doc_cache: A cache of retrieved docs. This will be added to.
            k_per_edge: The number of target docs to fetch for each edge.
            filter: Optional metadata to filter the results.
        Returns:
            Dictionary of adjacent nodes, keyed by node ID.
        """
        results: dict[str, Node] = set()
        for outgoing_edge in outgoing_edges:
            docs = self.store.similarity_search_with_embedding_by_vector(
                embedding=query_embedding,
                k=k_per_edge or 10,
                filter=self._get_metadata_filter(
                    metadata=filter, outgoing_edge=outgoing_edge
                ),
                **kwargs,
            )
            results.update(cache.add_documents(docs))
        return results

    async def _aget_adjacent(
        self,
        outgoing_edges: set[Edge],
        cache: DocumentCache,
        query_embedding: list[float],
        k_per_edge: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> dict[str, Node]:
        """Returns document nodes with incoming edges from any of the given edges.
        Args:
            edges: The edges to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            doc_cache: A cache of retrieved docs. This will be added to.
            k_per_edge: The number of target nodes to fetch for each edge.
            filter: Optional metadata to filter the results.
        Returns:
            Dictionary of adjacent nodes, keyed by node ID.
        """

        tasks = [
            self.store.asimilarity_search_with_embedding_by_vector(
                embedding=query_embedding,
                k=k_per_edge or 10,
                filter=self._get_metadata_filter(
                    metadata=filter, outgoing_edge=outgoing_edge
                ),
                **kwargs,
            )
            for outgoing_edge in outgoing_edges
        ]

        results: dict[str, Node] = {}
        for completed_task in asyncio.as_completed(tasks):
            docs = await completed_task
            results.update(cache.add_documents(docs))
        return results

    def _get_outgoing_edges(self, doc: Document) -> set[Edge]:
        edges = set()
        for source_key, target_key in self._edges:
            if source_key in doc.metadata:
                value = doc.metadata[source_key]
                if isinstance(value, BASIC_TYPES):
                    edges.add(Edge(key=target_key, value=value))
                elif isinstance(value, Iterable) and not isinstance(
                    value, (str, bytes)
                ):
                    if self.use_denormalized_metadata:
                        warnings.warn(
                            "Iterable metadata values are supported as"
                            " edges in denormalized metadata"
                        )
                    else:
                        for item in value:
                            if isinstance(item, BASIC_TYPES):
                                edges.add(Edge(key=target_key, value=item))
            elif self.use_denormalized_metadata:
                prefix = f"{source_key}{self.denormalized_path_delimiter}"
                matching_keys: list[str] = [
                    key for key in doc.metadata if key.startswith(prefix)
                ]
                for matching_key in matching_keys:
                    if doc.metadata[matching_key] == self.denormalized_static_value:
                        edges.add(
                            Edge(
                                key=target_key,
                                value=matching_key.removeprefix(prefix),
                                is_denormalized=True,
                            )
                        )
        return edges

    def _get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        edge: Edge | None = None,
    ) -> dict[str, Any]:
        """Builds a metadata filter to search for documents

        Args:
            metadata: Any metadata that should be used for hybrid search
            edge: An optional outgoing edge to add to the search
        """
        if edge is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        if edge.is_denormalized:
            metadata_filter[
                f"{edge.key}{self.denormalized_path_delimiter}{edge.value}"
            ] = self.denormalized_static_value
        else:
            metadata_filter[edge.key] = edge.value

        return metadata_filter
