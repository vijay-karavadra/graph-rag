
import dataclasses
from typing import (
    TYPE_CHECKING,
    Iterable,
)

import numpy as np
from langchain_core.documents import Document
from numpy.typing import NDArray


from graph_pancake.utils.math import cosine_similarity
from .embedded_document import EmbeddedDocument

if TYPE_CHECKING:
    from numpy.typing import NDArray

NEG_INF = float("-inf")

@dataclasses.dataclass
class _Candidate:
    doc: Document
    similarity: float
    weighted_similarity: float
    weighted_redundancy: float
    score: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.score = self.weighted_similarity - self.weighted_redundancy

    @property
    def id(self) -> str:
        if self.doc.id is None:
            msg = "All documents should have ids"
            raise ValueError(msg)
        return self.doc.id

    def update_redundancy(self, new_weighted_redundancy: float) -> None:
        if new_weighted_redundancy > self.weighted_redundancy:
            self.weighted_redundancy = new_weighted_redundancy
            self.score = self.weighted_similarity - self.weighted_redundancy


def _emb_to_ndarray(embedding: list[float]) -> NDArray[np.float32]:
    emb_array = np.array(embedding, dtype=np.float32)
    if emb_array.ndim == 1:
        emb_array = np.expand_dims(emb_array, axis=0)
    return emb_array


class MmrHelper:
    """Helper for executing an MMR traversal query.

    Args:
        query_embedding: The embedding of the query to use for scoring.
        lambda_mult: Number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding to maximum
            diversity and 1 to minimum diversity. Defaults to 0.5.
        score_threshold: Only documents with a score greater than or equal
            this threshold will be chosen. Defaults to -infinity.
    """

    dimensions: int
    """Dimensions of the embedding."""

    query_embedding: NDArray[np.float32]
    """Embedding of the query as a (1,dim) ndarray."""

    lambda_mult: float
    """Number between 0 and 1.

    Determines the degree of diversity among the results with 0 corresponding to
    maximum diversity and 1 to minimum diversity."""

    lambda_mult_complement: float
    """1 - lambda_mult."""

    score_threshold: float
    """Only documents with a score greater than or equal to this will be chosen."""

    selected_ids: list[str]
    """List of selected IDs (in selection order)."""

    selected_embeddings: NDArray[np.float32]
    """(N, dim) ndarray with a row for each selected node."""

    candidate_id_to_index: dict[str, int]
    """Dictionary of candidate IDs to indices in candidates and candidate_embeddings."""
    candidates: list[_Candidate]
    """List containing information about candidates.

    Same order as rows in `candidate_embeddings`.
    """
    candidate_embeddings: NDArray[np.float32]
    """(N, dim) ndarray with a row for each candidate."""

    best_score: float
    best_id: str | None

    def __init__(
        self,
        k: int,
        query_embedding: list[float],
        lambda_mult: float = 0.5,
        score_threshold: float = NEG_INF,
    ) -> None:
        """Create a new Traversal MMR helper."""
        self.query_embedding = _emb_to_ndarray(query_embedding)
        self.dimensions = self.query_embedding.shape[1]

        self.lambda_mult = lambda_mult
        self.lambda_mult_complement = 1 - lambda_mult
        self.score_threshold = score_threshold

        self.selected_ids = []

        # List of selected embeddings (in selection order).
        self.selected_embeddings = np.ndarray((k, self.dimensions), dtype=np.float32)

        self.candidate_id_to_index = {}

        # List of the candidates.
        self.candidates = []
        # numpy n-dimensional array of the candidate embeddings.
        self.candidate_embeddings = np.ndarray((0, self.dimensions), dtype=np.float32)

        self.best_score = NEG_INF
        self.best_id = None

    def candidate_ids(self) -> Iterable[str]:
        """Return the IDs of the candidates."""
        return self.candidate_id_to_index.keys()

    def _already_selected_embeddings(self) -> NDArray[np.float32]:
        """Return the selected embeddings sliced to the already assigned values."""
        selected = len(self.selected_ids)
        return np.vsplit(self.selected_embeddings, [selected])[0]

    def _pop_candidate(
        self, candidate_id: str
    ) -> tuple[Document, float, NDArray[np.float32]]:
        """Pop the candidate with the given ID.

        Returns:
            The document, similarity score, and embedding of the candidate.
        """
        # Get the embedding for the id.
        index = self.candidate_id_to_index.pop(candidate_id)
        candidate = self.candidates[index]
        if candidate.id != candidate_id:
            msg = (
                "ID in self.candidate_id_to_index doesn't match the ID of the "
                "corresponding index in self.candidates"
            )
            raise ValueError(msg)
        embedding: NDArray[np.float32] = self.candidate_embeddings[index].copy()

        # Swap that index with the last index in the candidates and
        # candidate_embeddings.
        last_index = self.candidate_embeddings.shape[0] - 1

        similarity = 0.0
        if index == last_index:
            # Already the last item. We don't need to swap.
            similarity = self.candidates.pop().similarity
        else:
            self.candidate_embeddings[index] = self.candidate_embeddings[last_index]

            similarity = self.candidates[index].similarity

            old_last = self.candidates.pop()
            self.candidates[index] = old_last
            self.candidate_id_to_index[old_last.id] = index

        self.candidate_embeddings = np.vsplit(self.candidate_embeddings, [last_index])[
            0
        ]

        return candidate.doc, similarity, embedding

    def pop_best(self) -> Document | None:
        """Select and pop the best item being considered.

        Updates the consideration set based on it.

        Returns:
            A tuple containing the ID of the best item.
        """
        if self.best_id is None or self.best_score < self.score_threshold:
            return None

        # Get the selection and remove from candidates.
        selected_id = self.best_id
        selected_doc, selected_similarity, selected_embedding = self._pop_candidate(
            selected_id
        )

        # Add the ID and embedding to the selected information.
        selection_index = len(self.selected_ids)
        self.selected_ids.append(selected_id)
        self.selected_embeddings[selection_index] = selected_embedding

        # Set the scores in the doc metadata
        selected_doc.metadata["_similarity_score"] = selected_similarity
        selected_doc.metadata["_mmr_score"] = self.best_score

        # Reset the best score / best ID.
        self.best_score = NEG_INF
        self.best_id = None

        # Update the candidates redundancy, tracking the best node.
        if self.candidate_embeddings.shape[0] > 0:
            similarity = cosine_similarity(
                self.candidate_embeddings, np.expand_dims(selected_embedding, axis=0)
            )
            for index, candidate in enumerate(self.candidates):
                candidate.update_redundancy(similarity[index][0])
                if candidate.score > self.best_score:
                    self.best_score = candidate.score
                    self.best_id = candidate.id

        return selected_doc

    def add_candidates(
        self, candidates: dict[str, EmbeddedDocument], depth_found: int
    ) -> None:
        """Add candidates to the consideration set."""
        # Determine the keys to actually include.
        # These are the candidates that aren't already selected
        # or under consideration.
        include_ids_set = set(candidates.keys())
        include_ids_set.difference_update(self.selected_ids)
        include_ids_set.difference_update(self.candidate_id_to_index.keys())
        include_ids = list(include_ids_set)

        # Now, build up a matrix of the remaining candidate embeddings.
        # And add them to the
        new_embeddings: NDArray[np.float32] = np.ndarray(
            (
                len(include_ids),
                self.dimensions,
            )
        )
        offset = self.candidate_embeddings.shape[0]
        for index, candidate_id in enumerate(include_ids):
            if candidate_id in include_ids:
                self.candidate_id_to_index[candidate_id] = offset + index
                new_embeddings[index] = candidates[candidate_id].embedding

        # Compute the similarity to the query.
        similarity = cosine_similarity(new_embeddings, self.query_embedding)

        # Compute the distance metrics of all of pairs in the selected set with
        # the new candidates.
        redundancy = cosine_similarity(
            new_embeddings, self._already_selected_embeddings()
        )
        for index, candidate_id in enumerate(include_ids):
            max_redundancy = 0.0
            if redundancy.shape[0] > 0:
                max_redundancy = redundancy[index].max()
            candidate = _Candidate(
                doc=candidates[candidate_id].document(),
                similarity=similarity[index][0],
                weighted_similarity=self.lambda_mult * similarity[index][0],
                weighted_redundancy=self.lambda_mult_complement * max_redundancy,
            )
            candidate.doc.metadata["_depth_found"] = depth_found
            self.candidates.append(candidate)

            if candidate.score >= self.best_score:
                self.best_score = candidate.score
                self.best_id = candidate.id

        # Add the new embeddings to the candidate set.
        self.candidate_embeddings = np.vstack(
            (
                self.candidate_embeddings,
                new_embeddings,
            )
        )

