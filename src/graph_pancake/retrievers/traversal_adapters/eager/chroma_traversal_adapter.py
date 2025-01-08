from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
)

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .traversal_adapter import TraversalAdapter


class ChromaTraversalAdapter(TraversalAdapter):
    def __init__(self, vector_store: VectorStore):
        try:
            from langchain_chroma import Chroma
        except (ImportError, ModuleNotFoundError):
            msg = "please `pip install langchain-chroma`"
            raise ImportError(msg)
        self._vector_store = cast(Chroma, vector_store)
        self._base_vector_store = vector_store

    def similarity_search_by_vector(  # type: ignore
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        if k > self._vector_store._collection.count():
            k = self._vector_store._collection.count()
        return self._vector_store.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
