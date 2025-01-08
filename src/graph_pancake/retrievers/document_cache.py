from typing import Iterable

from langchain_core.documents import Document

from .node import Node

METADATA_EMBEDDING_KEY = "__embedding"


class DocumentCache:
    documents: dict[str, Document] = {}

    def add_document(self, doc: Document, *, depth: int) -> Node:
        if doc.id is None:
            msg = "All documents should have ids"
            raise ValueError(msg)
        if doc.id not in self.documents:
            self.documents[doc.id] = doc
        return Node(
            id=doc.id,
            depth=depth,
            embedding=doc.metadata[METADATA_EMBEDDING_KEY],
            metadata=doc.metadata,
        )

    def add_documents(self, docs: Iterable[Document], *, depth: int) -> None:
        for doc in docs:
            self.add_document(doc, depth=depth)

    def get_document(self, node: Node) -> Document:
        doc: Document = self.documents.get(id, None)
        if doc is None:
            raise RuntimeError(f"unexpected, cache should contain doc id: {node.id}")

        # Create a copy since we're going to mutate metadata.
        doc = doc.copy()

        # Add the extra metadata.
        doc.metadata.update(node.extra_metadata)

        return doc

    def get_documents(self, nodes: Iterable[Node]) -> list[Document]:
        return [self.get_document(node) for node in nodes]
