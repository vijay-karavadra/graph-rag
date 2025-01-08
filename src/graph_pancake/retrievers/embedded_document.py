import dataclasses
from typing import Any

from langchain_core.documents import Document

METADATA_EMBEDDING_KEY = "__embedding"


@dataclasses.dataclass
class EmbeddedDocument:
    doc: Document

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddedDocument):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def id(self) -> str:
        if self.doc.id is None:
            msg = "All documents should have ids"
            raise ValueError(msg)
        return self.doc.id

    @property
    def metadata(self) -> dict[str, Any]:
        """Get the metadata from the document."""
        return self.doc.metadata

    @property
    def embedding(self) -> list[float]:
        """Get the embedding from the document."""
        return self.doc.metadata.get(METADATA_EMBEDDING_KEY, [])

    def document(self) -> Document:
        """Get the underlying document with the embedding removed."""
        self.doc.metadata.pop(METADATA_EMBEDDING_KEY, None)
        return self.doc
