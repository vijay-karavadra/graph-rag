import json
import random
from typing import Any, Iterable

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from graph_pancake.retrievers.graph_traversal_retriever import (
    Edge,
    GraphTraversalRetriever,
)
from graph_pancake.retrievers.traversal_adapters.eager import (
    InMemoryTraversalAdapter,
    TraversalAdapter,
)
from tests.embeddings import SimpleEmbeddings


class ParserEmbeddings(Embeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals


class EarthEmbeddings(Embeddings):
    def get_vector_near(self, value: float) -> list[float]:
        base_point = [value, (1 - value**2) ** 0.5]
        fluctuation = random.random() / 100.0
        return [base_point[0] + fluctuation, base_point[1] - fluctuation]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        words = set(text.lower().split())
        if "earth" in words:
            vector = self.get_vector_near(0.9)
        elif {"planet", "world", "globe", "sphere"}.intersection(words):
            vector = self.get_vector_near(0.8)
        else:
            vector = self.get_vector_near(0.1)
        return vector


class FakeAdapter(TraversalAdapter):
    def similarity_search_by_vector(
        self,
        **kwargs: Any,
    ) -> list[Document]:
        return []


def test_get_normalized_outgoing_edges() -> None:
    doc = Document(
        page_content="test",
        metadata={
            "place": ["berlin", "paris"],
            "incoming": ["one", 2],
            "outgoing": ["three", 4],
            "boolean": True,
            "string": "pizza",
            "number": 42,
        },
    )

    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=["place", ("outgoing", "incoming"), "boolean", ("number", "string")],
        use_denormalized_metadata=False,
    )

    edges = sorted(retriever._get_outgoing_edges(doc=doc))

    assert edges[0] == Edge(key="boolean", value=True, is_denormalized=False)
    assert edges[1] == Edge(key="incoming", value=4, is_denormalized=False)
    assert edges[2] == Edge(key="incoming", value="three", is_denormalized=False)
    assert edges[3] == Edge(key="place", value="berlin", is_denormalized=False)
    assert edges[4] == Edge(key="place", value="paris", is_denormalized=False)
    assert edges[5] == Edge(key="string", value=42, is_denormalized=False)


def test_get_denormalized_outgoing_edges() -> None:
    doc = Document(
        page_content="test",
        metadata={
            "place.berlin": True,
            "place.paris": True,
            "incoming.one": True,
            "incoming.2": True,
            "outgoing.three": True,
            "outgoing.4": True,
            "boolean": True,
            "string": "pizza",
            "number": 42,
        },
    )

    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=["place", ("outgoing", "incoming"), "boolean", ("number", "string")],
        use_denormalized_metadata=True,
    )

    edges = sorted(retriever._get_outgoing_edges(doc=doc))

    assert edges[0] == Edge(key="boolean", value=True, is_denormalized=False)
    assert edges[1] == Edge(key="incoming", value="4", is_denormalized=True)
    assert edges[2] == Edge(key="incoming", value="three", is_denormalized=True)
    assert edges[3] == Edge(key="place", value="berlin", is_denormalized=True)
    assert edges[4] == Edge(key="place", value="paris", is_denormalized=True)
    assert edges[5] == Edge(key="string", value=42, is_denormalized=False)


def test_get_normalized_metadata_filter() -> None:
    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=[],
        use_denormalized_metadata=False,
    )

    assert retriever._get_metadata_filter(
        edge=Edge(key="boolean", value=True, is_denormalized=False)
    ) == {"boolean": True}

    assert retriever._get_metadata_filter(
        edge=Edge(key="incoming", value=4, is_denormalized=False)
    ) == {"incoming": 4}

    assert retriever._get_metadata_filter(
        edge=Edge(key="place", value="berlin", is_denormalized=False)
    ) == {"place": "berlin"}


def test_get_denormalized_metadata_filter() -> None:
    retriever = GraphTraversalRetriever(
        store=FakeAdapter(),
        edges=[],
        use_denormalized_metadata=True,
    )

    assert retriever._get_metadata_filter(
        edge=Edge(key="boolean", value=True, is_denormalized=False)
    ) == {"boolean": True}

    assert retriever._get_metadata_filter(
        edge=Edge(key="incoming", value=4, is_denormalized=True)
    ) == {"incoming.4": True}

    assert retriever._get_metadata_filter(
        edge=Edge(key="place", value="berlin", is_denormalized=True)
    ) == {"place.berlin": True}


def _doc_ids(docs: Iterable[Document]) -> list[str]:
    return sorted([doc.id for doc in docs if doc.id is not None])


@pytest.fixture
def graph_vector_store_docs() -> list[Document]:
    """
    This is a set of Documents to pre-populate a graph vector store,
    with entries placed in a certain way.

    Space of the entries (under Euclidean similarity):

                      A0    (*)
        ....        AL   AR       <....
        :              |              :
        :              |  ^           :
        v              |  .           v
                       |   :
       TR              |   :          BL
    T0   --------------x--------------   B0
       TL              |   :          BR
                       |   :
                       |  .
                       | .
                       |
                    FL   FR
                      F0

    the query point is meant to be at (*).
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    docs_a = [
        Document(id="AL", page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(id="A0", page_content="[0, 10]", metadata={"label": "A0"}),
        Document(id="AR", page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(id="BL", page_content="[9, 1]", metadata={"label": "BL"}),
        Document(id="B0", page_content="[10, 0]", metadata={"label": "B0"}),
        Document(id="BR", page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(id="FL", page_content="[1, -9]", metadata={"label": "FL"}),
        Document(id="F0", page_content="[0, -10]", metadata={"label": "F0"}),
        Document(id="FR", page_content="[-1, -9]", metadata={"label": "FR"}),
    ]
    docs_t = [
        Document(id="TL", page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(id="T0", page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(id="TR", page_content="[-9, 1]", metadata={"label": "TR"}),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        doc_a.metadata["tag"] = f"ab_{suffix}"
        doc_a.metadata["out"] = f"at_{suffix}"
        doc_a.metadata["in"] = f"af_{suffix}"
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        doc_b.metadata["tag"] = f"ab_{suffix}"
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        doc_t.metadata["in"] = f"at_{suffix}"
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        doc_f.metadata["out"] = f"af_{suffix}"
    return docs_a + docs_b + docs_f + docs_t


def test_traversal() -> None:
    greetings = Document(
        id="greetings",
        page_content="Typical Greetings",
        metadata={
            "incoming": "parent",
        },
    )

    doc1 = Document(
        id="doc1",
        page_content="Hello World",
        metadata={"outgoing": "parent", "keywords": ["greeting", "world"]},
    )

    doc2 = Document(
        id="doc2",
        page_content="Hello Earth",
        metadata={"outgoing": "parent", "keywords": ["greeting", "earth"]},
    )
    vector_store = InMemoryVectorStore(embedding=EarthEmbeddings())
    vector_store.add_documents([greetings, doc1, doc2])

    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=True,
        ),
        edges=[("outgoing", "incoming"), "keywords"],
        start_k=2,
        depth=2,
    )

    docs = retriever.invoke("Earth", start_k=1, depth=0)
    assert _doc_ids(docs) == ["doc2"]

    docs = retriever.invoke("Earth", depth=0)
    assert _doc_ids(docs) == ["doc1", "doc2"]

    docs = retriever.invoke("Earth", start_k=1, depth=1)
    assert set(_doc_ids(docs)) == {"doc1", "doc2", "greetings"}


def assert_document_format(doc: Document) -> None:
    assert doc.id is not None
    assert doc.page_content is not None
    assert doc.metadata is not None
    assert "__embedding" not in doc.metadata


class TestGraphTraversal:
    def test_invoke_sync(
        self,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a vector store."""
        vector_store = InMemoryVectorStore(embedding=ParserEmbeddings(dimension=2))
        vector_store.add_documents(graph_vector_store_docs)

        retriever = GraphTraversalRetriever(
            store=InMemoryTraversalAdapter(vector_store=vector_store),
            edges=[("out", "in"), "tag"],
            depth=2,
            start_k=2,
        )

        docs = retriever.invoke(input="[2, 10]", depth=0)
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0"}
        assert_document_format(docs[0])

        docs = retriever.invoke(input="[2, 10]")
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in docs}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])

    async def test_invoke_async(
        self,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """Graph traversal search on a graph store."""
        vector_store = InMemoryVectorStore(embedding=ParserEmbeddings(dimension=2))
        await vector_store.aadd_documents(graph_vector_store_docs)

        retriever = GraphTraversalRetriever(
            store=InMemoryTraversalAdapter(vector_store=vector_store),
            edges=[("out", "in"), "tag"],
            depth=2,
            start_k=2,
        )
        docs = await retriever.ainvoke(input="[2, 10]", depth=0)
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0"}
        assert_document_format(docs[0])

        docs = await retriever.ainvoke(input="[2, 10]")
        ss_labels = {doc.metadata["label"] for doc in docs}
        assert ss_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        assert_document_format(docs[0])


@pytest.fixture
def animal_docs() -> list[Document]:
    documents = []
    with open("tests/data/animals.jsonl", "r") as file:
        for line in file:
            data = json.loads(line.strip())
            documents.append(
                Document(
                    id=data["id"],
                    page_content=data["text"],
                    metadata=data["metadata"],
                )
            )

    return documents


def test_animals_sync(animal_docs: list[Document]) -> None:
    vector_store = InMemoryVectorStore(embedding=SimpleEmbeddings())

    vector_store.add_documents(animal_docs)

    # test non-graph search
    docs = vector_store.similarity_search("hedgehog", k=2)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    # test graph-search without normalized support
    # on a normalized bi-directional edge
    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=False,
        ),
        edges=["keywords"],
        start_k=2,
    )

    docs = retriever.invoke("hedgehog", depth=0)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=1)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=2)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    # test graph-search with normalized support
    # on a normalized bi-directional edge
    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=True,
        ),
        edges=["keywords"],
        start_k=2,
    )

    docs = retriever.invoke("hedgehog", depth=0)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=1)
    assert _doc_ids(docs) == ["cat", "coyote", "fox", "gazelle", "hedgehog", "mongoose"]

    docs = retriever.invoke("hedgehog", depth=2)
    assert _doc_ids(docs) == [
        "alpaca",
        "bison",
        "cat",
        "chicken",
        "coyote",
        "crow",
        "dingo",
        "dog",
        "fox",
        "gazelle",
        "hedgehog",
        "horse",
        "hyena",
        "jackal",
        "llama",
        "mongoose",
        "ostrich",
    ]

    # test graph-search without normalized support
    # on a standard bi-directional edge
    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=False,
        ),
        edges=["habitat"],
        start_k=2,
    )

    docs = retriever.invoke("hedgehog", depth=0)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=1)
    assert _doc_ids(docs) == ["antelope", "buffalo", "coyote", "fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=2)
    assert _doc_ids(docs) == ["antelope", "buffalo", "coyote", "fox", "hedgehog"]

    # test graph-search with normalized support
    # on a standard bi-directional edge
    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=True,
        ),
        edges=["habitat"],
        start_k=2,
    )

    docs = retriever.invoke("hedgehog", depth=0)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=1)
    assert _doc_ids(docs) == ["antelope", "buffalo", "coyote", "fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=2)
    assert _doc_ids(docs) == ["antelope", "buffalo", "coyote", "fox", "hedgehog"]

    # test graph-search without normalized support
    # on a standard -> normalized edge
    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=False,
        ),
        edges=[("habitat", "keywords")],
        start_k=2,
    )

    docs = retriever.invoke("hedgehog", depth=0)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=1)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=2)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    # test graph-search with normalized support
    # on a standard -> normalized edge
    retriever = GraphTraversalRetriever(
        store=InMemoryTraversalAdapter(
            vector_store=vector_store,
            support_normalized_metadata=True,
        ),
        edges=[("habitat", "keywords")],
        start_k=2,
    )

    docs = retriever.invoke("hedgehog", depth=0)
    assert _doc_ids(docs) == ["fox", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=1)
    assert _doc_ids(docs) == ["bison", "fox", "gazelle", "hedgehog"]

    docs = retriever.invoke("hedgehog", depth=2)
    assert _doc_ids(docs) == ["aardvark", "bison", "fox", "gazelle", "hedgehog"]
