"""Test of Graph Traversal Retriever"""

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from graph_pancake.retrievers.graph_traversal_retriever import GraphTraversalRetriever
from graph_pancake.retrievers.traversal_adapters.eager import (
    AstraTraversalAdapter,
    CassandraTraversalAdapter,
    ChromaTraversalAdapter,
    OpenSearchTraversalAdapter,
    TraversalAdapter,
)
from tests.conftest import assert_document_format, sorted_doc_ids

vector_store_types = [
    "astra-db",
    "cassandra",
    "chroma-db",
    "open-search",
]


def get_adapter(vector_store: VectorStore, vector_store_type: str) -> TraversalAdapter:
    if vector_store_type == "astra-db":
        return AstraTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "cassandra":
        return CassandraTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "chroma-db":
        return ChromaTraversalAdapter(vector_store=vector_store)
    elif vector_store_type == "open-search":
        return OpenSearchTraversalAdapter(vector_store=vector_store)
    else:
        msg = f"Unknown vector store type: {vector_store_type}"
        raise ValueError(msg)


# this test has complex metadata fields (values with list type)
# only `astra-db`` and `open-search` can correctly handle
# complex metadata fields at this time.


@pytest.mark.parametrize("vector_store_type", ["astra-db", "open-search"])
@pytest.mark.parametrize("embedding_type", ["earth"])
def test_traversal(
    vector_store: VectorStore,
    vector_store_type: str,
    hello_docs: list[Document],
) -> None:
    vector_store.add_documents(hello_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphTraversalRetriever(
        store=vector_store_adapter,
        edges=[("outgoing", "incoming"), "keywords"],
        start_k=2,
        depth=2,
    )

    docs = retriever.invoke("Earth", start_k=1, depth=0)
    assert sorted_doc_ids(docs) == ["doc2"]

    docs = retriever.invoke("Earth", depth=0)
    assert sorted_doc_ids(docs) == ["doc1", "doc2"]

    docs = retriever.invoke("Earth", start_k=1, depth=1)
    assert sorted_doc_ids(docs) == ["doc1", "doc2", "greetings"]


# the tests below use simple metadata fields
# astra-db, cassandra, chroma-db, and open-search
# can all handle simple metadata fields


@pytest.mark.parametrize("vector_store_type", vector_store_types)
@pytest.mark.parametrize("embedding_type", ["parser-d2"])
def test_invoke_sync(
    vector_store: VectorStore,
    vector_store_type: str,
    graph_vector_store_docs: list[Document],
) -> None:
    """Graph traversal search on a vector store."""
    vector_store.add_documents(graph_vector_store_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphTraversalRetriever(
        store=vector_store_adapter,
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


@pytest.mark.parametrize("vector_store_type", vector_store_types)
@pytest.mark.parametrize("embedding_type", ["parser-d2"])
async def test_invoke_async(
    vector_store: VectorStore,
    vector_store_type: str,
    graph_vector_store_docs: list[Document],
) -> None:
    """Graph traversal search on a graph store."""
    await vector_store.aadd_documents(graph_vector_store_docs)

    vector_store_adapter = get_adapter(
        vector_store=vector_store,
        vector_store_type=vector_store_type,
    )

    retriever = GraphTraversalRetriever(
        store=vector_store_adapter,
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
