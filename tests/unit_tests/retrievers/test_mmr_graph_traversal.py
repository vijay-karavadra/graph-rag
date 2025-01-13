import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from graph_pancake.retrievers.graph_mmr_traversal_retriever import (
    GraphMMRTraversalRetriever,
)
from graph_pancake.retrievers.traversal_adapters.mmr import (
    InMemoryMMRTraversalAdapter,
)
from tests.conftest import assert_document_format, sorted_doc_ids
from tests.embeddings import (
    ParserEmbeddings,
)


@pytest.mark.parametrize("vector_store_type", ["in-memory"])
@pytest.mark.parametrize("embedding_type", ["angular"])
def test_mmr_traversal(vector_store: VectorStore, vector_store_type: str) -> None:
    """ Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          //      \\
         //        \\  v1
    v3  ||    .     || query
         \\        //  v0
          \\______//                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    v0 = Document(id="v0", page_content="-0.124")
    v1 = Document(id="v1", page_content="+0.127")
    v2 = Document(id="v2", page_content="+0.25")
    v3 = Document(id="v3", page_content="+1.0")

    v0.metadata["outgoing"] = "link"
    v2.metadata["incoming"] = "link"
    v3.metadata["incoming"] = "link"

    vector_store.add_documents([v0, v1, v2, v3])

    retriever = GraphMMRTraversalRetriever(
        store=InMemoryMMRTraversalAdapter(vector_store=vector_store),
        edges=[("outgoing", "incoming")],
        fetch_k=2,
        k=2,
        depth=2,
    )

    docs = retriever.invoke("0.0", k=2, fetch_k=2)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    docs = retriever.invoke("0.0", k=2, fetch_k=2, depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    docs = retriever.invoke("0.0", k=2, fetch_k=3, depth=0)
    assert sorted_doc_ids(docs) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    docs = retriever.invoke("0.0", k=2, score_threshold=0.2)
    assert sorted_doc_ids(docs) == ["v0"]

    # with k=4 we should get all of the documents.
    docs = retriever.invoke("0.0", k=4)
    assert sorted_doc_ids(docs) == ["v0", "v1", "v2", "v3"]


class TestMmrGraphTraversal:
    def test_invoke_sync(
        self,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """MMR Graph traversal search on a vector store."""
        vector_store = InMemoryVectorStore(embedding=ParserEmbeddings(2))
        vector_store.add_documents(graph_vector_store_docs)

        retriever = GraphMMRTraversalRetriever(
            store=InMemoryMMRTraversalAdapter(vector_store=vector_store),
            vector_store=vector_store,
            edges=[("out", "in"), "tag"],
            depth=2,
            k=2,
        )

        docs = retriever.invoke(input="[2, 10]")
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])

    async def test_invoke_async(
        self,
        graph_vector_store_docs: list[Document],
    ) -> None:
        """MMR Graph traversal search on a vector store."""
        vector_store = InMemoryVectorStore(embedding=ParserEmbeddings(2))
        await vector_store.aadd_documents(graph_vector_store_docs)

        retriever = GraphMMRTraversalRetriever(
            store=InMemoryMMRTraversalAdapter(vector_store=vector_store),
            vector_store=vector_store,
            edges=[("out", "in"), "tag"],
            depth=2,
            k=2,
        )
        mt_labels = set()
        docs = await retriever.ainvoke(input="[2, 10]")
        mt_labels = {doc.metadata["label"] for doc in docs}
        assert mt_labels == {"AR", "BR"}
        assert docs[0].metadata
        assert_document_format(docs[0])

    @pytest.mark.parametrize("vector_store_type", ["in-memory"])
    @pytest.mark.parametrize("embedding_type", ["animal"])
    @pytest.mark.parametrize("support_normalized_metadata", [False, True])
    def test_animals_sync(
        self,
        support_normalized_metadata: bool,
        vector_store: VectorStore,
        animal_docs: list[Document],
    ) -> None:
        vector_store.add_documents(animal_docs)

        query = "small agile mammal"
        depth_0_expected = ["fox", "mongoose"]

        # test non-graph search
        docs = vector_store.similarity_search(query, k=2)
        assert sorted_doc_ids(docs) == depth_0_expected

        # test graph-search on a normalized bi-directional edge
        retriever = GraphMMRTraversalRetriever(
            store=InMemoryMMRTraversalAdapter(
                vector_store=vector_store,
                support_normalized_metadata=support_normalized_metadata,
            ),
            edges=["keywords"],
            fetch_k=2,
        )

        docs = retriever.invoke(query, depth=0)
        assert sorted_doc_ids(docs) == depth_0_expected

        docs = retriever.invoke(query, depth=1)
        assert (
            sorted_doc_ids(docs) == ["fox", "mongoose"]
            if support_normalized_metadata
            else depth_0_expected
        )

        docs = retriever.invoke(query, depth=2)
        assert (
            sorted_doc_ids(docs)
            # WOULD HAVE EXPECTED THIS AT DEPTH 1
            == ["cat", "gazelle", "jackal", "mongoose"]
            if support_normalized_metadata
            else depth_0_expected
        )

        # test graph-search on a standard bi-directional edge
        retriever = GraphMMRTraversalRetriever(
            store=InMemoryMMRTraversalAdapter(
                vector_store=vector_store,
                support_normalized_metadata=support_normalized_metadata,
            ),
            edges=["habitat"],
            fetch_k=2,
        )

        docs = retriever.invoke(query, depth=0)
        assert sorted_doc_ids(docs) == depth_0_expected

        docs = retriever.invoke(query, depth=1)
        assert sorted_doc_ids(docs) == ["fox", "mongoose"]

        docs = retriever.invoke(query, depth=2)
        # WOULD HAVE EXPECTED THIS AT DEPTH 1
        assert sorted_doc_ids(docs) == ["bobcat", "deer", "fox", "mongoose"]

        # test graph-search on a standard -> normalized edge
        retriever = GraphMMRTraversalRetriever(
            store=InMemoryMMRTraversalAdapter(
                vector_store=vector_store,
                support_normalized_metadata=support_normalized_metadata,
            ),
            edges=[("habitat", "keywords")],
            fetch_k=2,
        )

        docs = retriever.invoke(query, depth=0)
        assert sorted_doc_ids(docs) == depth_0_expected

        docs = retriever.invoke(query, depth=1)
        assert (
            sorted_doc_ids(docs) == ["fox", "mongoose"]
            if support_normalized_metadata
            else depth_0_expected
        )

        docs = retriever.invoke(query, depth=2)
        assert (
            # WOULD HAVE EXPECTED THIS AT DEPTH 1
            sorted_doc_ids(docs) == ["bear", "bobcat", "fox", "mongoose"]
            if support_normalized_metadata
            else depth_0_expected
        )
