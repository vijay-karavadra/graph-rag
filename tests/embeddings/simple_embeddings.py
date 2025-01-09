import json
import random

import sentence_transformers
from langchain_core.embeddings import Embeddings


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


class SimpleEmbeddings(Embeddings):
    def __init__(self):
        self._client = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a text using the all-MiniLM-L6-v2 transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))

        embeddings = self._client.encode(texts)

        if isinstance(embeddings, list):
            raise TypeError(
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )

        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using the all-MiniLM-L6-v2 transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings the all-MiniLM-L6-v2 transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed([text])[0]
