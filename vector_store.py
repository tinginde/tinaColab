from typing import List, Dict
from math import sqrt
from collections import defaultdict
from text_utils import Document

class VectorIndex:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.vocab: Dict[str, int] = {}
        self.vectors: List[List[int]] = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in text.split()]

    def _build_index(self) -> None:
        # Build vocabulary
        for doc in self.documents:
            for token in self._tokenize(doc.page_content):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        # Create vectors
        for doc in self.documents:
            vec = [0] * len(self.vocab)
            for token in self._tokenize(doc.page_content):
                if token in self.vocab:
                    vec[self.vocab[token]] += 1
            self.vectors.append(vec)

    def _vectorize_query(self, query: str) -> List[int]:
        vec = [0] * len(self.vocab)
        for token in self._tokenize(query):
            if token in self.vocab:
                vec[self.vocab[token]] += 1
        return vec

    @staticmethod
    def _cosine(a: List[int], b: List[int]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sqrt(sum(x*x for x in a))
        norm_b = sqrt(sum(x*x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, text: str, k: int = 1) -> List[Document]:
        q_vec = self._vectorize_query(text)
        sims = [self._cosine(vec, q_vec) for vec in self.vectors]
        ranked = sorted(zip(sims, self.documents), key=lambda x: x[0], reverse=True)
        return [doc for _sim, doc in ranked[:k]]
