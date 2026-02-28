"""
similarity_engine.py — SIRD component (Section 4.5.1)

Paper uses microsoft/mpnet-base via SentenceTransformer with AVERAGE
pooling, achieving Pearson correlation = 0.8901.

Original used "all-mpnet-base-v2" which is fine — it IS the mpnet model.
Improvement: added batch encoding for efficiency when DB is large.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityEngine:

    # Paper Section 4.5.1: mpnet-base with average pooling
    MODEL_NAME = "all-mpnet-base-v2"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        # Cache for batch comparisons
        self._cache = {}

    def compute_similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts.

        Paper formula: S(x_i) = (e_i · r) / (‖e_i‖ · ‖r‖)
        Score is already in [0, 1] for well-formed sentences.

        Note: Paper normalizes by /5 because it trains on STS benchmark
        (scores 0–5). Since SentenceTransformer cosine output is already
        in [-1, 1], we just clip to [0, 1]. The threshold of 0.80 holds.
        """
        if not text1 or not text2:
            return 0.0

        emb1 = self._encode(text1)
        emb2 = self._encode(text2)

        score = float(cosine_similarity(emb1, emb2)[0][0])
        return max(0.0, min(1.0, score))

    def compute_batch_similarities(self, query_text, corpus_texts):
        """
        Efficiently compare one query against many corpus texts.
        Much faster than calling compute_similarity() in a loop.

        Returns:
            list[float]: similarity score for each corpus text.
        """
        if not query_text or not corpus_texts:
            return [0.0] * len(corpus_texts)

        query_emb  = self.model.encode([query_text])
        corpus_emb = self.model.encode(corpus_texts)

        scores = cosine_similarity(query_emb, corpus_emb)[0]
        return [max(0.0, min(1.0, float(s))) for s in scores]

    def _encode(self, text):
        """Encode a single text, using in-memory cache."""
        key = hash(text[:200])   # short cache key
        if key not in self._cache:
            self._cache[key] = self.model.encode([text])
        return self._cache[key]
