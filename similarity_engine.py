from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:

    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def compute_similarity(self, text1, text2):

        if not text1 or not text2:
            return 0.0

        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])

        score = cosine_similarity(emb1, emb2)[0][0]

        # 🔥 Normalize safely between 0 and 1
        score = float(score)
        score = max(0.0, min(1.0, score))

        return score
