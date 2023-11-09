# word2vec_distance/similarity.py
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

class SimilarityCalculator:
    def __init__(self, phrase_embeddings):
        self.phrase_embeddings = phrase_embeddings

    def calculate_cosine_similarity(self, phrase1, phrase2):
        emb1 = self.phrase_embeddings.create_phrase_embedding(phrase1)
        emb2 = self.phrase_embeddings.create_phrase_embedding(phrase2)
        if emb1 is not None and emb2 is not None:
            return cosine_similarity([emb1], [emb2])[0][0]
        return None

    def calculate_l2_distance(self, phrase1, phrase2):
        emb1 = self.phrase_embeddings.create_phrase_embedding(phrase1)
        emb2 = self.phrase_embeddings.create_phrase_embedding(phrase2)
        if emb1 is not None and emb2 is not None:
            return euclidean_distances([emb1], [emb2])[0][0]
        return None