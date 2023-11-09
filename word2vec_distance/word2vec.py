# word2vec_distance/word2vec.py
import gensim

class Word2VecLoader:
    def __init__(self, model_path, limit=1000000):
        self.model_path = model_path
        self.limit = limit
        self.model = None

    def load_word2vec_model(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            self.model_path, binary=True, limit=self.limit
        )

    def get_word_embedding(self, word):
        if self.model is not None:
            if word in self.model:
                return self.model[word]
        return None

# word2vec_distance/word2vec.py

class PhraseEmbeddings:
    def __init__(self, w2v_loader):
        self.w2v_loader = w2v_loader

    def create_phrase_embedding(self, phrase):
        words = phrase.split()
        embeddings = [self.w2v_loader.get_word_embedding(word) for word in words]
        
        # Filter out None embeddings
        embeddings = [embedding for embedding in embeddings if embedding is not None]
        
        if not embeddings:
            return None  # Handle the case where no valid word embeddings were found
        
        # Calculate the sum of word embeddings and normalize
        phrase_embedding = [0.0] * len(embeddings[0])
        for embedding in embeddings:
            phrase_embedding = [sum(x) for x in zip(phrase_embedding, embedding)]
        
        # Normalize
        norm = sum(x ** 2 for x in phrase_embedding) ** 0.5
        phrase_embedding = [x / norm for x in phrase_embedding]
        
        return phrase_embedding
