# main.py
from word2vec_distance.word2vec import Word2VecLoader, PhraseEmbeddings
from word2vec_distance.similarity import SimilarityCalculator
from word2vec_distance.utils import setup_logging, validate_file_existence

def main():
    setup_logging()
    validate_file_existence("phrases.csv")

    # Load Word2Vec model
    w2v_loader = Word2VecLoader("GoogleNews-vectors-negative300.bin", limit=1000000)
    w2v_loader.load_word2vec_model()

    # Create phrase embeddings
    phrase_embeddings = PhraseEmbeddings(w2v_loader)

    # Calculate similarities
    similarity_calculator = SimilarityCalculator(phrase_embeddings)
    with open("phrases.csv", "r", encoding="ISO-8859-1") as f:
    	phrases = f.readlines()


    for phrase1 in phrases:
        phrase1 = phrase1.strip()
        for phrase2 in phrases:
            phrase2 = phrase2.strip()
            if phrase1 != phrase2:
                cosine_similarity = similarity_calculator.calculate_cosine_similarity(phrase1, phrase2)
                l2_distance = similarity_calculator.calculate_l2_distance(phrase1, phrase2)
                print(f"Similarity between '{phrase1}' and '{phrase2}':")
                print(f"Cosine Similarity: {cosine_similarity}")
                print(f"L2 Distance: {l2_distance}")

if __name__ == "__main__":
    main()
