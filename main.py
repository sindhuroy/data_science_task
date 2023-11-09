import gdown
import os
import logging
import gensim
from word2vec_distance.word2vec import Word2VecLoader, PhraseEmbeddings
from word2vec_distance.similarity import SimilarityCalculator
from word2vec_distance.utils import setup_logging, validate_file_existence

# Define the Google Drive file ID
file_id = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'

# Define the destination path for the downloaded Word2Vec model
model_path = 'GoogleNews-vectors-negative300.bin'

output_vectors_file = 'word2vec_vectors.txt'

# Download the Word2Vec model if it doesn't exist
if not os.path.exists(model_path):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)
    
def save_vectors_as_flat_file(word2vec_model, output_file, limit=1000000):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, word in enumerate(word2vec_model.index_to_key[:limit]):
            vector = word2vec_model[word]
            vector_str = ' '.join(map(str, vector))
            file.write(f'{word} {vector_str}\n')
def main():
    setup_logging()
    validate_file_existence("phrases.csv")

    # Load Word2Vec model
    w2v_loader = Word2VecLoader(model_path, limit=1000000)
    w2v_loader.load_word2vec_model()
    save_vectors_as_flat_file(w2v_loader.model, output_vectors_file)

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
                logging.info(f"Similarity between '{phrase1}' and '{phrase2}':")
                logging.info(f"Cosine Similarity: {cosine_similarity}")
                logging.info(f"L2 Distance: {l2_distance}")

if __name__ == "__main__":
    main()