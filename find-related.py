from gensim.models import KeyedVectors
import sys

def find_related(model_path, word, top_n=10):
    # Load model
    print(f"Loading model from {model_path}...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print(f"Model loaded! Vocabulary size: {len(model.index_to_key)}\n")
    
    # Check if word exists
    if word not in model:
        print(f"'{word}' not found in vocabulary!")
        print("\nShowing first 20 words in vocabulary:")
        for i, w in enumerate(model.index_to_key[:20]):
            print(f"{i+1}. {w}")
        return
    
    # Find similar words
    print(f"Most similar words to '{word}':\n")
    print(f"{'Rank':<6} {'Word':<30} {'Similarity':<10}")
    print("-" * 50)
    
    similar_words = model.most_similar(word, topn=top_n)
    for i, (similar_word, score) in enumerate(similar_words, 1):
        print(f"{i:<6} {similar_word:<30} {score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python find-related.py <model.bin> <word> [top_n]")
        print("Example: python find-related.py plotfiles/s100w8n5a0.025i7skipgram.bin ուսանող 20")
        sys.exit(1)
    
    model_path = sys.argv[1]
    word = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    find_related(model_path, word, top_n)