import pickle

def load_glove_embeddings(glove_file_path, embedding_dim):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings[word] = vector
    embeddings['unk'] = [0.0] * embedding_dim 
    return embeddings

if __name__ == "__main__":
    glove_file_path = "glove.6B.50d.txt"
    embedding_dim = 50  
    embeddings = load_glove_embeddings(glove_file_path, embedding_dim)
    with open('./word_embedding.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    print("Word embeddings saved to word_embedding.pkl")
