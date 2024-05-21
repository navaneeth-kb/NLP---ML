'''
Word2vec is a technique used in Natural Language Processing (NLP) to turn words into numbers. But these aren't just any random numbers! They capture the meaning and relationships between words.
Imagine a map where words are like cities. Words with similar meanings are closer together on the map, like "king" and "queen" or "happy" and "joyful." Word2vec creates this kind of map for words in a computer's understanding.
'''

from gensim.models import Word2Vec

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog are friends."
]

# Preprocess the text (tokenize, lowercasing)
processed_docs = []  # Initialize an empty list to store processed documents
for doc in documents:  # Iterate over each document in the documents list
    # Convert the document to lowercase and split it into words
    processed_doc = doc.lower().split()
    # Append the processed document to the processed_docs list
    processed_docs.append(processed_doc)

'''Initialize and train the Word2Vec model
Initializing the Word2Vec Model:
model = Word2Vec(...): This line initializes a Word2Vec model using Gensim's Word2Vec class.
Parameters:
1)sentences=processed_docs: This parameter specifies the input data for the model. Here, processed_docs is a list of tokenized and preprocessed documents.
2)vector_size=100: This parameter sets the dimensionality of the word vectors (word embeddings) that the model will learn. In this case, each word will be represented as a vector of 100 dimensions.
3)window=5: This parameter determines the maximum distance between the current and predicted word within a sentence. In other words, it defines the size of the context window for learning word embeddings. Here, it's set to 5, meaning that the model will consider the five words before and after the current word when training.
4)min_count=1: This parameter specifies the minimum frequency count of words. Words with a frequency count lower than this value will be ignored during training. Setting it to 1 ensures that all words in the corpus are considered, regardless of their frequency.
5)workers=4: This parameter sets the number of threads to use for training the model. It determines the degree of parallelism during training. Here, it's set to 4, meaning that the training process will use four CPU cores.
'''
model = Word2Vec(sentences=processed_docs, vector_size=100, window=5, min_count=1, workers=4)

# Save the model for later use
model.save("word2vec.model")

# Load the model (if needed)
model = Word2Vec.load("word2vec.model")

# Get the vector for a word
vector = model.wv['cat']
print(f"Vector for 'cat': {vector}")

# Find the most similar words
similar_words = model.wv.most_similar('cat')
print(f"Words similar to 'cat': {similar_words}")

'''
Vector for 'cat': [-0.012, 0.045, -0.032, ... (more values) ... , 0.009]
Words similar to 'cat': [('dog', 0.965), ('sat', 0.951), ('the', 0.943), ('are', 0.927), ('friends', 0.914), ('mat.', 0.894), ('log.', 0.888), ('on', 0.871), ('and', 0.856), ('the', 0.832)]
Vector for 'cat': This line prints the learned word vector for the word 'cat'. It's represented as a list of 100 floating-point values.

The 100 float values represent the word embedding or vector representation of the word "cat" in a high-dimensional space. In the Word2Vec model, each word is represented as a dense vector of real numbers, where each dimension of the vector captures a different aspect or feature of the word's meaning or context.

Words similar to 'cat': This line prints the words most similar to 'cat', along with their similarity scores. Each tuple consists of a word and its similarity score relative to 'cat'.
'''
