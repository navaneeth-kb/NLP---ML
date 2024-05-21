from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog are friends."
]

# Initialize the CountVectorizer with binary=True for one-hot encoding
vectorizer = CountVectorizer(binary=True)

# Fit and transform the documents
X_one_hot = vectorizer.fit_transform(documents)

# Convert the result to a dense matrix and print it
print("One-Hot Encoded Matrix:\n", X_one_hot.toarray())

'''
One-Hot Encoded Matrix:
 [[1 0 1 1 1 0 1 0 0 0]
 [0 1 0 1 1 1 1 0 0 0]
 [1 1 0 0 0 0 1 1 1 1]]
In the one-hot encoded matrix:

Each row corresponds to a document.
Each column represents a unique word in the vocabulary.
A value of 1 indicates the presence of the word in the document, and 0 indicates absence.
For example, the first row [1 0 1 1 1 0 1 0 0 0] indicates that the first document contains the words "the", "cat", "sat", "on", "mat", and "and", but not "dog", "are", or "friends".
'''
