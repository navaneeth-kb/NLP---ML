from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog are friends."
]

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the documents
X = vectorizer.fit_transform(documents)

# Convert the result to a dense matrix and print it
print("Vocabulary:", vectorizer.vocabulary_)
print("Document-Term Matrix:\n", X.toarray())

'''
1)This line imports the CountVectorizer class from the sklearn.feature_extraction.text module. CountVectorizer is a method for converting text documents into a matrix of token counts.
3)This line initializes a CountVectorizer object. It creates an instance of the CountVectorizer class with default parameters. Later, we'll use this object to transform the text data into a document-term matrix.
4)vectorizer.fit_transform(documents) fits the CountVectorizer object to the documents and transforms the documents into a document-term matrix. It learns the vocabulary of the corpus (the unique words present) and converts each document into a vector representing word counts.
17,18)
vectorizer.vocabulary_ returns a dictionary where keys are the terms (words) in the vocabulary and values are the indices of the terms in the vocabulary.
X.toarray() converts the sparse matrix X (which represents the document-term matrix) into a dense NumPy array for better visualization.
Finally, it prints out the vocabulary and the document-term matrix. The vocabulary shows the unique words in the documents and their corresponding indices, while the document-term matrix shows the word counts for each word in each document.

O/P:
Vocabulary: {'the': 6, 'cat': 0, 'sat': 4, 'on': 3, 'mat': 2, 'dog': 1, 'log': 5, 'and': 7, 'are': 8, 'friends': 9}
Document-Term Matrix:
 [[1 0 1 1 1 0 1 0 0 0]
 [0 1 0 1 1 1 1 0 0 0]
 [1 1 0 0 0 0 2 1 1 1]]

'''
