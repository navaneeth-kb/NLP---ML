from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
X_tfidf = tfidf_vectorizer.fit_transform(documents)

# Convert the result to a dense matrix and print it
print("TF-IDF Matrix:\n", X_tfidf.toarray())

'''
TF-IDF Matrix:
 [[0.43370786 0.         0.57735027 0.43370786 0.43370786 0.        0.22479055 0.         0.         0.        ]
 [0.         0.43370786 0.         0.43370786 0.43370786 0.57735027 0.29884525 0.         0.         0.        ]
 [0.35745504 0.35745504 0.         0.         0.         0.         0.18534691 0.35745504 0.35745504 0.35745504]]
 In the TF-IDF matrix:

Each row corresponds to a document.
Each column represents a unique word in the vocabulary.
The values represent the TF-IDF score of each word in each document.
For example, the first row [0.43370786 0. 0.57735027 0.43370786 0.43370786 0. 0.22479055 0. 0. 0.] indicates the TF-IDF scores for the words "the", "cat", "sat", "on", "mat", "and", "dog", "are", and "friends" in the first document.

The TF-IDF (Term Frequency-Inverse Document Frequency) value for each word in a document is calculated using the following formula:

TF-IDF=TF*IDF
For example, let's say we want to calculate the TF-IDF value of the word "cat" in the first document "The cat sat on the mat." in a corpus containing 3 documents:


TF(t,d)) for "cat" in the first document is 1/6
Document Frequency (
DF(t)) for "cat" is 2 (it appears in 2 out of 3 documents)=2/3
Inverse Document Frequency =log(3/2)
TF-IDF value for "cat" in the first document is= 1/6 *log(3/2)
'''


