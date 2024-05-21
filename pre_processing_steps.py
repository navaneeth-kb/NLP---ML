import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Example text
text = "Hello! This is an example sentence to demonstrate preprocessing steps."

# Convert to lowercase
text = text.lower()

# Remove punctuation
# Create translation table
translation_table = str.maketrans('', '', string.punctuation)
# Apply translation table to text
text = text.translate(translation_table)

# Tokenize the text
tokens = word_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = []
for word in tokens:
    if word not in stop_words:
        filtered_tokens.append(word)

# Perform stemming
stemmer = PorterStemmer()
stemmed_tokens = []
for word in filtered_tokens:
    stemmed_word = stemmer.stem(word)
    stemmed_tokens.append(stemmed_word)

print("Original Text:", text)
print("Filtered Tokens:", filtered_tokens)
print("Stemmed Tokens:", stemmed_tokens)

'''Here are the typical preprocessing steps:

1)Lowercasing: Convert all characters to lowercase to ensure uniformity.
2)Removing Punctuation: Eliminate punctuation marks to avoid treating "word." and "word" as different tokens.
3)Removing Stop Words: Stop words (e.g., "and", "the", "is") are common words that usually don't carry significant meaning and can be removed.
4)Tokenization: Split the text into individual words or tokens.
5)Stemming/Lemmatization: Reduce words to their base or root form (e.g., "running" to "run").

str.maketrans() Function:

The str.maketrans() function creates a mapping table that can be used with the translate() method to replace specified characters.
Syntax: str.maketrans(x, y, z)
x: A string where each character in the string is replaced by the corresponding character in y.
y: A string with characters that correspond to characters in x.
z: A string with characters that should be deleted.
Arguments to str.maketrans('', '', string.punctuation):

'': The first argument is an empty string, meaning there are no characters to be replaced by other characters (because we're not replacing but deleting).
'': The second argument is also an empty string, for the same reason as above.
string.punctuation: The third argument is a string containing all punctuation characters that need to be removed.
string.punctuation includes characters like !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`.
text.translate(mapping_table)

translate() Method:
The translate() method returns a copy of the string where each character has been mapped through the given translation table.
The mapping_table is created by str.maketrans().
How It Works Together
str.maketrans('', '', string.punctuation) creates a translation table where each punctuation character maps to None (i.e., they are to be removed).
text.translate(mapping_table) uses this translation table to remove all punctuation characters from text.'''
