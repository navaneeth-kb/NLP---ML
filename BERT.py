from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
# - 'bert-base-uncased': Specifies the version of BERT to load, which is a smaller, uncased version of BERT.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
# - input_text: The text that we want to tokenize and feed into the model.
# - return_tensors='pt': Returns PyTorch tensors, suitable for use with the BERT model.
input_text = "The cat sat on the mat."
inputs = tokenizer(input_text, return_tensors='pt')

# Get BERT embeddings
# - The model processes the inputs and outputs the embeddings.
outputs = model(**inputs)

# Extract the last hidden states
# - last_hidden_state: Contains the hidden states output by the model.
last_hidden_states = outputs.last_hidden_state

# Print the embeddings
print(last_hidden_states)

'''
Overview:
BERT, developed by Google, is a state-of-the-art pre-trained transformer model designed for a variety of natural language processing (NLP) tasks. It captures bidirectional context, making it extremely powerful for understanding the nuances of language.

Key Points:

Transformer Architecture: BERT is based on the transformer architecture, specifically the encoder part. Transformers use self-attention mechanisms to weigh the importance of different words in a sentence relative to each other, enabling the model to capture complex dependencies.

Bidirectional Training: Unlike traditional models that read text sequentially (left-to-right or right-to-left), BERT reads the entire sequence of words at once. This allows it to understand the context of a word based on all the words surrounding it, both before and after.

Pre-training and Fine-tuning: BERT is pre-trained on large corpora using two tasks:

Masked Language Modeling (MLM): Randomly masks some of the tokens in the input, and the model is trained to predict the masked tokens.
Next Sentence Prediction (NSP): Trains the model to understand the relationship between two sentences by predicting if one sentence follows another.
After pre-training, BERT can be fine-tuned on specific tasks such as question answering, sentiment analysis, or named entity recognition.
'''
