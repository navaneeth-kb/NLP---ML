from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Create a Sequential model
model = Sequential()

# Add a Bidirectional LSTM layer
# - LSTM(50): LSTM layer with 50 units (hidden states)
# - activation='tanh': Use the tanh activation function
# - input_shape=(100, 1): Input shape is (timesteps, features), here 100 timesteps with 1 feature each
model.add(Bidirectional(LSTM(50, activation='tanh'), input_shape=(100, 1)))

# Add a Dense layer
# - Dense(1): Fully connected layer with 1 unit (for regression tasks, change this for classification)
model.add(Dense(1))

# Compile the model
# - optimizer='adam': Use the Adam optimizer
# - loss='mean_squared_error': Use mean squared error as the loss function (suitable for regression tasks)
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
# - Displays the layers and their shapes, number of parameters, etc.
model.summary()


'''
Overview:
A BiLSTM, or Bidirectional Long Short-Term Memory, is a type of Recurrent Neural Network (RNN) that consists of two LSTM layers: one processing the input sequence forward and the other processing it backward. This structure allows the model to capture context from both past and future states, providing a more comprehensive understanding of the sequence data.

Key Points:

LSTM (Long Short-Term Memory): LSTM networks are designed to overcome the limitations of traditional RNNs, particularly the vanishing gradient problem. They use gates (input, forget, and output) to regulate the flow of information and maintain long-term dependencies.

Bidirectional Nature: In a BiLSTM, two separate LSTM networks are used. One processes the input sequence from the beginning to the end (forward LSTM), and the other processes the sequence from the end to the beginning (backward LSTM).

Combining Outputs: The outputs of the forward and backward LSTMs are concatenated or summed, providing a richer representation of the input data by considering both past and future contexts.
'''
