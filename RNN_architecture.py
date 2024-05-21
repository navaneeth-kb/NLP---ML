import tensorflow as tf
from tensorflow.keras import layers, models

# Initialize a Sequential model
model = models.Sequential()

# Add a SimpleRNN layer with 50 units
# - activation='tanh': The activation function used in the RNN cells.
# - input_shape=(100, 1): The shape of the input data (timesteps, features).
#   - timesteps=100: The length of the input sequences.
#   - features=1: The number of features at each timestep.
model.add(layers.SimpleRNN(50, activation='tanh', input_shape=(100, 1)))

# Add a Dense (fully connected) layer with 1 unit
# - This layer is used to produce the final output of the network.
# - Since it's a regression task, we use 1 unit.
model.add(layers.Dense(1))

# Compile the model
# - optimizer='adam': Adam optimizer is used to minimize the loss.
# - loss='mean_squared_error': Mean Squared Error (MSE) is used as the loss function, which is common for regression tasks.
model.compile(optimizer='adam', loss='mean_squared_error')

# Print a summary of the model
# - This shows the layers of the model, output shapes, and the number of parameters.
model.summary()
