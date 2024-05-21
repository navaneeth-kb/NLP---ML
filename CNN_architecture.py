import tensorflow as tf
from tensorflow.keras import layers, models

# Initialize the Sequential model
model = models.Sequential()

# Add a convolutional layer with 32 filters, 3x3 kernel size, ReLU activation, and input shape (64, 64, 3)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# Add a max pooling layer with 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Add another convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add another max pooling layer with 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the feature maps into a 1D vector
model.add(layers.Flatten())

# Add a fully connected (dense) layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))

# Add the output layer with 10 units (for 10 classes) and softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

'''
Convolutional Layers: This is the layer, which is used to extract the feature from the input dataset.
Pooling layer: This layer is periodically inserted in the covnets and its main function is to reduce the size of volume which makes the computation fast reduces memory and also prevents overfitting.
Flattening: The resulting feature maps are flattened into a one-dimensional vector after the convolution and pooling layers so they can be passed into a completely linked layer for categorization or regression.
Fully Connected Layers/Dense: It takes the input from the previous layer and computes the final classification or regression task.
Output Layer: The output from the fully connected layers is then fed into a logistic function for classification tasks like sigmoid or softmax which converts the output of each class into the probability score of each class.

Libraries and Model Initialization:

1)import tensorflow as tf: Imports the TensorFlow library as tf. This library provides the core functionalities for building and training machine learning models.
2)from tensorflow.keras import layers, models: Imports specific components from the Keras API within TensorFlow. layers provides building blocks for neural network layers, and models allows us to construct sequential models like this one.
3)model = models.Sequential(): Initializes a sequential model. This type of model adds layers sequentially, one after the other.

Convolutional Layers and Pooling:

1)model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3))): This line defines the first convolutional layer. Let's unpack it:
layers.Conv2D: This is a 2D convolutional layer. It's used for extracting features from images.
32: This is the number of filters in the layer. Each filter learns to detect specific features in the image.
(3, 3): This defines the size of the kernel (filter). A 3x3 kernel means it will look at a 3x3 pixel area in the input image.

2)activation='relu': This specifies the activation function as ReLU (Rectified Linear Unit). It introduces non-linearity to the model, helping it learn complex patterns.

3)input_shape=(64, 64, 3): This defines the expected input shape of the image data. This image is expected to be 64 pixels high, 64 pixels wide, and have 3 color channels (RGB).

4)model.add(layers.MaxPooling2D((2, 2))): This adds a max pooling layer with a pool size of 2x2. It downsamples the feature maps by taking the maximum value from each 2x2 region. This helps reduce the dimensionality of data and introduces some level of translation invariance (the model becomes less sensitive to small shifts in the image).
We repeat steps 1 and 2 with a higher number of filters (64) in the second convolutional layer to extract more complex features.

Flattening and Dense Layers:

1)model.add(layers.Flatten()): This layer flattens the multidimensional feature maps from the convolutional layers into a 1D vector. This allows them to be fed into fully connected layers.
2)model.add(layers.Dense(64, activation='relu')): This adds a fully connected (dense) layer with 64 neurons. Dense layers connect every neuron from the previous layer to every neuron in this layer. Here, it takes the flattened feature vector and learns more abstract representations.
3)model.add(layers.Dense(10, activation='softmax')): This defines the output layer. It has 10 neurons because we're assuming the model classifies images into 10 different categories. The softmax activation ensures the output probabilities for each class sum to 1.

Compilation and Summary:

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']): This line compiles the model. It specifies three key aspects:
1)optimizer='adam': This defines the optimizer used to train the model. Adam is a popular optimizer that efficiently updates the model weights during training.
2)loss='sparse_categorical_crossentropy': This defines the loss function used to measure how well the model performs. In this case, we're using sparse categorical crossentropy, which is suitable for multi-class classification problems.
3)metrics=['accuracy']: This tells the model to track the accuracy metric during training and evaluation. Accuracy measures the percentage of correctly classified images.

model.summary(): This prints a summary of the model architecture, including the type and number o
'''
