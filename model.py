import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(60, 160, 3)):
    """
    Builds a Convolutional Neural Network for autonomous driving.
    Input: image (120x160x3)
    Output: [speed, direction]
    """
    model = models.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Flattening
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    
    # Output layer: 2 values [speed, direction] in range [-1, 1]
    model.add(layers.Dense(2, activation='tanh'))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
