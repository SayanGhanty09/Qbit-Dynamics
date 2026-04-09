import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os

def load_data(csv_path):
    """
    Loads data from CSV file with columns: [image_path, speed, direction]
    """
    df = pd.read_csv(csv_path)
    return df

def augment_image(image, speed, direction):
    """
    Apply data augmentation: brightness, flipping and noise.
    """
    # Random brightness
    brightness = np.random.uniform(0.7, 1.3)
    aug_img = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Horizontal flipping
    if np.random.rand() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
        direction = -direction  # Flip direction label
        
    # Add random noise
    noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
    aug_img = cv2.add(aug_img, noise)
    
    return aug_img, speed, direction

def convert_to_tflite(keras_model_path, tflite_path):
    """
    Converts a saved Keras model to TFLite format.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(keras_model_path)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to {tflite_path}")

def save_image(image, path):
    cv2.imwrite(path, image)
