import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import build_cnn_model
from utils import augment_image, load_data
from vision import preprocess_image

def data_generator(df, batch_size=32, is_training=True):
    """
    Generator that yields batches of (image, [speed, direction]).
    """
    while True:
        batch_df = df.sample(n=batch_size)
        images = []
        labels = []
        
        for index, row in batch_df.iterrows():
            img_path = row['image_path']
            speed = row['speed']
            direction = row['direction']
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Augmentation during training
            if is_training:
                img, speed, direction = augment_image(img, speed, direction)
            
            # Preprocess (Resize, ROI, Normalize)
            img = preprocess_image(img)
            
            images.append(img)
            labels.append([speed, direction])
            
        yield np.array(images), np.array(labels)

def train_model(csv_path, epochs=10, batch_size=32):
    # Load data
    df = load_data(csv_path)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2)
    print(f"Total samples: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Build model
    model = build_cnn_model()
    
    # Train
    history = model.fit(
        data_generator(train_df, batch_size=batch_size, is_training=True),
        steps_per_epoch=len(train_df) // batch_size,
        validation_data=data_generator(val_df, batch_size=batch_size, is_training=False),
        validation_steps=len(val_df) // batch_size,
        epochs=epochs
    )
    
    # Save model
    model_save_path = 'autonomous_car_model.keras'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Print results
    print("Training finished.")
    print(f"Final training loss: {history.history['loss'][-1]}")
    print(f"Final validation loss: {history.history['val_loss'][-1]}")

if __name__ == "__main__":
    # Example usage:
    train_model('dataset.csv', epochs=10)
