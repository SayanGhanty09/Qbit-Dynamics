import tensorflow as tf
import os

h5_path = r'c:\Users\ghant\OneDrive\Desktop\Ai Car\autonomous_car_model.h5'
keras_path = r'c:\Users\ghant\OneDrive\Desktop\Ai Car\autonomous_car_model.keras'

if os.path.exists(h5_path):
    print(f"Loading legacy model from {h5_path}...")
    try:
        # Try loading with compile=False to avoid layer/optimizer issues
        model = tf.keras.models.load_model(h5_path, compile=False)
        print(f"Saving to native format at {keras_path}...")
        model.save(keras_path)
        print("Success!")
    except Exception as e:
        print(f"Failed to migrate format: {e}")
else:
    print(f"Source model not found at {h5_path}")
