import tensorflow as tf
import os

model_path = r'c:\Users\ghant\OneDrive\Desktop\Ai Car\autonomous_car_model.keras'
tflite_path = r'c:\Users\ghant\OneDrive\Desktop\Ai Car\autonomous_car_model.tflite'

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Successfully converted to {tflite_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Model not found at {model_path}")
