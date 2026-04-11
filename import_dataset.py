import pandas as pd
import os

def prepare_dataset(input_csv, img_dir, output_csv):
    # The downloaded dataset has no headers.
    # Standard format: center, left, right, steering, throttle, brake, speed
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv, names=columns)
    
    # We only need center, speed, direction (steering)
    # The images in 'center' have absolute paths from the original creator's PC.
    # We need to extract just the filename and prepend the actual img_dir.
    
    # Extract filename: it's the last part of the path separated by '\' or '/'
    df['filename'] = df['center'].apply(lambda x: x.split('\\')[-1].split('/')[-1])
    df['image_path'] = df['filename'].apply(lambda x: os.path.join(img_dir, x).replace('\\', '/'))
    
    df['direction'] = df['steering']
    
    # Normalize speed: divide by 40.0 (max speed in dataset is ~30.6)
    # This ensures speed fits within the model's tanh output range [-1, 1]
    df['speed'] = df['speed'] / 40.0
    
    # Create final df with columns expected by train.py
    final_df = df[['image_path', 'speed', 'direction']]
    
    # Filter out rows where image doesn't exist (safety check)
    print("Validating image paths...")
    valid_mask = final_df['image_path'].apply(os.path.exists)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} images not found, removing from dataset.")
    final_df = final_df[valid_mask].reset_index(drop=True)
    
    print(f"Saving to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print(f"Done! {len(final_df)} rows saved. Dataset is ready.")

if __name__ == "__main__":
    # Using the jungle dataset - white dotted center line, matches real road look
    downloaded_csv = r"c:\Users\ghant\OneDrive\Desktop\Ai Car\archive\self_driving_car_dataset_jungle\driving_log.csv"
    img_folder = r"c:\Users\ghant\OneDrive\Desktop\Ai Car\archive\self_driving_car_dataset_jungle\IMG"
    output_dataset = r"c:\Users\ghant\OneDrive\Desktop\Ai Car\dataset.csv"
    
    if os.path.exists(downloaded_csv):
        prepare_dataset(downloaded_csv, img_folder, output_dataset)
    else:
        print(f"Could not find dataset at: {downloaded_csv}")
        print("Trying fallback: self_driving_car_dataset_make...")
        downloaded_csv = r"c:\Users\ghant\Downloads\archive\self_driving_car_dataset_make\driving_log.csv"
        img_folder = r"c:\Users\ghant\Downloads\archive\self_driving_car_dataset_make\IMG"
        if os.path.exists(downloaded_csv):
            prepare_dataset(downloaded_csv, img_folder, output_dataset)
        else:
            print("Fallback dataset also not found. Please check dataset paths.")
