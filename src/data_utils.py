import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import zipfile
import shutil
from google.cloud import storage
import io
import json
from pathlib import Path
# import kaggle

def download_scin(data_dir):
    """Download and prepare SCIN dataset from Google Cloud Storage."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize GCS client
    storage_client = storage.Client(project='dx-scin-public')
    bucket = storage_client.bucket('dx-scin-public-data')
    
    # Define paths
    cases_csv = 'dataset/scin_cases.csv'
    labels_csv = 'dataset/scin_labels.csv'
    gcs_images_dir = 'dataset/images/'
    
    # Download and save CSV files
    def download_and_save_csv(csv_path, output_path):
        print(f"Downloading {csv_path}...")
        df = pd.read_csv(
            io.BytesIO(bucket.blob(csv_path).download_as_string()),
            dtype={'case_id': str}
        )
        df['case_id'] = df['case_id'].astype(str)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        return df
    
    # Download cases and labels
    cases_df = download_and_save_csv(
        cases_csv,
        os.path.join(data_dir, "scin_cases.csv")
    )
    
    labels_df = download_and_save_csv(
        labels_csv,
        os.path.join(data_dir, "scin_labels.csv")
    )
    
    # Merge cases and labels
    merged_df = pd.merge(cases_df, labels_df, on='case_id')
    merged_df.to_csv(os.path.join(data_dir, "scin_merged.csv"), index=False)
    
    # Download images
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Get unique image paths
    image_path_columns = ['image_1_path', 'image_2_path', 'image_3_path']
    image_paths = []
    for col in image_path_columns:
        image_paths.extend(merged_df[col].dropna().unique())
    
    print(f"Downloading {len(image_paths)} images...")
    for image_path in tqdm(image_paths):
        # Remove 'dataset/images/' prefix if present
        clean_path = image_path.replace('dataset/images/', '')
        
        # Get blob
        blob = bucket.blob(os.path.join(gcs_images_dir, clean_path))
        
        # Create output path
        output_path = os.path.join(image_dir, clean_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download if not exists
        if not os.path.exists(output_path):
            try:
                blob.download_to_filename(output_path)
            except Exception as e:
                print(f"Error downloading {image_path}: {str(e)}")
    
    return True

def prepare_dataset(data_dir, dataset_name, test_size=0.2, val_size=0.1, random_state=42):
    """Prepare dataset for training and evaluation."""
    if dataset_name == "ham10000":
        metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    elif dataset_name == "scin":
        metadata_path = os.path.join(data_dir, "scin_merged.csv")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Split into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state)
    
    # Save splits
    splits_dir = os.path.join(data_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    
    return train_df, val_df, test_df

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input."""
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        return image
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def get_class_distribution(df, label_column):
    """Get class distribution from dataset."""
    return df[label_column].value_counts()

def visualize_class_distribution(df, label_column, title, save_path=None):
    """Visualize class distribution."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=label_column)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 