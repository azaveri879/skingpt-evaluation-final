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
import kaggle

def setup_kaggle_credentials():
    """Setup Kaggle credentials."""
    print("Setting up Kaggle credentials...")
    
    # Create .kaggle directory
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Check if credentials already exist
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json):
        print("\nPlease follow these steps to set up Kaggle credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to the 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This will download a kaggle.json file")
        print("\nThen, please provide the path to your kaggle.json file:")
        
        json_path = input().strip()
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find kaggle.json at {json_path}")
        
        # Copy the file to the correct location
        shutil.copy2(json_path, kaggle_json)
        
        # Set correct permissions
        os.chmod(kaggle_json, 0o600)
    
    print("Kaggle credentials setup complete!")

class HAM10000Downloader:
    def __init__(self):
        # Kaggle dataset details
        self.dataset_name = "kmader/skin-cancer-mnist-ham10000"
        
    def download_and_extract(self, output_dir: str):
        """Download and extract dataset from Kaggle."""
        print(f"Downloading HAM10000 dataset from Kaggle...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            self.dataset_name,
            path=output_dir,
            unzip=True
        )
        
        # Move files to correct structure
        print("Organizing dataset files...")
        
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Move all jpg files to images directory
        for file in Path(output_dir).glob("*.jpg"):
            shutil.move(str(file), os.path.join(images_dir, file.name))
        
        # Ensure metadata file is in the right place
        metadata_file = os.path.join(output_dir, "HAM10000_metadata.csv")
        if not os.path.exists(metadata_file):
            # If metadata is in a subdirectory, find and move it
            for file in Path(output_dir).rglob("HAM10000_metadata.csv"):
                shutil.move(str(file), metadata_file)
                break
    
    def download_dataset(self, output_dir: str = "data/ham10000"):
        """Download and organize the entire HAM10000 dataset."""
        try:
            # Download and organize dataset
            self.download_and_extract(output_dir)
            
            # Verify dataset
            metadata_path = os.path.join(output_dir, "HAM10000_metadata.csv")
            if os.path.exists(metadata_path):
                df = pd.read_csv(metadata_path)
                print(f"Dataset downloaded and organized in {output_dir}")
                print(f"Total images: {len(df)}")
                return True
            else:
                raise FileNotFoundError("Metadata file not found after download")
            
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return False

def download_ham10000(data_dir):
    """Download and prepare HAM10000 dataset."""
    try:
        # Setup Kaggle credentials
        setup_kaggle_credentials()
        
        # Download dataset
        downloader = HAM10000Downloader()
        return downloader.download_dataset(data_dir)
    except Exception as e:
        print(f"Error in download_ham10000: {str(e)}")
        return False

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