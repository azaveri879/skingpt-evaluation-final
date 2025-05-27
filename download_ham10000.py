
"""
Script to download and organize the HAM10000 dataset from Kaggle.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import json
import sys

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

# Setup credentials before importing kaggle
setup_kaggle_credentials()
import kaggle

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
            else:
                raise FileNotFoundError("Metadata file not found after download")
            
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            raise

if __name__ == "__main__":
    downloader = HAM10000Downloader()
    downloader.download_dataset() 
