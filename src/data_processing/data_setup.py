#!/usr/bin/env python3
"""
Data setup script for Brain Tumor Detection project.
Downloads the dataset using kagglehub and performs preprocessing.
"""

import os
import shutil
import random
import argparse
import kagglehub
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

# Image processing
from PIL import Image
import matplotlib.pyplot as plt

def download_dataset(dataset_dir):
    """
    Download the brain MRI dataset using kagglehub.
    
    Args:
        dataset_dir: Path to store the dataset
    
    Returns:
        Path to the downloaded dataset
    """
    print("Downloading Brain MRI dataset...")
    
    # Download latest version
    path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
    
    print(f"Dataset downloaded to: {path}")
    
    # Create organized directory structure in our project
    organize_dataset(path, dataset_dir)
    
    return path

def organize_dataset(source_path, dataset_dir):
    """
    Organize downloaded dataset into train/val/test splits within our project structure.
    
    Args:
        source_path: Path to the downloaded dataset
        dataset_dir: Path to the project's data directory
    """
    # Create directories
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define paths for yes (tumor) and no (no tumor) images
    yes_dir = os.path.join(source_path, 'yes')
    no_dir = os.path.join(source_path, 'no')
    
    if not os.path.exists(yes_dir) or not os.path.exists(no_dir):
        print(f"Expected directory structure not found in {source_path}")
        print("Looking for alternative structure...")
        
        # Try to find the correct paths
        for root, dirs, files in os.walk(source_path):
            for d in dirs:
                if d.lower() == 'yes':
                    yes_dir = os.path.join(root, d)
                    print(f"Found 'yes' directory at {yes_dir}")
                elif d.lower() == 'no':
                    no_dir = os.path.join(root, d)
                    print(f"Found 'no' directory at {no_dir}")
    
    # Get all image files
    yes_images = [os.path.join(yes_dir, f) for f in os.listdir(yes_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    no_images = [os.path.join(no_dir, f) for f in os.listdir(no_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create dataframe with file paths and labels
    files_df = pd.DataFrame({
        'filepath': yes_images + no_images,
        'label': ['tumor'] * len(yes_images) + ['no_tumor'] * len(no_images)
    })
    
    # Split into train, validation, and test sets (70%, 15%, 15%)
    train_df, temp_df = train_test_split(files_df, test_size=0.3, stratify=files_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    # Create directories for organized dataset
    for split in ['train', 'val', 'test']:
        for label in ['tumor', 'no_tumor']:
            os.makedirs(os.path.join(dataset_dir, split, label), exist_ok=True)
    
    # Copy files to appropriate directories
    def copy_files(df, split):
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split} files"):
            dest_path = os.path.join(dataset_dir, split, row['label'], os.path.basename(row['filepath']))
            shutil.copy2(row['filepath'], dest_path)
    
    copy_files(train_df, 'train')
    copy_files(val_df, 'val')
    copy_files(test_df, 'test')
    
    # Save the splits for reference
    train_df.to_csv(os.path.join(dataset_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(dataset_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_dir, 'test_split.csv'), index=False)
    
    print("Dataset organized successfully!")

def preprocess_images(dataset_dir, output_dir, img_size=(224, 224)):
    """
    Preprocess all images in the dataset.
    
    Args:
        dataset_dir: Path to the organized dataset
        output_dir: Path to store preprocessed images
        img_size: Target image size (width, height)
    """
    print("Preprocessing images...")
    
    # Create directories for preprocessed images
    for split in ['train', 'val', 'test']:
        for label in ['tumor', 'no_tumor']:
            os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        for label in ['tumor', 'no_tumor']:
            src_dir = os.path.join(dataset_dir, split, label)
            dst_dir = os.path.join(output_dir, split, label)
            
            # Get all image files
            image_files = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files, desc=f"Preprocessing {split}/{label}"):
                src_path = os.path.join(src_dir, img_file)
                dst_path = os.path.join(dst_dir, img_file)
                
                # Apply preprocessing
                try:
                    # Read image
                    img = cv2.imread(src_path)
                    
                    # Convert to RGB (OpenCV uses BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize
                    img = cv2.resize(img, img_size)
                    
                    # Normalize pixel values to [0, 1]
                    img = img / 255.0
                    
                    # Apply additional preprocessing
                    img = apply_preprocessing(img, split)
                    
                    # Convert back to uint8 for saving
                    img = (img * 255).astype(np.uint8)
                    
                    # Save preprocessed image
                    cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
    
    print("Preprocessing completed!")
    
    # Create and save metadata
    create_metadata(output_dir)

def apply_preprocessing(img, split):
    """
    Apply specific preprocessing techniques to an image.
    
    Args:
        img: Input image (normalized to [0, 1])
        split: Dataset split ('train', 'val', or 'test')
    
    Returns:
        Preprocessed image
    """
    # Apply histogram equalization to enhance contrast
    # Convert to grayscale first, then back to RGB
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    eq_gray = cv2.equalizeHist(gray)
    
    # Convert back to RGB
    eq_rgb = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2RGB)
    eq_rgb = eq_rgb / 255.0  # Normalize back to [0, 1]
    
    # For training data, optionally apply data augmentation
    if split == 'train':
        # Apply random augmentation with 50% probability
        if random.random() > 0.5:
            # Randomly apply one of several augmentations
            aug_type = random.choice(['flip', 'rotate', 'brightness'])
            
            if aug_type == 'flip':
                # Horizontal flip
                eq_rgb = cv2.flip(eq_rgb, 1)
            elif aug_type == 'rotate':
                # Random rotation between -15 and 15 degrees
                angle = random.uniform(-15, 15)
                h, w = eq_rgb.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                eq_rgb = cv2.warpAffine(eq_rgb, M, (w, h))
            elif aug_type == 'brightness':
                # Adjust brightness
                factor = random.uniform(0.8, 1.2)
                eq_rgb = np.clip(eq_rgb * factor, 0, 1)
    
    return eq_rgb

def create_metadata(output_dir):
    """
    Create metadata file with dataset statistics.
    
    Args:
        output_dir: Path to the preprocessed images
    """
    metadata = {
        'split': [],
        'label': [],
        'count': []
    }
    
    for split in ['train', 'val', 'test']:
        for label in ['tumor', 'no_tumor']:
            dir_path = os.path.join(output_dir, split, label)
            image_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            metadata['split'].append(split)
            metadata['label'].append(label)
            metadata['count'].append(image_count)
    
    # Create dataframe and save to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    print(f"Dataset metadata:\n{metadata_df}")

def visualize_samples(output_dir, samples_per_class=5):
    """
    Visualize sample images from each class after preprocessing.
    
    Args:
        output_dir: Path to the preprocessed images
        samples_per_class: Number of samples to visualize per class
    """
    plt.figure(figsize=(15, 10))
    
    for i, label in enumerate(['tumor', 'no_tumor']):
        images = []
        for split in ['train', 'val', 'test']:
            dir_path = os.path.join(output_dir, split, label)
            image_files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue
                
            # Select random samples
            selected_files = random.sample(image_files, min(samples_per_class, len(image_files)))
            
            for j, img_file in enumerate(selected_files):
                img_path = os.path.join(dir_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(2, samples_per_class, i * samples_per_class + j + 1)
                plt.imshow(img)
                plt.title(f"{label} ({split})")
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    plt.close()
    
    print(f"Sample visualization saved to {os.path.join(output_dir, 'sample_images.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and preprocess the Brain MRI dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store the dataset')
    parser.add_argument('--processed_dir', type=str, default='./data/processed', 
                        help='Directory to store preprocessed images')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (width, height)')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    
    # Download dataset
    download_path = download_dataset(os.path.join(args.data_dir, 'raw'))
    
    # Preprocess images
    preprocess_images(
        os.path.join(args.data_dir, 'raw'),
        args.processed_dir,
        tuple(args.img_size)
    )
    
    # Visualize samples
    visualize_samples(args.processed_dir)
    
    print("Data preparation completed successfully!") 