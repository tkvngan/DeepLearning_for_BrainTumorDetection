#!/usr/bin/env python3
"""
Brain MRI Preprocessor for Brain Tumor Detection project.
Pre-processing steps for the brain MRI dataset.
"""

import os
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2 # type: ignore
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

class BrainMRIPreprocessor:
    """
    Comprehensive class for preprocessing Brain MRI images.
    Handles downloading, organizing, preprocessing, and data loading.
    """
    
    def __init__(self, data_dir='./data', processed_dir='./data/processed', img_size=(224, 224), batch_size=32):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory to store the raw dataset
            processed_dir: Directory to store preprocessed images
            img_size: Target image size (width, height)
            batch_size: Batch size for data loaders
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.raw_dir = os.path.join(data_dir, 'raw')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Class attributes
        self.class_names = ['tumor', 'no_tumor']
        self.metadata = None
    
    def download_dataset(self):
        """
        Download the brain MRI dataset using Kaggle API.
        
        Returns:
            Path to the downloaded dataset
        """
        print("Downloading Brain MRI dataset...")
        
        # Try to download the dataset using Kaggle API
        try:
            # Import Kaggle API - using the correct import path
            from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
            
            # Initialize and authenticate
            api = KaggleApi()
            api.authenticate()
            
            print("Authenticated with Kaggle API")
            print(f"Downloading to {self.raw_dir}...")
            
            # Download the dataset
            api.dataset_download_files(
                dataset="navoneel/brain-mri-images-for-brain-tumor-detection",
                path=self.raw_dir,
                unzip=True
            )
            
            print("Dataset downloaded successfully")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nManual download instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
            print("2. Click the 'Download' button")
            print(f"3. Extract the downloaded zip file to: {self.raw_dir}")
            print("4. Ensure the directory structure is:")
            print(f"   {self.raw_dir}/yes/ - contains tumor images")
            print(f"   {self.raw_dir}/no/ - contains non-tumor images")
            raise
        
        # Verify the download
        yes_dir = os.path.join(self.raw_dir, 'yes')
        no_dir = os.path.join(self.raw_dir, 'no')
        
        # Check if we need to look deeper for the data directories
        if not (os.path.exists(yes_dir) and os.path.exists(no_dir)):
            print("Looking for dataset in subdirectories...")
            # Look for yes/no directories in subdirectories
            for root, dirs, files in os.walk(self.raw_dir):
                for d in dirs:
                    if d.lower() == 'yes':
                        yes_dir = os.path.join(root, d)
                        print(f"Found 'yes' directory at {yes_dir}")
                    elif d.lower() == 'no':
                        no_dir = os.path.join(root, d)
                        print(f"Found 'no' directory at {no_dir}")
        
        if not (os.path.exists(yes_dir) and os.path.exists(no_dir)):
            raise FileNotFoundError("Could not find 'yes' and 'no' directories in the downloaded dataset.")
        
        return self.raw_dir
    
    def organize_dataset(self):
        """
        Organize downloaded dataset into train/val/test splits.
        """
        print("Organizing dataset...")
        
        # Define paths for yes (tumor) and no (no tumor) images
        yes_dir = os.path.join(self.raw_dir, 'yes')
        no_dir = os.path.join(self.raw_dir, 'no')
        
        if not (os.path.exists(yes_dir) and os.path.exists(no_dir)):
            print("Looking for dataset in subdirectories...")
            # Look for yes/no directories in subdirectories
            for root, dirs, files in os.walk(self.raw_dir):
                for d in dirs:
                    if d.lower() == 'yes':
                        yes_dir = os.path.join(root, d)
                        print(f"Found 'yes' directory at {yes_dir}")
                    elif d.lower() == 'no':
                        no_dir = os.path.join(root, d)
                        print(f"Found 'no' directory at {no_dir}")
        
        if not (os.path.exists(yes_dir) and os.path.exists(no_dir)):
            raise FileNotFoundError("Could not find 'yes' and 'no' directories in the downloaded dataset.")
        
        # Get all image files
        yes_images = [os.path.join(yes_dir, f) for f in os.listdir(yes_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        no_images = [os.path.join(no_dir, f) for f in os.listdir(no_dir) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(yes_images)} tumor images and {len(no_images)} non-tumor images")
        
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
            for label in self.class_names:
                os.makedirs(os.path.join(self.raw_dir, split, label), exist_ok=True)
        
        # Copy files to appropriate directories
        def copy_files(df, split):
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split} files"):
                dest_path = os.path.join(self.raw_dir, split, row['label'], os.path.basename(row['filepath']))
                shutil.copy2(row['filepath'], dest_path)
        
        copy_files(train_df, 'train')
        copy_files(val_df, 'val')
        copy_files(test_df, 'test')
        
        # Save the splits for reference
        train_df.to_csv(os.path.join(self.raw_dir, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(self.raw_dir, 'val_split.csv'), index=False)
        test_df.to_csv(os.path.join(self.raw_dir, 'test_split.csv'), index=False)
        
        print("Dataset organized successfully!")
    
    def preprocess_images(self):
        """
        Preprocess all images in the dataset.
        """
        print("Preprocessing images...")
        
        # Create directories for preprocessed images
        for split in ['train', 'val', 'test']:
            for label in self.class_names:
                os.makedirs(os.path.join(self.processed_dir, split, label), exist_ok=True)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            for label in self.class_names:
                src_dir = os.path.join(self.raw_dir, split, label)
                dst_dir = os.path.join(self.processed_dir, split, label)
                
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
                        img = cv2.resize(img, self.img_size)
                        
                        # Normalize pixel values to [0, 1]
                        img = img / 255.0
                        
                        # Apply additional preprocessing
                        img = self._apply_preprocessing(img, split)
                        
                        # Convert back to uint8 for saving
                        img = (img * 255).astype(np.uint8)
                        
                        # Save preprocessed image
                        cv2.imwrite(dst_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        
                    except Exception as e:
                        print(f"Error processing {src_path}: {e}")
        
        print("Preprocessing completed!")
        
        # Create and save metadata
        self.create_metadata()
    
    def _apply_preprocessing(self, img, split):
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
                aug_type = random.choice(['flip', 'rotate', 'brightness', 'contrast'])
                
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
                elif aug_type == 'contrast':
                    # Adjust contrast
                    factor = random.uniform(0.8, 1.2)
                    gray = cv2.cvtColor((eq_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    gray = gray * factor
                    gray = np.clip(gray, 0, 255).astype(np.uint8)
                    eq_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) / 255.0
        
        return eq_rgb
    
    def create_metadata(self):
        """
        Create metadata file with dataset statistics.
        """
        metadata = {
            'split': [],
            'label': [],
            'count': []
        }
        
        for split in ['train', 'val', 'test']:
            for label in self.class_names:
                dir_path = os.path.join(self.processed_dir, split, label)
                image_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                
                metadata['split'].append(split)
                metadata['label'].append(label)
                metadata['count'].append(image_count)
        
        # Create dataframe and save to CSV
        self.metadata = pd.DataFrame(metadata)
        self.metadata.to_csv(os.path.join(self.processed_dir, 'metadata.csv'), index=False)
        
        print(f"Dataset metadata:\n{self.metadata}")
    
    def visualize_samples(self, samples_per_class=5):
        """
        Visualize sample images from each class after preprocessing.
        
        Args:
            samples_per_class: Number of samples to visualize per class
        """
        plt.figure(figsize=(15, 10))
        
        for i, label in enumerate(self.class_names):
            images = []
            for split in ['train', 'val', 'test']:
                dir_path = os.path.join(self.processed_dir, split, label)
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
        plt.savefig(os.path.join(self.processed_dir, 'sample_images.png'))
        plt.close()
        
        print(f"Sample visualization saved to {os.path.join(self.processed_dir, 'sample_images.png')}")
    
    def get_torch_dataloaders(self, transforms=None):
        """
        Create PyTorch DataLoaders for all splits.
        
        Args:
            transforms: Dict of transforms for each split
            
        Returns:
            Dictionary of DataLoaders for train, val, and test splits
        """
        # Set default transforms if not provided
        if transforms is None:
            transforms = self.get_pytorch_transforms()
        
        # Create datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            datasets[split] = BrainMRIDataset(
                data_dir=self.processed_dir,
                split=split,
                transform=transforms.get(split)
            )
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                datasets['train'], 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=4
            ),
            'val': DataLoader(
                datasets['val'], 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4
            ),
            'test': DataLoader(
                datasets['test'], 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4
            )
        }
        
        return dataloaders
    
    def get_tf_datasets(self):
        """
        Create TensorFlow datasets for all splits.
        
        Returns:
            Dictionary of TensorFlow datasets for train, val, and test splits
        """
        def _parse_function(img_path, label):
            # Read image file
            img = tf.io.read_file(img_path)
            # Decode to tensor
            img = tf.image.decode_image(img, channels=3)
            # Resize
            img = tf.image.resize(img, self.img_size)
            # Normalize
            img = tf.cast(img, tf.float32) / 255.0
            return img, label
        
        # Create datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            # Get image paths and labels
            image_paths = []
            labels = []
            
            for label_idx, label in enumerate(self.class_names):
                dir_path = os.path.join(self.processed_dir, split, label)
                if not os.path.exists(dir_path):
                    continue
                    
                for img_file in os.listdir(dir_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(dir_path, img_file))
                        labels.append(label_idx)
            
            # Create TensorFlow dataset
            ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
            ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            
            if split == 'train':
                # Shuffle and batch training data
                ds = ds.shuffle(buffer_size=len(image_paths))
                ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
                
                # Apply data augmentation to training data
                ds = ds.map(
                    lambda x, y: (
                        tf.image.random_flip_left_right(x),
                        y
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            else:
                # Just batch validation and test data
                ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            
            datasets[split] = ds
            print(f"Created {split} dataset with {len(image_paths)} images")
        
        return datasets
    
    def get_pytorch_transforms(self):
        """
        Get PyTorch transformation pipelines for different splits.
        
        Returns:
            Dictionary of transforms for train, val, and test splits
        """
        # Common transformations
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
        
        # Validation and test transforms (no augmentation)
        val_test_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            normalize
        ])
        
        return {
            'train': train_transforms,
            'val': val_test_transforms,
            'test': val_test_transforms
        }
    
    def get_tf_augmentation_layer(self):
        """
        Get a TensorFlow data augmentation layer for training.
        
        Returns:
            TensorFlow Sequential model for augmentation
        """
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2)
        ])
    
    def get_class_weights(self):
        """
        Calculate class weights to handle class imbalance.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        # Read metadata
        metadata_path = os.path.join(self.processed_dir, 'metadata.csv')
        if not os.path.exists(metadata_path):
            print("Metadata file not found. Creating one...")
            self.create_metadata()
        
        metadata = pd.read_csv(metadata_path)
        
        # Filter for training set
        train_data = metadata[metadata['split'] == 'train']
        
        # Get class counts
        class_counts = train_data.groupby('label')['count'].sum().to_dict()
        
        # Calculate weights (inversely proportional to class frequency)
        total_samples = sum(class_counts.values())
        class_weights = {}
        
        for label, count in class_counts.items():
            # Map string labels to indices
            class_idx = 1 if label == 'tumor' else 0
            class_weights[class_idx] = total_samples / (len(class_counts) * count)
        
        return class_weights
    
    def run_pipeline(self):
        """
        Run the complete preprocessing pipeline.
        """
        print("Starting data preprocessing pipeline...")
        
        # Step 1: Download dataset
        self.download_dataset()
        
        # Step 2: Organize dataset
        self.organize_dataset()
        
        # Step 3: Preprocess images
        self.preprocess_images()
        
        # Step 4: Visualize samples
        self.visualize_samples()
        
        print("Data preprocessing pipeline completed successfully!")
        print(f"Processed data is available at: {self.processed_dir}")


class BrainMRIDataset(Dataset):
    """PyTorch Dataset for Brain MRI images."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing processed images
            split: Data split ('train', 'val', or 'test')
            transform: PyTorch transforms to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for label in ['tumor', 'no_tumor']:
            dir_path = os.path.join(data_dir, split, label)
            if not os.path.exists(dir_path):
                continue
                
            for img_file in os.listdir(dir_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(dir_path, img_file))
                    self.labels.append(1 if label == 'tumor' else 0)
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default conversion to tensor
            image = np.array(image) / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
        
        # Get label
        label = self.labels[idx]
        
        return image, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the complete data preprocessing pipeline')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to the data directory')
    parser.add_argument('--processed_dir', type=str, default='./data/processed',
                       help='Path to store preprocessed images')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (width, height)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loaders')
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = BrainMRIPreprocessor(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size
    )
    
    # Run pipeline
    preprocessor.run_pipeline() 