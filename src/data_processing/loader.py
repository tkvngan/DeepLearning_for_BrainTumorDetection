"""
Data loader module for the Brain Tumor Detection project.
Provides functionality to load and batch preprocessed MRI images.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

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

def get_torch_dataloaders(data_dir, batch_size=32, transforms=None):
    """
    Create PyTorch DataLoaders for all splits.
    
    Args:
        data_dir: Root directory containing processed images
        batch_size: Batch size for the dataloaders
        transforms: Dict of transforms for each split
        
    Returns:
        Dictionary of DataLoaders for train, val, and test splits
    """
    # Set default transforms if not provided
    if transforms is None:
        transforms = {split: None for split in ['train', 'val', 'test']}
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = BrainMRIDataset(
            data_dir=data_dir,
            split=split,
            transform=transforms.get(split)
        )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            datasets['train'], 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        ),
        'val': DataLoader(
            datasets['val'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        ),
        'test': DataLoader(
            datasets['test'], 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
    }
    
    return dataloaders

def get_tf_datasets(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Create TensorFlow datasets for all splits.
    
    Args:
        data_dir: Root directory containing processed images
        batch_size: Batch size for the datasets
        img_size: Target image size (height, width)
        
    Returns:
        Dictionary of TensorFlow datasets for train, val, and test splits
    """
    def _parse_function(img_path, label):
        # Read image file
        img = tf.io.read_file(img_path)
        # Decode to tensor
        img = tf.image.decode_image(img, channels=3)
        # Resize
        img = tf.image.resize(img, img_size)
        # Normalize
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        # Get image paths and labels
        image_paths = []
        labels = []
        
        for label_idx, label in enumerate(['tumor', 'no_tumor']):
            dir_path = os.path.join(data_dir, split, label)
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
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
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
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        datasets[split] = ds
        print(f"Created {split} dataset with {len(image_paths)} images")
    
    return datasets

def get_class_weights(data_dir):
    """
    Calculate class weights to handle class imbalance.
    
    Args:
        data_dir: Root directory containing processed images
        
    Returns:
        Dictionary mapping class indices to weights
    """
    # Read metadata
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print("Metadata file not found. Creating one...")
        create_metadata(data_dir)
    
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

def create_metadata(data_dir):
    """
    Create metadata file with dataset statistics if it doesn't exist.
    
    Args:
        data_dir: Root directory containing processed images
    """
    metadata = {
        'split': [],
        'label': [],
        'count': []
    }
    
    for split in ['train', 'val', 'test']:
        for label in ['tumor', 'no_tumor']:
            dir_path = os.path.join(data_dir, split, label)
            if os.path.exists(dir_path):
                image_count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                
                metadata['split'].append(split)
                metadata['label'].append(label)
                metadata['count'].append(image_count)
    
    # Create dataframe and save to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)
    
    print(f"Dataset metadata created at {os.path.join(data_dir, 'metadata.csv')}") 