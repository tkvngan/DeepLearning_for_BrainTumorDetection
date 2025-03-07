"""
Preprocessor module for the Brain Tumor Detection project.
Contains utilities for image preprocessing and augmentation.
"""

import cv2 # type: ignore
import numpy as np
import random
import tensorflow as tf
import torch
from torchvision import transforms

def preprocess_image(image, target_size=(224, 224), normalize=True, augment=False):
    """
    Preprocess a single image.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        normalize: Whether to normalize the image
        augment: Whether to apply data augmentation
        
    Returns:
        Preprocessed image
    """
    # Resize image
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, target_size)
    
    # Normalize
    if normalize and isinstance(image, np.ndarray):
        image = image / 255.0
    
    # Apply augmentation if requested
    if augment:
        image = apply_augmentation(image)
    
    return image

def apply_augmentation(image):
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image (numpy array normalized to [0, 1])
        
    Returns:
        Augmented image
    """
    # Randomly apply augmentation
    aug_type = random.choice(['flip', 'rotate', 'brightness', 'contrast', 'none'])
    
    if aug_type == 'flip':
        # Horizontal flip
        image = cv2.flip(image, 1)
    elif aug_type == 'rotate':
        # Random rotation between -15 and 15 degrees
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
    elif aug_type == 'brightness':
        # Adjust brightness
        factor = random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 1)
    elif aug_type == 'contrast':
        # Adjust contrast
        factor = random.uniform(0.8, 1.2)
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray * factor
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) / 255.0
    
    return image

def enhance_contrast(image):
    """
    Enhance image contrast using histogram equalization.
    
    Args:
        image: Input image (numpy array normalized to [0, 1])
        
    Returns:
        Enhanced image
    """
    # Convert to grayscale
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization
    eq_gray = cv2.equalizeHist(gray)
    
    # Convert back to RGB
    eq_rgb = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2RGB)
    
    # Normalize
    eq_rgb = eq_rgb / 255.0
    
    return eq_rgb

# Define PyTorch transformation pipelines
def get_pytorch_transforms(img_size=(224, 224)):
    """
    Get PyTorch transformation pipelines for different splits.
    
    Args:
        img_size: Target image size (width, height)
        
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
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation and test transforms (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])
    
    return {
        'train': train_transforms,
        'val': val_test_transforms,
        'test': val_test_transforms
    }

# Define TensorFlow data augmentation
def get_tf_augmentation_layer():
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