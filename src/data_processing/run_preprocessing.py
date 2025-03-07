#!/usr/bin/env python3
"""
Script to run the complete data preprocessing pipeline.
"""

import os
import sys
import argparse

# Add the project root to Python path to import our modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

from src.data_processing.data_setup import download_dataset, organize_dataset, preprocess_images, visualize_samples

def main():
    parser = argparse.ArgumentParser(description='Run the complete data preprocessing pipeline')
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data'),
                       help='Path to the data directory')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (width, height)')
    args = parser.parse_args()
    
    # Create directories
    raw_dir = os.path.join(args.data_dir, 'raw')
    processed_dir = os.path.join(args.data_dir, 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Run pipeline
    print("Starting data preprocessing pipeline...")
    
    # Step 1: Download dataset
    download_path = download_dataset(raw_dir)
    
    # Step 2: Organize dataset
    organize_dataset(download_path, raw_dir)
    
    # Step 3: Preprocess images
    preprocess_images(
        raw_dir,
        processed_dir,
        tuple(args.img_size)
    )
    
    # Step 4: Visualize samples
    visualize_samples(processed_dir)
    
    print("Data preprocessing pipeline completed successfully!")
    print(f"Processed data is available at: {processed_dir}")

if __name__ == "__main__":
    main() 