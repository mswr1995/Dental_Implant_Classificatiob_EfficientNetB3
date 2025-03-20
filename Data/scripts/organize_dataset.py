import os
import shutil
import yaml
import glob
import pandas as pd
from pathlib import Path
import random
from collections import Counter
import cv2
import numpy as np
from tqdm import tqdm

# Update source paths
source_paths = [
    '../../data/data_collected/IMPLANT SYSTEM DETECTION.v7i.yolov5pytorch',
    '../../data/data_collected/implants.v2i.yolov5pytorch',
    '../../data/data_collected/Dental Implants 2.0.v3i.yolov5pytorch',
    '../../data/data_collected/The dental implant brands recognition system.v6i.yolov5pytorch (1)',
    '../../data/data_collected/Implant Doctor.v1i.yolov5pytorch',
    '../../data/data_collected/dental.v5i.yolov5pytorch',
]

output_dir = '../../data/data_raw'


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_yaml_config(yaml_path):
    """Load YAML configuration file"""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {yaml_path}: {e}")
        return None


def extract_implant_brand_from_yolo_annotation(img_path, txt_path, class_names):
    """
    Extract implant brand from YOLO annotation.
    For simplicity, we'll use the class of the first detected implant in the image.
    """
    if not os.path.exists(txt_path):
        return None
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return None
            
        # Parse first annotation (class_id x_center y_center width height)
        parts = lines[0].strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            if class_id < len(class_names):
                return class_names[class_id]
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
    
    return None


def process_dataset(source_path, output_dir):
    """Process a single YOLOv5 dataset"""
    # Find dataset YAML file (data.yaml)
    yaml_files = glob.glob(os.path.join(source_path, "*.yaml"))
    data_yaml = next((f for f in yaml_files if "data.yaml" in f), None)
    
    if not data_yaml:
        print(f"No data.yaml found in {source_path}")
        return []
    
    # Load dataset configuration
    config = load_yaml_config(data_yaml)
    if not config or 'names' not in config:
        print(f"Invalid or missing class names in {data_yaml}")
        return []
    
    class_names = config['names']
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Find train, val, test directories
    data_processed = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(source_path, split)
        if not os.path.exists(split_dir):
            continue
            
        # Find images and labels
        images_dir = os.path.join(split_dir, 'images')
        labels_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Missing images or labels directory in {split_dir}")
            continue
            
        image_files = glob.glob(os.path.join(images_dir, "**/*.*"), recursive=True)
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images in {split}/{split_dir}")
        
        for img_path in image_files:
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            
            # Find corresponding label file
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            
            # Extract brand from annotation
            brand = extract_implant_brand_from_yolo_annotation(img_path, label_path, class_names)
            
            if brand:
                data_processed.append({
                    'source_img_path': img_path,
                    'brand': brand,
                    'split': split,
                    'dataset': os.path.basename(source_path)
                })
    
    return data_processed


def copy_images_to_output(data_processed, output_dir):
    """Copy images to output directory structure"""
    ensure_directory_exists(output_dir)
    
    # Create directories for each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        ensure_directory_exists(split_dir)
    
    # Track copied files to avoid duplicates
    copied_files = set()
    
    # Copy images
    for item in tqdm(data_processed, desc="Copying images"):
        source_path = item['source_img_path']
        brand = item['brand']
        split = item['split']
        
        # Create brand directory if it doesn't exist
        brand_dir = os.path.join(output_dir, split, brand)
        ensure_directory_exists(brand_dir)
        
        # Create unique filename to avoid collisions
        img_filename = os.path.basename(source_path)
        base_name, ext = os.path.splitext(img_filename)
        new_filename = f"{base_name}_{item['dataset']}{ext}"
        
        # Avoid duplicate filenames
        if new_filename in copied_files:
            new_filename = f"{base_name}_{item['dataset']}_{random.randint(1000, 9999)}{ext}"
        
        dest_path = os.path.join(brand_dir, new_filename)
        
        # Copy file
        try:
            shutil.copy2(source_path, dest_path)
            copied_files.add(new_filename)
        except Exception as e:
            print(f"Error copying {source_path} to {dest_path}: {e}")


def main():
    """Main function to process all datasets"""
    # Process each dataset
    all_data = []
    for source_path in source_paths:
        if os.path.exists(source_path):
            print(f"\nProcessing dataset: {source_path}")
            dataset_data = process_dataset(source_path, output_dir)
            all_data.extend(dataset_data)
            print(f"Extracted {len(dataset_data)} valid implant images")
        else:
            print(f"Source path does not exist: {source_path}")
    
    # Print statistics
    print(f"\nTotal images found: {len(all_data)}")
    
    # Count by brand
    brand_counts = Counter([item['brand'] for item in all_data])
    print("\nImages per brand:")
    for brand, count in brand_counts.most_common():
        print(f"  {brand}: {count}")
    
    # Count by split
    split_counts = Counter([item['split'] for item in all_data])
    print("\nImages per split:")
    for split, count in split_counts.items():
        print(f"  {split}: {count}")
    
    # Copy images to output directory structure
    print("\nOrganizing dataset...")
    copy_images_to_output(all_data, output_dir)
    
    # Save dataset summary
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, 'dataset_summary.csv'), index=False)
    print(f"\nDataset summary saved to {os.path.join(output_dir, 'dataset_summary.csv')}")
    
    print("\nDataset organization complete!")


if __name__ == "__main__":
    main()