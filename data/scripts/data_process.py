import os
import cv2
import numpy as np
from pathlib import Path
import random
import logging
from PIL import Image
import imagehash
from typing import Dict, List
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DuplicateInfo:
    def __init__(self):
        self.duplicates_found = 0
        self.duplicate_groups = defaultdict(list)
        self.kept_images = set()
        self.removed_images = set()
        self.hash_size = 16  # Larger hash size for more precise matching

class DataProcessor:
    def __init__(self, raw_data_path: str, output_path: str, image_size: int = 512):
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        
        if not self.raw_data_path.is_absolute():
            self.raw_data_path = self.project_root / self.raw_data_path
        if not self.output_path.is_absolute():
            self.output_path = self.project_root / self.output_path
            
        self.image_size = image_size
        self.duplicate_info = DuplicateInfo()
        
        self.stats = {
            'processed_images': 0,
            'failed_images': 0,
            'class_distribution': defaultdict(int),
            'split_distribution': defaultdict(int),
            'duplicates_found': 0,
            'duplicates_removed': 0
        }

    def process_image(self, image_path: Path) -> np.ndarray:
        """Process a single image with minimal transformations."""
        # Read image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Calculate padding to maintain aspect ratio
        h, w = image.shape
        scale = min(self.image_size / h, self.image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize maintaining aspect ratio
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        h_offset = (self.image_size - new_h) // 2
        w_offset = (self.image_size - new_w) // 2
        padded[h_offset:h_offset+new_h, w_offset:w_offset+new_w] = resized
        
        # Normalize to [0, 255]
        if padded.max() > padded.min():
            padded = ((padded - padded.min()) * 255 / (padded.max() - padded.min())).astype(np.uint8)
            
        return padded

    def detect_duplicates(self, image_paths: List[Path]) -> Dict[str, List[Path]]:
        """Detect duplicates using conservative matching."""
        hash_dict = {}
        logger.info("Starting duplicate detection...")
        
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'L':
                        img = img.convert('L')
                    img_hash = str(imagehash.whash(img, hash_size=self.duplicate_info.hash_size))
                
                if img_hash in hash_dict:
                    # Only consider as duplicate if extremely similar
                    existing_path = hash_dict[img_hash][0]
                    if self._are_similar_filenames(img_path, existing_path):
                        hash_dict[img_hash].append(img_path)
                        self.duplicate_info.duplicates_found += 1
                        logger.info(f"Found duplicate: {img_path}")
                else:
                    hash_dict[img_hash] = [img_path]
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
        
        return {k: v for k, v in hash_dict.items() if len(v) > 1}

    def _are_similar_filenames(self, path1: Path, path2: Path) -> bool:
        """Check if filenames suggest images are duplicates."""
        name1 = path1.stem.lower()
        name2 = path2.stem.lower()
        
        # Remove common suffixes and numbers
        base1 = ''.join(c for c in name1 if not c.isdigit())
        base2 = ''.join(c for c in name2 if not c.isdigit())
        
        return base1 == base2 or base1.startswith(base2) or base2.startswith(base1)

    def create_splits(self):
        """Create train/val/test splits."""
        train_ratio, val_ratio = 0.7, 0.15  # test = 0.15
        
        for class_dir in self.raw_data_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            # Get all valid images (excluding duplicates)
            images = [p for p in class_dir.glob('*.jpg') 
                     if p not in self.duplicate_info.removed_images]
            
            if not images:
                continue
                
            # Create splits
            random.shuffle(images)
            n = len(images)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            splits = {
                'train': images[:train_idx],
                'val': images[train_idx:val_idx],
                'test': images[val_idx:]
            }
            
            # Process and save each split
            for split_name, split_images in splits.items():
                self._process_split(split_name, split_images, class_dir.name)

    def _process_split(self, split_name: str, image_paths: List[Path], class_name: str):
        """Process and save images for a specific split."""
        output_dir = self.output_path / split_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_paths:
            try:
                processed_image = self.process_image(img_path)
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), processed_image)
                
                self.stats['processed_images'] += 1
                self.stats['split_distribution'][split_name] += 1
                self.stats['class_distribution'][class_name] += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                self.stats['failed_images'] += 1

    def process_dataset(self):
        """Main processing method."""
        try:
            logger.info("Starting dataset processing...")
            
            # Collect all image paths
            all_images = []
            for class_dir in self.raw_data_path.iterdir():
                if class_dir.is_dir():
                    all_images.extend(list(class_dir.glob('*.jpg')))
            
            # Detect and handle duplicates
            duplicate_groups = self.detect_duplicates(all_images)
            
            # Keep only one image from each duplicate group
            for group in duplicate_groups.values():
                best_image = max(group, key=lambda x: x.stat().st_size)
                self.duplicate_info.kept_images.add(best_image)
                self.duplicate_info.removed_images.update(set(group) - {best_image})
            
            # Create directory structure and process images
            self.create_splits()
            self._log_statistics()
            
            logger.info("Dataset processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during dataset processing: {e}")
            raise

    def _log_statistics(self):
        """Log processing statistics."""
        logger.info("\n=== Processing Statistics ===")
        logger.info(f"Total images processed: {self.stats['processed_images']}")
        logger.info(f"Failed images: {self.stats['failed_images']}")
        logger.info(f"Duplicates found: {self.duplicate_info.duplicates_found}")
        logger.info(f"Duplicates removed: {len(self.duplicate_info.removed_images)}")
        
        logger.info("\nClass Distribution:")
        for class_name, count in self.stats['class_distribution'].items():
            logger.info(f"{class_name}: {count}")
        
        logger.info("\nSplit Distribution:")
        for split_name, count in self.stats['split_distribution'].items():
            logger.info(f"{split_name}: {count}")

def main():
    processor = DataProcessor(
        raw_data_path='data/data_raw',
        output_path='data/data_processed',
        image_size=512
    )
    processor.process_dataset()

if __name__ == "__main__":
    main()
