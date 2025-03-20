import os
import shutil
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCombiner:
    def __init__(self, collected_data_path, output_path):
        # Get the project root directory (2 levels up from the script)
        self.project_root = Path(__file__).parent.parent.parent
        
        # Convert relative paths to absolute paths
        self.collected_data_path = self.project_root / collected_data_path
        self.output_path = self.project_root / output_path
        
        logger.info(f"Data collection path: {self.collected_data_path}")
        logger.info(f"Output path: {self.output_path}")
        
        self.valid_classes = {'Bego', 'Bicon', 'ITI', 'ADIN', 'DIONAVI', 
                            'Dentium', 'MIS', 'NORIS', 'nobel', 'osstem'}
        
        # Class name mappings for normalization
        self.class_mappings = {
            'nobel': 'nobel',
            'Nobel': 'nobel',
            'NOBEL': 'nobel',
            'osstem': 'osstem',
            'Osstem': 'osstem',
            'OSSTEM': 'osstem',
            # Add more mappings as needed
        }

    def normalize_class_name(self, name):
        """Normalize class names to match valid_classes format."""
        if not name:
            return None
        
        # Remove any leading/trailing whitespace and quotes
        name = name.strip().strip('"\'')
        
        # Check direct mapping
        if name in self.class_mappings:
            return self.class_mappings[name]
        
        # Try capitalizing first letter
        capitalized = name.capitalize()
        if capitalized in self.valid_classes:
            return capitalized
            
        # Try upper case
        if name.upper() in self.valid_classes:
            return name.upper()
            
        # Try lower case
        if name.lower() in self.valid_classes:
            return name.lower()
            
        return None

    def get_class_from_filename(self, filename):
        """Extract class name from filename."""
        # Split by common separators and get the first part
        for separator in ['-', '_', ' ']:
            parts = filename.split(separator)
            if parts:
                normalized = self.normalize_class_name(parts[0])
                if normalized:
                    return normalized
        return None

    def read_yaml_config(self, yaml_path):
        """Read YAML configuration file."""
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def setup_output_directories(self):
        """Create output directories for each class."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        for class_name in self.valid_classes:
            (self.output_path / class_name).mkdir(parents=True, exist_ok=True)

    def process_dataset(self, dataset_path):
        """Process a single dataset directory."""
        yaml_file = next(dataset_path.glob('*.yaml'), None)
        if not yaml_file:
            logger.warning(f"No YAML file found in {dataset_path}")
            return

        config = self.read_yaml_config(yaml_file)
        class_names = config.get('names', [])
        
        # Create mapping from numeric indices to class names
        class_mapping = {}
        for idx, name in enumerate(class_names):
            normalized = self.normalize_class_name(name)
            if normalized:
                class_mapping[str(idx)] = normalized

        # Process train/val/test directories
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split / 'images'
            if not split_dir.exists():
                continue

            labels_dir = dataset_path / split / 'labels'
            if not labels_dir.exists():
                continue

            self.copy_images_with_labels(split_dir, labels_dir, class_mapping)

    def copy_images_with_labels(self, images_dir, labels_dir, class_mapping):
        """Copy images that have corresponding labels to appropriate class folders."""
        processed = 0
        skipped = 0
        
        for image_file in images_dir.glob('*.*'):
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            label_file = labels_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                continue

            # Try to get class from filename first
            class_name = self.get_class_from_filename(image_file.stem)
            
            # If no class from filename, try to get from label file
            if not class_name and label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            class_idx = first_line.split()[0]
                            class_name = class_mapping.get(class_idx)
                except Exception as e:
                    logger.warning(f"Error reading label file {label_file}: {e}")

            if class_name and class_name in self.valid_classes:
                dest_path = self.output_path / class_name / image_file.name
                shutil.copy2(image_file, dest_path)
                processed += 1
                logger.debug(f"Copied {image_file} to {dest_path}")
            else:
                skipped += 1
                logger.debug(f"Skipped {image_file} - no valid class found")

        logger.info(f"Processed {processed} images, skipped {skipped} images in {images_dir}")

    def combine_datasets(self):
        """Main method to combine all datasets."""
        logger.info("Starting dataset combination process...")
        self.setup_output_directories()

        # Process each dataset directory
        for dataset_dir in self.collected_data_path.iterdir():
            if dataset_dir.is_dir():  # Remove the yolov5pytorch check
                logger.info(f"Processing dataset: {dataset_dir}")
                self.process_dataset(dataset_dir)
            
        # Log summary of processed datasets
        logger.info("Processed datasets:")
        for dataset_dir in self.collected_data_path.iterdir():
            if dataset_dir.is_dir():
                logger.info(f"- {dataset_dir.name}")

        logger.info("Dataset combination completed!")

def main():
    combiner = DatasetCombiner(
        collected_data_path='data/data_collected',
        output_path='data/data_raw'
    )
    combiner.combine_datasets()

if __name__ == "__main__":
    main()
