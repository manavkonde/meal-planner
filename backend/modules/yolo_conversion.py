"""
Day 4 — YOLO Annotation Conversion Pipeline
Converts cleaned, taxonomy-mapped datasets into YOLOv8 format:
- Normalizes bounding boxes from pixel coords to YOLO format (0-1 range)
- Handles classification-only datasets (whole-image box fallback)
- Creates 80/10/10 stratified train/val/test split
- Produces data.yaml for YOLOv8 training
"""

import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image
from collections import defaultdict
import pandas as pd


class YOLOConverter:
    """
    Converts datasets to YOLO format with proper split management.
    """

    def __init__(self, base_dir, raw_dir, processed_dir, class_mapping_path):
        """
        Args:
            base_dir: Project root
            raw_dir: data/raw directory
            processed_dir: data/processed directory
            class_mapping_path: Path to class_mapping.json
        """
        self.base_dir = base_dir
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            data = json.load(f)
            self.raw_to_canonical = data.get('mapping', {})
        
        # Extract unique canonical classes (ordered)
        self.canonical_classes = sorted(set(self.raw_to_canonical.values()))
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.canonical_classes)}
        
        # Create output structure
        self.images_dir = os.path.join(processed_dir, "images")
        self.labels_dir = os.path.join(processed_dir, "labels")
        
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_dir, split), exist_ok=True)
        
        # Track conversions per split
        self.split_data = defaultdict(list)  # split -> list of (image_path, label_path, class_id)
        
        print(f"✓ YOLOConverter initialized")
        print(f"  - {len(self.canonical_classes)} canonical classes")
        print(f"  - Output dirs created at {self.images_dir} and {self.labels_dir}")

    @staticmethod
    def convert_bbox_to_yolo(box, img_w, img_h):
        """
        Convert pixel bounding box to YOLO normalized format.
        
        Args:
            box: (x_min, y_min, x_max, y_max) in pixel coordinates
            img_w, img_h: Image dimensions in pixels
            
        Returns:
            (x_center, y_center, width, height) in normalized [0, 1] range
        """
        x_min, y_min, x_max, y_max = box
        x_center = ((x_min + x_max) / 2.0) / img_w
        y_center = ((y_min + y_max) / 2.0) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        return x_center, y_center, width, height

    @staticmethod
    def whole_image_box():
        """Return YOLO format for whole-image bounding box."""
        return 0.5, 0.5, 1.0, 1.0

    def process_fruits360(self):
        """
        Convert fruits360 dataset (classification-only) to YOLO whole-image boxes.
        Structure: Training/{class_name}/{image}.jpg or Test/{class_name}/{image}.jpg
        """
        print("\n📦 Processing Fruits-360 (classification-only)...")
        
        fruits_base = os.path.join(self.raw_dir, "fruits360")
        if not os.path.exists(fruits_base):
            print("  ⚠ Fruits-360 not found, skipping")
            return
        
        count = 0
        skipped = 0
        
        # Walk through Training and Test directories
        for split_folder in ["Training", "Test"]:
            split_path = os.path.join(fruits_base, split_folder)
            if not os.path.exists(split_path):
                continue
            
            for class_folder in os.listdir(split_path):
                class_path = os.path.join(split_path, class_folder)
                if not os.path.isdir(class_path):
                    continue
                
                # Map raw class to canonical class
                canonical_class = self.raw_to_canonical.get(class_folder)
                if canonical_class is None:
                    skipped += 1
                    continue
                
                class_id = self.class_to_id[canonical_class]
                
                # Process each image in the class folder
                for img_file in os.listdir(class_path):
                    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = Image.open(img_path)
                        img_w, img_h = img.size
                    except Exception as e:
                        print(f"  ⚠ Failed to open {img_path}: {e}")
                        skipped += 1
                        continue
                    
                    # Create label with whole-image box
                    x_center, y_center, width, height = self.whole_image_box()
                    label_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
                    
                    # Unique filename based on canonical class
                    base_filename = f"{canonical_class}_{count}"
                    img_ext = os.path.splitext(img_file)[1]
                    
                    self.split_data['fruits360'].append({
                        'src_img': img_path,
                        'src_label_line': label_line,
                        'canonical_class': canonical_class,
                        'base_filename': base_filename,
                        'img_ext': img_ext
                    })
                    
                    count += 1
        
        print(f"  ✓ Fruits-360: {count} images prepared ({skipped} skipped)")

    def process_grocery(self):
        """
        Convert Grocery Store Dataset (classification-only) to YOLO whole-image boxes.
        Structure: {train,test,val}/{category}/{subcategory}/*.jpg
        """
        print("\n📦 Processing Grocery Store Dataset (classification-only)...")
        
        grocery_base = os.path.join(self.raw_dir, "grocery", "dataset")
        if not os.path.exists(grocery_base):
            print("  ⚠ Grocery dataset not found, skipping")
            return
        
        count = 0
        skipped = 0
        
        # Walk through all images
        for root, dirs, files in os.walk(grocery_base):
            for img_file in files:
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(root, img_file)
                
                # Extract label: use immediate parent folder name
                raw_class = os.path.basename(root)
                canonical_class = self.raw_to_canonical.get(raw_class)
                
                if canonical_class is None:
                    skipped += 1
                    continue
                
                class_id = self.class_to_id[canonical_class]
                
                try:
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                except Exception as e:
                    print(f"  ⚠ Failed to open {img_path}: {e}")
                    skipped += 1
                    continue
                
                # Create label with whole-image box
                x_center, y_center, width, height = self.whole_image_box()
                label_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
                
                # Unique filename
                base_filename = f"{canonical_class}_{count}"
                img_ext = os.path.splitext(img_file)[1]
                
                self.split_data['grocery'].append({
                    'src_img': img_path,
                    'src_label_line': label_line,
                    'canonical_class': canonical_class,
                    'base_filename': base_filename,
                    'img_ext': img_ext
                })
                
                count += 1
        
        print(f"  ✓ Grocery: {count} images prepared ({skipped} skipped)")

    def perform_stratified_split(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Perform stratified train/val/test split per canonical class.
        
        Args:
            train_ratio: Proportion for training (default 0.8)
            val_ratio: Proportion for validation (default 0.1)
            seed: Random seed for reproducibility
        """
        print("\n🔀 Performing stratified 80/10/10 split...")
        
        random.seed(seed)
        
        # Group all data by canonical class
        class_data = defaultdict(list)
        for dataset_name, items in self.split_data.items():
            for item in items:
                cls = item['canonical_class']
                class_data[cls].append(item)
        
        # Split each class independently
        splits = {'train': [], 'val': [], 'test': []}
        
        for cls, items in class_data.items():
            random.shuffle(items)
            n = len(items)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits['train'].extend(items[:train_end])
            splits['val'].extend(items[train_end:val_end])
            splits['test'].extend(items[val_end:])
        
        print(f"  ✓ Split complete:")
        for split_name, items in splits.items():
            print(f"    - {split_name}: {len(items)} images")
        
        return splits

    def write_dataset(self, splits):
        """
        Write images and labels to the output directory structure.
        
        Args:
            splits: Dict mapping split name to list of items
        """
        print("\n📝 Writing images and labels...")
        
        total_written = 0
        
        for split_name, items in splits.items():
            for idx, item in enumerate(items):
                src_img = item['src_img']
                label_line = item['src_label_line']
                base_filename = item['base_filename']
                img_ext = item['img_ext']
                
                # Create output filenames
                img_filename = f"{base_filename}{img_ext}"
                label_filename = f"{base_filename}.txt"
                
                dst_img = os.path.join(self.images_dir, split_name, img_filename)
                dst_label = os.path.join(self.labels_dir, split_name, label_filename)
                
                try:
                    # Copy image
                    shutil.copy2(src_img, dst_img)
                    
                    # Write label
                    with open(dst_label, 'w') as f:
                        f.write(label_line)
                    
                    total_written += 1
                except Exception as e:
                    print(f"  ⚠ Failed to write {img_filename}: {e}")
        
        print(f"  ✓ Wrote {total_written} image-label pairs")

    def create_data_yaml(self, yaml_path):
        """
        Create data.yaml for YOLOv8.
        
        Args:
            yaml_path: Path to write data.yaml
        """
        print("\n📄 Creating data.yaml...")
        
        yaml_content = f"""# YOLOv8 Dataset Config
# Generated for AI Meal Planner Project

path: {self.processed_dir}  # dataset root
train: images/train
val: images/val
test: images/test

nc: {len(self.canonical_classes)}  # number of classes
names: {self.canonical_classes}  # class names
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"  ✓ data.yaml created at {yaml_path}")
        print(f"    - Classes: {len(self.canonical_classes)}")
        print(f"    - Class names: {', '.join(self.canonical_classes[:5])}...")

    def validate_dataset(self):
        """
        Validate that all images have corresponding labels and vice versa.
        """
        print("\n✅ Validating dataset structure...")
        
        errors = []
        
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(self.images_dir, split)
            label_dir = os.path.join(self.labels_dir, split)
            
            img_files = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)))
            label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f)))
            
            # Check for orphaned files
            orphaned_imgs = img_files - label_files
            orphaned_labels = label_files - img_files
            
            if orphaned_imgs:
                errors.append(f"  ⚠ {split}: {len(orphaned_imgs)} images without labels")
            if orphaned_labels:
                errors.append(f"  ⚠ {split}: {len(orphaned_labels)} labels without images")
            
            if not orphaned_imgs and not orphaned_labels:
                print(f"  ✓ {split}: {len(img_files)} images/labels (consistent)")
        
        if errors:
            print("\n⚠ Validation errors found:")
            for error in errors:
                print(error)
            return False
        else:
            print("\n✓ All validation checks passed!")
            return True


def run_yolo_conversion(base_dir, raw_dir, processed_dir, class_mapping_path):
    """
    Main entry point for YOLO conversion pipeline.
    """
    print("=" * 70)
    print("🚀 YOLO Conversion Pipeline - Day 4")
    print("=" * 70)
    
    # Initialize converter
    converter = YOLOConverter(base_dir, raw_dir, processed_dir, class_mapping_path)
    
    # Process datasets
    converter.process_fruits360()
    converter.process_grocery()
    
    # Perform split
    splits = converter.perform_stratified_split()
    
    # Write dataset
    converter.write_dataset(splits)
    
    # Create config
    yaml_path = os.path.join(processed_dir, "data.yaml")
    converter.create_data_yaml(yaml_path)
    
    # Validate
    converter.validate_dataset()
    
    print("\n" + "=" * 70)
    print("✅ YOLO Conversion Complete!")
    print("=" * 70)
    print(f"\n📦 Output structure:")
    print(f"   {converter.images_dir}/")
    print(f"   {converter.labels_dir}/")
    print(f"   {yaml_path}")


if __name__ == "__main__":
    # Navigate from backend/modules back to project root
    # __file__ -> backend/modules/yolo_conversion.py
    # dirname(__file__) -> backend/modules
    # dirname(dirname(__file__)) -> backend
    # dirname(dirname(dirname(__file__))) -> meal-planner (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    class_mapping_path = os.path.join(processed_dir, "class_mapping.json")
    
    run_yolo_conversion(base_dir, raw_dir, processed_dir, class_mapping_path)
