"""
Day 4 — YOLO Conversion Validation & Visualization
Visualizes converted YOLO labels by drawing bounding boxes back onto images
to verify the conversion accuracy and consistency.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import random


class YOLOVisualizer:
    """
    Visualizes YOLO format labels by drawing bounding boxes on images.
    """
    
    def __init__(self, processed_dir, class_mapping_path):
        """
        Args:
            processed_dir: Path to data/processed directory
            class_mapping_path: Path to class_mapping.json
        """
        self.processed_dir = processed_dir
        self.images_dir = os.path.join(processed_dir, "images")
        self.labels_dir = os.path.join(processed_dir, "labels")
        
        # Load class info from data.yaml
        yaml_path = os.path.join(processed_dir, "data.yaml")
        self.class_names = self._load_classes_from_yaml(yaml_path)
        self.colors = self._generate_colors(len(self.class_names))
        
        print(f"✓ YOLOVisualizer initialized with {len(self.class_names)} classes")
    
    @staticmethod
    def _load_classes_from_yaml(yaml_path):
        """Load class names from data.yaml."""
        import re
        with open(yaml_path, 'r') as f:
            content = f.read()
            # Extract names list
            match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
            if match:
                names_str = match.group(1)
                # Parse quoted names
                names = re.findall(r"'([^']+)'", names_str)
                return names
        return []
    
    @staticmethod
    def _generate_colors(num_classes):
        """Generate distinct colors for each class."""
        random.seed(42)
        colors = {}
        for i in range(num_classes):
            colors[i] = tuple(random.randint(0, 255) for _ in range(3))
        return colors
    
    def visualize_image(self, img_path, label_path):
        """
        Draw YOLO labels on an image.
        
        Args:
            img_path: Path to image file
            label_path: Path to corresponding .txt label file
            
        Returns:
            Image array with drawn bounding boxes
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ Could not read image: {img_path}")
            return None
        
        h, w = img.shape[:2]
        
        # Read labels
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except:
            print(f"  ⚠ Could not read labels: {label_path}")
            return img
        
        # Draw each bounding box
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])
                
                # Convert from normalized to pixel coordinates
                x1 = int((x_center - box_width / 2) * w)
                y1 = int((y_center - box_height / 2) * h)
                x2 = int((x_center + box_width / 2) * w)
                y2 = int((y_center + box_height / 2) * h)
                
                # Clamp coordinates
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Draw rectangle
                color = self.colors.get(class_id, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label text
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                label_text = f"{class_name} ({class_id})"
                cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            except (ValueError, IndexError) as e:
                print(f"  ⚠ Error parsing label line: {line.strip()} - {e}")
                continue
        
        return img
    
    def sample_and_visualize(self, split='train', num_samples=10, output_dir=None):
        """
        Visualize a sample of images from a given split.
        
        Args:
            split: 'train', 'val', or 'test'
            num_samples: Number of samples to visualize
            output_dir: Directory to save visualizations (if None, no save)
        """
        print(f"\n🖼️  Sampling {num_samples} images from {split} split...")
        
        split_img_dir = os.path.join(self.images_dir, split)
        split_label_dir = os.path.join(self.labels_dir, split)
        
        if not os.path.exists(split_img_dir):
            print(f"  ⚠ Split directory not found: {split_img_dir}")
            return
        
        # Get all images
        img_files = [f for f in os.listdir(split_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample random images
        sample_files = random.sample(img_files, min(num_samples, len(img_files)))
        
        print(f"  Selected {len(sample_files)} images for visualization")
        
        for idx, img_file in enumerate(sample_files):
            img_path = os.path.join(split_img_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            label_path = os.path.join(split_label_dir, f"{base_name}.txt")
            
            # Visualize
            vis_img = self.visualize_image(img_path, label_path)
            
            if vis_img is not None:
                # Save if output_dir provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{split}_{idx:03d}_{base_name}.jpg")
                    cv2.imwrite(output_path, vis_img)
                    print(f"    ✓ Saved {os.path.basename(output_path)}")
    
    def generate_split_report(self):
        """Generate a report of the dataset split statistics."""
        print(f"\n📊 Dataset Split Report")
        print(f"{'Split':<10} {'Images':<10} {'Labels':<10} {'Status':<15}")
        print(f"{'-'*45}")
        
        for split in ['train', 'val', 'test']:
            split_img_dir = os.path.join(self.images_dir, split)
            split_label_dir = os.path.join(self.labels_dir, split)
            
            img_count = len([f for f in os.listdir(split_img_dir) if os.path.isfile(os.path.join(split_img_dir, f))])
            label_count = len([f for f in os.listdir(split_label_dir) if os.path.isfile(os.path.join(split_label_dir, f))])
            
            status = "✓ OK" if img_count == label_count else "⚠ Mismatch"
            print(f"{split:<10} {img_count:<10} {label_count:<10} {status:<15}")


def run_validation(processed_dir, class_mapping_path, output_dir=None):
    """
    Main entry point for YOLO validation.
    """
    print("=" * 70)
    print("🔍 YOLO Conversion Validation & Visualization - Day 4")
    print("=" * 70)
    
    visualizer = YOLOVisualizer(processed_dir, class_mapping_path)
    
    # Generate report
    visualizer.generate_split_report()
    
    # Visualize samples from each split
    viz_output_dir = os.path.join(processed_dir, "visualization_samples")
    for split in ['train', 'val', 'test']:
        visualizer.sample_and_visualize(split=split, num_samples=5, output_dir=viz_output_dir)
    
    print("\n" + "=" * 70)
    print("✅ Validation Complete!")
    if output_dir:
        print(f"📁 Visualizations saved to: {viz_output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(base_dir, "data", "processed")
    class_mapping_path = os.path.join(processed_dir, "class_mapping.json")
    
    run_validation(processed_dir, class_mapping_path)
