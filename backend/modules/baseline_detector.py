"""
Day 5 — Baseline Detection Test with Pre-trained YOLOv8
Runs COCO-pretrained YOLOv8 on real fridge photos to establish baseline
performance and identify gaps that justify fine-tuning in Week 2.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO


class BaselineDetector:
    """
    Runs pre-trained YOLOv8 on test fridge photos and logs results.
    """
    
    def __init__(self, base_dir, class_mapping_path, confidence_threshold=0.25):
        """
        Args:
            base_dir: Project root
            class_mapping_path: Path to class_mapping.json (Day 3 output)
            confidence_threshold: Confidence threshold for detections
        """
        self.base_dir = base_dir
        self.test_fridges_dir = os.path.join(base_dir, 'data', 'raw', 'test_fridges')
        self.results_dir = os.path.join(base_dir, 'results', 'day5_baseline')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load unified taxonomy
        with open(class_mapping_path, 'r') as f:
            data = json.load(f)
            self.unified_classes = sorted(set(data.get('mapping', {}).values()))
        
        # Load pre-trained model
        print(f"Loading pre-trained YOLOv8 model (COCO weights)...")
        self.model = YOLO('yolov8n.pt')
        self.coco_classes = self.model.names
        
        self.confidence_threshold = confidence_threshold
        self.results = []
        
        print(f"✓ BaselineDetector initialized")
        print(f"  - COCO classes: {len(self.coco_classes)}")
        print(f"  - Unified classes: {len(self.unified_classes)}")
        print(f"  - Test fridges: {self.test_fridges_dir}")
    
    def get_test_images(self):
        """Get all test images from test_fridges directory."""
        if not os.path.exists(self.test_fridges_dir):
            print(f"⚠ Test fridges directory not found: {self.test_fridges_dir}")
            return []
        
        image_files = []
        for f in os.listdir(self.test_fridges_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(self.test_fridges_dir, f))
        
        return sorted(image_files)
    
    def run_inference(self):
        """Run inference on all test images."""
        print(f"\n🔍 Running baseline inference on test fridges...")
        
        test_images = self.get_test_images()
        if not test_images:
            print("⚠ No test images found")
            return []
        
        print(f"Found {len(test_images)} test images")
        
        for idx, img_path in enumerate(test_images):
            print(f"  [{idx+1}/{len(test_images)}] Processing {os.path.basename(img_path)}...")
            
            # Run inference
            results = self.model.predict(
                source=img_path,
                save=False,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            if results and len(results) > 0:
                r = results[0]
                detections = self._parse_detections(r, img_path)
                self.results.append(detections)
        
        print(f"✓ Inference complete: {len(self.results)} images processed")
        return self.results
    
    def _parse_detections(self, result, img_path):
        """Parse YOLOv8 result object."""
        img_name = os.path.basename(img_path)
        
        detections = {
            'image': img_name,
            'path': img_path,
            'timestamp': datetime.now().isoformat(),
            'detections': []
        }
        
        # Extract boxes and classes
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                class_name = self.coco_classes.get(class_id, f"Class {class_id}")
                
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': xyxy.tolist()  # [x1, y1, x2, y2]
                }
                
                detections['detections'].append(detection)
        
        return detections
    
    def save_annotated_images(self):
        """Save images with drawn detections."""
        print(f"\n💾 Saving annotated images...")
        
        annotated_dir = os.path.join(self.results_dir, 'annotated_images')
        os.makedirs(annotated_dir, exist_ok=True)
        
        for result_data in self.results:
            img_path = result_data['path']
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"  ⚠ Failed to read {img_path}")
                continue
            
            # Draw detections
            for detection in result_data['detections']:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Draw box (green for correct COCO detection)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save annotated image
            output_path = os.path.join(annotated_dir, result_data['image'])
            cv2.imwrite(output_path, img)
        
        print(f"✓ Saved {len(self.results)} annotated images to {annotated_dir}")
        return annotated_dir
    
    def generate_results_csv(self):
        """Save detection results as CSV for analysis."""
        print(f"\n📊 Generating results CSV...")
        
        rows = []
        for result_data in self.results:
            img_name = result_data['image']
            
            if not result_data['detections']:
                rows.append({
                    'image': img_name,
                    'detection_count': 0,
                    'classes_detected': '',
                    'confidences': ''
                })
            else:
                classes = [d['class_name'] for d in result_data['detections']]
                confidences = [f"{d['confidence']:.3f}" for d in result_data['detections']]
                
                rows.append({
                    'image': img_name,
                    'detection_count': len(result_data['detections']),
                    'classes_detected': ', '.join(classes),
                    'confidences': ', '.join(confidences)
                })
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.results_dir, 'baseline_detections.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Results saved to {csv_path}")
        print(f"\nDetection Summary:")
        print(f"  - Total images: {len(df)}")
        print(f"  - Images with detections: {(df['detection_count'] > 0).sum()}")
        print(f"  - Images with no detections: {(df['detection_count'] == 0).sum()}")
        print(f"  - Average detections per image: {df['detection_count'].mean():.1f}")
        
        return csv_path, df
    
    def analyze_coco_coverage(self):
        """Analyze COCO class coverage vs unified taxonomy."""
        print(f"\n🎯 Analyzing COCO coverage vs unified taxonomy...")
        
        # Find overlap
        unified_set = set(self.unified_classes)
        coco_set = set(self.coco_classes.values())
        
        # Normalize for fuzzy matching
        overlap = []
        missing = []
        
        for cls in self.unified_classes:
            cls_lower = cls.lower()
            found = False
            
            for coco_cls in self.coco_classes.values():
                coco_lower = coco_cls.lower()
                if cls_lower == coco_lower or cls_lower in coco_lower or coco_lower in cls_lower:
                    overlap.append((cls, coco_cls))
                    found = True
                    break
            
            if not found:
                missing.append(cls)
        
        coverage_report = {
            'total_unified_classes': len(self.unified_classes),
            'coco_coverage': len(overlap),
            'coverage_percentage': (len(overlap) / len(self.unified_classes)) * 100,
            'missing_classes': missing,
            'overlapping_classes': overlap,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✓ Coverage Analysis:")
        print(f"  - Unified classes: {coverage_report['total_unified_classes']}")
        print(f"  - COCO coverage: {coverage_report['coco_coverage']} ({coverage_report['coverage_percentage']:.1f}%)")
        print(f"  - Missing classes: {len(missing)}")
        print(f"\nMissing classes in COCO:")
        for cls in sorted(missing):
            print(f"    - {cls}")
        
        return coverage_report
    
    def save_results_json(self):
        """Save all results as JSON for detailed analysis."""
        json_path = os.path.join(self.results_dir, 'baseline_detections.json')
        
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Detailed results saved to {json_path}")
        return json_path


def run_day5_baseline(base_dir, class_mapping_path):
    """Main entry point for Day 5 baseline testing."""
    print("=" * 70)
    print("🚀 Day 5: Baseline Detection Test - Pre-trained YOLOv8")
    print("=" * 70)
    
    detector = BaselineDetector(base_dir, class_mapping_path)
    
    # Run inference
    detector.run_inference()
    
    # Save results
    annotated_dir = detector.save_annotated_images()
    csv_path, csv_df = detector.generate_results_csv()
    json_path = detector.save_results_json()
    coverage = detector.analyze_coco_coverage()
    
    print("\n" + "=" * 70)
    print("✅ Day 5 Baseline Testing Complete!")
    print("=" * 70)
    print(f"\n📦 Output saved to: {detector.results_dir}")
    print(f"  - Annotated images: {annotated_dir}")
    print(f"  - CSV results: {csv_path}")
    print(f"  - JSON details: {json_path}")
    
    return detector, coverage


if __name__ == "__main__":
    # Navigate from backend/modules back to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    class_mapping_path = os.path.join(base_dir, "data", "processed", "class_mapping.json")
    
    detector, coverage = run_day5_baseline(base_dir, class_mapping_path)
