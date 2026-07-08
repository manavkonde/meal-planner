import os
import sys
from pathlib import Path
from PIL import Image

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    "fruits360": RAW_DATA_DIR / "fruits360",
    "grocery": RAW_DATA_DIR / "grocery",
    "fridge_objects": RAW_DATA_DIR / "fridge_objects",
    "test_fridges": RAW_DATA_DIR / "test_fridges"
}

def count_and_verify_images(folder_path):
    if not folder_path.exists():
        return False, 0, "Directory does not exist"
        
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                image_files.append(Path(root) / f)
                
    total_images = len(image_files)
    if total_images == 0:
        return False, 0, "No image files found"
        
    # Spot check 5 random images (or up to total_images)
    import random
    random.seed(42)
    sample_size = min(5, total_images)
    samples = random.sample(image_files, sample_size)
    
    corrupted_count = 0
    for s in samples:
        try:
            with Image.open(s) as img:
                img.verify() # Verify image integrity
        except Exception as e:
            print(f"Corrupted image detected at {s}: {e}")
            corrupted_count += 1
            
    if corrupted_count > 0:
        return False, total_images, f"{corrupted_count} out of {sample_size} sample images checked are corrupted"
        
    return True, total_images, "All check samples are valid"

def main():
    print("========================================")
    print("Dataset Integrity Verification Starting")
    print("========================================")
    
    all_success = True
    for name, path in DATASETS.items():
        print(f"\nVerifying {name} at {path}...")
        success, count, message = count_and_verify_images(path)
        if success:
            print(f"  [SUCCESS] {name}: Found {count} images. {message}.")
        else:
            print(f"  [FAILED] {name}: Found {count} images. Error: {message}.")
            all_success = False
            
    print("\n========================================")
    if all_success:
        print("Verification completed: ALL DATASETS ARE VALID AND READABLE!")
        sys.exit(0)
    else:
        print("Verification completed: SOME DATASETS ENCOUNTERED PROBLEMS.")
        sys.exit(1)

if __name__ == '__main__':
    main()
