import os
import sys
import zipfile
import urllib.request
import shutil
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Datasets URLs and IDs
DATASETS = {
    "fruits360": {
        "kaggle": "moltean/fruits",
        "fallback_url": "https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip",
        "zip_name": "fruits360.zip",
        "extract_to": RAW_DATA_DIR / "fruits360"
    },
    "grocery": {
        "kaggle": "marlonrcfranco/grocery-store-dataset",
        "fallback_url": "https://github.com/marcusklasson/GroceryStoreDataset/archive/refs/heads/master.zip",
        "zip_name": "grocery.zip",
        "extract_to": RAW_DATA_DIR / "grocery"
    },
    "fridge_objects": {
        "kaggle": "surendraallam/refrigerator-contents",
        "fallback_url": "https://automlsamplenotebookdata-adcuc7f7bqhhh8a4.b02.azurefd.net/image-object-detection/odFridgeObjects.zip",
        "zip_name": "fridge_objects.zip",
        "extract_to": RAW_DATA_DIR / "fridge_objects"
    }
}

def check_kaggle_creds():
    # Check ~/.kaggle/kaggle.json or env vars
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    
    # Custom opener to handle user-agent blocks
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')]
    urllib.request.install_opener(opener)
    
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 1e2 / total_size
            sys.stdout.write(f"\rProgress: {percent:3.1f}% ({read_so_far}/{total_size} bytes)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rRead {read_so_far} bytes")
            sys.stdout.flush()
            
    urllib.request.urlretrieve(url, str(dest_path), reporthook=report_hook)
    print("\nDownload finished.")

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

def download_via_kaggle(dataset_info, name):
    kaggle_id = dataset_info["kaggle"]
    zip_name = dataset_info["zip_name"]
    extract_to = dataset_info["extract_to"]
    
    dest_zip = RAW_DATA_DIR / zip_name
    print(f"\n[Kaggle API] Downloading {name} from Kaggle ({kaggle_id})...")
    
    try:
        # Run kaggle datasets download -d <id> -p <path>
        cmd = ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(RAW_DATA_DIR)]
        subprocess.run(cmd, check=True)
        
        # Kaggle CLI saves zip file named after the dataset basename
        actual_zip_name = kaggle_id.split("/")[-1] + ".zip"
        downloaded_zip = RAW_DATA_DIR / actual_zip_name
        
        if downloaded_zip.exists():
            unzip_file(downloaded_zip, extract_to)
            os.remove(downloaded_zip)
            return True
        else:
            print(f"Error: Expected downloaded zip at {downloaded_zip} but not found.")
            return False
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False

def download_via_http(dataset_info, name):
    url = dataset_info["fallback_url"]
    zip_name = dataset_info["zip_name"]
    extract_to = dataset_info["extract_to"]
    
    dest_zip = RAW_DATA_DIR / zip_name
    print(f"\n[Direct HTTP] Downloading {name} from {url}...")
    
    try:
        download_file(url, dest_zip)
        unzip_file(dest_zip, extract_to)
        os.remove(dest_zip)
        
        # Some GitHub repositories zip inside a master folder (e.g. Fruit-Images-Dataset-master)
        # Let's clean up folder levels if needed so the data is directly in the dataset folder
        subdirs = list(extract_to.glob("*"))
        if len(subdirs) == 1 and subdirs[0].is_dir():
            print(f"Flatteng master folder {subdirs[0].name}...")
            temp_dir = extract_to.parent / f"{extract_to.name}_temp"
            os.makedirs(temp_dir, exist_ok=True)
            # Move contents
            for item in subdirs[0].glob("*"):
                shutil.move(str(item), str(temp_dir))
            shutil.rmtree(extract_to)
            os.rename(temp_dir, extract_to)
            
        return True
    except Exception as e:
        print(f"Direct download failed: {e}")
        return False

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    use_kaggle = check_kaggle_creds()
    
    if use_kaggle:
        print("Kaggle credentials found. Using Kaggle API for downloads.")
    else:
        print("Kaggle credentials not found. Using Direct HTTP Fallback Mode.")
        
    for name, info in DATASETS.items():
        success = False
        if use_kaggle:
            success = download_via_kaggle(info, name)
            
        if not success:
            if use_kaggle:
                print("Kaggle download failed. Falling back to HTTP...")
            success = download_via_http(info, name)
            
        if not success:
            print(f"CRITICAL: Failed to download dataset: {name}")
            sys.exit(1)
            
    print("\nAll datasets downloaded, extracted, and cleaned up successfully!")

if __name__ == '__main__':
    main()
