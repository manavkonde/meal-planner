import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
TEST_FRIDGES_DIR = BASE_DIR / "data" / "raw" / "test_fridges"

# Curated fallback Unsplash photo IDs in case NAPI fails
FALLBACK_IDS = [
    "photo-1571175432247-fe0a28f8284f",
    "photo-1543083477-4f785ae82753",
    "photo-1590779033100-9f60a05a013d",
    "photo-1574781330855-d0db8cc6a79c",
    "photo-1598449356475-b9f71ef73f47",
    "photo-1584269600464-37b1b58a9fe7",
    "photo-1563245372-f21724e3856d",
    "photo-1542838132-92c53300491e",
    "photo-1506806732259-39c2d0268443",
    "photo-1540340061720-ee2c48371891",
    "photo-1588964895597-cfccd6e2dbf9",
    "photo-1606787366850-de6330128bfc",
    "photo-1618220179428-22790b461013",
    "photo-1556910103-1c02745aae4d",
    "photo-1601004890684-d8cbf643f5f2",
    "photo-1615485290382-441e4d049cb5",
    "photo-1518843025960-d60217f226f5",
    "photo-1498837167922-ddd27525d352",
    "photo-1490645935967-10de6ba17061",
    "photo-1504674900247-0877df9cc836",
    "photo-1567306226416-28f0efdc88ce",
    "photo-1512621776951-a57141f2eefd",
    "photo-1551248429-40975aa4de74",
    "photo-1478144542123-af7e8298f690",
    "photo-1482049016688-2d3e1b311543",
    "photo-1473093295043-cdd812d0e601",
    "photo-1484723091739-30a097e8f929",
    "photo-1504754524776-8f4f37790ca0",
    "photo-1529042410759-befb1204b468",
    "photo-1515003197210-e0cd71810b5f",
    "photo-1546069901-ba9599a7e63c",
    "photo-1565299624946-b28f40a0ae38",
    "photo-1467003909585-2f8a72700288",
    "photo-1432139555190-58524dae6a55",
    "photo-1555939594-58d7cb561ad1",
    "photo-1544025162-d76694265947",
    "photo-1493770308161-b68565312786",
    "photo-1532980400857-e8d9d275727b",
    "photo-1501959915014-722129306b39",
    "photo-1565958011703-44f9829ba187"
]

def fetch_unsplash_urls(query, count=30):
    url = f"https://unsplash.com/napi/search/photos?query={urllib.parse.quote(query)}&per_page={count}"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = data.get('results', [])
            urls = []
            for r in results:
                img_url = r.get('urls', {}).get('regular')
                if img_url:
                    urls.append(img_url)
            return urls
    except Exception as e:
        print(f"Unsplash NAPI fetch failed for '{query}': {e}")
        return []

def download_image(url, dest_path):
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    )
    with urllib.request.urlopen(req) as response:
        with open(dest_path, 'wb') as f:
            f.write(response.read())

def main():
    os.makedirs(TEST_FRIDGES_DIR, exist_ok=True)
    print("Collecting stock photos of refrigerators/groceries...")
    
    # Try fetching dynamically first
    urls = []
    for query in ["open fridge", "refrigerator food"]:
        urls.extend(fetch_unsplash_urls(query, count=25))
        
    # De-duplicate URLs
    urls = list(set(urls))
    
    # If dynamic search failed or returned too few results, fall back to our curated list
    if len(urls) < 30:
        print(f"Dynamic fetch returned only {len(urls)} images. Using curated fallbacks.")
        urls = [f"https://images.unsplash.com/{photo_id}?auto=format&fit=crop&w=800&q=80" for photo_id in FALLBACK_IDS]
    
    total_to_download = min(len(urls), 40)  # We target 30-50 images
    print(f"Starting download of {total_to_download} test fridge photos...")
    
    success_count = 0
    for i in range(total_to_download):
        url = urls[i]
        dest_file = TEST_FRIDGES_DIR / f"fridge_{i+1:03d}.jpg"
        sys.stdout.write(f"\rDownloading image {i+1}/{total_to_download}...")
        sys.stdout.flush()
        try:
            download_image(url, dest_file)
            success_count += 1
        except Exception as e:
            # If download fails, continue to next
            pass
            
    print(f"\nFinished. Successfully downloaded {success_count} real-world fridge test photos to {TEST_FRIDGES_DIR}.")

if __name__ == '__main__':
    main()
