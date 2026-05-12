import requests
import pandas as pd
import os
import time


def fetch_vlm_dataset(total_target=200):
    """
    Fetches visually rich celestial bodies from NASA's Image and Video Library.
    Perfect for training a Vision-Language Model (VLM) for image-captioning.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "dataset")
    images_dir = os.path.join(dataset_dir, "vlm_images")
    csv_path = os.path.join(dataset_dir, "vlm_captions_dataset.csv")
    
    os.makedirs(images_dir, exist_ok=True)
    
    # The varied celestial bodies you requested for the VLM
    categories = [
        "planet", "galaxy", "nebula", "star cluster",
        "moon", "comet", "supernova"
    ]
    
    # Calculate how many images per category we need
    limit_per_category = (total_target // len(categories)) + 2 
    
    all_data = []
    print(f"--> Collecting visually rich images for VLM captioning...")
    
    for category in categories:
        print(f"    Querying NASA for: {category}...")
        url = f"https://images-api.nasa.gov/search?q={category}&media_type=image"
        
        try:
            response = requests.get(url, timeout=10).json()
            items = response.get("collection", {}).get("items", [])
            
            count = 0
            for item in items:
                if count >= limit_per_category:
                    break
                    
                data_block = item.get("data", [])
                links_block = item.get("links", [])
                
                if data_block and links_block:
                    nasa_id = data_block[0].get("nasa_id")
                    # NASA provides rich descriptions which act as perfect ground-truth captions!
                    description = data_block[0].get("description", "").strip() 
                    title = data_block[0].get("title", "")
                    img_url = links_block[0].get("href")
                    
                    # Skip if description is too short or missing
                    if len(description) < 20:
                        continue
                        
                    all_data.append({
                        "object_id": nasa_id,
                        "category": category,
                        "title": title,
                        "caption": description,  # The crucial VLM text pairing
                        "image_url": img_url,
                        "image_file_path": f"vlm_images/{nasa_id}.jpg"
                    })
                    count += 1
            time.sleep(1)  # Be polite to API
        except Exception as e:
            print(f"Error fetching {category}: {e}")
            
    print(f"\nTotal visual objects compiled: {len(all_data)}")
    if not all_data:
        return
        
    print("Downloading VLM training images...")
    downloaded_records = []
    
    for i, item in enumerate(all_data):
        # Stop exactly at targeted amount (e.g. 200)
        if len(downloaded_records) >= total_target:
            break
            
        img_path = os.path.join(dataset_dir, item["image_file_path"])
        
        if not os.path.exists(img_path):
            try:
                img_data = requests.get(item["image_url"], timeout=10).content
                # Save only valid images
                if b"<html" not in img_data[:10]:
                    with open(img_path, "wb") as f: 
                        f.write(img_data)
                else:
                    continue
            except Exception:
                continue
                
        # Remove URL from final dataset
        del item["image_url"]
        downloaded_records.append(item)
        if (i + 1) % 50 == 0: 
            print(f"    Downloaded {i+1}/{total_target} VLM images...")
            
    # Save perfectly prepped VLM dataset
    df = pd.DataFrame(downloaded_records)
    df.to_csv(csv_path, index=False)
    print(f"\nSUCCESS! VLM dataset with {len(df)} images and rich captions saved to {csv_path}")


if __name__ == "__main__":
    # Let's pull slightly more than 200, e.g., 210, to guarantee high variance
    fetch_vlm_dataset(total_target=210)
