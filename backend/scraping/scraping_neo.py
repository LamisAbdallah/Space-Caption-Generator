"""
NASA Near-Earth Object (NEO) Scraper
Fetches asteroid data from NASA's NeoWs API with real hazard labels.
"""
import requests
import pandas as pd
import os
import time

NASA_API_KEY = os.environ.get("NASA_API_KEY", "vPZVibI9TQXU0bxT6TcfezBSgnP1QUeFeQohsn7Z")

def fetch_neo_data(target_count=1500):
    """
    Fetches NEO browse data from NASA API.
    Each NEO has close approach data, diameter estimates, and the real hazard label.
    """
    print(f"Fetching {target_count} NEOs from NASA NeoWs API...")
    all_neos = []
    page = 0
    page_size = 20  # NASA API max per page

    while len(all_neos) < target_count:
        url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={NASA_API_KEY}&page={page}&size={page_size}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 429:
                print("Rate limited. Waiting 60s...")
                time.sleep(60)
                continue
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text[:200]}")
                break

            data = response.json()
            neos = data.get("near_earth_objects", [])
            if not neos:
                print("No more NEOs returned.")
                break

            for neo in neos:
                close_approaches = neo.get("close_approach_data", [])
                if not close_approaches:
                    continue

                # Use the most recent close approach
                ca = close_approaches[-1]
                diameter = neo.get("estimated_diameter", {}).get("kilometers", {})

                record = {
                    "neo_id": neo.get("id"),
                    "name": neo.get("name", "Unknown"),
                    "absolute_magnitude_h": neo.get("absolute_magnitude_h"),
                    "estimated_diameter_min_km": diameter.get("estimated_diameter_min"),
                    "estimated_diameter_max_km": diameter.get("estimated_diameter_max"),
                    "relative_velocity_kmh": float(ca.get("relative_velocity", {}).get("kilometers_per_hour", 0)),
                    "miss_distance_km": float(ca.get("miss_distance", {}).get("kilometers", 0)),
                    "orbiting_body": ca.get("orbiting_body", "Earth"),
                    "is_potentially_hazardous": neo.get("is_potentially_hazardous_asteroid", False),
                }
                all_neos.append(record)

            print(f"  Page {page}: collected {len(all_neos)}/{target_count} NEOs")
            page += 1
            time.sleep(0.3)  # Be respectful to the API

        except requests.exceptions.Timeout:
            print("Timeout, retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"Error: {e}")
            break

    return all_neos[:target_count]


def build(target_count=1500):
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, "neo_hazard_dataset.csv")

    # Don't re-scrape if we already have the data
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if len(existing) >= target_count:
            print(f"Dataset already exists with {len(existing)} records. Skipping scrape.")
            return

    records = fetch_neo_data(target_count)
    df = pd.DataFrame(records)

    # Drop rows with missing critical values
    df = df.dropna(subset=[
        "absolute_magnitude_h",
        "estimated_diameter_min_km",
        "estimated_diameter_max_km",
        "relative_velocity_kmh",
        "miss_distance_km",
    ])

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} NEOs to {output_path}")
    print(f"Hazardous: {df['is_potentially_hazardous'].sum()}")
    print(f"Not Hazardous: {(~df['is_potentially_hazardous']).sum()}")
    print("Done!")


if __name__ == "__main__":
    build(1500)
