import os
import requests
import json
from pathlib import Path

def download_geojson():
    # URL to a reliable Brazil states GeoJSON
    url = "https://raw.githubusercontent.com/fititnt/gis-dataset-brasil/master/uf/geojson/uf.json"
    
    # Path from Config
    # We'll hardcode it here for simplicity or try to import Config
    try:
        from src.olist_pipeline.utils.config_loader import Config
        path = Config.get_path("data", "external_geojson")
    except ImportError:
        path = Path("data/external/br_states.geojson")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        print(f"File already exists at {path}")
        return

    print(f"Downloading Brazil states GeoJSON from {url}...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Compatibility fix: map 'UF_05' (abbreviation) to 'abbrev_state' 
        # which is expected by the Streamlit app and matches geobr format.
        print("Applying compatibility mapping (UF_05 -> abbrev_state)...")
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            if 'UF_05' in props:
                props['abbrev_state'] = props['UF_05']
            elif 'sigla' in props:
                props['abbrev_state'] = props['sigla']
                
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Successfully saved to {path}")
    except Exception as e:
        print(f"Error downloading GeoJSON: {e}")
        print("Please manually download a Brazil states GeoJSON and save it to data/external/br_states.geojson")

if __name__ == "__main__":
    download_geojson()
