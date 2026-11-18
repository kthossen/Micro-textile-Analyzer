import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_IMG_DIR = os.path.join(BASE_DIR, "fabricnew")
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")

def parse_percentages_from_folder(folder_name):
    pct_dict = {}
    matches = re.findall(r'([a-zA-Z]+)_?(\d+)', folder_name)
    for fiber, val in matches:
        fiber_key = f"{fiber.lower()}_pct"
        pct_dict[fiber_key] = float(val)
    return pct_dict

def generate_csv():
    print(f"Scanning directory: {BASE_IMG_DIR}")
    records = []
    
    if not os.path.exists(BASE_IMG_DIR):
        print(f"Error: Directory {BASE_IMG_DIR} does not exist.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

    for root, dirs, files in os.walk(BASE_IMG_DIR):
        folder_name = os.path.basename(root)
        if folder_name == "fabricnew" or not folder_name:
            continue
            
        parsed_pcts = parse_percentages_from_folder(folder_name)
        if not parsed_pcts:
            print(f"Skipping folder (could not parse percentages): {folder_name}")
            continue

        for file in files:
            if file.lower().endswith(valid_extensions):
                row = {
                    'folder': folder_name,
                    'filename': file,
                    **parsed_pcts
                }
                records.append(row)

    if not records:
        print("No valid fabric images found matching the naming convention.")
        return

    df = pd.DataFrame(records).fillna(0.0)
    df.to_csv(LABELS_PATH, index=False)
    print(f"\n[Success] Generated new labels CSV at: {LABELS_PATH}")
    print(f"Total images cataloged: {len(df)}")

if __name__ == "__main__":
    generate_csv()
