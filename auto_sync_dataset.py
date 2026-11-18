import os
import re
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_IMG_DIR = os.path.join(BASE_DIR, "fabricnew")
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")
INDEX_PATH = os.path.join(BASE_DIR, "fabric_index.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNeXtExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = base.avgpool
        self.flatten = nn.Flatten(1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.flatten(x)

def parse_percentages_from_folder(folder_name):
    pct_dict = {}
    matches = re.findall(r'([a-zA-Z]+)_?(\d+)', folder_name)
    for fiber, val in matches:
        fiber_key = f"{fiber.lower()}_pct"
        pct_dict[fiber_key] = float(val)
    return pct_dict

def sync_and_build():
    print("Scanning dataset folders...")
    records = []
    
    if not os.path.exists(BASE_IMG_DIR):
        print(f"Error: Directory {BASE_IMG_DIR} not found.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

    for root, dirs, files in os.walk(BASE_IMG_DIR):
        folder_name = os.path.basename(root)
        if folder_name == "fabricnew" or not folder_name:
            continue
            
        parsed_pcts = parse_percentages_from_folder(folder_name)
        if not parsed_pcts:
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
    print(f"[Success] Updated {LABELS_PATH} with {len(df)} total samples.")

    print("\nLoading ConvNeXt-Base backbone for feature extraction...")
    extractor = ConvNeXtExtractor().to(DEVICE)
    extractor.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    embeddings_list = []
    metadata_list = []

    print("Extracting 1024-dim micro-texture embeddings...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(BASE_IMG_DIR, row['folder'], row['filename'])
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            feat = extractor(tensor)
            feat = feat.view(feat.size(0), -1)
            feat = F.normalize(feat, p=2, dim=1)
            
            embeddings_list.append(feat.cpu())
            metadata_list.append(row.to_dict())

    if embeddings_list:
        all_embeddings = torch.cat(embeddings_list, dim=0)
        torch.save({
            'embeddings': all_embeddings,
            'metadata': metadata_list
        }, INDEX_PATH)
        print(f"\n[Success] Built and saved embedding index to: {INDEX_PATH}")
    else:
        print("Error: No embeddings were generated.")

if __name__ == "__main__":
    sync_and_build()
