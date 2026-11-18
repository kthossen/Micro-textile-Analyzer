import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Fabric Composition & Precision Analyzer",
    page_icon="🔬",
    layout="wide"
)

# ==========================================
# PATHS & CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "fabric_index.pt")
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")
BASE_IMG_DIR = os.path.join(BASE_DIR, "fabricnew")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OOD_THRESHOLD = 0.50  # Similarity gating threshold for unknown fabrics

# ==========================================
# MODEL ARCHITECTURE (Feature Extractor)
# ==========================================
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

@st.cache_resource
def load_pipeline():
    if not os.path.exists(INDEX_PATH):
        return None, None, None
    
    extractor = ConvNeXtExtractor().to(DEVICE)
    extractor.eval()

    index_data = torch.load(INDEX_PATH, map_location=DEVICE)
    index_embeddings = index_data['embeddings'].to(DEVICE)
    metadata = index_data['metadata']

    return extractor, index_embeddings, metadata

extractor, index_embeddings, metadata = load_pipeline()

# ==========================================
# UI LAYOUT & SIDEBAR SETTINGS
# ==========================================
st.title("🔬 Micro-Textile Precision Composition Analyzer")
st.markdown("Advanced multi-fiber micro-texture analysis powered by **ConvNeXt-Base** and **k-NN Retrieval**.")

if extractor is None:
    st.error(f"Feature index not found at `{INDEX_PATH}`. Please run `python auto_sync_dataset.py` first.")
else:
    st.sidebar.header("Input Settings")
    input_mode = st.sidebar.radio("Select Image Source:", ["Upload Micro-Image", "USB Micro-Camera (Live Capture)"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("Retrieval Parameters")
    k_neighbors = st.sidebar.slider("k-NN Neighbors (Blend Fusion)", min_value=1, max_value=10, value=5)
    temperature = st.sidebar.slider("Similarity Sharpness", min_value=1.0, max_value=20.0, value=10.0)

    # Image input handling
    query_image = None
    if input_mode == "Upload Micro-Image":
        uploaded_file = st.sidebar.file_uploader("Choose a fabric patch...", type=["jpg", "jpeg", "png", "bmp", "tif"])
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
    else:
        st.sidebar.info("Ensure your USB micro-camera is active.")
        camera_file = st.sidebar.camera_input("Capture Micro-Texture Frame")
        if camera_file is not None:
            query_image = Image.open(camera_file).convert("RGB")

    # ==========================================
    # INFERENCE PIPELINE
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if query_image is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Query Image Preview")
            st.image(query_image, width="stretch")

        with col2:
            st.subheader("Composition Prediction Results")
            
            tensor = transform(query_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                query_emb = extractor(tensor)
                query_emb = query_emb.view(query_emb.size(0), -1)
                query_emb = F.normalize(query_emb, p=2, dim=1)
                
                current_index_embeddings = index_embeddings
                if current_index_embeddings.dim() > 2:
                    current_index_embeddings = current_index_embeddings.view(current_index_embeddings.size(0), -1)
                
                similarities = torch.mm(query_emb, current_index_embeddings.T).squeeze(0)
                
                top_sims, top_indices = torch.topk(similarities, k=min(k_neighbors, len(current_index_embeddings)))
                top_sim_score = top_sims[0].item()

            if top_sim_score < OOD_THRESHOLD:
                st.warning(f"⚠️ **Novel / Unregistered Fabric Detected** (Max Similarity: {top_sim_score:.3f})")
                st.info("This micro-texture lies outside your current database distribution.")
            else:
                weights = torch.softmax(top_sims * temperature, dim=0).cpu().numpy()
                retrieved_rows = [metadata[idx.item()] for idx in top_indices]

                all_keys = set()
                for row in retrieved_rows:
                    for k_col in row.keys():
                        if k_col.endswith('_pct'):
                            all_keys.add(k_col)

                metric_cols = st.columns(2)
                metric_idx = 0

                for fiber_key in sorted(all_keys):
                    weighted_sum = sum(row.get(fiber_key, 0.0) * w for row, w in zip(retrieved_rows, weights))
                    pct = round(float(weighted_sum), 1)
                    if pct > 0.5:
                        fiber_name = fiber_key.replace('_pct', '').upper()
                        with metric_cols[metric_idx % 2]:
                            st.metric(label=fiber_name, value=f"{pct}%")
                        metric_idx += 1

        st.markdown("---")
        st.subheader("Top Nearest Reference Matches (Database Context)")
        
        match_cols = st.columns(k_neighbors)
        for i, (score, idx) in enumerate(zip(top_sims, top_indices)):
            meta = metadata[idx.item()]
            match_img_path = os.path.join(BASE_IMG_DIR, meta['folder'], meta['filename'])
            with match_cols[i]:
                if os.path.exists(match_img_path):
                    st.image(
                        match_img_path, 
                        caption=f"Sim: {score.item():.3f}\n{meta['folder']}", 
                        width="stretch"
                    )
                else:
                    st.write(f"Image not found: {meta['filename']}")
