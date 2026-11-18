# Micro-Textile Precision Composition Analyzer

An AI-driven micro-textile recognition and fiber blend percentage estimation pipeline powered by **ConvNeXt-Base** embeddings and **Weighted k-NN Retrieval**, featuring open-set out-of-distribution (OOD) detection and live USB micro-camera support.

## Features
- **1024-D Visual Embeddings**: Uses `ConvNeXt-Base` to extract microscopic yarn and fiber structure representations.
- **Weighted k-NN Blend Estimation**: Predicts multi-fiber blend percentages via temperature-scaled cosine similarity retrieval.
- **Open-Set OOD Gating**: Automatically flags unknown or unregistered fabric blends below the similarity threshold (`0.50`).
- **Automated Dataset Management**: Folder-based taxonomy (`Cotton70_Polyester30`) automatically syncs to `labels.csv` and `fabric_index.pt`.
- **Interactive Streamlit UI**: Supports both batch image uploads and live USB micro-camera capture.

## Project Structure
```text
├── app.py                     # Main Streamlit web dashboard
├── auto_sync_dataset.py       # Scans folders, updates labels.csv, and builds fabric_index.pt
├── generate_labels_csv.py     # Standalone script to regenerate labels.csv from folders
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes large image datasets and .pt index files
└── fabricnew/                 # Place your composition-named image folders here
```

## Quickstart

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize Dataset**
   Place your micro-camera images inside `fabricnew/` using composition folder names (e.g., `fabricnew/Cotton70_Polyester30/`).

3. **Build Labels & Embedding Index**
   ```bash
   python auto_sync_dataset.py
   ```

4. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```
