# 🌿 Sentinel Deforestation Detection

A deep learning pipeline for detecting deforestation events in tropical regions using Sentinel-2 multitemporal satellite imagery. The system trains and compares three models — Random Forest, 1D CNN, and LSTM — on pixel-level time series data, and produces geographic deforestation maps validated against Global Forest Watch reference data.

---

## 📌 Project Overview

Deforestation is one of the most urgent environmental crises of our time. Ground-based monitoring is impractical at scale. This project leverages freely available Sentinel-2 satellite imagery to build an automated detection pipeline that can identify where and when forest loss occurred — using only the spectral time series of each pixel as input.

The core idea: instead of classifying a single satellite image, we observe the same location across 24 months and teach a model to recognize the temporal signature of deforestation — a stable high-vegetation signal that permanently collapses.

**Key features:**
- Full data engineering pipeline: cloud masking, monthly compositing, feature engineering, geographic train/test splitting
- Three model comparison: Random Forest baseline, 1D CNN, LSTM
- Honest evaluation using F1, recall, and precision on the deforested class
- Geographic deforestation maps saved as GeoTIFF and interactive HTML
- Exploratory transfer to Algerian forest regions

---

## 👥 Team

| Name | Role |
|------|------|
| Nour Zamiche | Team Leader |
| Sirine Atoum | Team Member |
| Maroua Sayah | Team Member |
| Kaouther Bensaddek | Team Member |
| Fatima Zohra Douaa Bourzak | Team Member |

**Supervisor:** Dr. Meziane Iftene  
**Institution:** ENSIA — École Nationale Supérieure d'Intelligence Artificielle, Algeria  
**Year:** 3rd Year Group Project, 2025–2026

---

## 🗂️ Repository Structure

```
sentinel-deforestation-detection/
│
├── README.md                        ← You are here
├── .gitignore
│
├── data/
│   ├── raw/                         ← Raw GEE exports (not tracked by Git)
│   ├── processed/                   ← Cloud-masked, composited arrays
│   └── final/                       ← Train/val/test splits ready for modeling
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
│
├── models/                          ← Saved model weights
│   ├── random_forest.joblib
│   ├── cnn_best_weights.pt
│   └── lstm_best_weights.pt
│
├── notebooks/                       ← Run these in order
│   ├── 01_data_collection.ipynb     ← GEE data acquisition
│   ├── 02_preprocessing.ipynb       ← Cloud masking, compositing, features
│   ├── 03_baseline_rf.ipynb         ← Random Forest baseline
│   ├── 04_cnn.ipynb                 ← 1D CNN training and evaluation
│   ├── 05_lstm.ipynb                ← LSTM training and evaluation
│   └── 06_evaluation_maps.ipynb     ← Final comparison and deforestation maps
│
├── src/                             ← Shared functions used across notebooks
│   ├── preprocessing.py             ← Cloud masking, compositing, feature engineering
│   ├── models.py                    ← CNN and LSTM class definitions
│   └── evaluation.py               ← Metrics, confusion matrix, map generation
│
└── results/
    ├── figures/                     ← All saved plots (training curves, confusion matrices)
    └── maps/                        ← GeoTIFF and HTML deforestation maps
```

---

## ⚙️ Environment Setup

```bash
# Create environment
conda create -n deforestation python=3.10
conda activate deforestation

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install earthengine-api rasterio xarray numpy pandas
pip install scikit-learn matplotlib seaborn folium geopandas
pip install imbalanced-learn joblib jupyter notebook ipywidgets

# Authenticate Google Earth Engine (one-time)
python -c "import ee; ee.Authenticate()"
```

---

## 🚀 How to Run

Run the notebooks in order. Each notebook saves its output so the next one can load it directly without rerunning previous steps.

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | `01_data_collection.ipynb` | Downloads Sentinel-2 time series via GEE for your study area |
| 2 | `02_preprocessing.ipynb` | Cloud masking, monthly compositing, feature engineering, geographic split |
| 3 | `03_baseline_rf.ipynb` | Trains and evaluates the Random Forest baseline |
| 4 | `04_cnn.ipynb` | Trains and evaluates the 1D CNN |
| 5 | `05_lstm.ipynb` | Trains and evaluates the LSTM |
| 6 | `06_evaluation_maps.ipynb` | Final model comparison and deforestation map generation |

---

## 📊 Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| Sentinel-2 L2A | Copernicus / Google Earth Engine | Satellite imagery — model input |
| Global Forest Watch (Hansen) | University of Maryland / GFW | Deforestation labels |
| PRODES | INPE — Brazil | High-accuracy Amazon deforestation reference |

All data sources are freely and publicly accessible.

---

## 📈 Model Comparison

Three models are trained and compared on the same geographic test set:

| Model | Type | Temporal Handling |
|-------|------|------------------|
| Random Forest | Classical ML | Temporal statistics (flattened) |
| 1D CNN | Deep Learning | Local pattern detection via convolution |
| LSTM | Deep Learning | Sequential memory across full time series |

Primary evaluation metric: **F1 score on the deforested class**.  
Evaluation uses a **geographic train/test split** — never random — to prevent spatial autocorrelation from inflating results.

---

## 🌍 Transfer to Algeria

After training and evaluating on the Brazilian Amazon, the best model is applied exploratorily to forested regions of northern Algeria (Tell Atlas / Kabylie). This transfer is presented as exploratory — no quantitative evaluation is performed due to the absence of verified reference labels comparable to PRODES. Domain shift between tropical Amazon forest and Mediterranean Algerian forest is acknowledged and discussed in the analysis report.

---

## 📄 License

This project was developed for academic purposes at ENSIA. All satellite data used is openly licensed through the Copernicus program and Global Forest Watch.

---

