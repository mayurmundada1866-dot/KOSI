# KOSI
### Multimodal AI for Microplastic and River Environmental Intelligence


> **Live Demo:** [KOSI Dashboard](https://mayurmundada1866-dot-kosi-scripts19-dashboard-u3qtvo.streamlit.app)

KOSI is a multimodal environmental intelligence platform combining **computer vision, Raman spectroscopy, water-quality modelling, and temporal forecasting** into one integrated workflow — from microplastic detection to future river-state forecasting.


---

## Modules at a Glance

| Modality | Purpose | Model |
|---|---|---|
| Visual | Detect microplastic particles | YOLOv8 |
| Spectral | Identify polymer type | RBF SVM |
| Environmental | Estimate water quality | Random Forest |
| Temporal | Forecast future river state | Random Forest |

### Workflow
```text
           IMAGE        RAMAN        WATER
             |            |            |
           YOLO          SVM           RF
             |            |            |
         Detection    Polymer         WQI
             |       Identification    |
         Morphology                   |
             └────────────┬───────────┘
                          |
                    DIGITAL TWIN
                          |
                  Future River State
```

---

## 1. Microplastic Vision (YOLOv8)

Single-class microplastic detector trained on microscope images.

**Dataset:** 781 images · 7,126 annotated objects · 577 train / 204 val

| Metric | Score |
|---|---|
| Precision | 0.800 |
| Recall | 0.650 |
| mAP@50 | 0.734 |

---

## 2. Morphology Analysis

Aspect ratio from YOLO bounding box → morphology heuristic:

| Aspect Ratio | Morphology |
|---|---|
| 0.8 – 1.0 | Filament |
| 1.0 – 1.3 | Pellet |
| 1.3 – 4.0 | Fragment |


> Labels are aspect-ratio derived, not independently human-annotated.

---

## 3. Raman Spectroscopy (RBF SVM)

Identifies polymer type from preprocessed Raman spectra.

**Pipeline:** Raw Spectrum → Grid Alignment → Interpolation → Baseline Correction → Savitzky–Golay Smoothing → L2 Normalization → 1400 features → RBF SVM

**Dataset:** 173 spectra · 165 labelled · 7 polymer classes (PE, PP, PET, PVC, PA, PS, PC)

| Model | Accuracy | Macro F1 |
|---|---|---|
| **RBF SVM** (selected) | **0.939** | **0.922** |
| Random Forest | 0.909 | 0.764 |
| XGBoost | 0.909 | 0.752 |

---

## 4. Water Quality (Random Forest)

Predicts Water Quality Index (WQI) from 9 environmental parameters.

**Dataset:** 295 samples · `Results_MADE.csv`

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** (selected) | **11.359** | **37.921** | **0.933** |
| XGBoost | 13.411 | 54.715 | 0.862 |

> Conductivity dominates prediction (feature importance ≈ 0.945, correlation with WQI ≈ 0.9997). R² drops to 0.163 without it.

---

## 5. Digital Twin (Temporal Forecasting)

Next-state WQI forecasting using lagged features from Ganga & Sangam river time-series.

**Datasets:** Ganga (46,528 obs) · Sangam (52,363 obs)

**Lagged features:** WQI, DO, pH, ORP, Conductivity, Temperature — lags 1–3

| Metric | Score |
|---|---|
| MAE | 0.269 |
| RMSE | 0.467 |
| R² | 0.987 |

---

## 6. Performance Summary

| Module | Model | Metric | Score |
|---|---|---|---|
| Microplastic Detection | YOLOv8n | mAP@50 | 0.734 |
| Raman Polymer ID | RBF SVM | Accuracy / Macro F1 | 0.939 / 0.922 |
| Water Quality | Random Forest | R² | 0.933 |
| Digital Twin | Random Forest | R² | 0.987 |

---

## 7. Project Structure

```
KOSI/
├── scripts/
│   ├── 01_yolo_dataset_audit.py
│   ├── 02_visualize_yolo.py
│   ├── 03_train_yolo.py
│   ├── 04_yolo_predictions.py
│   ├── 05_morphology_classification.py
│   ├── 06_morphology_evaluation.py
│   ├── 07_raman_dataset_audit.py
│   ├── 08_raman_eda.py
│   ├── 09_raman_preprocessing.py
│   ├── 10_raman_models.py
│   ├── 11_raman_evaluation.py
│   ├── 12_river_eda.py
│   ├── 13_river_models.py
│   ├── 14_digital_twin_eda.py
│   ├── 15_digital_twin_model.py
│   ├── 16_digital_twin_evaluation.py
│   ├── 17_digital_twin_forecast.py
│   ├── 18_save_models.py
│   └── 19_dashboard.py
├── models/
│   ├── yolo_best.pt
│   ├── raman_svm.pkl
│   └── river_rf.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

> Large raw datasets and the Digital Twin model are excluded from this repo.

---

## 8. Installation

```bash
git clone https://github.com/mayurmundada1866-dot/KOSI.git
cd KOSI
pip install -r requirements.txt
```

## 9. Run Dashboard

```bash
streamlit run scripts/19_dashboard.py
```

---


## Built With

`Python` `YOLOv8` `Scikit-learn` `XGBoost` `OpenCV` `SciPy` `Pandas` `NumPy` `Plotly` `Streamlit`

---

## Author

**Mayur Mundada** · [GitHub](https://github.com/mayurmundada1866-dot)
