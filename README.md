# AI-Based Insider Threat Detection
## Mohammed Faizudden, Anna Wille, Maya Wyganowska

### Offline XGBoost Model
This repository contains the Semester 1 deliverables for the CS492 AI-Based Insider Threat Detection project:
a complete offline machine-learning pipeline for detecting insider threats using the CERT r4.2 Insider Threat Dataset.

The goal of this semester is to:
-build a reproducible modeling pipeline
-perform feature optimization
-tune a high-recall and high-precision classifier
-calibrate model thresholds
-prepare for deployment in Semester 2

All modeling was performed offline on static CERT logs.

### Repository Structure
'''
Senior-Design-Insider-Threat-Detection/
│
├── data/
│   ├── raw/                
│   ├── processed/          
│   └── features.csv        
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_xgboost_tuning.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_threshold_evaluation.ipynb
│   └── 06_calibration_curves.ipynb (optional)
│
├── models/
│   ├── xgb_final_model.joblib
│   └── feature_list.json
│
├── src/
│   └── utils.py
│
├── figures/
│   └── *plots used for final report*
│
├── requirements.txt
└── README.md
'''
### Environment Setup

