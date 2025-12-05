# AI-Based Insider Threat Detection
### Mohammed Faizudden, Anna Wille, Maya Wyganowska
Project Goal:

Create a complete offline machine-learning pipeline for detecting insider threats using the CERT r4.2 Insider Threat Dataset.

This repository will guide you through how to:
- build a reproducible modeling pipeline
- perform feature optimization
- tune a high-recall and high-precision classifier
- calibrate model thresholds
- prepare for deployment

All modeling was performed offline on static CERT r4.2 logs.

### Quick Results Summary

| Mode | Threshold | Precision | Recall | F1-Score | False Positives | Use Case |
|------|-----------|-----------|--------|----------|-----------------|----------|
| **Alert** | 0.26 | 77.78% | 77.78% | 0.7778 | 14 (0.10%) | Early warning, comprehensive monitoring |
| **Critical** | 0.65 | 97.78% | 69.84% | 0.8148 | 1 (0.007%) | High-confidence, immediate action |

### Preliminary Requirements:
Clone repository using `git clone <URL>`

Open project root directory using `cd Senior-Design-Insider-Threat-Detection`

### Repository Structure
```
Senior-Design-Insider-Threat-Detection/
│
├── examples/
|   ├── figures/
|   ├── models/
│
├── data/            
│   ├── processed/          
│   └── features.csv        
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_modelComparison.ipynb
│   ├── 03_modelTuning.ipynb
│   ├── 04_featureSelection.ipynb
│   └── 05_modelEvaluation.ipynb
│
├── models/
│
├── src/
│   └── utils.py
│
├── figures/
│
├── requirements.txt
└── README.md
```
# Offline XGBoost Model
### Environment Setup
Create environment: 
```
conda create -n certml python=3.10 -y
conda activate certml
```
Install dependencies: 
```
pip install -r requirements.txt
```
Launch Jupyter Notebook **through project root directory**:
```
jupyter notebook
```
### Dataset
This project uses the CERT Insider Threat Dataset, a simulated environment containing user behavior logs such as:
- logon/logoff activity
- file access
- USB events
- web browsing
- email activity
- psychometric scores
- HR events

[Download the CERT r4.2 Dataset](https://doi.org/10.1184/R1/12841247)

### Feature Extraction
CERT provides a feature extraction script to be used on their datasets. This script merges all raw logs (logon, file, email, USB, web, HR, psychometric)
into a single feature matrix representing user–day behavioral data.

Use the repository's documentation to install feature extraction script dependencies and run.


[Download the Feature Extraction Script](https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets)

After running the CERT feature extraction script, you will have a local `ExtractedData/` folder containing aggregated behavioral logs. **Copy `ExtractedData/` into the project root.**

For this project, `ExtractedData/` is treated as a local source and is not tracked in the GitHub repository.
**Notebook `01_preprocessing.ipynb` expects an `ExtractedData/` folder in the project root.**

It reads from `ExtractedData/`, performs cleaning and splitting, and then writes the processed modeling files into the `data/` directory:
- `data/features.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## Users must download CERT dataset, run the feature extraction script, and save `ExtractedData/` to project root before running Notebook 01.



# Let's get down to business...


### Preprocessing
### Preprocessing (Notebook 01)

Notebook 01_preprocessing reads the raw extracted CERT data located in the `ExtractedData/` directory (produced by the feature_extraction.py script).

This notebook:
- loads the multiple extracted CSV files from `ExtractedData/`
- merges and cleans them into a unified dataset
- saves this dataset as `data/features.csv`
- creates train/validation/test splits
- saves them into `data/processed/`

### Model Comparison (Notebook 02)
Notebook 02_modelComparison outlines the process we used to select our best model. In models that struggle with heavy class imbalances, we implemented a SMOTE pipeline to create more synthetic insider threats and see if a more balanced dataset could produce more accurate results. We trained these models using 5-fold Stratified Cross-Validation on the training dataset. No test data will be used at this stage.

We evaluated four baseline models:
- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost

And achieved these results:
```
Model                 Precision   Recall   F1 Score   Precision Rank   Recall Rank   F1 Rank
--------------------------------------------------------------------------------------------
XGBoost               0.796       0.642    0.708      2                2             1
Linear SVM + SMOTE    0.260       0.605    0.363      3                3             2
LogReg + SMOTE        0.190       0.663    0.295      4                1             3
Random Forest         0.911       0.174    0.290      1                4             4
```
XGBoost achieved the highest F1-score and recall, making it the best fit for imbalanced insider classification.

### Model Tuning (Notebook 03)
Notebook 03_modelTuning outlines the process we used to tune our XGBoost model to the best parameters. It trains multiple XGBoost models with different hyperparameter 
settings. Model performance is evaluated on validation folds only.

RandomizedSearchCV produced the final optimized parameters:
```
Best mean CV F1: 0.7143576006609965
Best params:
{'subsample': 0.7,
 'scale_pos_weight': 158.325,
 'reg_lambda': 3,
 'reg_alpha': 0,
 'n_estimators': 300,
 'min_child_weight': 1,
 'max_depth': 6,
 'learning_rate': 0.1,
 'gamma': 0,
 'colsample_bytree': 0.8}
```
This parameter set allows XGBoost to learn complex insider-threat patterns without overfitting, while properly correcting for the dataset’s strong class imbalance.

### Feature Selection (Notebook 04)
Notebook 04_featureSelection uses the tuned XGBoost model to identify and remove low-importance features. XGBoost provides built-in feature importance scores based on how much each
feature contributes to decision-tree splits. It retrains the tuned XGBoost model using only the selected 
reduced feature set. Training occurs on the training split, and evaluation is performed on the validation split. This reduced set becomes the final model used for threshold calibration.

Results:
```
Precision    Recall  F1-Score   Support
----------------------------------------
Class 0       1.00     1.00      1.00     13370
Class 1       0.88     0.70      0.78        63

Accuracy                          1.00     13433
Macro Avg     0.94     0.85      0.89     13433
Weighted Avg  1.00     1.00      1.00     13433

Confustion Matrix:
[[13364     6]
 [   19    44]]
```
### Threshold Evaluation  (Notebook 05)
Notebook 05 will calibrate the model, select optimal thresholds, evaluate on test set

#### Model Calibration
- Calibration curve: agreement between predicted and actual probabilities
- Assessment: Model is well-calibrated, especially at high confidence levels
- Conclusion: Probability scores can be trusted for threshold-based decisions

#### Threshold Sweep
- Method: Evaluate 500 thresholds from 0.0 to 1.0 on validation set
- Metrics: Precision, Recall, F1-Score at each threshold
- Optimal F1: 0.789 at threshold 0.649

#### Threshold Selection

**Alert Mode (0.26)**:
- Goal: Maximize recall while maintaining acceptable precision
- Selection criteria: Recall ≥ 75%, maximize F1
- Validation performance: Precision 74.24%, Recall 77.78%, F1 0.7597

**Critical Mode (0.65)**:
- Goal: Maximize precision for actionable alerts
- Selection criteria: Global F1 maximum
- Validation performance: Precision 93.48%, Recall 68.25%, F1 0.7890

#### Test Set Evaluation

**Alert Mode Results (Threshold = 0.26)**:
```
Precision: 77.78%
Recall:    77.78%
F1-Score:  0.7778
Accuracy:  99.79%

Confusion Matrix:
                Predicted
                Normal  Threat
Actual Normal   13,357     14  
       Threat      14     49   

Key Insight: Perfect precision-recall balance (14 FP = 14 FN)
```

**Critical Mode Results (Threshold = 0.65)**:
```
Precision: 97.78%
Recall:    69.84%
F1-Score:  0.8148
Accuracy:  99.85%

Confusion Matrix:
                Predicted
                Normal  Threat
Actual Normal   13,370      1   
       Threat      19     44   

Key Insight: Near-perfect precision with only 1 FP out of 13,371 normal
```

#### Model Discrimination

**ROC Curve**:
- AUC: 0.997
- Interpretation: 99.7% probability model ranks threats higher than normals

**Precision-Recall Curve**:
- AUC: 0.840 
- Interpretation: Robust performance despite class imbalance

#### Probability Distribution Analysis
- Normal users: Tightly clustered near 0.0 (99.9% < 0.1)
- Threat users: Bimodal with peak at 0.9-1.0
- Separation: Clear between classes 
- Threat median: 0.65 (matches Critical Mode threshold!)

# Using the Model
The final XGBoost model and the reduced feature list are stored in the models/ directory and can be used to generate predictions on new user–day behavioral data. 

### Example Usage:
```
import joblib
import json
import pandas as pd

# Load model and features
model = joblib.load("models/xgb_reduced_model.joblib")
with open("models/feature_list.json", "r") as f:
    features = json.load(f)

# Prepare new data (must contain the same 121 features)
df = pd.read_csv("path/to/new_user_data.csv")
X = df[features]

# Generate probability scores
probs = model.predict_proba(X)[:, 1]

# Apply thresholds
alert_predictions = (probs >= 0.26).astype(int)
critical_predictions = (probs >= 0.65).astype(int)

# Generate alerts
alert_cases = df[alert_predictions == 1]
critical_cases = df[critical_predictions == 1]

print(f"Alert Mode: {alert_predictions.sum()} alerts")
print(f"Critical Mode: {critical_predictions.sum()} critical alerts")

# Get probability scores for alerts
for idx in critical_cases.index:
    user = df.loc[idx, 'user']
    prob = probs[idx]
    print(f"CRITICAL: User {user} - Threat probability: {prob:.2%}")
```
These prediction outputs will support the alerting system that we will be deploying for Semester 2.

# So What's Next?
In Semester 2, this offline model will be integrated into a live runtime environment. Planned components include:
- a FastAPI inference service for real-time prediction
- Docker containerization for portability
- a GUI dashboard to display alerts
- real time predictions on live or simulated user activity logs
