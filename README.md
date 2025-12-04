# AI-Based Insider Threat Detection
### Mohammed Faizudden, Anna Wille, Maya Wyganowska
This repository contains the Semester 1 deliverables for the CS492 AI-Based Insider Threat Detection project:
a complete offline machine-learning pipeline for detecting insider threats using the CERT r4.2 Insider Threat Dataset.

The goal of this semester is to:
- build a reproducible modeling pipeline
- perform feature optimization
- tune a high-recall and high-precision classifier
- calibrate model thresholds
- prepare for deployment in Semester 2

All modeling was performed offline on static CERT r4.2 logs.
### Preliminary Requirements:
Clone repository using `git clone <URL>`

Open project root directory using `cd Senior-Design-Insider-Threat-Detection`

### Repository Structure
```
Senior-Design-Insider-Threat-Detection/
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
│   └── 05_thresholdEvaluation.ipynb
│
├── models/
│   ├── xgb_final_model.joblib
│   └── feature_list.json
│
├── src/
│   └── utils.py
│
├── figures/
│   └── // add plots
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

### Model Comparison
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

### Model Tuning
Notebook 03_modelTuning outlines the process we used to tune our XGBoost model to the best parameters. It trains multiple XGBoost models with different hyperparameter 
settings. Model performance is evaluated on validation folds only.

RandomizedSearchCV produced the final optimized parameters:
```
learning_rate = 0.1
max_depth = 4
n_estimators = 400
scale_pos_weight = 100
```
These parameters balance underfitting/overfitting and handle extreme class imbalance without using SMOTE.

### Feature Selection
Notebook 04_featureSelection uses the tuned XGBoost model to identify and remove low-importance features. XGBoost provides built-in feature importance scores based on how much each
feature contributes to decision-tree splits. It retrains the tuned XGBoost model using only the selected 
reduced feature set. Training occurs on the training split, and evaluation is performed on the validation split. This reduced set becomes the final model used for threshold calibration.

### Threshold Evaluation 
Notebook 05_evaluation performs the final evaluation of your optimized XGBoost model on the test dataset. 

Before selecting operational thresholds, we evaluated the calibration of the model's predicted probabilities to ensure that the output score from 
the XGBoost model reflects the true likelihood of insider threat activity. This calibration process allowed us to treat the model's output as a 
reliable risk score that could be evaluated to create operational alert modes.

To determine the optimal thresholds, first load the reduced model and feature list, then generate probability scores for each sample.

A full threshold sweep from 0.0 to 1.0 is used to evaluate precision, recall, F1, and confusion matrix values at different operating points. Based on this
analysis, two thresholds were selected:

**Alert Mode (0.18):** Prioritizes recall for broad detection.

**Critical Mode (0.64):** Prioritizes precision for critical alerts.

These two modes will aid in the design of our deployed model, which will alert differently for threats based on their threshold. Lower thresholds produce early-warning alerts, while 
higher thresholds produce high-confidence, critical alerts.

# Model Testing & Evaluation
The final reduced XGBoost model was tested on the test dataset. The model generated probability scores for each sample, which were then evaluated at 
the two chosen thresholds (0.18 and 0.64). For each threshold, we computed precision, recall, F1-score, and confusion matrices. Since the test set was not 
used during training or tuning, these results reflect the true performance of the model on unseen data.

### Alert Mode Results (threshold = 0.18)
```
Precision: 0.7423  
Recall:    0.7579  
F1-score:  0.7500  
Accuracy:  0.9976  

Confusion Matrix:
[[20030   25]
 [   23   72]]
```
### Critical Mode Results (threshold = 0.64)
```
Precision: 0.9194  
Recall:    0.6000  
F1-score:  0.7261  
Accuracy:  0.9979  

Confusion Matrix:
[[20050     5]
 [   38    57]]
```
# Using the Model
The final XGBoost model and the reduced feature list are stored in the models/ directory and can be used to generate predictions on new user–day behavioral data. This will serve as the basis for the deployed runtime system in Semester 2.

### Example Usage:
```
import joblib, json
import pandas as pd

# Load model
model = joblib.load("models/xgb_final_model.joblib")

# Load the required feature list
with open("models/feature_list.json", "r") as f:
    features = json.load(f)

# Prepare new data (must contain the same features)
df = pd.read_csv("path/to/new_data.csv")
X = df[features]

# Generate probability scores
probs = model.predict_proba(X)[:, 1]

# Apply thresholds
alert_predictions = (probs >= 0.18).astype(int)
critical_predictions = (probs >= 0.64).astype(int)

print("Alert Mode Flags:", alert_predictions.sum())
print("Critical Mode Flags:", critical_predictions.sum())
```
These prediction outputs will support the alerting system that we will be deploying for Semester 2.

# So What's Next?
In Semester 2, this offline model will be integrated into a live runtime environment. Planned components include:
- a FastAPI inference service for real-time prediction
- Docker containerization for portability
- a GUI dashboard to display alerts
- real time predictions on live or simulated user activity logs
