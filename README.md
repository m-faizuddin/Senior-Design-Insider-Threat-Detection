## AI-Based Insider Threat Detection
### Mohammed Faizudden, Anna Wille, Maya Wyganowska

# Offline XGBoost Model
This repository contains the Semester 1 deliverables for the CS492 AI-Based Insider Threat Detection project:
a complete offline machine-learning pipeline for detecting insider threats using the CERT r4.2 Insider Threat Dataset.

The goal of this semester is to:
- build a reproducible modeling pipeline
- perform feature optimization
- tune a high-recall and high-precision classifier
- calibrate model thresholds
- prepare for deployment in Semester 2

All modeling was performed offline on static CERT r4.2 logs.

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
```
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
Launch Jupyter Notebook:
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

The script outputs a structured `features.csv` file that is used as input to the offline XGBoost modeling pipeline.

### Preprocessing
Notebook 01_preprocessing outlines the process we used to clean and split the extracted data. 

This step:
- loads features.csv
- cleans unused fields
- identifies feature columns
- creates train/validation/test splits
- saves processed files for modeling

### Model Comparison
Notebook 02_modelComparison outlines the process we used to select our best model. In models that struggle with heavy class imbalances, we implemented a SMOTE pipeline to create more synthetic insider threats and see if a more balanced dataset could produce more accurate results. We trained these models using 5-fold Stratified Cross-Validation on the training dataset. No test data is used at this stage.

We evaluated four baseline models:
- Logistic Regression
- Linear SVM
- Random Forest
- XGBoost

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

The final reduced XGBoost model was tested on the test dataset. The model generated probability scores for each sample, which were then evaluated at 
the two chosen thresholds (0.18 and 0.64). For each threshold, we computed precision, recall, F1-score, and confusion matrices. Since the test set was not 
used during training or tuning, these results reflect the true performance of the model on unseen data.

### Alert Mode Results
```
Precision: 0.7423  
Recall:    0.7579  
F1-score:  0.7500  
Accuracy:  0.9976  

Confusion Matrix:
[[20030   25]
 [   23   72]]
```
### Critical Mode Results
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
