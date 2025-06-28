# 📊 Predictive Maintenance using Machine Learning

This project builds a machine learning pipeline to predict the failure of machines in advance based on sensor readings and operating conditions. It enables proactive maintenance to avoid costly breakdowns, thus improving industrial reliability and efficiency.

---

## 📁 Dataset

The dataset used is from Predictive Maintenance Dataset on Kaggle.

**File used**: `predictive_maintenance.csv`

### 🔍 Dataset Description:

Each row in the dataset represents one machine with the following features:

| Feature Name             | Description                                           |
|--------------------------|-------------------------------------------------------|
| Product ID              | Unique identifier for the machine product             |
| Type                    | Type of machine (L, M, H)                              |
| Air Temperature [K]     | Sensor reading (in Kelvin)                            |
| Process Temperature [K] | Process sensor (in Kelvin)                            |
| Rotational Speed [rpm]  | RPM of rotating parts                                 |
| Torque [Nm]             | Torque reading                                        |
| Tool Wear [min]         | Tool usage in minutes                                 |
| Target                  | 1 = Failure, 0 = No Failure                            |
| Failure Type            | Type of failure (e.g. Heat Dissipation, etc.)         |

---

## 🚀 Project Pipeline

### 1. Data Preprocessing
- Dropped irrelevant columns like `UDI`, `Product ID`, and `Failure Type`
- One-hot encoded the `Type` column
- Cleaned column names for XGBoost compatibility

### 2. Data Splitting
- Split into training, validation, and test sets
- Used stratified splitting to maintain class distribution

### 3. Handling Imbalanced Data
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** on training data

### 4. Model Training
- Used **XGBoost Classifier** with:
  - `scale_pos_weight` to handle imbalance
  - `logloss` as evaluation metric
  - Monitored validation performance

### 5. Model Evaluation
- Metrics used:
  - Accuracy
  - Classification report
  - Confusion matrix
  - ROC Curve and AUC score
  - Feature importance plot
  - Log loss (Train vs Validation)

### 6. Saving the Model
- Saved trained model as `xgb_model.pkl` using `joblib`

---

## ✅ Output Example

**Model Accuracy**: `97%`

| Failure_Probability | Prediction | Risk_Level |
|---------------------|------------|------------|
| 0.83                | 1          | High       |
| 0.51                | 1          | Medium     |
| 0.02                | 0          | Low        |

---

## 🛠 How to Use This Project

### Install the requirements:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn shap joblib
```

### Train and Save the Model:
- Run the training script to build the model.
- Model is saved as `xgb_model.pkl`.

---

# 🔧 Predictive Maintenance – Failure Risk Prediction Script

This script loads the trained model and predicts failure probability from new incoming machine data.

## 📦 Requirements
```bash
pip install pandas joblib
```

## 📂 Files
- `xgb_model.pkl`: Pre-trained XGBoost model.
- `incoming_machine_data.csv`: New sensor data for prediction (must be preprocessed like training data).

## 🧠 What the Script Does

### ✅ Step-by-step:
1. **Load model**: Load the saved XGBoost model from disk.
2. **Load data**: Load new machine data (in `.csv` format).
3. **Clean columns**: Clean column names for compatibility.
4. **Predict**:
   - Calculate failure probability using `predict_proba`.
   - Get class prediction (0/1).
5. **Assign Risk Level**:
   - Use quantiles of probability to categorize into:
     - Low
     - Medium
     - High

### 🧪 Sample Output:
| Failure_Probability | Prediction | Risk_Level |
|---------------------|------------|------------|
| 0.0003              | 0          | Low        |
| 0.5231              | 1          | High       |
| 0.2749              | 1          | Medium     |

---

## ⚠️ Notes
- Make sure the input data is preprocessed like the training data.
- This script does not retrain the model—just performs inference.

---

## 📈 Use Case
- Helps **industries** monitor machine health.
- Enables **proactive maintenance** and reduces unplanned downtime.
- Can be integrated into dashboards or alerting systems.

---

## 📄 License

This project is open-source and free to use for educational or non-commercial purposes.
