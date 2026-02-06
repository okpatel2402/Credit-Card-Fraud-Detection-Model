# Credit-Card-Fraud-Detection-Model
This project implements an end-to-end machine learning system for detecting credit card fraud on a highly imbalanced dataset (~2.7M transactions). The system focuses on minimizing false negatives while maintaining strong overall predictive performance.. Models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC

## Problem Statement
Financial fraud detection is a binary classification problem where fraudulent transactions represent less than 1% of total records. Traditional accuracy-based evaluation is misleading in such scenarios, as a model can achieve over 99% accuracy while failing to detect fraud entirely.

**Goal:**  
Detect fraudulent transactions effectively by prioritizing Recall, ROC-AUC, and Precision-Recall trade-offs rather than raw accuracy.

---

## Dataset
- Transaction-level financial dataset
- Highly imbalanced:
  - Fraudulent transactions: < 1%
  - Legitimate transactions: > 99%
- Features include anonymized transaction attributes and transaction amount
- Target variable:
  - `0` – Legitimate
  - `1` – Fraudulent

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Verified extreme class imbalance
- Analyzed transaction amount distributions
- Identified outliers using Z-score analysis
- Checked for missing values and data consistency

### 2. Data Preprocessing
- Feature scaling for numerical variables
- Outlier analysis on transaction amounts
- Train-test split with stratification
- Addressed class imbalance using:
  - Class weighting
  - SMOTE (Synthetic Minority Over-sampling Technique)

---

## Models Implemented
The following supervised learning models were trained and evaluated:

- Logistic Regression (Baseline)
- Random Forest Classifier
- Gradient Boosting / XGBoost
- SMOTE + Random Forest (Final Model)

Each model was tuned with recall-oriented objectives rather than default probability thresholds.

---

## Evaluation Metrics
Due to class imbalance, the following metrics were prioritized:

- Recall (Fraud Detection Rate)
- ROC-AUC
- Precision
- F1-score
- Confusion Matrix Analysis

Accuracy was explicitly not used as the primary decision metric.

---

## Results Summary

### Key Observations
- Standard models achieved ~99.9% accuracy but near 0% recall, missing almost all fraud cases
- Class-weighted models improved recall to approximately 70–72%
- SMOTE-based models significantly improved minority class learning

### Best Performing Model
**SMOTE + Random Forest**

- Fraud Recall: ~89%
- ROC-AUC: Strong class separability
- False positives increased but remained within acceptable operational limits

---

## Model Comparison (High-Level)

| Model | Fraud Recall | Key Insight |
|------|-------------|------------|
| Logistic Regression | Low | Poor minority detection |
| Random Forest (Weighted) | ~70% | Conservative but stable |
| Gradient Boosting | Slightly lower than RF | Struggles with extreme imbalance |
| **SMOTE + Random Forest** | **~89%** | Best trade-off between recall and false positives |

---

## Final Conclusion
- Fraud detection requires recall-first optimization
- Accuracy alone is misleading for imbalanced datasets
- SMOTE combined with Random Forest provides the best balance between fraud detection and operational feasibility
- The final model substantially increases fraud detection while maintaining a manageable false alarm rate

This approach is suitable for deployment in real-world fraud monitoring systems where missing fraud is more costly than reviewing false alerts.

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

---

## Author
**Om Patel**  
MS in Data Science  
GitHub: https://github.com/okpatel2402  
LinkedIn: https://linkedin.com/in/okpatel
