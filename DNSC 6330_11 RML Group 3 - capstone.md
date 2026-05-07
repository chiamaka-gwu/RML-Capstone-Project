# Responsible ML Audit of HMDA Loan Approval Model Group 3

This project was completed as part of our Responsible Machine Learning capstone. Our group worked on building and auditing a machine learning model that predicts whether a mortgage loan application is approved or denied using HMDA loan application data.

Instead of only focusing on model accuracy, we treated this as a responsible ML audit. Since loan approval is a high-stakes decision that can affect access to credit, we evaluated the model for performance, fairness, explainability, robustness, and remaining deployment risks.

## Project Goal

The main goal of this project was to audit a loan approval model and answer a practical question:

	⁠Can a machine learning model predict loan approval decisions while still being fair, explainable, and defensible for use in a high-stakes financial setting?

To answer this, we built a classification model, tested it across racial groups, identified possible bias risks, applied mitigation, and documented the risks that still remained.

## Dataset

We used 2024 HMDA loan application data. The full dataset had more than 12 million records, so we used DuckDB and chunk-based processing to handle the data without running into memory issues.

For modeling, we worked with a stratified sample of the data so that the approved and denied loan classes were represented properly.

### Target Variable

We converted the HMDA ⁠ action_taken ⁠ column into a binary label:

•⁠  ⁠⁠ 1 ⁠ or ⁠ 2 ⁠ → Approved  
•⁠  ⁠⁠ 3 ⁠ → Denied  

Other action values were removed because they did not directly answer our approval vs denial question.

## Why This Problem Matters

Loan approval models can look accurate overall but still create unfair outcomes for specific groups. Because mortgage lending is connected to financial access, housing opportunity, and fair lending laws, we wanted to check more than just predictive performance.

Our audit focused on:

•⁠  ⁠Whether the model performs well overall
•⁠  ⁠Whether approval rates differ across racial groups
•⁠  ⁠Whether some groups experience higher false negative rates
•⁠  ⁠Whether financial variables act as proxies for protected attributes
•⁠  ⁠Whether the model remains stable under simulated distribution shifts

## Tools and Libraries Used

•⁠  ⁠Python
•⁠  ⁠DuckDB
•⁠  ⁠Pandas
•⁠  ⁠NumPy
•⁠  ⁠Scikit-learn
•⁠  ⁠XGBoost
•⁠  ⁠SHAP
•⁠  ⁠Matplotlib
•⁠  ⁠Seaborn
•⁠  ⁠Google Colab

## Project Workflow

### 1. Data Loading and Cleaning

We first explored the full HMDA dataset using DuckDB because the dataset was too large to load directly into memory. After that, we created a stratified sample and cleaned the data.

Main cleaning steps included:

•⁠  ⁠Filtering valid loan approval/denial labels
•⁠  ⁠Creating the binary target variable
•⁠  ⁠Replacing HMDA sentinel values such as ⁠ Exempt ⁠, ⁠ 1111 ⁠, and ⁠ 9999 ⁠ with missing values
•⁠  ⁠Imputing missing numerical values using the median
•⁠  ⁠Imputing missing categorical values using the most frequent value
•⁠  ⁠Removing features that could create leakage or unfairness concerns
•⁠  ⁠Keeping protected attributes separately only for fairness auditing

Protected attributes were not used as model input features.

### 2. Model Training

We trained and compared two models:

•⁠  ⁠Random Forest baseline
•⁠  ⁠XGBoost Gradient Boosting model

Random Forest performed better in terms of balanced performance, but XGBoost was selected as the main model for the audit because it works well for structured/tabular data, supports class imbalance handling, and is easier to explain using SHAP.

The class distribution was imbalanced, so we used ⁠ scale_pos_weight ⁠ in XGBoost to reduce the chance of the model ignoring the denied class.

## Model Performance

The models were evaluated using:

•⁠  ⁠Macro F1
•⁠  ⁠ROC-AUC
•⁠  ⁠Balanced accuracy
•⁠  ⁠F1 score for the denied class
•⁠  ⁠Confusion matrix

From our results, Random Forest was more balanced, while XGBoost tended to favor approvals. This was important because over-approving may look good from an accuracy perspective, but it can hide weaknesses in detecting denied applications.

## Explainability

We used SHAP to understand which features had the strongest influence on model predictions.

The most important features included:

•⁠  ⁠Loan amount
•⁠  ⁠Property value
•⁠  ⁠Income
•⁠  ⁠Debt-to-income ratio

These features are financially meaningful, but they can also act as proxy variables for race or other protected characteristics. Because of that, we treated them as proxy bias risks and included them in the risk documentation.

## Fairness Audit

We evaluated fairness mainly across race groups using:

•⁠  ⁠Adverse Impact Ratio (AIR)
•⁠  ⁠False Positive Rate (FPR)
•⁠  ⁠False Negative Rate (FNR)
•⁠  ⁠Intersectional analysis using race and sex

The AIR threshold used in the project was ⁠ 0.80 ⁠. Any group below this threshold would raise fair lending concerns.

After mitigation, all eight race groups passed the AIR ≥ 0.80 threshold. The lowest AIR was still above the threshold, but we also noted that smaller subgroups were less stable because of limited sample size.

## Mitigation Strategy

We used threshold adjustment as the mitigation strategy.

Instead of retraining the model, we tested decision thresholds from ⁠ 0.30 ⁠ to ⁠ 0.60 ⁠ and selected the threshold that gave the best balance between fairness and performance.

The final selected threshold was:

```text
0.60