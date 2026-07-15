# Churn-Prediction-Naive-Bayes-vs-Tuned-Decision-Tree
Predicting telecom customer churn using two methods.

## Customer Churn Classification - GaussianNB

Predicting telecom customer churn and identifying the factors that drive it,
using Gaussian Naive Bayes with permutation-importance analysis for the drivers.

### Research Question

Which factors most influence customer churn, and can churn be predicted
accurately enough to support retention decisions?

### Dataset

A 10,000-record telecom customer dataset with a 26.5% churn rate. The data file
(`churn_clean.csv`) is not included in this repository; place your own copy in
the project root before running.

### Approach

- **Preprocessing:** mapped binary service fields to 0/1, collapsed raw timezones
  into standard US zones, one-hot encoded categorical predictors.
- **Model:** Gaussian Naive Bayes on 16 selected predictors.
- **Split:** stratified 80/20 to preserve the churn rate on imbalanced data.
- **Driver analysis:** permutation importance for magnitude, a logistic-regression
  companion for direction of effect.
- **Evaluation:** ROC AUC, precision, recall, F1, and a confusion matrix.

### Results

- **ROC AUC: 0.89**
- **Recall: 0.87** — the model catches 87% of customers who actually churn
- Accuracy 0.76, which is only modestly above the 0.735 majority-class baseline,
  so AUC and recall are the metrics that matter here, not accuracy

### Key Findings

- **Tenure** is the strongest driver, and longer tenure lowers churn. New and
  short-tenure customers are the highest-risk group.
- **Higher monthly charges** raise churn.
- **One- and two-year contracts** lower churn substantially versus month-to-month.
- Streaming add-ons show a slight *positive* association with churn, so promoting
  them as a retention tool is not supported by this data.

### Recommendations

- Focus retention on the first year of tenure, where risk is highest.
- Review pricing and value for high monthly-charge segments.
- Incentivize month-to-month customers toward longer contracts.

### Tech Stack

Python, pandas, scikit-learn, matplotlib

### Running It

1. Place `churn_clean.csv` in the project root.
2. Install dependencies: `pip install pandas scikit-learn matplotlib`
3. Open `churn-classification.ipynb` and run all cells.

### Updates

- **2026-07:** Modernized for pandas 3.x, added permutation-importance driver
  analysis, and expanded evaluation with precision, recall, F1, and a confusion
  matrix.

## Customer Churn Prediction: Decision Tree

Predicting telecom customer churn and identifying the factors that drive it,
using a cross-validated decision tree classifier.

### Research Question

Which factors most influence customer churn, and can churn be predicted
accurately enough to support retention decisions?

### Dataset

A 10,000-record telecom customer dataset with a 26.5% churn rate. The data file
(`churn_clean.csv`) is not included in this repository; place your own copy in
the project root before running.

### Approach

- **Preprocessing:** mapped binary service fields to 0/1, collapsed raw timezones
  into standard US zones, one-hot encoded categorical predictors.
- **Model:** decision tree on 16 selected predictors, with depth and leaf size
  tuned by 5-fold cross-validated grid search to control overfitting.
- **Split:** stratified 80/20 to preserve the churn rate on imbalanced data.
- **Driver analysis:** the tree's built-in feature importances rank the drivers.
- **Evaluation:** ROC AUC, precision, recall, F1, and a confusion matrix.

### Results

- **ROC AUC: 0.94**
- **Accuracy: 0.89**, with balanced precision and recall (0.79 each)
- Tuning was the key step: an untuned tree overfit badly (perfect training
  accuracy, 0.80 test AUC). Cross-validated depth and leaf-size limits lifted
  test AUC to 0.94.

### Key Findings

- **Tenure** and **monthly charge** are the dominant drivers, together accounting
  for roughly three-quarters of the model's splitting decisions.
- **Contract length** and **fiber-optic service** are the next most influential.
- Longer tenure and longer contracts are associated with lower churn; higher
  monthly charges with higher churn.

### Recommendations

- Focus retention on the first year of tenure, where risk is highest.
- Review pricing and value for high monthly-charge segments.
- Incentivize month-to-month customers toward longer contracts.

### Tech Stack

Python, pandas, scikit-learn, matplotlib

### Running It

1. Place `churn_clean.csv` in the project root.
2. Install dependencies: `pip install pandas scikit-learn matplotlib`
3. Open the notebook and run all cells.

### Updates

- **2026-07:** Modernized for pandas 3.x, tuned the tree with cross-validation to
  fix overfitting (test AUC 0.80 to 0.94), added feature-importance driver
  analysis, and expanded evaluation with precision, recall, F1, and a confusion
  matrix.
