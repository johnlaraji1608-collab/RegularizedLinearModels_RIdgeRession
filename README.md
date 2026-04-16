# RegularizedLinearModels_RIdgeRession
# 📊 Ridge Regression Model

## 📌 Overview

This project implements **Ridge Regression**, a regularized version of linear regression that helps reduce overfitting by adding a penalty to large coefficients.

The model is trained and evaluated using cross-validation to ensure reliable performance.

---

## 🎯 Objective

* Build a regression model using Ridge Regression
* Reduce overfitting in linear models
* Find the optimal value of **alpha (regularization strength)**
* Evaluate model performance using RMSE

---

## 🧠 What is Ridge Regression?

Ridge Regression adds an **L2 penalty** to the loss function:

[
\text{Loss} = \text{MSE} + \alpha \sum (w^2)
]

* **MSE** → Mean Squared Error
* **α (alpha)** → Regularization parameter
* Penalizes large weights
* Improves generalization

---

## ⚙️ Data Preprocessing

* Removed rows with missing target values
* Converted categorical data using **One-Hot Encoding**
* Filled missing values with 0
* Applied feature scaling
* Applied log transformation (optional)

---

## 🔍 Model Implementation

* Used `Ridge` from `sklearn.linear_model`
* Tested multiple values of alpha
* Selected the best alpha based on performance

---

## 🔄 Cross Validation

* Used **5-Fold Cross Validation**
* Ensures model works well on unseen data

---

## 🧪 Hyperparameter Tuning

```python
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
```

* Choose alpha with lowest RMSE

---

## 📏 Evaluation Metric

* **RMSE (Root Mean Squared Error)**
* Lower RMSE = Better model

---

## 💻 Implementation (Python)

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error",
                                    cv=5))
    return rmse

model = Ridge(alpha=0.1)
score = rmse_cv(model, X, y).mean()

print("RMSE:", score)
```

---

## 🚀 Results

* Reduced overfitting
* Improved model stability
* Found optimal alpha value

---


### ✅ Solution:

* Ensure dataset is not empty
* Handle missing values
* Convert categorical data to numeric
* Check X and y shapes

---

## 📌 Conclusion

Ridge Regression is a simple and powerful technique for improving linear models by controlling overfitting and stabilizing predictions.
---
