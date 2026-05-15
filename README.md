# 🌿 Leaf Classification with Machine Learning Pipelines

A complete machine learning experimentation pipeline for **leaf classification** using multiple models, feature selection techniques, nested cross-validation, statistical testing, and automated performance visualization.

This project goes beyond simple model training by building a **research-style evaluation framework** that compares different machine learning algorithms under controlled experimental conditions.

---

## 🚀 Features

- ✅ Multiple ML Models
  - Multi-Layer Perceptron (MLP) - Deep learning-based classification
  - Random Forest (RF) - Ensemble tree-based learning
  - K-Nearest Neighbors (KNN) -  Distance-based classification
  - Support Vector Machine (SVM) - Kernel Based classification
  - XGBOOST - Gradient Boosting   

- ✅ Feature Selection Support
  - Recursive Feature Elimination (RFE)
  - Full feature comparison

- ✅ Proper ML Pipeline Design
  - Prevents data leakage using `Pipeline`
  - Automatic scaling where necessary

- ✅ Nested Cross-Validation
  - Inner CV for hyperparameter tuning
  - Outer CV for unbiased evaluation

- ✅ Hyperparameter Optimization
  - Uses `GridSearchCV`

- ✅ Statistical Evaluation
  - ANOVA test
  - Paired t-tests

- ✅ Automatic Visualization
  - Performance bar plots
  - Box plots
  - Confusion matrix

- ✅ Exported Results
  - CSV summaries
  - Raw fold scores
  - Statistical reports
  - Saved plots

---

## ⚙️ Experimental Workflow

1. Load dataset  
2. Apply feature selection  
3. Build preprocessing pipeline  
4. Tune hyperparameters with GridSearchCV  
5. Evaluate using nested cross-validation  
6. Compare models statistically  
7. Generate plots and reports automatically  

---

## 📊 Outputs Generated

The script automatically saves:

- Cross-validation scores
- Performance summary table
- Confidence intervals
- Statistical significance tests
- Confusion matrix
- Bar charts & boxplots

All generated outputs are stored inside the `outputs/` folder.

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- SciPy

---
🔬 Key Highlight

Unlike basic ML projects, this implementation uses nested cross-validation, which provides a more reliable estimate of real-world model performance and reduces the risk of overfitting during hyperparameter tuning.

📌 Future Improvements
SHAP explainability
Automated experiment tracking

👨‍💻 Author

Built as part of a machine learning experimentation and evaluation study focused on robust classification workflows and reproducible AI research.
