End-to-End Machine Learning with Scikit-Learn

Project Overview
This project demonstrates a comprehensive, end-to-end Machine Learning workflow using the Scikit-Learn library in Python. It covers the two primary branches of supervised learning: Classification and Regression. The focus is on writing clean, production-ready code by utilizing Pipelines, handling missing data, and performing rigorous model evaluation.

Technologies & Libraries
Python 3: The core programming language.
Pandas & NumPy: For data manipulation, exploration, and numerical operations.
Matplotlib: For data visualization and plotting evaluation curves.
Scikit-Learn: The primary framework for building and evaluating ML models.
Joblib: For efficient model persistence (exporting and importing trained models).

Datasets Used
The project utilizes two real-world datasets:
1. Heart Disease Dataset (`heart-disease.csv`): Used for classification to predict whether a patient has heart disease based on medical attributes.
2. Car Sales Dataset (`car-sales-extended-missing-data.csv`): Used for regression to predict car sale prices. This dataset specifically includes missing values to demonstrate data imputation techniques.

Key Learning Paths & Workflows

1. Classification Workflow
Model Comparison: Evaluated multiple algorithms including `RandomForestClassifier`, `LogisticRegression`, `SVC`, and `KNeighborsClassifier`.
Hyperparameter Tuning: Optimized the `LogisticRegression` model using `RandomizedSearchCV` to find the best configuration automatically.
Advanced Evaluation: Moved beyond simple accuracy to include:
    Confusion Matrix: Visualized using `ConfusionMatrixDisplay`.
    Classification Report: Precision, Recall, and F1-Score (Achieved ~87% accuracy on test data).
    ROC Curve & AUC: Plotted performance trade-offs using `RocCurveDisplay`.
Cross-Validation: Implemented `cross_val_score` to ensure model reliability across different data splits.

2. Regression Workflow
Data Preprocessing Pipeline: Built a robust `ColumnTransformer` to handle different data types simultaneously:
    Categorical Features: Used `SimpleImputer` and `OneHotEncoder` for "Make" and "Colour".
    Numerical Features: Imputed missing values using the **Median** for odometer readings.
Regression Modeling: Tested various regressors such as `Ridge`, `SVR`, and `RandomForestRegressor`.
Error Metrics: Evaluated financial predictions using:
    Mean Absolute Error (MAE): To understand the average dollar-amount deviation.
    Mean Squared Error (MSE): To penalize larger outliers.
    R^2 Score: Achieved a baseline coefficient of determination of ~0.254.

3. Model Persistence
Export/Import: Successfully saved the trained classifier using `joblib.dump()` and reloaded it via `joblib.load()` to perform predictions on new data without re-training
