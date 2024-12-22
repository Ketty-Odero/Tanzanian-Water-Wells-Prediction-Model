# Tanzanian-Water-Wells-Prediction-Model
This repository builds a model to predict functionality of pumps using the Tanzanian Water Wells dataset

## Overview
This project analyzes Tanzanian water wells to assess their functionality and identify key factors affecting well performance. Using a dataset of over 59,000 water points, we classify wells as:

- Functional
- Non-functional
- Functional but needs repair

The goal is to provide actionable insights to improve water resource management in Tanzania and prioritize repairs or replacements.

## Project Objectives

- Understand factors contributing to well performance (geographic, population, funding, etc.).
- Predict well status using machine learning models.
- Optimize model performance for binary classification (Functional vs. Non-functional/Needs Repair).
- Provide data-driven recommendations for stakeholders.

## Project Structure
Tanzanian-Water-Wells-Prediction-Model/
├── data/
│   ├── wells_data_cleaned.csv        # Cleaned dataset
│   ├── validation_data_cleaned.csv   # Validation dataset
├── modules/
│   ├── dataprocessor.py              # Preprocessing class for training data
│   ├── testprocessor.py              # Preprocessing class for validation data
│   ├── EDA.py                        # Exploratory Data Analysis helper functions
├── notebooks/
│   ├── 1.0 Data Preparation.ipynb    # Data cleaning and preprocessing
│   ├── 2.0 Exploratory Analysis.ipynb # Exploratory data analysis (EDA)
│   ├── 3.0 Model Training.ipynb      # Machine learning model training
│   ├── 4.0 Validation & Results.ipynb # Model evaluation and validation
├── images/                           # Visualizations and figures
├── README.md                         # Project documentation
└── columns.json                      # Reference columns from training data
└── index.ipynb                       # Final jupyter notebook

#### Technologies Used
- Programming: Python, IDE : Jupyter Notebooks
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Visualization: Heatmaps, Confusion matrix and Stacked bar charts
- Version Control: Git, GitHub

## Analysis Steps
####  Data Cleaning:

- Handled missing values for features like funder, installer, permit, etc.
- Dropped irrelevant features like id and date_recorded.
#### Exploratory Data Analysis:

- Investigated geographic, demographic, and operational factors.
- Visualized relationships using stacked bar charts and heatmaps.
#### Feature Engineering:

- Encoded categorical variables with label encoding and one-hot encoding.
#### Modeling:

- *Dummy Classifier*: Achieved 50% accuracy with stratified predictions.
- Machine Learning Models: *Tested Logistic Regression*, *Decision Tree*, and *Random Forest*.
- Hyperparameter Tuning: Optimized **Random Forest with GridSearchCV**.
#### Validation:
- Preprocessed validation data to align with training features.
- Evaluated model predictions with the test data.

## Key Results
- Best Model: **Tuned Random Forest**
- Testing Accuracy: **82.1%**
- Cross-Validation Score: **81.6%**
Important Features:
- *Longitude*, *Latitude*, *GPS Height*, *Construction Year*, *Population*

#### **Recommendation**
- Focus on class 1 improvement by using advanced techniques such class weighting to improve on dataset balance.
- Further analyze feature importance to actually understand which variable are really driving predictions.
- Explore advanced ensemble methods such as boosting, inorder to optimize models ability to predict minority class.
- Data quality: Engage with key stakeholders to ensure quality data is collected, to ensure proper handling of outliers or missing values in future with data quality controls put in place.
- Deployment and Monitoring: Develop a robust system for deploying the model into a production environment and monitor its performance over time.



