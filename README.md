# Tanzanian-Water-Wells-Prediction-Model
This repository builds a model to predict functionality of pumps using the Tanzanian Water Wells dataset

## Business Understanding
### Overview

Tanzania is in the midst of a crisis, out of its 65 million population, 55%  and 11% of rural and urban population respectively, do not have access to clean water. People living under these circumstances, particularly women and girls, spend a significant amount of time traveling long distances to collect water.This poses significant risks in public health, economic productivity and educational opportunities. Now more than everaccess to safe water at home is critical to families in Tanzania.

This project analyzes Tanzanian water wells to assess their functionality and identify key factors affecting well performance. Using a dataset of over 59,000 water points, we classify wells as:

- Functional
- Non-functional
- Functional but needs repair

The goal is to provide actionable insights to improve water resource management in Tanzania and prioritize repairs or replacements.

## Business goals
- **Optimize resource allocation :** Predict well functionality to prioritize repairs for non-functional and poorly functioning wells . 

- **Improve community access to clean water :** Reduce repairs downtime and increase availability of functional water points

- **Support sustainability and durability of wells :** Provide insights to future installations and maintenance strategies. What factors contribute to well failures?

#### Project objectives
- Understand factors contributing to well performance (geographic, population, funding, etc.).
- Predict well status using machine learning models.
- Optimize model performance for **binary classification** (Functional vs. Non-functional/Needs Repair).
- Provide data-driven recommendations for stakeholders.

## Data source
The dataset provided on https://www.drivendata.org/ by **Taarifa** and the **Tanzanian Ministry of Water**. More details on the competition could be found [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/).
Feature description for the data can be found in [data description](data_description.txt).
The two datasets we will use are [Training set values](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/) and [Training set labels](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/)
##### **Data assumptions :**
- The dataset is representative of all wells in Tanzania
- Historical data trends will hold for future predictions

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
#### Visualizations
The following visualizations are included in this analysis:

[well status by region]
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

#### Future Work
- Integrate additional datasets (e.g.maintenance records).
- Experiment with other machine learning algorithms (e.g., XGBoost, LightGBM).



