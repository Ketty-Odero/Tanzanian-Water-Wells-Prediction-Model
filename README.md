# Tanzanian-Water-Wells-Prediction-Model
This repository builds a model to predict functionality of pumps using the Tanzanian Water Wells dataset

# 📖 Overview
This project analyzes Tanzanian water wells to assess their functionality and identify key factors affecting well performance. Using a dataset of over 59,000 water points, we classify wells as:
Functional
Non-functional
Functional but needs repair
The goal is to provide actionable insights to improve water resource management in Tanzania and prioritize repairs or replacements.

# Project Structure
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
