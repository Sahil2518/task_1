# ML Pipeline Project

## Overview
A machine learning pipeline demonstrating preprocessing, training, and evaluation.

## Dataset
Iris Dataset from scikit-learn.

## Preprocessing
- Missing value imputation
- Feature scaling
- Categorical encoding
- ColumnTransformer
- Pipeline

## Model 
- Baseline: DummyClassifier (majority class)
- First Model: DecisionTreeClassifier

## Data Split
- Train / Validation / Test (60% / 16% / 20%)
- Stratified splits with fixed seed (42)

## Run

python -m src.train

## Outputs
- models/model.pkl
- models/preprocessor.pkl
- logs/results.csv