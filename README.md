# Diabetes Prediction Project

## Overview

This project uses a Support Vector Machine (SVM) model to predict whether a person has diabetes based on various health features. The dataset used for training the model is the [Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) which includes information about pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age.

## Project Structure

1. **Data Preprocessing**
    - Load the dataset
    - Analyze the data
    - Handle missing values (if any)
    - Standardize the features

2. **Model Training**
    - Split the data into training and testing sets
    - Train an SVM classifier with a linear kernel

3. **Evaluation**
    - Evaluate the model on both training and testing data
    - Calculate accuracy scores

4. **Prediction**
    - Create a function to make predictions based on user input

## Requirements

- Python 3.6
- `numpy`
- `pandas`
- `scikit-learn`

## Installation

To install the necessary libraries, use the following command:

```bash
pip install numpy pandas scikit-learn
