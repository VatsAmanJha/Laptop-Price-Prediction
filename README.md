# Laptop Price Prediction

This repository contains code and data for predicting laptop prices using various machine learning models. The goal is to compare different regression models and select the best one based on performance metrics such as R-squared (R2) and Mean Absolute Error (MAE).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)

## Introduction

This project aims to predict the prices of laptops using features like brand, processor, RAM, storage, etc. Various machine learning models are trained and evaluated to find the most accurate one.

## Dataset

The dataset used in this project contains information about various laptops along with their prices. The features include:

- Brand
- Processor
- RAM
- Storage
- Screen Size
- Operating System
- Weight
- ... and more

## Models Used

The following regression models are trained and evaluated in this project:

- Linear Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Support Vector Regressor
- Random Forest Regressor

## Evaluation Metrics

The models are evaluated based on the following metrics:

- **R-squared (R2)**: Indicates how well the model explains the variance in the target variable.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in predictions.

## Results

The performance of each model is as follows:

| Model Name               | R2 Score | Mean Absolute Error |
|--------------------------|----------|---------------------|
| Linear Regression        | 0.808    | 0.210               |
| K-Neighbors Regressor    | 0.805    | 0.193               |
| Decision Tree Regressor  | 0.845    | 0.181               |
| Support Vector Regressor | 0.819    | 0.198               |
| Random Forest Regressor  | 0.886    | 0.155               |

The **Random Forest Regressor** performed the best with an R-squared of 0.886 and an MAE of 0.155.

## Usage

To use the model for predicting laptop prices:

1. Load the saved model using `pickle`.
2. Prepare the input data in the same format as the training data.
3. Use the model to predict prices.

```python
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example prediction
example_data = [...]  # Replace with your input data
predicted_price = model.predict([example_data])
print(f"Predicted Laptop Price: {predicted_price}")
```

## Installation

1. Clone this repository:

```sh
git clone https://github.com/VatsAmanJha/Laptop-Price-Prediction.git
```

2. Install the required packages:

```sh
pip install -r requirements.txt
```

3. Run the Jupyter Notebook or Python script to train and evaluate the models.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.
