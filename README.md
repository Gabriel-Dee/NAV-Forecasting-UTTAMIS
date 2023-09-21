# Net Asset Value (NAV) Forecasting Project

![Project Logo](project_logo.png)

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Time Series Forecasting](#time-series-forecasting)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Welcome to the Net Asset Value (NAV) Forecasting project! This project aims to forecast the NAV of various investment funds managed by UttaMIS, providing valuable insights for investors and financial institutions.

## Project Overview
- **Problem Statement:** Predict the future NAV of investment funds.
- **Dataset:** We utilize historical NAV data for multiple funds.
- **Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Statsmodels.

## Getting Started
To get started with this project, follow these steps:
1. Clone the repository: `git clone https://github.com/your-username/net-asset-value-forecast.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Explore the project's Jupyter notebooks in the `notebooks` directory.

## Data Preparation
Before analysis, the dataset is cleaned, missing values are handled, and features are selected. This ensures a high-quality dataset for modeling.

## Exploratory Data Analysis
- Gain insights into the historical NAV data.
- Visualize trends, seasonality, and anomalies in the data.
- Identify key features influencing NAV.

## Time Series Forecasting
- Utilize Time Series models like ARIMA for NAV forecasting.
- Tune model hyperparameters for accurate predictions.
- Evaluate model performance using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Results and Visualizations
- Present the forecasted NAV values against actual values.
- Visualize key trends and seasonal components for each fund.
- Interpret model performance and provide insights for investors.

## Feature Engineering

- Additional features such as market indices, economic indicators, and temporal information (Year, Month, Day, Weekday) were engineered to capture the complex relationships within the data.
- Logarithmic transformation of the "ONU" feature was applied to normalize its distribution.

## Data Splitting

- The dataset was split into training and testing sets (80% training and 20% testing) using the `train_test_split` function.

## Model Selection

Several models were explored to find the best fit for the NAV forecasting task. These models included:

1. **LSTM (Long Short-Term Memory)**
   - LSTM is a deep learning model designed for sequence prediction tasks.
   - It was considered due to its capability to capture complex temporal dependencies.

2. **ARIMA (AutoRegressive Integrated Moving Average)**
   - ARIMA, a traditional time series model, served as a baseline for comparison.

3. **SARIMA (Seasonal ARIMA)**
   - SARIMA was applied to account for seasonality in the data.

4. **Ensemble Model (Lasso Regression + Ridge Regression)**
   - An ensemble model combining Lasso and Ridge Regression was selected for its ability to handle feature selection and regularization simultaneously.

## Model Training and Evaluation

### Ensemble Model

- Lasso Regression and Ridge Regression models were trained with the best hyperparameters determined through experimentation (Lasso Alpha = 0.01, Ridge Alpha = 0.1).
- The ensemble model, combining these two models using the `VotingRegressor`, was trained on the training dataset.

### Model Evaluation

- The model's performance was evaluated using key metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)

- **Mean Absolute Percentage Error (MAPE)**
  - MAPE was calculated to assess the percentage error in predicting NAV values.

- **Sharpe Ratio and Annualized Returns**
  - Daily returns were calculated from the predicted NAV values, and annualized returns were computed to gauge investment performance.
  - The Sharpe ratio was used to measure the risk-adjusted returns.

## Results and Findings

After extensive experimentation and evaluation, the ensemble model combining Lasso Regression and Ridge Regression exhibited superior performance:

- It outperformed other models in terms of forecasting accuracy, as evidenced by lower MAE, MSE, and RMSE.
- The model achieved higher R² values, indicating a better fit to the data.
- Its robustness in handling feature selection and regularization made it the preferred choice.

## Conclusion

The model development phase for forecasting Net Asset Values of UTT AMIS schemes involved a systematic approach to data preparation, model selection, and evaluation. The ensemble model combining Lasso and Ridge Regression proved to be a reliable and accurate solution for NAV forecasting across all six schemes. This approach ensures that UTT AMIS can make informed investment decisions, mitigate financial risk, and maximize returns for its investors.

Continued monitoring and periodic retraining of the model will be essential to adapt to evolving market dynamics and ensure the accuracy and relevance of predictions. This model development effort exemplifies the significance of data-driven decision-making in the financial industry, benefiting both fund managers and investors.


## Contributing
We welcome contributions to this project. If you find any issues or have ideas for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
