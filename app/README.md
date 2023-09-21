# UTT AMIS NAV FORECASTING APP

## Introduction
This README file provides instructions on how to use the project's Streamlit app, including details on the app's structure and the steps to run it locally.

## Project Structure
The project has the following structure:

- `requirements.txt`: Contains a list of required Python packages and dependencies.
- `app` (folder): Contains the main application files.
  - `app.py`: The main Python script to run the Streamlit app.
  - `assets` (folder): Contains images and other assets such as JSON files.
  - `notebooks` (folder): Contains Jupyter notebooks used in the project.
  - `data` (folder): Contains datasets used in the project.
  - `models` (folder): Contains saved machine learning models.
  - `preprocessing` (folder): Contains modules for data preprocessing.
  - `documents` (folder): Contains project documents and resources.

## Running the App
To run the Streamlit app, follow these steps:

1. Install the required packages using `pip` by running the following command in your terminal:
   ```
   pip install -r requirements.txt
   ```

2. Navigate to the `app` folder using the command:
   ```
   cd app
   ```

3. Run the Streamlit app using the following command:
   ```
   streamlit run app.py
   ```

The app will start locally, and you can access it through the following URLs:

- Local URL: [http://localhost:8501](http://localhost:8501)
- Network URL: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## App Pages
The app consists of the following pages:

1. **Home Page**: Provides an introduction to the project and its objectives.

2. **Dashboard Page**: Allows you to view visualizations and data insights. Explore charts and graphs related to the project's datasets.

3. **Prediction Page**: Use this page to predict the Net Asset Value (NAV) per unit. Provide input data, and the app will make predictions based on the selected fund.

4. **NAV Prediction Page**: Predict the Net Asset Value (NAV) using this page. Input the required features, and the app will provide predictions.

## Additional Information
For any further assistance or inquiries, please refer to the project documents located in the `documents` folder. You can also explore the Jupyter notebooks in the `notebooks` folder for detailed analysis and code explanations.
