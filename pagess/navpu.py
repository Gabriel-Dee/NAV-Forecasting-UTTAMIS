import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor

# Define the pages
PAGES = {
    "Home Page": "home_page",
    "Select Scheme": "scheme_selection",
    "Visualization": "visualization",
}

# Create a sidebar to select pages
page = st.selectbox("Navigation", list(PAGES.keys()))

# Display content based on the selected page
if page == "Home Page":
    st.title("Welcome to NAV Forecasting System")
    st.write("Unlock the Future of Your Investments!")
    
    st.image("Assets/hero.jpeg", use_column_width=True)

    st.markdown(
        """
        Welcome to the NAV Forecasting System, your gateway to informed investment decisions. 
        We empower you to anticipate and plan for the future of your investments with precision 
        and confidence. Whether you're a seasoned investor or just starting your journey, 
        our cutting-edge forecasting tools and insights will be your financial compass.

        **Why Choose Us?**

        - **Accurate Predictions**: Our advanced algorithms provide highly accurate 
          forecasts of Net Asset Value (NAV) for various investment schemes.

        - **Diverse Schemes**: Explore and forecast NAV for six different investment schemes, 
          including mutual funds, ETFs, and more.

        - **User-Friendly Interface**: Our intuitive interface makes it easy for anyone, 
          regardless of experience, to access powerful investment analytics.

        - **In-Depth Analysis**: Dive deeper into historical NAV data, trends, and visualizations, 
          enabling you to make data-driven investment decisions.

        - **Plan Your Future**: Secure your financial future by making informed investment choices.

        **Get Started**:

        1. Select your preferred investment scheme.
        2. Explore historical data and trends.
        3. Get highly accurate forecasts for your investments.
        
        Start now and watch your investments thrive!

        """
    )

elif page == "Select Scheme":
    st.info("You are on the 'Select Scheme' page.")
    # Add your code for scheme selection here
    st.subheader('Select Scheme From the Side Bar')
    # Function to calculate annualized returns
    def calculate_annualized_returns(returns):
        return (returns.mean() + 1) ** 252 - 1

    # Function to calculate Sharpe Ratio
    def calculate_sharpe_ratio(returns):
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    # Define a function to format money values in Tanzanian Shillings (TZS)
    def format_money_tzs(value):
        return f'TZS {value:,.2f}'

    # Load datasets
    bond_data = pd.read_csv('Data/NAV Bond Fund.csv')
    jikimu_data = pd.read_csv('Data/NAV Jikimu Fund.csv')
    watoto_data = pd.read_csv('Data/NAV Watoto Fund.csv')
    liquid_data = pd.read_csv('Data/NAV Liquid Fund.csv')
    umoja_data = pd.read_csv('Data/NAV Umoja Fund.csv')
    wekeza_maisha_data = pd.read_csv('Data/NAV Wekeza Maisha Fund.csv')

    # Sidebar
    st.sidebar.title('Fund Selection')
    selected_fund = st.sidebar.selectbox('Select a Fund', ('Bond Fund', 'Jikimu Fund', 'Watoto Fund', 'Liquid Fund', 'Umoja Fund', 'Wekeza Maisha Fund'))

    if selected_fund == 'Bond Fund':
        st.subheader('Bond Fund Features:')

        year_bond = st.number_input('Year', min_value=0)
        month_bond = st.number_input('Month', min_value=1, max_value=12)
        day_bond = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Bond Fund
            st.subheader('Prediction Result for Bond Fund:')

            X_bond = bond_data[['Year', 'Month', 'Day']]
            y_bond = bond_data['Nav Per Unit']
            X_train_bond, X_test_bond, y_train_bond, y_test_bond = train_test_split(X_bond, y_bond, test_size=0.2, random_state=42)

            best_lasso_alpha_bond = 0.01
            best_ridge_alpha_bond = 0.1
            lasso_model_bond = Lasso(alpha=best_lasso_alpha_bond)
            ridge_model_bond = Ridge(alpha=best_ridge_alpha_bond)

            models_bond = [('Lasso', lasso_model_bond), ('Ridge', ridge_model_bond)]
            ensemble_model_bond = VotingRegressor(models_bond)
            ensemble_model_bond.fit(X_train_bond, y_train_bond)
            ensemble_predictions_bond = ensemble_model_bond.predict(X_test_bond)

            mae_bond = mean_absolute_error(y_test_bond, ensemble_predictions_bond)
            mse_bond = mean_squared_error(y_test_bond, ensemble_predictions_bond)
            rmse_bond = np.sqrt(mse_bond)
            r2_bond = r2_score(y_test_bond, ensemble_predictions_bond)

            mape_bond = np.mean(np.abs((y_test_bond - ensemble_predictions_bond) / y_test_bond)) * 100
            daily_returns_bond = y_test_bond.pct_change().dropna()
            annualized_returns_bond = calculate_annualized_returns(daily_returns_bond)
            sharpe_ratio_bond = calculate_sharpe_ratio(daily_returns_bond)

            # Calculate and display the predicted NAV value
            predicted_sqrt_nav_bond = ensemble_model_bond.predict([[year_bond, month_bond, day_bond]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_bond = predicted_sqrt_nav_bond ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_bond_str = f"{predicted_sqrt_nav_bond:,.2f} TZS"

            # Calculate R-squared (accuracy)
            accuracy = r2_bond

            # Define a message and container based on accuracy
            if accuracy > 0.8:
                message = f"Predicted the Net Asset Value with an accuracy of {accuracy:.2%}"
                container = st.success(message)
            elif 0.5 <= accuracy <= 0.8:
                message = f"Predicted the Net Asset Value with an accuracy of {accuracy:.2%}"
                container = st.warning(message)
            else:
                message = f"Predicted the Net Asset Value with an accuracy of {accuracy:.2%}"
                container = st.error(message)

            # Display the predicted NAV value as TZS
            st.info(f"Predicted NAV: {predicted_nav_bond_str}")

            # Create a toggle box to display accuracy metrics
            with st.expander("Click to display the accuracy metrics"):
                st.write(f'MAE: {mae_bond}')
                st.write(f'MSE: {mse_bond}')
                st.write(f'RMSE: {rmse_bond}')
                st.write(f'R-squared: {r2_bond}')
                st.write(f'MAPE: {mape_bond}')
                st.write(f'Annualized Returns: {annualized_returns_bond}')
                st.write(f'Sharpe Ratio: {sharpe_ratio_bond}')

    elif selected_fund == 'Jikimu Fund':
        st.subheader('Jikimu Fund Features:')
        # Add input fields for Jikimu Fund features
        year_jikimu = st.number_input('Year', min_value=0)
        month_jikimu = st.number_input('Month', min_value=1, max_value=12)
        day_jikimu = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Jikimu Fund
            st.subheader('Prediction Result for Jikimu Fund:')

            X_jikimu = jikimu_data[['Year', 'Month', 'Day']]
            y_jikimu = jikimu_data['Nav Per Unit']
            X_train_jikimu, X_test_jikimu, y_train_jikimu, y_test_jikimu = train_test_split(X_jikimu, y_jikimu, test_size=0.2, random_state=42)

            best_lasso_alpha_jikimu = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_jikimu = 0.1  # Replace with the best alpha you found
            lasso_model_jikimu = Lasso(alpha=best_lasso_alpha_jikimu)
            ridge_model_jikimu = Ridge(alpha=best_ridge_alpha_jikimu)

            models_jikimu = [('Lasso', lasso_model_jikimu), ('Ridge', ridge_model_jikimu)]
            ensemble_model_jikimu = VotingRegressor(models_jikimu)
            ensemble_model_jikimu.fit(X_train_jikimu, y_train_jikimu)
            ensemble_predictions_jikimu = ensemble_model_jikimu.predict(X_test_jikimu)

            mae_jikimu = mean_absolute_error(y_test_jikimu, ensemble_predictions_jikimu)
            mse_jikimu = mean_squared_error(y_test_jikimu, ensemble_predictions_jikimu)
            rmse_jikimu = np.sqrt(mse_jikimu)
            r2_jikimu = r2_score(y_test_jikimu, ensemble_predictions_jikimu)

            mape_jikimu = np.mean(np.abs((y_test_jikimu - ensemble_predictions_jikimu) / y_test_jikimu)) * 100
            daily_returns_jikimu = y_test_jikimu.pct_change().dropna()
            annualized_returns_jikimu = calculate_annualized_returns(daily_returns_jikimu)
            sharpe_ratio_jikimu = calculate_sharpe_ratio(daily_returns_jikimu)

            # Calculate and display the predicted NAV value for Jikimu Fund
            predicted_sqrt_nav_jikimu = ensemble_model_jikimu.predict([[year_jikimu, month_jikimu, day_jikimu]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_jikimu = predicted_sqrt_nav_jikimu ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_jikimu_str = f"{predicted_sqrt_nav_jikimu:,.2f} TZS"

            # Calculate R-squared (accuracy) for Jikimu Fund
            accuracy_jikimu = r2_jikimu

            # Define a message and container based on accuracy for Jikimu Fund
            if accuracy_jikimu > 0.8:
                message_jikimu = f"Predicted the Net Asset Value with an accuracy of {accuracy_jikimu:.2%}"
                container_jikimu = st.success(message_jikimu)
            elif 0.5 <= accuracy_jikimu <= 0.8:
                message_jikimu = f"Predicted the Net Asset Value with an accuracy of {accuracy_jikimu:.2%}"
                container_jikimu = st.warning(message_jikimu)
            else:
                message_jikimu = f"Predicted the Net Asset Value with an accuracy of {accuracy_jikimu:.2%}"
                container_jikimu = st.error(message_jikimu)

            # Display the predicted NAV value as TZS
            st.info(f"Predicted NAV: {predicted_nav_jikimu_str}")

            # Create a toggle box to display accuracy metrics for Jikimu Fund
            with st.expander("Click to display the accuracy metrics for Jikimu Fund"):
                st.write(f'MAE: {mae_jikimu}')
                st.write(f'MSE: {mse_jikimu}')
                st.write(f'RMSE: {rmse_jikimu}')
                st.write(f'R-squared: {r2_jikimu}')
                st.write(f'MAPE: {mape_jikimu}')
                st.write(f'Annualized Returns: {annualized_returns_jikimu}')
                st.write(f'Sharpe Ratio: {sharpe_ratio_jikimu}')

    elif selected_fund == 'Watoto Fund':
        st.subheader('Watoto Fund Features:')
        # Add input fields for Watoto Fund features
        year_watoto = st.number_input('Year', min_value=0)
        month_watoto = st.number_input('Month', min_value=1, max_value=12)
        day_watoto = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Watoto Fund
            st.subheader('Prediction Result for Watoto Fund:')

            X_watoto = watoto_data[['Year', 'Month', 'Day']]
            y_watoto = watoto_data['Nav Per Unit']
            X_train_watoto, X_test_watoto, y_train_watoto, y_test_watoto = train_test_split(X_watoto, y_watoto, test_size=0.2, random_state=42)

            best_lasso_alpha_watoto = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_watoto = 0.1  # Replace with the best alpha you found
            lasso_model_watoto = Lasso(alpha=best_lasso_alpha_watoto)
            ridge_model_watoto = Ridge(alpha=best_ridge_alpha_watoto)

            models_watoto = [('Lasso', lasso_model_watoto), ('Ridge', ridge_model_watoto)]
            ensemble_model_watoto = VotingRegressor(models_watoto)
            ensemble_model_watoto.fit(X_train_watoto, y_train_watoto)
            ensemble_predictions_watoto = ensemble_model_watoto.predict(X_test_watoto)

            mae_watoto = mean_absolute_error(y_test_watoto, ensemble_predictions_watoto)
            mse_watoto = mean_squared_error(y_test_watoto, ensemble_predictions_watoto)
            rmse_watoto = np.sqrt(mse_watoto)
            r2_watoto = r2_score(y_test_watoto, ensemble_predictions_watoto)

            mape_watoto = np.mean(np.abs((y_test_watoto - ensemble_predictions_watoto) / y_test_watoto)) * 100
            daily_returns_watoto = y_test_watoto.pct_change().dropna()
            annualized_returns_watoto = calculate_annualized_returns(daily_returns_watoto)
            sharpe_ratio_watoto = calculate_sharpe_ratio(daily_returns_watoto)

            # Calculate and display the predicted NAV value for Watoto Fund
            predicted_sqrt_nav_watoto = ensemble_model_watoto.predict([[year_watoto, month_watoto, day_watoto]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_watoto = predicted_sqrt_nav_watoto ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_watoto_str = f"{predicted_sqrt_nav_watoto:,.2f} TZS"

            # Calculate R-squared (accuracy) for Watoto Fund
            accuracy_watoto = r2_watoto

            # Define a message and container based on accuracy for Watoto Fund
            if accuracy_watoto > 0.8:
                message_watoto = f"Predicted the Net Asset Value with an accuracy of {accuracy_watoto:.2%}"
                container_watoto = st.success(message_watoto)
            elif 0.5 <= accuracy_watoto <= 0.8:
                message_watoto = f"Predicted the Net Asset Value with an accuracy of {accuracy_watoto:.2%}"
                container_watoto = st.warning(message_watoto)
            else:
                message_watoto = f"Predicted the Net Asset Value with an accuracy of {accuracy_watoto:.2%}"
                container_watoto = st.error(message_watoto)

            # Display the predicted NAV value as TZS
            st.info(f"Predicted NAV: {predicted_nav_watoto_str}")

            # Create a toggle box to display accuracy metrics for Watoto Fund
            with st.expander("Click to display the accuracy metrics for Watoto Fund"):
                st.write(f'MAE: {mae_watoto}')
                st.write(f'MSE: {mse_watoto}')
                st.write(f'RMSE: {rmse_watoto}')
                st.write(f'R-squared: {r2_watoto}')
                st.write(f'MAPE: {mape_watoto}')
                st.write(f'Annualized Returns: {annualized_returns_watoto}')
                st.write(f'Sharpe Ratio: {sharpe_ratio_watoto}')

    elif selected_fund == 'Liquid Fund':
        st.subheader('Liquid Fund Features:')
        # Add input fields for Liquid Fund features
        year_liquid = st.number_input('Year', min_value=0)
        month_liquid = st.number_input('Month', min_value=1, max_value=12)
        day_liquid = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Liquid Fund
            st.subheader('Prediction Result for Liquid Fund:')

            X_liquid = liquid_data[['Year', 'Month', 'Day']]
            y_liquid = liquid_data['Nav Per Unit']
            X_train_liquid, X_test_liquid, y_train_liquid, y_test_liquid = train_test_split(X_liquid, y_liquid, test_size=0.2, random_state=42)

            best_lasso_alpha_liquid = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_liquid = 0.1  # Replace with the best alpha you found
            lasso_model_liquid = Lasso(alpha=best_lasso_alpha_liquid)
            ridge_model_liquid = Ridge(alpha=best_ridge_alpha_liquid)

            models_liquid = [('Lasso', lasso_model_liquid), ('Ridge', ridge_model_liquid)]
            ensemble_model_liquid = VotingRegressor(models_liquid)
            ensemble_model_liquid.fit(X_train_liquid, y_train_liquid)
            ensemble_predictions_liquid = ensemble_model_liquid.predict(X_test_liquid)

            mae_liquid = mean_absolute_error(y_test_liquid, ensemble_predictions_liquid)
            mse_liquid = mean_squared_error(y_test_liquid, ensemble_predictions_liquid)
            rmse_liquid = np.sqrt(mse_liquid)
            r2_liquid = r2_score(y_test_liquid, ensemble_predictions_liquid)

            mape_liquid = np.mean(np.abs((y_test_liquid - ensemble_predictions_liquid) / y_test_liquid)) * 100
            daily_returns_liquid = y_test_liquid.pct_change().dropna()
            annualized_returns_liquid = calculate_annualized_returns(daily_returns_liquid)
            sharpe_ratio_liquid = calculate_sharpe_ratio(daily_returns_liquid)

            # Calculate and display the predicted NAV value for Liquid Fund
            predicted_sqrt_nav_liquid = ensemble_model_liquid.predict([[year_liquid, month_liquid, day_liquid]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_liquid = predicted_sqrt_nav_liquid ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_liquid_str = f"{predicted_sqrt_nav_liquid:,.2f} TZS"

            # Calculate R-squared (accuracy) for Liquid Fund
            accuracy_liquid = r2_liquid

            # Define a message and container based on accuracy for Liquid Fund
            if accuracy_liquid > 0.8:
                message_liquid = f"Predicted the Net Asset Value with an accuracy of {accuracy_liquid:.2%}"
                container_liquid = st.success(message_liquid)
            elif 0.5 <= accuracy_liquid <= 0.8:
                message_liquid = f"Predicted the Net Asset Value with an accuracy of {accuracy_liquid:.2%}"
                container_liquid = st.warning(message_liquid)
            else:
                message_liquid = f"Predicted the Net Asset Value with an accuracy of {accuracy_liquid:.2%}"
                container_liquid = st.error(message_liquid)

            # Display the predicted NAV value as TZS
            st.info(f"Predicted NAV: {predicted_nav_liquid_str}")

            # Create a toggle box to display accuracy metrics for Liquid Fund
            with st.expander("Click to display the accuracy metrics for Liquid Fund"):
                st.write(f'MAE: {mae_liquid}')
                st.write(f'MSE: {mse_liquid}')
                st.write(f'RMSE: {rmse_liquid}')
                st.write(f'R-squared: {r2_liquid}')
                st.write(f'MAPE: {mape_liquid}')
                st.write(f'Annualized Returns: {annualized_returns_liquid}')
                st.write(f'Sharpe Ratio: {sharpe_ratio_liquid}')

    elif selected_fund == 'Umoja Fund':
        st.subheader('Umoja Fund Features:')
        # Features input
        year_umoja = st.number_input('Year', min_value=0)
        month_umoja = st.number_input('Month', min_value=1, max_value=12)
        day_umoja = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Umoja Fund
            st.subheader('Prediction Result for Umoja Fund:')

            X_umoja = umoja_data[['Year', 'Month', 'Day']]
            y_umoja = umoja_data['Nav Per Unit']
            X_train_umoja, X_test_umoja, y_train_umoja, y_test_umoja = train_test_split(X_umoja, y_umoja, test_size=0.2, random_state=42)

            best_lasso_alpha_umoja = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_umoja = 0.1  # Replace with the best alpha you found
            lasso_model_umoja = Lasso(alpha=best_lasso_alpha_umoja)
            ridge_model_umoja = Ridge(alpha=best_ridge_alpha_umoja)

            models_umoja = [('Lasso', lasso_model_umoja), ('Ridge', ridge_model_umoja)]
            ensemble_model_umoja = VotingRegressor(models_umoja)
            ensemble_model_umoja.fit(X_train_umoja, y_train_umoja)
            ensemble_predictions_umoja = ensemble_model_umoja.predict(X_test_umoja)

            mae_umoja = mean_absolute_error(y_test_umoja, ensemble_predictions_umoja)
            mse_umoja = mean_squared_error(y_test_umoja, ensemble_predictions_umoja)
            rmse_umoja = np.sqrt(mse_umoja)
            r2_umoja = r2_score(y_test_umoja, ensemble_predictions_umoja)

            mape_umoja = np.mean(np.abs((y_test_umoja - ensemble_predictions_umoja) / y_test_umoja)) * 100
            daily_returns_umoja = y_test_umoja.pct_change().dropna()
            annualized_returns_umoja = calculate_annualized_returns(daily_returns_umoja)
            sharpe_ratio_umoja = calculate_sharpe_ratio(daily_returns_umoja)

            # Calculate and display the predicted NAV value for Umoja Fund
            predicted_sqrt_nav_umoja = ensemble_model_umoja.predict([[year_umoja, day_umoja, month_umoja]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_umoja = predicted_sqrt_nav_umoja ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_umoja_str = f"{predicted_sqrt_nav_umoja:,.2f} TZS"

            # Calculate R-squared (accuracy) for Umoja Fund
            accuracy_umoja = r2_umoja

            # Define a message and container based on accuracy for Umoja Fund
            if accuracy_umoja > 0.8:
                message_umoja = f"Predicted the Net Asset Value with an accuracy of {accuracy_umoja:.2%}"
                container_umoja = st.success(message_umoja)
            elif 0.5 <= accuracy_umoja <= 0.8:
                message_umoja = f"Predicted the Net Asset Value with an accuracy of {accuracy_umoja:.2%}"
                container_umoja = st.warning(message_umoja)
            else:
                message_umoja = f"Predicted the Net Asset Value with an accuracy of {accuracy_umoja:.2%}"
                container_umoja = st.error(message_umoja)

            # Display the predicted NAV value for Umoja Fund
            st.info(f"Predicted NAV: {predicted_nav_umoja_str}")

            # Create a toggle box to display accuracy metrics for Umoja Fund
            with st.expander("Click to display the accuracy metrics for Umoja Fund"):
                st.write(f'MAE: {mae_umoja}')
                st.write(f'MSE: {mse_umoja}')
                st.write(f'RMSE: {rmse_umoja}')
                st.write(f'R-squared: {r2_umoja}')
                st.write(f'MAPE: {mape_umoja}')
                st.write(f'Annualized Returns: {annualized_returns_umoja}')
                st.write(f'Sharpe Ratio: {sharpe_ratio_umoja}')

    elif selected_fund == 'Wekeza Maisha Fund':
        st.subheader('Wekeza Maisha Fund Features:')
        # Add input fields for Wekeza Maisha Fund features
        year_wekeza_maisha = st.number_input('Year', min_value=0)
        month_wekeza_maisha = st.number_input('Month', min_value=1, max_value=12)
        day_wekeza_maisha = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Wekeza Maisha Fund
            st.subheader('Prediction Result for Wekeza Maisha Fund:')

            X_wekeza_maisha = wekeza_maisha_data[['Year', 'Month', 'Day']]
            y_wekeza_maisha = wekeza_maisha_data['Nav Per Unit']
            X_train_wekeza_maisha, X_test_wekeza_maisha, y_train_wekeza_maisha, y_test_wekeza_maisha = train_test_split(X_wekeza_maisha, y_wekeza_maisha, test_size=0.2, random_state=42)

            best_lasso_alpha_wekeza_maisha = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_wekeza_maisha = 0.1  # Replace with the best alpha you found
            lasso_model_wekeza_maisha = Lasso(alpha=best_lasso_alpha_wekeza_maisha)
            ridge_model_wekeza_maisha = Ridge(alpha=best_ridge_alpha_wekeza_maisha)

            models_wekeza_maisha = [('Lasso', lasso_model_wekeza_maisha), ('Ridge', ridge_model_wekeza_maisha)]
            ensemble_model_wekeza_maisha = VotingRegressor(models_wekeza_maisha)
            ensemble_model_wekeza_maisha.fit(X_train_wekeza_maisha, y_train_wekeza_maisha)
            ensemble_predictions_wekeza_maisha = ensemble_model_wekeza_maisha.predict(X_test_wekeza_maisha)

            mae_wekeza_maisha = mean_absolute_error(y_test_wekeza_maisha, ensemble_predictions_wekeza_maisha)
            mse_wekeza_maisha = mean_squared_error(y_test_wekeza_maisha, ensemble_predictions_wekeza_maisha)
            rmse_wekeza_maisha = np.sqrt(mse_wekeza_maisha)
            r2_wekeza_maisha = r2_score(y_test_wekeza_maisha, ensemble_predictions_wekeza_maisha)

            mape_wekeza_maisha = np.mean(np.abs((y_test_wekeza_maisha - ensemble_predictions_wekeza_maisha) / y_test_wekeza_maisha)) * 100
            daily_returns_wekeza_maisha = y_test_wekeza_maisha.pct_change().dropna()
            annualized_returns_wekeza_maisha = calculate_annualized_returns(daily_returns_wekeza_maisha)
            sharpe_ratio_wekeza_maisha = calculate_sharpe_ratio(daily_returns_wekeza_maisha)

            # Calculate and display the predicted NAV value for Wekeza Maisha Fund
            predicted_sqrt_nav_wekeza_maisha = ensemble_model_wekeza_maisha.predict([[year_wekeza_maisha, month_wekeza_maisha, day_wekeza_maisha]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_wekeza_maisha = predicted_sqrt_nav_wekeza_maisha ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_wekeza_maisha_str = f"{predicted_sqrt_nav_wekeza_maisha:,.2f} TZS"

            # Calculate R-squared (accuracy) for Wekeza Maisha Fund
            accuracy_wekeza_maisha = r2_wekeza_maisha

            # Define a message and container based on accuracy for Wekeza Maisha Fund
            if accuracy_wekeza_maisha > 0.8:
                message_wekeza_maisha = f"Predicted the Net Asset Value with an accuracy of {accuracy_wekeza_maisha:.2%}"
                container_wekeza_maisha = st.success(message_wekeza_maisha)
            elif 0.5 <= accuracy_wekeza_maisha <= 0.8:
                message_wekeza_maisha = f"Predicted the Net Asset Value with an accuracy of {accuracy_wekeza_maisha:.2%}"
                container_wekeza_maisha = st.warning(message_wekeza_maisha)
            else:
                message_wekeza_maisha = f"Predicted the Net Asset Value with an accuracy of {accuracy_wekeza_maisha:.2%}"
                container_wekeza_maisha = st.error(message_wekeza_maisha)

            # Display the predicted NAV value for Wekeza Maisha Fund
            st.info(f"Predicted NAV: {predicted_nav_wekeza_maisha_str}")

            # Create a toggle box to display accuracy metrics for Wekeza Maisha Fund
            with st.expander("Click to display the accuracy metrics for Wekeza Maisha Fund"):
                st.write(f'MAE: {mae_wekeza_maisha}')
                st.write(f'MSE: {mse_wekeza_maisha}')
                st.write(f'RMSE: {rmse_wekeza_maisha}')
                st.write(f'R-squared: {r2_wekeza_maisha}')
                st.write(f'MAPE: {mape_wekeza_maisha}')
                st.write(f'Annualized Returns: {annualized_returns_wekeza_maisha}')
                st.write(f'Sharpe Ratio: {sharpe_ratio_wekeza_maisha}')

elif page == "Visualization":
    st.info("You are on the Visualization page.")
    # Add your code for model input here
# elif page == "Result Prediction":
#     st.info("You are on the 'Result Prediction' page.")
#     # Add your code for result prediction here

# Example of a button to trigger navigation
if page == "Select Scheme":
    if st.button("Go to Visualizations"):
        page = "Visualization"
elif page == "Visualization":
    if st.button("Go to Scheme Selection"):
        page = "Select Scheme"