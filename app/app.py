import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import requests
import json
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from matplotlib.ticker import ScalarFormatter 
import statsmodels.api as sm
import pathlib
import joblib
# import preprocessing

# Get the parent directory of the current script's file
BASE_DIR = pathlib.Path(__file__).resolve().parent

selected = option_menu(
    menu_title=None,
    options=['Home', 'Dashboard', 'Prediction', 'NAV Prediction'],
    icons=['house', 'laptop', 'eye', 'eye'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#11e2"},
        "icon": {},
        "nav-link": {
            "font-size": "14px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "green"}
    },
)

if selected == "Home":
    with st.container():
        st.subheader("UTT AMIS NAV FORECASTING")
        st.write('Your Obvious Investment Partner')

    with st.container():
        left_col, right_col = st.columns(2)
        with left_col:
            st.write("Welcome to the NAV Forecasting System, your gateway to informed investment decisions. We empower you to anticipate and plan for the future of your investments with precision and confidence. Whether you're a seasoned investor or just starting your journey, our cutting-edge forecasting tools and insights will be your financial compass.")
            st.write("[Learn more](https://www.uttamis.co.tz/)")
        with right_col:
            investor_path = BASE_DIR / "Assets/investor.json"

            def load_lottiefile(investor_path: str):
                with open(investor_path, "r") as f:
                    return json.load(f)

            lottie_inv = load_lottiefile(investor_path)
            st.lottie(
                lottie_inv,
                speed=1,
                reverse=False,
                loop=True,
                quality="medium",
                # renderer="svg",
                height=None,
                width=None,
                key=None,
            )

    with st.container():
        st.write("---")
        st.header('Why Choose Us?')
        st.write("1. Accurate Predictions: Our advanced algorithms provide highly accurate forecasts of Net Asset Value (NAV) for various investment schemes.")
        st.write("2. Diverse Schemes: Explore and forecast NAV for six different investment schemes, including mutual funds, ETFs, and more.")
        st.write("3. User-Friendly Interface: Our intuitive interface makes it easy for anyone, regardless of experience, to access powerful investment analytics.")
        st.write("4. In-Depth Analysis: Dive deeper into historical NAV data, trends, and visualizations, enabling you to make data-driven investment decisions.")
        st.write("5. Plan Your Future: Secure your financial future by making informed investment choices.")

    with st.container():
        st.markdown(
            """<div style="text-align:center">
            <h3>Our Product</h3>
            </div>""",
            unsafe_allow_html=True
        )
        umoja, wekeza, watoto = st.columns(3)
        with umoja:
            imageU = Image.open(BASE_DIR / 'Assets/umoja-fund-logo.gif')
            st.image(imageU, caption='')
            st.subheader('Umoja Fund')
            st.write('It is an open-ended balance fund that invests in a diversified portfolio which makes this fund best suited for a medium risk profile.')
        with wekeza:
            imageU = Image.open(BASE_DIR / 'Assets/wekeza maisha.jpeg')
            st.image(imageU, caption='')
            st.subheader('Wekeza Fund')
            st.write('The Wekeza Maisha/Invest life Unit Trust is the first investment cum insurance scheme to be established by UTT AMIS in the country.')
        with watoto:
            imageU = Image.open(BASE_DIR / 'Assets/watoto1.jpeg')
            st.image(imageU, caption='')
            st.subheader('Watoto Fund')
            st.write('A child benefits an open-end balanced fund, which seeks to generate long-term capital appreciation.')
            
        jikimu, liquid, bond = st.columns(3)
        with jikimu:
            imageJ = Image.open(BASE_DIR / 'Assets/jikimu logo.jpeg')
            st.image(imageJ, caption='')
            st.subheader('Jikimu Fund')
            st.write('It is an open-ended balance fund that invests in a diversified portfolio which makes this fund best suited for a medium risk profile.')
        with liquid:
            imageL = Image.open(BASE_DIR / 'Assets/liquid_logo.jpeg')
            st.image(imageL, caption='')
            st.subheader('Liquid Fund')
            st.write('The Wekeza Maisha/Invest life Unit Trust is the first investment cum insurance scheme to be established by UTT AMIS in the country.')
        with bond:
            imageB = Image.open(BASE_DIR / 'Assets/bondfund.jpeg')
            st.image(imageB, caption='')
            st.subheader('Bond Fund')
            st.write('A child benefits an open-end balanced fund, which seeks to generate long-term capital appreciation') 

    with st.container():
        st.subheader('Get started:')
        st.write('1. Select your preferred investment scheme.')
        st.write('2. Explore historical data and trends.')
        st.write('3. Get highly accurate forecasts for your investments.')
            
            
            
if selected == "Dashboard":

    @st.cache_data
    def load_data():
        df = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/net_asset_value.csv') 
        import re

        # Clean and extract numbers
        def clean_and_extract_number(s):
            cleaned_value = re.sub(r'[^\d.]', '', str(s))
            return cleaned_value

        numeric_columns = ['Net Asset Value', 'Outstanding Number of Units']
        for col in numeric_columns:
            df[col] = df[col].apply(clean_and_extract_number)
        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric, errors='coerce')

        # Custom date conversion
        def custom_date_conversion(date_str):
            try:
                return pd.to_datetime(date_str)
            except:
                pass
            return pd.NaT

        df["Date Valued"] = df["Date Valued"].apply(custom_date_conversion)
        df['Year'] = df['Date Valued'].dt.year
        df['Month'] = df['Date Valued'].dt.month
        df['Week'] = df['Date Valued'].dt.isocalendar().week
        df['Day'] = df['Date Valued'].dt.day

        return df, df["Date Valued"].isna().sum()

    df, nat_values = load_data()

    st.sidebar.image('Assets/uttamislogof.png',
                     caption='Your Obvious Investment Partner')

    # Display alerts based on nat_values
    if nat_values > 0:
        st.warning(f"Warning: {nat_values} values could not be converted.")
    else:
        st.success("Date conversion successful!")

   # Data Overview inside an expander
    with st.expander("Data Preview"):
        st.write("## Data Overview")
        st.write("Here's a glimpse of the dataset:")

    # Display the first few rows of the dataframe
        st.write("### Top Rows")
        st.write(df.head())

    # Display the statistical summary of the dataframe
        st.write("### Statistical Summary")
        st.write(df.describe().T)

    # Display the date
    st.write("Date: Tuesday, 19th of September 2023")

    # Define the funds and their respective values
    funds = [
        {"name": "Watoto Fund", "value": "12B"},
        {"name": "Jikimu Fund", "value": "21B"},
        {"name": "Liquid Fund", "value": "813B"},
        {"name": "Bond Fund", "value": "477B"},
        {"name": "Wekeza Maisha Fund", "value": "10B"},
        {"name": "Umoja Fund", "value": "330B"},
    ]

    # Split the funds into two lists
    funds_first_row = funds[:3]
    funds_second_row = funds[3:]

    # Create columns for the first row and display metrics
    columns_first_row = st.columns(len(funds_first_row), gap='large')
    for idx, fund in enumerate(funds_first_row):
        with columns_first_row[idx]:
            st.info(fund["name"])
            st.metric(label='TZS', value=fund["value"])

    # Create columns for the second row and display metrics
    columns_second_row = st.columns(len(funds_second_row), gap='large')
    for idx, fund in enumerate(funds_second_row):
        with columns_second_row[idx]:
            st.info(fund["name"])
            st.metric(label='TZS', value=fund["value"])

    # Create a color palette with unique colors for each scheme name
    unique_colors = sns.color_palette('Set3', n_colors=len(df['Scheme Name'].unique()))

    # Create the countplot with the specified color palette
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Scheme Name', palette=unique_colors)
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Net Asset Values by Scheme Name')

    # Display the plot using Streamlit
    st.pyplot(plt)

    # Pie chart
    funds_counts = df['Scheme Name'].value_counts()
    explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.pie(funds_counts, explode=explode, labels=funds_counts.index, autopct='%1.1f%%', startangle=60,
            wedgeprops={"edgecolor": "black", 'linewidth': 2, 'antialiased': True}, textprops={'fontsize': 6})
    ax1.set_title('Distribution of Net Asset Values by Scheme Name', fontsize=6)
    st.pyplot(fig1)
    
    # Sidebar additions
    st.sidebar.write("Choose a visualization:")

    # Unified sidebar settings for scheme, year, and month selection
    schemes = df['Scheme Name'].unique().tolist()
    selected_scheme = st.sidebar.selectbox(
        'Select a scheme for visualization:', schemes, key='scheme_select')
    years = sorted(df['Year'].unique().tolist())
    selected_year = st.sidebar.selectbox(
        'Select a year:', years, key='year_select')
    months = list(range(1, 13))
    selected_month = st.sidebar.selectbox(
        'Select a month:', months, key='month_select')

    monthly_data = df[(df['Scheme Name'] == selected_scheme) &
                      (df['Year'] == selected_year) &
                      (df['Month'] == selected_month)]

    filtered_df = df[df['Scheme Name'] == selected_scheme]

    col1, col2 = st.columns(2)

    # Time Series Plot for NAV
    with col1:
        st.write("#### Time Series Plot of NAV values")
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=filtered_df, x='Year', y='Net Asset Value')
        plt.title(
            f"Net Asset Value Trend for {selected_scheme} Over the Years")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Time Series Plot for Nav Per Unit
    with col2:
        st.write("#### Time Series Plot of Nav Per Unit")
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=filtered_df, x='Year', y='Nav Per Unit')
        plt.title(f"Nav Per Unit Trend for {selected_scheme} Over the Years")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Monthly Trend Plot for Nav Per Unit
    with col2:
        st.write(
            f"#### Monthly Trend Plot of Nav Per Unit for {selected_month}/{selected_year}")
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=monthly_data, x='Day', y='Nav Per Unit')
        plt.title(
            f"Nav Per Unit for {selected_month}/{selected_year} for {selected_scheme}")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Monthly Trend Plot for NAV
    with col1:
        st.write(
            f"#### Monthly Trend Plot of NAV values for {selected_month}/{selected_year}")
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=monthly_data, x='Day', y='Net Asset Value')
        plt.title(
            f"NAV values for {selected_month}/{selected_year} for {selected_scheme}")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Monthly Trend Plot with Outstanding Number of Units Bar Overlay
    st.write("## Monthly Trend Plot of NAV values with Outstanding Number of Units")
    st.write(
        f"Displaying NAV values for {selected_month}/{selected_year} for scheme: {selected_scheme}")
    monthly_data = df[(df['Scheme Name'] == selected_scheme) &
                      (df['Year'] == selected_year) &
                      (df['Month'] == selected_month)]
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar plot for Outstanding Number of Units
    ax1.bar(monthly_data['Day'], monthly_data['Outstanding Number of Units'],
            color='gray', alpha=0.5, label='Outstanding Number of Units')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Outstanding Number of Units', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    ax1.set_title(
        f"NAV values & Outstanding Units for {selected_month}/{selected_year} for {selected_scheme}")

    # Line plot for NAV values
    ax2 = ax1.twinx()
    sns.lineplot(data=monthly_data, x='Day', y='Net Asset Value',
                 ax=ax2, color='blue', label='NAV Value')
    ax2.set_ylabel('Net Asset Value', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    st.write("\n")

    # Group data by 'Scheme Name' and calculate the average outstanding units
    average_units = df.groupby('Scheme Name')['Outstanding Number of Units'].mean()
    # Define a color palette for each scheme
    colors = sns.color_palette("husl", len(df['Scheme Name'].unique()))

    st.title('Average Outstanding Number of Units by Scheme Name')

    # Create a bar plot using Matplotlib
    plt.figure(figsize=(10, 6))
    average_units.plot(kind='bar', color=colors)
    plt.title('Average Outstanding Number of Units by Scheme Name')
    plt.xlabel('Scheme Name')
    plt.ylabel('Average Outstanding Number of Units')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # Modify the Y-axis formatting to display real numbers
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))


    # Display the plot in the Streamlit app
    st.pyplot(plt)

    # Define a color palette for each scheme
    colors = sns.color_palette("husl", len(df['Scheme Name'].unique()))

    # Group data by 'Scheme Name' and calculate the average Net Asset Value
    average_nav = df.groupby('Scheme Name')['Net Asset Value'].mean()

    # Create a Streamlit web app
    st.title('Average Net Asset Value by Scheme Name')

    # Create a bar plot using Matplotlib with different colors
    plt.figure(figsize=(10, 6))
    average_nav.plot(kind='bar', color=colors)
    plt.title('Average Net Asset Value by Scheme Name')
    plt.xlabel('Scheme Name')
    plt.ylabel('Average Net Asset Value')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Modify the Y-axis formatting to display real numbers
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Display the plot in the Streamlit app
    st.pyplot(plt)

    # Create a Streamlit web app
    # Group data by 'Year' and 'Scheme Name' and calculate the average Net Asset Value
    scheme_avg_nav = df.groupby(['Year', 'Scheme Name'])['Net Asset Value'].mean().reset_index()
    st.title('Average Net Asset Value of Schemes Over the Years')

    # Create a bar plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(data=scheme_avg_nav, x='Year', y='Net Asset Value', hue='Scheme Name')
    plt.xlabel('Year')
    plt.ylabel('Average Net Asset Value')
    plt.title('Average Net Asset Value of Schemes Over the Years')
    plt.xticks(rotation=45)
    plt.legend(title='Scheme Name', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(plt)


    # Create a Streamlit web app
    # Group data by 'Month' and 'Scheme Name' and calculate the average Net Asset Value
    scheme_avg_nav = df.groupby(['Month', 'Scheme Name'])['Net Asset Value'].mean().reset_index()

    # Define a list of month names for labeling
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    st.title('Average Net Asset Value of Schemes by Month')

    # Create a bar plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(data=scheme_avg_nav, x='Month', y='Net Asset Value', hue='Scheme Name')
    plt.xlabel('Month')
    plt.ylabel('Average Net Asset Value')
    plt.title('Average Net Asset Value of Schemes by Month')
    plt.xticks(ticks=range(12), labels=month_names, rotation=45)
    plt.legend(title='Scheme Name', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(plt)
    
    image = Image.open('Assets/time.png')
    image_with_opacity = image.copy()
    image_with_opacity.putalpha(255)  # Set the alpha value (0-255), where 0 is fully transparent and 255 is fully opaque

    # Display the image with adjusted opacity
    st.image(image_with_opacity, caption='Time series decomposition!', width=700)


if selected == "Prediction":
    st.sidebar.image('Assets/uttamislogof.png',caption='Your Obvious Investment Partner')
    with st.sidebar:
        selected_fund=option_menu(
        menu_title=None,
        options=['Bond Fund','Jikimu Fund','Watoto Fund','Liquid Fund','Umoja Fund','Wekeza Maisha Fund'],
        )  

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
    bond_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Bond Fund.csv')
    jikimu_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Jikimu Fund.csv')
    watoto_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Watoto Fund.csv')
    liquid_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Liquid Fund.csv')
    umoja_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Umoja Fund.csv')
    wekeza_maisha_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Wekeza Maisha Fund.csv')

    if selected_fund == 'Bond Fund':
        st.subheader('Bond Fund Features:')
        # Add input fields for Jikimu Fund features
        current_nav = st.number_input('Enter Current NAV', min_value=0.0)  # User input for current NAV
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
            
#             # Save the trained model to a file
#             model_filename = "bond_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_bond, model_file)

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
            st.info(f"Predicted NAV per unit: {predicted_nav_bond_str}")
            
            # Calculate profit or loss
            profit_loss = predicted_sqrt_nav_bond - current_nav

            # Display profit or loss message based on the result
            if profit_loss > 0:
                container = st.success(f"Profit: {profit_loss:.2f} TZS Per Unit of Bond Fund.")
            elif profit_loss < 0:
                container = st.error(f"Loss: {profit_loss:.2f} TZS Per Unit of Bond Fund.")
            else:
                container = st.warning("No Profit or Loss Per Unit of Bond Fund.")

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
        current_nav = st.number_input('Enter Current NAV', min_value=0.0)  # User input for current NAV
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
            
#             # Save the trained model to a file
#             model_filename = "jikimu_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_jikimu, model_file)

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
            st.info(f"Predicted NAV per unit: {predicted_nav_jikimu_str}")
            
            # Calculate profit or loss
            profit_loss = predicted_sqrt_nav_jikimu - current_nav

            # Display profit or loss message based on the result
            if profit_loss > 0:
                container = st.success(f"Profit: {profit_loss:.2f} TZS Per Unit of Jikimu Fund.")
            elif profit_loss < 0:
                container = st.error(f"Loss: {profit_loss:.2f} TZS Per Unit of Jikimu Fund.")
            else:
                container = st.warning("No Profit or Loss Per Unit of Jikimu Fund.")
                
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
        current_nav = st.number_input('Enter Current NAV', min_value=0.0)  # User input for current NAV
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
            
#             # Save the trained model to a file
#             model_filename = "watoto_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_watoto, model_file)

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
            st.info(f"Predicted NAV per unit: {predicted_nav_watoto_str}")
            
            # Calculate profit or loss
            profit_loss = predicted_sqrt_nav_watoto - current_nav

            # Display profit or loss message based on the result
            if profit_loss > 0:
                container = st.success(f"Profit: {profit_loss:.2f} TZS Per Unit of Watoto Fund.")
            elif profit_loss < 0:
                container = st.error(f"Loss: {profit_loss:.2f} TZS Per Unit of Watoto Fund.")
            else:
                container = st.warning("No Profit or Loss Per Unit of Watoto Fund.")
                
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
        current_nav = st.number_input('Enter Current NAV', min_value=0.0)  # User input for current NAV
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
            
#             # Save the trained model to a file
#             model_filename = "liquid_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_liquid, model_file)

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
            st.info(f"Predicted NAV per unit: {predicted_nav_liquid_str}")
            
            # Calculate profit or loss
            profit_loss = predicted_sqrt_nav_liquid - current_nav

            # Display profit or loss message based on the result
            if profit_loss > 0:
                container = st.success(f"Profit: {profit_loss:.2f} TZS Per Unit of Liquid Fund.")
            elif profit_loss < 0:
                container = st.error(f"Loss: {profit_loss:.2f} TZS Per Unit of Liquid Fund.")
            else:
                container = st.warning("No Profit or Loss Per Unit of Liquid Fund.")
                
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
        current_nav = st.number_input('Enter Current NAV', min_value=0.0)  # User input for current NAV
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
            
#             # Save the trained model to a file
#             model_filename = "umoja_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_umoja, model_file)

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
            st.info(f"Predicted NAV per unit: {predicted_nav_umoja_str}")

            # Calculate profit or loss
            profit_loss = predicted_sqrt_nav_umoja - current_nav

            # Display profit or loss message based on the result
            if profit_loss > 0:
                container = st.success(f"Profit: {profit_loss:.2f} TZS Per Unit of Umoja Fund.")
            elif profit_loss < 0:
                container = st.error(f"Loss: {profit_loss:.2f} TZS Per Unit of Umoja Fund.")
            else:
                container = st.warning("No Profit or Loss Per Unit of Umoja Fund.")
            
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
        current_nav = st.number_input('Enter Current NAV', min_value=0.0)  # User input for current NAV
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
            
#             # Save the trained model to a file
#             model_filename = "wekeza_maisha_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_wekeza_maisha, model_file)

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
            st.info(f"Predicted NAV per unit: {predicted_nav_wekeza_maisha_str}")
            
            # Calculate profit or loss
            profit_loss = predicted_sqrt_nav_wekeza_maisha - current_nav

            # Display profit or loss message based on the result
            if profit_loss > 0:
                container = st.success(f"Profit: {profit_loss:.2f} TZS Per Unit of Wekeza Maisha Fund.")
            elif profit_loss < 0:
                container = st.error(f"Loss: {profit_loss:.2f} TZS Per Unit of Wekeza Maisha Fund.")
            else:
                container = st.warning("No Profit or Loss Per Unit of Wekeza Maisha Fund.")
            
            # Create a toggle box to display accuracy metrics for Wekeza Maisha Fund
            with st.expander("Click to display the accuracy metrics for Wekeza Maisha Fund"):
                st.write(f'MAE: {mae_wekeza_maisha}')
                st.write(f'MSE: {mse_wekeza_maisha}')
                st.write(f'RMSE: {rmse_wekeza_maisha}')
                st.write(f'R-squared: {r2_wekeza_maisha}')
                st.write(f'MAPE: {mape_wekeza_maisha}')
                st.write(f'Annualized Returns: {annualized_returns_wekeza_maisha}')
                st.write(f'Sharpe Ratio: {sharpe_ratio_wekeza_maisha}')

if selected == "NAV Prediction":
    st.sidebar.image('Assets/uttamislogof.png',caption='Your Obvious Investment Partner')
    with st.sidebar:
        selected_fund=option_menu(
        menu_title=None,
        options=['Bond Fund','Jikimu Fund','Watoto Fund','Liquid Fund','Umoja Fund','Wekeza Maisha Fund'],
            )

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
    bond_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Bond Fund.csv')
    jikimu_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Jikimu Fund.csv')
    watoto_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Watoto Fund.csv')
    liquid_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Liquid Fund.csv')
    umoja_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Umoja Fund.csv')
    wekeza_maisha_data = pd.read_csv('/Users/gabe/NAV-Forecasting-UTTAMIS/Data/NAV Wekeza Maisha Fund.csv')

#     # Sidebar
#     st.sidebar.title('Fund Selection')
#     selected_fund = st.sidebar.selectbox('Select a Fund', ('Bond Fund', 'Jikimu Fund', 'Watoto Fund', 'Liquid Fund', 'Umoja Fund', 'Wekeza Maisha Fund'))

    if selected_fund == 'Bond Fund':
        st.subheader('Bond Fund Features:')
        # Add input fields for Bond Fund features

        # Add input field for Outstanding Number of Units (ONU)
        onu_bond = st.number_input('Outstanding Number of Units (ONU)', min_value=0.0)

        # Calculate Log_ONU
        log_onu_bond = np.log(onu_bond) if onu_bond > 0 else 0.0  # Avoid log(0) which is undefined
#         nav_per_unit_bond = st.number_input('Nav Per Unit', min_value=0.0)
#         sale_price_per_unit_bond = st.number_input('Sale Price per Unit', min_value=0.0)
#         repurchase_price_per_unit_bond = st.number_input('Repurchase Price per Unit', min_value=0.0)
        year_bond = st.number_input('Year', min_value=0)
        month_bond = st.number_input('Month', min_value=1, max_value=12)
        day_bond = st.number_input('Day', min_value=1, max_value=31)

        # Add a "Forecast" button
        if st.button("Forecast"):
            # Model loading and prediction for Bond Fund
            st.subheader('Prediction Result for Bond Fund:')

            X_bond = bond_data[['Log_ONU', 'Year', 'Month', 'Day']]
            y_bond = bond_data['Sqrt_NAV']
            X_train_bond, X_test_bond, y_train_bond, y_test_bond = train_test_split(X_bond, y_bond, test_size=0.2, random_state=42)

            best_lasso_alpha_bond = 0.01
            best_ridge_alpha_bond = 0.1
            lasso_model_bond = Lasso(alpha=best_lasso_alpha_bond)
            ridge_model_bond = Ridge(alpha=best_ridge_alpha_bond)

            models_bond = [('Lasso', lasso_model_bond), ('Ridge', ridge_model_bond)]
            ensemble_model_bond = VotingRegressor(models_bond)
            ensemble_model_bond.fit(X_train_bond, y_train_bond)
            ensemble_predictions_bond = ensemble_model_bond.predict(X_test_bond)
            
#             # Save the trained model to a file
#             model_filename = "nav_bond_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_bond, model_file)

            mae_bond = mean_absolute_error(y_test_bond, ensemble_predictions_bond)
            mse_bond = mean_squared_error(y_test_bond, ensemble_predictions_bond)
            rmse_bond = np.sqrt(mse_bond)
            r2_bond = r2_score(y_test_bond, ensemble_predictions_bond)

            mape_bond = np.mean(np.abs((y_test_bond - ensemble_predictions_bond) / y_test_bond)) * 100
            daily_returns_bond = y_test_bond.pct_change().dropna()
            annualized_returns_bond = calculate_annualized_returns(daily_returns_bond)
            sharpe_ratio_bond = calculate_sharpe_ratio(daily_returns_bond)

            # Calculate and display the predicted NAV value
            predicted_sqrt_nav_bond = ensemble_model_bond.predict([[log_onu_bond, year_bond, month_bond, day_bond]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_bond = predicted_sqrt_nav_bond ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_bond_str = f"{predicted_nav_bond:,.2f} TZS"

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
        
#         # Add input field for the number of months ahead to forecast
#         forecast_months = st.number_input('Number of Months Ahead to Forecast', min_value=1, value=6)
        
        # Add input field for Outstanding Number of Units (ONU)
        onu_jikimu = st.number_input('Outstanding Number of Units (ONU)', min_value=0.0)
        
        # Calculate Log_ONU
        log_onu_jikimu = np.log(onu_jikimu) if onu_jikimu > 0 else 0.0  # Avoid log(0) which is undefined
        nav_per_unit_jikimu = st.number_input('Nav Per Unit', min_value=0.0)
#         sale_price_per_unit_jikimu = st.number_input('Sale Price per Unit', min_value=0.0)
        repurchase_price_per_unit_jikimu = st.number_input('Repurchase Price per Unit', min_value=0.0)
        year_jikimu = st.number_input('Year', min_value=0)
        month_jikimu = st.number_input('Month', min_value=1, max_value=12)
        day_jikimu = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Jikimu Fund
            st.subheader('Prediction Result for Jikimu Fund:')

            X_jikimu = jikimu_data[['Log_ONU', 'Nav Per Unit', 'Repurchase Price/Unit', 'Year', 'Month', 'Day']]
            y_jikimu = jikimu_data['Sqrt_NAV']
            X_train_jikimu, X_test_jikimu, y_train_jikimu, y_test_jikimu = train_test_split(X_jikimu, y_jikimu, test_size=0.2, random_state=42)

            best_lasso_alpha_jikimu = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_jikimu = 0.1  # Replace with the best alpha you found
            lasso_model_jikimu = Lasso(alpha=best_lasso_alpha_jikimu)
            ridge_model_jikimu = Ridge(alpha=best_ridge_alpha_jikimu)

            models_jikimu = [('Lasso', lasso_model_jikimu), ('Ridge', ridge_model_jikimu)]
            ensemble_model_jikimu = VotingRegressor(models_jikimu)
            ensemble_model_jikimu.fit(X_train_jikimu, y_train_jikimu)
            ensemble_predictions_jikimu = ensemble_model_jikimu.predict(X_test_jikimu)
            
#             # Save the trained model to a file
#             model_filename = "nav_jikimu_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_jikimu, model_file)

            mae_jikimu = mean_absolute_error(y_test_jikimu, ensemble_predictions_jikimu)
            mse_jikimu = mean_squared_error(y_test_jikimu, ensemble_predictions_jikimu)
            rmse_jikimu = np.sqrt(mse_jikimu)
            r2_jikimu = r2_score(y_test_jikimu, ensemble_predictions_jikimu)

            mape_jikimu = np.mean(np.abs((y_test_jikimu - ensemble_predictions_jikimu) / y_test_jikimu)) * 100
            daily_returns_jikimu = y_test_jikimu.pct_change().dropna()
            annualized_returns_jikimu = calculate_annualized_returns(daily_returns_jikimu)
            sharpe_ratio_jikimu = calculate_sharpe_ratio(daily_returns_jikimu)

            # Calculate and display the predicted NAV value for Jikimu Fund
            predicted_sqrt_nav_jikimu = ensemble_model_jikimu.predict([[log_onu_jikimu, nav_per_unit_jikimu, repurchase_price_per_unit_jikimu, year_jikimu, month_jikimu, day_jikimu]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_jikimu = predicted_sqrt_nav_jikimu ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_jikimu_str = f"{predicted_nav_jikimu:,.2f} TZS"

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
        
        # Add input field for Outstanding Number of Units (ONU)
        onu_watoto = st.number_input('Outstanding Number of Units (ONU)', min_value=0.0)
        
        # Calculate Log_ONU
        log_onu_watoto = np.log(onu_watoto) if onu_watoto > 0 else 0.0  # Avoid log(0) which is undefined
        nav_per_unit_watoto = st.number_input('Nav Per Unit', min_value=0.0)
#         sale_price_per_unit_watoto = st.number_input('Sale Price per Unit', min_value=0.0)
        repurchase_price_per_unit_watoto = st.number_input('Repurchase Price per Unit', min_value=0.0)
        year_watoto = st.number_input('Year', min_value=0)
        month_watoto = st.number_input('Month', min_value=1, max_value=12)
        day_watoto = st.number_input('Day', min_value=1, max_value=31)

        
        if st.button("Forecast"):
            # Model loading and prediction for Watoto Fund
            st.subheader('Prediction Result for Watoto Fund:')

            X_watoto = watoto_data[['Log_ONU', 'Nav Per Unit', 'Repurchase Price/Unit', 'Year', 'Month', 'Day']]
            y_watoto = watoto_data['Sqrt_NAV']
            X_train_watoto, X_test_watoto, y_train_watoto, y_test_watoto = train_test_split(X_watoto, y_watoto, test_size=0.2, random_state=42)

            best_lasso_alpha_watoto = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_watoto = 0.1  # Replace with the best alpha you found
            lasso_model_watoto = Lasso(alpha=best_lasso_alpha_watoto)
            ridge_model_watoto = Ridge(alpha=best_ridge_alpha_watoto)

            models_watoto = [('Lasso', lasso_model_watoto), ('Ridge', ridge_model_watoto)]
            ensemble_model_watoto = VotingRegressor(models_watoto)
            ensemble_model_watoto.fit(X_train_watoto, y_train_watoto)
            ensemble_predictions_watoto = ensemble_model_watoto.predict(X_test_watoto)
            
#             # Save the trained model to a file
#             model_filename = "nav_watoto_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_watoto, model_file)

            mae_watoto = mean_absolute_error(y_test_watoto, ensemble_predictions_watoto)
            mse_watoto = mean_squared_error(y_test_watoto, ensemble_predictions_watoto)
            rmse_watoto = np.sqrt(mse_watoto)
            r2_watoto = r2_score(y_test_watoto, ensemble_predictions_watoto)

            mape_watoto = np.mean(np.abs((y_test_watoto - ensemble_predictions_watoto) / y_test_watoto)) * 100
            daily_returns_watoto = y_test_watoto.pct_change().dropna()
            annualized_returns_watoto = calculate_annualized_returns(daily_returns_watoto)
            sharpe_ratio_watoto = calculate_sharpe_ratio(daily_returns_watoto)

            # Calculate and display the predicted NAV value for Watoto Fund
            predicted_sqrt_nav_watoto = ensemble_model_watoto.predict([[log_onu_watoto, nav_per_unit_watoto, repurchase_price_per_unit_watoto, year_watoto, month_watoto, day_watoto]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_watoto = predicted_sqrt_nav_watoto ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_watoto_str = f"{predicted_nav_watoto:,.2f} TZS"

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
        
#         # Add input field for the number of months ahead to forecast
#         forecast_months = st.number_input('Number of Months Ahead to Forecast', min_value=1, value=6)
        
        # Add input field for Outstanding Number of Units (ONU)
        onu_liquid = st.number_input('Outstanding Number of Units (ONU)', min_value=0.0)
        
        # Calculate Log_ONU
        log_onu_liquid = np.log(onu_liquid) if onu_liquid > 0 else 0.0  # Avoid log(0) which is undefined
        nav_per_unit_liquid = st.number_input('Nav Per Unit', min_value=0.0)
#         sale_price_per_unit_liquid = st.number_input('Sale Price per Unit', min_value=0.0)
        repurchase_price_per_unit_liquid = st.number_input('Repurchase Price per Unit', min_value=0.0)
        year_liquid = st.number_input('Year', min_value=0)
        month_liquid = st.number_input('Month', min_value=1, max_value=12)
        day_liquid = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Liquid Fund
            st.subheader('Prediction Result for Liquid Fund:')

            X_liquid = liquid_data[['Log_ONU', 'Nav Per Unit', 'Repurchase Price/Unit', 'Year', 'Month', 'Day']]
            y_liquid = liquid_data['Sqrt_NAV']
            X_train_liquid, X_test_liquid, y_train_liquid, y_test_liquid = train_test_split(X_liquid, y_liquid, test_size=0.2, random_state=42)

            best_lasso_alpha_liquid = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_liquid = 0.1  # Replace with the best alpha you found
            lasso_model_liquid = Lasso(alpha=best_lasso_alpha_liquid)
            ridge_model_liquid = Ridge(alpha=best_ridge_alpha_liquid)

            models_liquid = [('Lasso', lasso_model_liquid), ('Ridge', ridge_model_liquid)]
            ensemble_model_liquid = VotingRegressor(models_liquid)
            ensemble_model_liquid.fit(X_train_liquid, y_train_liquid)
            ensemble_predictions_liquid = ensemble_model_liquid.predict(X_test_liquid)
            
#             # Save the trained model to a file
#             model_filename = "nav_liquid_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_liquid, model_file)

            mae_liquid = mean_absolute_error(y_test_liquid, ensemble_predictions_liquid)
            mse_liquid = mean_squared_error(y_test_liquid, ensemble_predictions_liquid)
            rmse_liquid = np.sqrt(mse_liquid)
            r2_liquid = r2_score(y_test_liquid, ensemble_predictions_liquid)

            mape_liquid = np.mean(np.abs((y_test_liquid - ensemble_predictions_liquid) / y_test_liquid)) * 100
            daily_returns_liquid = y_test_liquid.pct_change().dropna()
            annualized_returns_liquid = calculate_annualized_returns(daily_returns_liquid)
            sharpe_ratio_liquid = calculate_sharpe_ratio(daily_returns_liquid)

            # Calculate and display the predicted NAV value for Liquid Fund
            predicted_sqrt_nav_liquid = ensemble_model_liquid.predict([[log_onu_liquid, nav_per_unit_liquid, repurchase_price_per_unit_liquid, year_liquid, month_liquid, day_liquid]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_liquid = predicted_sqrt_nav_liquid ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_liquid_str = f"{predicted_nav_liquid:,.2f} TZS"

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
        
        # Add input field for Outstanding Number of Units (ONU)
        onu_umoja = st.number_input('Outstanding Number of Units (ONU)', min_value=0.0)
        
        # Calculate Log_ONU
        log_onu_umoja = np.log(onu_umoja) if onu_umoja > 0 else 0.0  # Avoid log(0) which is undefined
        nav_per_unit_umoja = st.number_input('Nav Per Unit', min_value=0.0)
#         sale_price_per_unit_umoja = st.number_input('Sale Price per Unit', min_value=0.0)
        repurchase_price_per_unit_umoja = st.number_input('Repurchase Price per Unit', min_value=0.0)
        year_umoja = st.number_input('Year', min_value=0)
        month_umoja = st.number_input('Month', min_value=1, max_value=12)
        day_umoja = st.number_input('Day', min_value=1, max_value=31)

        if st.button("Forecast"):
            # Model loading and prediction for Umoja Fund
            st.subheader('Prediction Result for Umoja Fund:')

            X_umoja = umoja_data[['Log_ONU', 'Nav Per Unit', 'Repurchase Price/Unit', 'Year', 'Month', 'Day']]
            y_umoja = umoja_data['Sqrt_NAV']
            X_train_umoja, X_test_umoja, y_train_umoja, y_test_umoja = train_test_split(X_umoja, y_umoja, test_size=0.2, random_state=42)

            best_lasso_alpha_umoja = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_umoja = 0.1  # Replace with the best alpha you found
            lasso_model_umoja = Lasso(alpha=best_lasso_alpha_umoja)
            ridge_model_umoja = Ridge(alpha=best_ridge_alpha_umoja)

            models_umoja = [('Lasso', lasso_model_umoja), ('Ridge', ridge_model_umoja)]
            ensemble_model_umoja = VotingRegressor(models_umoja)
            ensemble_model_umoja.fit(X_train_umoja, y_train_umoja)
            ensemble_predictions_umoja = ensemble_model_umoja.predict(X_test_umoja)
            
#             # Save the trained model to a file
#             model_filename = "nav_umoja_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_umoja, model_file)

            mae_umoja = mean_absolute_error(y_test_umoja, ensemble_predictions_umoja)
            mse_umoja = mean_squared_error(y_test_umoja, ensemble_predictions_umoja)
            rmse_umoja = np.sqrt(mse_umoja)
            r2_umoja = r2_score(y_test_umoja, ensemble_predictions_umoja)

            mape_umoja = np.mean(np.abs((y_test_umoja - ensemble_predictions_umoja) / y_test_umoja)) * 100
            daily_returns_umoja = y_test_umoja.pct_change().dropna()
            annualized_returns_umoja = calculate_annualized_returns(daily_returns_umoja)
            sharpe_ratio_umoja = calculate_sharpe_ratio(daily_returns_umoja)

            # Calculate and display the predicted NAV value for Umoja Fund
            predicted_sqrt_nav_umoja = ensemble_model_umoja.predict([[log_onu_umoja, nav_per_unit_umoja, repurchase_price_per_unit_umoja, year_umoja, day_umoja, month_umoja]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_umoja = predicted_sqrt_nav_umoja ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_umoja_str = f"{predicted_nav_umoja:,.2f} TZS"

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
        
#         # Add input field for the number of months ahead to forecast
#         forecast_months = st.number_input('Number of Months Ahead to Forecast', min_value=1, value=6)
        
        # Add input field for Outstanding Number of Units (ONU)
        onu_wekeza_maisha = st.number_input('Outstanding Number of Units (ONU)', min_value=0.0)
        
        # Calculate Log_ONU
        log_onu_wekeza_maisha = np.log(onu_wekeza_maisha) if onu_wekeza_maisha > 0 else 0.0  # Avoid log(0) which is undefined
        nav_per_unit_wekeza_maisha = st.number_input('Nav Per Unit', min_value=0.00)
#         sale_price_per_unit_wekeza_maisha = st.number_input('Sale Price per Unit', min_value=0.00)
        repurchase_price_per_unit_wekeza_maisha = st.number_input('Repurchase Price per Unit', min_value=0.00)
        year_wekeza_maisha = st.number_input('Year', min_value=0)
        month_wekeza_maisha = st.number_input('Month', min_value=1, max_value=12)
        day_wekeza_maisha = st.number_input('Day', min_value=1, max_value=31)

        
        if st.button("Forecast"):
            # Model loading and prediction for Wekeza Maisha Fund
            st.subheader('Prediction Result for Wekeza Maisha Fund:')
                     
            X_wekeza_maisha = wekeza_maisha_data[['Log_ONU', 'Nav Per Unit', 'Repurchase Price/Unit', 'Year', 'Month', 'Day']]
            y_wekeza_maisha = wekeza_maisha_data['Sqrt_NAV']
            X_train_wekeza_maisha, X_test_wekeza_maisha, y_train_wekeza_maisha, y_test_wekeza_maisha = train_test_split(X_wekeza_maisha, y_wekeza_maisha, test_size=0.2, random_state=42)

            best_lasso_alpha_wekeza_maisha = 0.01  # Replace with the best alpha you found
            best_ridge_alpha_wekeza_maisha = 0.1  # Replace with the best alpha you found
            lasso_model_wekeza_maisha = Lasso(alpha=best_lasso_alpha_wekeza_maisha)
            ridge_model_wekeza_maisha = Ridge(alpha=best_ridge_alpha_wekeza_maisha)

            models_wekeza_maisha = [('Lasso', lasso_model_wekeza_maisha), ('Ridge', ridge_model_wekeza_maisha)]
            ensemble_model_wekeza_maisha = VotingRegressor(models_wekeza_maisha)
            ensemble_model_wekeza_maisha.fit(X_train_wekeza_maisha, y_train_wekeza_maisha)
            ensemble_predictions_wekeza_maisha = ensemble_model_wekeza_maisha.predict(X_test_wekeza_maisha)
            
#             # Save the trained model to a file
#             model_filename = "nav_wekeza_maisha_model.pkl"
#             with open(model_filename, 'wb') as model_file:
#                 joblib.dump(ensemble_model_wekeza_maisha, model_file)

            mae_wekeza_maisha = mean_absolute_error(y_test_wekeza_maisha, ensemble_predictions_wekeza_maisha)
            mse_wekeza_maisha = mean_squared_error(y_test_wekeza_maisha, ensemble_predictions_wekeza_maisha)
            rmse_wekeza_maisha = np.sqrt(mse_wekeza_maisha)
            r2_wekeza_maisha = r2_score(y_test_wekeza_maisha, ensemble_predictions_wekeza_maisha)

            mape_wekeza_maisha = np.mean(np.abs((y_test_wekeza_maisha - ensemble_predictions_wekeza_maisha) / y_test_wekeza_maisha)) * 100
            daily_returns_wekeza_maisha = y_test_wekeza_maisha.pct_change().dropna()
            annualized_returns_wekeza_maisha = calculate_annualized_returns(daily_returns_wekeza_maisha)
            sharpe_ratio_wekeza_maisha = calculate_sharpe_ratio(daily_returns_wekeza_maisha)

            # Calculate and display the predicted NAV value for Wekeza Maisha Fund
            predicted_sqrt_nav_wekeza_maisha = ensemble_model_wekeza_maisha.predict([[log_onu_wekeza_maisha, nav_per_unit_wekeza_maisha, repurchase_price_per_unit_wekeza_maisha, year_wekeza_maisha, month_wekeza_maisha, day_wekeza_maisha]])[0]

            # Square the predicted sqrt NAV value to get the exact value
            predicted_nav_wekeza_maisha = predicted_sqrt_nav_wekeza_maisha ** 2

            # Format the predicted NAV as a currency string (TZS)
            predicted_nav_wekeza_maisha_str = f"{predicted_nav_wekeza_maisha:,.2f} TZS"

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

#theme
hide_st_style="""

<style>



</style>



"""
