import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data
path = 'Data/Net Asset Value.csv'
data = pd.read_csv(path)

# Data preprocessing (cleaning and transformation)
def clean_and_extract_number(s):
    cleaned_value = re.sub(r'[^\d.]', '', str(s))
    return cleaned_value

numeric_columns = ['Net Asset Value', 'Outstanding Number of Units']
for col in numeric_columns:
    data[col] = data[col].apply(clean_and_extract_number)

data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data['Date Valued'] = pd.to_datetime(data['Date Valued'].str.replace('/', '-'), format='%d-%m-%Y', errors='coerce')
data['Year'] = pd.to_datetime(data['Date Valued']).dt.year
data['Month'] = pd.to_datetime(data['Date Valued']).dt.month

# Set a consistent style for Seaborn plots
sns.set_style("whitegrid")

# Streamlit app layout
st.title("Net Asset Value Analysis Dashboard")

# Add a sidebar for user interaction
st.sidebar.header("Options")
selected_hypothesis = st.sidebar.selectbox("Select Hypothesis", ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3", "Hypothesis 4", "Hypotheses 5-6", "Hypotheses 7-8", "Hypothesis 9", "Hypothesis 10", "Hypothesis 11", "Hypothesis 12"])

if selected_hypothesis == "Hypothesis 1":
    # Hypothesis 1: Time Series Plot of NAV
    st.header("Hypothesis 1: Time Series Plot of NAV")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Date Valued', y='Net Asset Value', hue='Scheme Name')
    plt.title("Time Series Plot of NAV")
    plt.xlabel("Date")
    plt.ylabel("Net Asset Value")
    plt.xticks(rotation=45)
    st.pyplot()

elif selected_hypothesis == "Hypothesis 2":
    # Hypothesis 2: Seasonal Decomposition Plot
    st.header("Hypothesis 2: Seasonal Decomposition Plot")
    fund_names = data['Scheme Name'].unique()

    # Loop through each fund for seasonal decomposition
    for fund_name in fund_names:
        fund_data = data[data['Scheme Name'] == fund_name]
        nav_series = fund_data['Net Asset Value']

        # Manually specify the seasonal period based on your data
        seasonal_period = 12  # Example: assuming monthly data (adjust as needed)

        # Calculate the rolling mean for the seasonal component
        seasonal_component = nav_series.rolling(window=seasonal_period, center=True).mean()

        # Calculate the trend (detrended) component
        trend_component = nav_series - seasonal_component

        # Plot the seasonal decomposition for each fund
        plt.figure(figsize=(10, 6))
        plt.plot(nav_series.index, nav_series.values, label='Original NAV')
        plt.plot(nav_series.index, trend_component, label='Trend')
        plt.plot(nav_series.index, seasonal_component, label='Seasonal')
        plt.title(f"Seasonal Decomposition of {fund_name} NAV")
        plt.xlabel("Date")
        plt.ylabel("Net Asset Value")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot()

elif selected_hypothesis == "Hypothesis 3":
    # Hypothesis 3: Volatility Comparison
    st.header("Hypothesis 3: Volatility Comparison")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Scheme Name', y='Net Asset Value', showfliers=False)
    plt.title("Volatility Comparison Across Funds")
    plt.xlabel("Scheme Name")
    plt.ylabel("Net Asset Value")
    plt.xticks(rotation=45)
    st.pyplot()

elif selected_hypothesis == "Hypothesis 4":
    # Hypothesis 4: Total Net Asset Value by Scheme
    st.header("Hypothesis 4: Total Net Asset Value by Scheme")
    grouping = data.groupby('Scheme Name')['Net Asset Value'].sum()
    plt.figure(figsize=(10, 6))
    ax = grouping.plot(kind='bar', stacked=True)
    plt.xlabel('Scheme Name')
    plt.ylabel('Total Net Asset Value')
    plt.title('Total Net Asset Value by Scheme')
    plt.xticks(rotation=45)
    st.pyplot()

elif selected_hypothesis == "Hypotheses 5-6":
    # Hypothesis 5: Average Net Asset Value Over the Years
    st.header("Hypothesis 5: Average Net Asset Value Over the Years")
    scheme_avg_nav = data.groupby(['Year', 'Scheme Name'])['Net Asset Value'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=scheme_avg_nav, x='Year', y='Net Asset Value', hue='Scheme Name')
    plt.xlabel('Year')
    plt.ylabel('Average Net Asset Value')
    plt.title('Average Net Asset Value of Schemes Over the Years')
    plt.xticks(rotation=45)
    plt.legend(title='Scheme Name', loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot()

    # Hypothesis 6: Net Asset Value Trends Over Time
    st.header("Hypothesis 6: Net Asset Value Trends Over Time")
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=data, x='Year', y='Net Asset Value', hue='Scheme Name')
    plt.xlabel('Year')
    plt.ylabel('Net Asset Value')
    plt.title('Net Asset Value Trends Over Time')
    plt.xticks(rotation=45)
    plt.legend(title='Scheme Name', loc='upper left', bbox_to_anchor=(1, 1))

    # Create subplots for Hypotheses 5 and 6 side by side
    st.pyplot()

elif selected_hypothesis == "Hypotheses 7-8":
    # Hypothesis 7: Average Net Asset Value by Month
    st.header("Hypothesis 7: Average Net Asset Value by Month")
    data['Month'] = pd.to_datetime(data['Date Valued']).dt.month
    scheme_avg_nav = data.groupby(['Month', 'Scheme Name'])['Net Asset Value'].mean().reset_index()
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plt.figure(figsize=(12, 6))
    sns.barplot(data=scheme_avg_nav, x='Month', y='Net Asset Value', hue='Scheme Name')
    plt.xlabel('Month')
    plt.ylabel('Average Net Asset Value')
    plt.title('Average Net Asset Value of Schemes by Month')
    plt.xticks(ticks=range(12), labels=month_names, rotation=45)
    plt.legend(title='Scheme Name', loc='upper left', bbox_to_anchor=(1, 1))

    # Hypothesis 8: Net Asset Value Trends by Month
    st.header("Hypothesis 8: Net Asset Value Trends by Month")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Month', y='Net Asset Value', hue='Scheme Name')
    plt.xlabel('Month')
    plt.ylabel('Net Asset Value')
    plt.title('Net Asset Value Trends by Month')
    plt.xticks(rotation=45)
    plt.legend(title='Scheme Name', loc='upper left', bbox_to_anchor=(1, 1))

    # Create subplots for Hypotheses 7 and 8 side by side
    st.pyplot()
    
elif selected_hypothesis == "Hypothesis 9":
    # Hypothesis 9: Net Asset Value Distribution by Scheme Name
    st.header("Hypothesis 9: Net Asset Value Distribution by Scheme Name")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x="Scheme Name", y="Net Asset Value")
    plt.title("Net Asset Value Distribution by Scheme Name")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Scheme Name")
    plt.ylabel("Net Asset Value")
    st.pyplot()

elif selected_hypothesis == "Hypothesis 10":
    # Hypothesis 10: Highest and Lowest Net Asset Values by Year
    st.header("Hypothesis 10: Highest and Lowest Net Asset Values by Year")
    max_nav = data.groupby("Year")["Net Asset Value"].max()
    min_nav = data.groupby("Year")["Net Asset Value"].min()
    plt.figure(figsize=(10, 6))
    plt.stem(max_nav.index, max_nav, basefmt=" ", linefmt="-g", markerfmt="go", label="Max NAV")
    plt.stem(min_nav.index, min_nav, basefmt=" ", linefmt="-r", markerfmt="ro", label="Min NAV")
    plt.title("Highest and Lowest Net Asset Values by Year")
    plt.xlabel("Year")
    plt.ylabel("Net Asset Value")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot()

elif selected_hypothesis == "Hypothesis 11":
    # Hypothesis 11: Average Net Asset Value of Schemes Over the Months (2017-2020)
    st.header("Hypothesis 11: Average Net Asset Value of Schemes Over the Months (2017-2020)")
    filtered_data = data[(data['Year'] >= 2017) & (data['Year'] <= 2020)]
    month_nav_avg = filtered_data.groupby(['Month', 'Scheme Name'])['Net Asset Value'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=month_nav_avg, x='Month', y='Net Asset Value', hue='Scheme Name')
    plt.xlabel('Month')
    plt.ylabel('Average Net Asset Value')
    plt.title('Average Net Asset Value of Schemes Over the Months (2017-2020)')
    plt.xticks(rotation=45)
    plt.legend(title='Scheme Name', loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot()

elif selected_hypothesis == "Hypothesis 12":
    # Hypothesis 12: Distribution of Net Asset Values
    st.header("Hypothesis 12: Distribution of Net Asset Values")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Net Asset Value'], kde=True)
    plt.title('Distribution of Net Asset Values')
    plt.xlabel('Net Asset Value')
    plt.ylabel('Frequency')
    st.pyplot()

# Run the Streamlit app
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable deprecated warning
    st.set_option('deprecation.showfileUploaderEncoding', False)  # Disable deprecated warning