from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta

# Set up the Chrome driver
driver = webdriver.Chrome()  # Make sure ChromeDriver is in PATH

# Calculate the date 10 years ago from today
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)  # 10 years ago
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Open the Yahoo Finance page for a stock
stock_symbol = "AAPL"
url = f"https://finance.yahoo.com/quote/AAPL/history?p=AAPL&period1={int(start_date.timestamp())}&period2={int(end_date.timestamp())}"
driver.get(url)

# Create the dataset folder if it does not exist
dataset_folder = os.path.join(os.getcwd(), 'dataset')
os.makedirs(dataset_folder, exist_ok=True)

try:
    # Wait for the page to load (use sleep as backup for debugging)
    time.sleep(5)

    # Wait for the historical prices table to load
    table = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/div[1]/div[3]/table'))
    )

    # Extract rows from the table
    rows = table.find_elements(By.TAG_NAME, 'tr')

    # Extract headers
    headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, 'th')]

    # Extract data rows
    data = []
    for row in rows[1:]:
        columns = row.find_elements(By.TAG_NAME, 'td')
        if len(columns) > 0:  # Skip empty rows
            data.append([col.text for col in columns])

    # Create a DataFrame
    historical_data = pd.DataFrame(data, columns=headers)

    # Save the raw data to CSV
    raw_file_path = os.path.join(dataset_folder, 'historical_stock_data_last_10_years.csv')
    historical_data.to_csv(raw_file_path, index=False)
    print(f"Raw data saved successfully to {raw_file_path}")

    # # Specify the full path for saving the CSV file
    # output_file_path = os.path.join(os.getcwd(), 'historical_stock_data_last_10_years.csv')  # Saves in current directory

    # # Save to CSV
    # historical_data.to_csv(output_file_path, index=False)
    # print(f"Data saved successfully to {output_file_path}")

    # Data cleaning steps:
    # 1. Drop the 'Adj Close' column if it exists
    if 'Adj Close*' in historical_data.columns:
        historical_data.drop(columns=['Adj Close*'], inplace=True)

    # 2. Remove rows with 'Dividend' in any of the price columns
    historical_data = historical_data[~historical_data['Open'].str.contains('Dividend', na=False)]

    # 3. Handle missing or None values in 'Volume'
    historical_data['Volume'] = historical_data['Volume'].str.replace(',', '').astype(float)

    # 4. Convert 'Date' column to datetime format
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%b %d, %Y', errors='coerce')

    # Drop rows with missing values in key columns
    cleaned_data = historical_data.dropna(subset=['High', 'Low', 'Close', 'Volume'])

    # Sort the data by 'Date' column in ascending order
    cleaned_data = cleaned_data.sort_values('Date', ascending=True)


    # # 4. Ensure 'Date' column is properly parsed and handle any errors
    # if 'Date' in historical_data.columns:
    #     try:
    #         # First, try parsing as ISO8601
    #         historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%Y-%m-%d', errors='coerce')
    #     except ValueError:
    #         # If ISO8601 parsing fails, try another format
    #         historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%b %d, %Y', errors='coerce')

    # # Drop rows with invalid dates
    # historical_data = historical_data.dropna(subset=['Date'])


    # Save the cleaned data to a new CSV file
    cleaned_file_path = os.path.join(dataset_folder, 'cleaned_historical_stock_data.csv')
    cleaned_data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved successfully to {cleaned_file_path}")

    # Split the dataset into training and test data
    train_size = int(0.8 * len(cleaned_data))
    training_data = cleaned_data[:train_size]
    test_data = cleaned_data[train_size:]

    # Save training and test datasets to CSV files
    training_data.to_csv(os.path.join(dataset_folder,'training_data.csv'), index=False)
    test_data.to_csv(os.path.join(dataset_folder,'test_data.csv'), index=False)

    # Save one data entry from the test dataset to a separate file
    if len(test_data) > 0:
        single_entry_path = os.path.join(dataset_folder,'activation_data.csv')
        test_data.iloc[0:1].to_csv(single_entry_path, index=False)
        print(f"Single data entry saved to {single_entry_path}")

    # Visualization
    # Load the cleaned data for plotting
    df = pd.read_csv(cleaned_file_path)

    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Plot the closing price over time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Closing Price', color='blue')

    # Add titles and labels
    plt.title('Stock Closing Price Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price (USD)', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show gridlines
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Print the first few rows of cleaned data
    print("Cleaned Historical Stock Data:")
    print(cleaned_data.head())

except Exception as e:
    print("Error while scraping or processing:", e)

finally:
    # Quit the browser
    driver.quit()