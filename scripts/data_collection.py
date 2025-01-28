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

    # Specify the full path for saving the CSV file
    output_file_path = os.path.join(os.getcwd(), 'historical_stock_data_last_10_years.csv')  # Saves in current directory

    # Save to CSV
    historical_data.to_csv(output_file_path, index=False)
    print(f"Data saved successfully to {output_file_path}")

    # Data cleaning steps:
    # 1. Drop the 'Adj Close' column
    historical_data.drop(columns=['Adj Close'], inplace=True)

    # 2. Remove rows with 'Dividend' in any of the price columns
    # Filter out rows where any price-related column contains 'Dividend'
    historical_data = historical_data[~historical_data['Open'].str.contains('Dividend', na=False)]

    historical_data['Volume'] = historical_data['Volume'].str.replace(',', '').astype(float)


    # Now save the cleaned data to a new CSV file
    historical_data.to_csv('cleaned_historical_stock_data.csv', index=False)

    df = pd.read_csv('cleaned_historical_stock_data.csv')

    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

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

    # Print the data
    print("Historical Stock Data:")
    print(historical_data.head())

except Exception as e:
    print("Error while scraping:", e)

finally:
    # Quit the browser
    driver.quit()
