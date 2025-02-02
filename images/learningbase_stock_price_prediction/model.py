import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Check if running inside a Docker container
if os.path.exists('/.dockerenv'):
    # Running inside a container
    data_path = '/tmp/learningBase/train/training_data.csv'
    test_data_path = '/tmp/learningBase/validation/test_data.csv'
    model_save_path = '/tmp/learningbase/currentAiSolution.keras'
    history_save_path = '/tmp/learningbase/training_history.csv'
    plots_save_path = '/tmp/learningbase/training_validation_plots.png'
    scatter_plot_save_path = '/tmp/learningbase/predictions_vs_actual.png'
else:
    # Running on local system
    data_path = "learningbase_stock_price_prediction\\train\\training_data.csv"  # Adjust path for local data
    test_data_path = "learningbase_stock_price_prediction\\validation\\test_data.csv"  # Adjust path for local test data
    model_save_path = r"learningbase_stock_price_prediction\currentAiSolution.keras" 
    history_save_path = r"learningbase_stock_price_prediction\training_history.csv"  # Adjust history save path
    plots_save_path = r"learningbase_stock_price_prediction\training_validations_plots.png"  # Adjust plots save path
    scatter_plot_save_path = r"learningbase_stock_price_prediction\predictions_vs_actual.png"  # Adjust scatter plot save path

# Load and preprocess data
data = pd.read_csv(data_path)  # Path to training data
data["Date"] = pd.to_datetime(data["Date"])  # Convert to datetime format
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day

# Remove the Date column before using as features
data = data.drop('Date', axis=1)

test_data = pd.read_csv(test_data_path)  # Path to test data
test_data["Date"] = pd.to_datetime(test_data["Date"])  # Convert to datetime format
test_data["Year"] = test_data["Date"].dt.year
test_data["Month"] = test_data["Date"].dt.month
test_data["Day"] = test_data["Date"].dt.day

# Remove the Date column before using as features in the test data
test_data = test_data.drop('Date', axis=1)

# Assume that 'X' are features and 'y' is the target variable
X = data.drop('Close', axis=1).values
y = data['Close'].values

# Scaling features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM (samples, timesteps, features)
timesteps = 30  # Look back over the past 30 days
X_lstm = []

# Create sequences of past `timesteps` days as input for LSTM
for i in range(timesteps, len(X_scaled)):
    X_lstm.append(X_scaled[i-timesteps:i])  # Get past `timesteps` days
X_lstm = np.array(X_lstm)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_lstm, y[timesteps:], test_size=0.2, random_state=42)

# Define LSTM model
model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(1)  # Output layer for regression (next day's stock price)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save(model_save_path)  # Save model to the specified path

# Save training and validation metrics
history_df = pd.DataFrame(history.history)
history_df.to_csv(history_save_path, index=False)

# Plot and save training and validation curves
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# MAE plot (mean absolute error)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')

plt.legend()

plt.tight_layout()
plt.savefig(plots_save_path)
plt.show()

# Create scatter plot of predictions vs actual
y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred) 

plt.scatter(y_val, y_pred)
plt.title('Predictions vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig(scatter_plot_save_path)
plt.show()

print(f"Model training completed and saved. R² score: {r2:.4f}")

# Load the test data for future prediction
X_test = test_data.drop('Close', axis=1).values
X_test_scaled = scaler.transform(X_test)

# Prepare input for LSTM - take the last `timesteps` days
X_input = X_test_scaled[-timesteps:]  # Last `timesteps` days for prediction
X_input = np.array(X_input).reshape(1, timesteps, X_test.shape[1])

# Generate future predictions
future_predictions = []
X_input = X_scaled[-timesteps:].reshape(1, timesteps, X_scaled.shape[1])  # Get last 'timesteps' days

for _ in range(30):  # Predict the next 30 days
    next_pred = model.predict(X_input)  # Predict next day's close price
    future_predictions.append(next_pred[0, 0])  # Store predicted value

    # Create new input row by copying last timestep
    new_row = X_input[:, -1, :].copy()  # Copy last feature row (shape: 1, num_features)
    new_row[0, -1] = next_pred[0, 0]  # Replace last column (Close price)

    # Append new row and shift the sequence
    new_input = np.append(X_input[:, 1:, :], new_row.reshape(1, 1, -1), axis=1)  # Keep shape consistent
    X_input = new_input  # Update for next iteration

# Convert predictions into DataFrame
future_dates = pd.date_range(start=pd.Timestamp.today(), periods=30, freq='D')
future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_predictions})

# Create a dummy array with the same shape as the original scaled data
dummy_input = np.zeros((30, X_scaled.shape[1]))  # Shape (30, 8) if there are 8 features
dummy_input[:, 0] = future_predictions  # Replace only the 'Close' column with predictions

# Perform inverse transform
predicted_prices = scaler.inverse_transform(dummy_input)[:, 0]  # Extract only the 'Close' values


# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(future_df["Date"], future_df["Predicted_Close"], marker='o', linestyle='-', color='b', label="Predicted Prices")

# Formatting the date axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.xticks(rotation=45)

plt.title("Next 30 Days Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Predicted Close Price")
plt.legend()
plt.grid()

# Save the future prediction plot
future_plot_save_path = os.path.join(os.path.dirname(history_save_path), "future_predictions_plot.png")
plt.savefig(future_plot_save_path)
plt.show()

print(f"Future 30-day prediction plot saved at {future_plot_save_path}")

# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# # Check if running inside a Docker container
# if os.path.exists('/.dockerenv'):
#     # Running inside a container
#     data_path = '/tmp/learningBase/train/training_data.csv'
#     test_data_path = '/tmp/learningBase/validation/test_data.csv'
#     model_save_path = '/tmp/learningbase/currentAiSolution.keras'
#     history_save_path = '/tmp/learningbase/training_history.csv'
#     plots_save_path = '/tmp/learningbase/training_validation_plots.png'
#     scatter_plot_save_path = '/tmp/learningbase/predictions_vs_actual.png'
# else:
#     # Running on local system
#     data_path = "C:\\Users\\HP\\AI-CPS\\learningbase_stock_price_prediction\\train\\training_data.csv"  # Adjust path for local data
#     test_data_path = "C:\\Users\\HP\\AI-CPS\\learningbase_stock_price_prediction\\validation\\test_data.csv"  # Adjust path for local test data
#     model_save_path = r"C:\Users\HP\AI-CPS\learningbase_stock_price_prediction\currentAiSolution.keras" 
#     history_save_path = r"C:\Users\HP\AI-CPS\learningbase_stock_price_prediction\training_history.csv"  # Adjust history save path
#     plots_save_path = r"C:\Users\HP\AI-CPS\learningbase_stock_price_prediction\training_validations_plots.png"  # Adjust plots save path
#     scatter_plot_save_path = r"C:\Users\HP\AI-CPS\learningbase_stock_price_prediction\predictions_vs_actual.png"  # Adjust scatter plot save path

# # Load and preprocess data
# data = pd.read_csv(data_path)  # Path to training data
# data["Date"] = pd.to_datetime(data["Date"])  # Convert to datetime format
# data["Year"] = data["Date"].dt.year
# data["Month"] = data["Date"].dt.month
# data["Day"] = data["Date"].dt.day

# # Remove the Date column before using as features
# data = data.drop('Date', axis=1)

# test_data = pd.read_csv(test_data_path)  # Path to test data
# test_data["Date"] = pd.to_datetime(test_data["Date"])  # Convert to datetime format
# test_data["Year"] = test_data["Date"].dt.year
# test_data["Month"] = test_data["Date"].dt.month
# test_data["Day"] = test_data["Date"].dt.day

# # Remove the Date column before using as features in the test data
# test_data = test_data.drop('Date', axis=1)

# # Assume that 'X' are features and 'y' is the target variable
# X = data.drop('Close', axis=1).values
# y = data['Close'].values

# # Scaling features
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Define a simple neural network model
# model = models.Sequential([
#     layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1)  # Output layer for regression (use softmax/sigmoid for classification)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# # Save the model
# model.save(model_save_path)  # Save model to the specified path

# # Save training and validation metrics
# history_df = pd.DataFrame(history.history)
# history_df.to_csv(history_save_path, index=False)

# # Plot and save training and validation curves
# plt.figure(figsize=(12, 6))

# # Loss plot
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Accuracy plot
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.savefig(plots_save_path)

# # Create scatter plot of predictions vs actual
# y_pred = model.predict(X_val)
# plt.scatter(y_val, y_pred)
# plt.title('Predictions vs Actual')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.savefig(scatter_plot_save_path)

# print("Model training completed and saved.")



# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# # Load and preprocess data (adjust this part based on your actual data format)
# data = pd.read_csv('/tmp/learningBase/train/training_data.csv')  # Path to training data in the container
# data["Date"] = pd.to_datetime(data["Date"])  # Convert to datetime format
# data["Year"] = data["Date"].dt.year
# data["Month"] = data["Date"].dt.month
# data["Day"] = data["Date"].dt.day

# # Remove the Date column before using as features
# data = data.drop('Date', axis=1)

# test_data = pd.read_csv('/tmp/learningBase/validation/test_data.csv')  # Path to test data in the container
# test_data["Date"] = pd.to_datetime(test_data["Date"])  # Convert to datetime format
# test_data["Year"] = test_data["Date"].dt.year
# test_data["Month"] = test_data["Date"].dt.month
# test_data["Day"] = test_data["Date"].dt.day

# # Remove the Date column before using as features in the test data
# test_data = test_data.drop('Date', axis=1)

# # Assume that 'X' are features and 'y' is the target variable
# X = data.drop('Close', axis=1).values
# y = data['Close'].values

# # Scaling features
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Define a simple neural network model
# model = models.Sequential([
#     layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1)  # Output layer for regression (use softmax/sigmoid for classification)
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# # Save the model
# model.save('/tmp/learningbase/currentAiSolution.h5')  # Save model to /learningBase/ directory

# # Save training and validation metrics
# history_df = pd.DataFrame(history.history)
# history_df.to_csv('/tmp/learningbase/training_history.csv', index=False)

# # Plot and save training and validation curves
# plt.figure(figsize=(12, 6))

# # Loss plot
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Accuracy plot
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.savefig('/tmp/learningbase/training_validation_plots.png')

# # Create scatter plot of predictions vs actual
# y_pred = model.predict(X_val)
# plt.scatter(y_val, y_pred)
# plt.title('Predictions vs Actual')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.savefig('/tmp/learningbase/predictions_vs_actual.png')

# print("Model training completed and saved.")
