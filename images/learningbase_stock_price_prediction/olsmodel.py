import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load your training and testing data
train_df = pd.read_csv('learningbase_stock_price_prediction\\train\\training_data.csv')
test_df = pd.read_csv('learningbase_stock_price_prediction\\validation\\test_data.csv')

# Prepare data for OLS (train and test sets)
# 'Close' is the target variable, 'Date' is the feature
X_train = pd.to_datetime(train_df['Date']).map(lambda x: x.toordinal()).values.reshape(-1, 1)  # Convert Date to numeric (ordinal)
y_train = train_df['Close']

X_test = pd.to_datetime(test_df['Date']).map(lambda x: x.toordinal()).values.reshape(-1, 1)  # Convert Date to numeric (ordinal)
y_test = test_df['Close']

# Add constant to the independent variables for the intercept term
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Step 1: Fit the OLS model
ols_model = sm.OLS(y_train, X_train).fit()

# Step 2: Make predictions
y_pred_train = ols_model.predict(X_train)
y_pred_test = ols_model.predict(X_test)

# Step 3: Save the trained OLS model using pickle
with open('currentOlsSolution.pkl', 'wb') as file:
    pickle.dump(ols_model, file)

# Step 4: Evaluate the model using performance metrics
train_mse = mean_squared_error(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# Print out the performance metrics
print(f'Train MSE: {train_mse}, Train MAE: {train_mae}, Train R2: {train_r2}')
print(f'Test MSE: {test_mse}, Test MAE: {test_mae}, Test R2: {test_r2}')

# Step 5: Create diagnostic plots (residuals plot and QQ plot)
residuals = y_train - y_pred_train

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Residuals plot
sns.residplot(x=y_pred_train, y=residuals, lowess=True, line_kws={'color': 'red'}, ax=ax[0])
ax[0].set_title('Residuals Plot')

# QQ plot
sm.qqplot(residuals, line='45', ax=ax[1])
ax[1].set_title('QQ Plot')

plt.show()

# Step 6: Scatter plot of predicted vs actual values (for test set)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
