import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Load the data from the CSV file
data = pd.read_csv('PBF\PBF_Data.csv')

# Extract input features (X) and target labels (y)
X = data.drop(columns=['Date', 'Volume', 'Open', 'High', 'Low'])
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Initialize the Random Forest classifier
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=15)

# Train the classifier on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


#Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

#predict future values
future_vals = rf_regressor.predict(X)
future_vals_range = 200

# Generate future dates for plotting
def generate_dates(start_date, end_date):
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday (0 to 4)
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

date_holder =  data.iloc[0,0] #string holder to be converted to an int
start_date = datetime.strptime(date_holder, "%m/%d/%Y")
end_date = start_date + timedelta(days=future_vals_range)   # End date
future_dates = generate_dates(start_date, end_date)
 

 #Convert future_vals to a pandas Series
future_vals_series = pd.Series(future_vals)

# Calculate percent change
future_percent_change = future_vals_series.pct_change() * 100


# Filter out NaN and Inf values
future_percent_change_filtered = future_percent_change[~future_percent_change.isnull() & ~future_percent_change.isin([np.inf, -np.inf])]

percent = False # show percent, set false for actual values
if(percent):
  # Plot predicted future values
  plt.figure(figsize=(8, 6))
  plt.plot(future_dates[:future_vals_range], future_percent_change_filtered[:future_vals_range], color='blue', label='Predicted Percent Change')
  #plt.plot(future_dates[:20], future_vals[:20], color='blue', label='Predicted Percent Change')
  plt.grid(True)
  plt.xlabel('Date')
  plt.ylabel('Percent Change')
  plt.title('PBF Predicted Percent Change')
  plt.legend()  # Add a legend with the label specified in the plot function
  data_range = max(abs(future_percent_change_filtered))
  plt.ylim(-data_range, data_range)  # Set y-axis limits symmetrically around 0
  plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
  plt.show()
elif(percent == False):
  # Plot predicted future values
  plt.figure(figsize=(8, 6))
  plt.plot(future_dates, future_vals[:len(future_dates)], color='blue', label='Predicted Percent Change')
  plt.grid(True)
  plt.xlabel('Date')
  plt.ylabel('Stock Price')
  plt.title('PBF Predicted Stock Price')
  plt.legend()  # Add a legend with the label specified in the plot function
  plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
  plt.show()