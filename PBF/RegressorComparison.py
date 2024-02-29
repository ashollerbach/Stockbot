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

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.to_csv('PBF\PBF_Accuracy.csv')

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


dates = list(range(len(y_pred)))
holder = np.arange(30)


plt.figure(figsize=(8, 6))
plt.plot(holder, y_pred[-30:], color='blue', label='Predicted Percent Change')
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('PBF Predicted Stock Price')
plt.legend()  # Add a legend with the label specified in the plot function
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()