import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\Program Files\linear_regression_data.csv'
data = pd.read_csv(file_path)

# Extracting independent and dependent variables
# Assuming the first column is the independent variable (X) and the second column is the dependent variable (Y)
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

# Compute the mean of X and Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Compute the covariance and variance
cov_XY = np.sum((X - mean_X) * (Y - mean_Y))
var_X = np.sum((X - mean_X) ** 2)

# Calculate the coefficients (slope and intercept) using the covariance approach
slope = cov_XY / var_X
intercept = mean_Y - slope * mean_X

# Predict Y values using the linear model
Y_pred = intercept + slope * X

# Plotting the data points and the linear regression line
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, Y_pred, color='red', label='Regression line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (Y)')
plt.title('Linear Regression using Covariance Approach')
plt.legend()
plt.show()

# Output the slope and intercept
slope, intercept
