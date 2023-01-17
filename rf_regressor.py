import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import max_error as max_e
from sklearn.metrics import median_absolute_error as med_ae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_gamma_deviance as mgd
from sklearn.metrics import mean_poisson_deviance as mpd
from sklearn.metrics import mean_tweedie_deviance as mtd


"""
Load and prepare dataset
"""
data = pd.read_csv('data.csv')
y = data['T2M']
X = data.drop('T2M', axis=1)
# print(X.shape)
# print(y.shape)


"""
Train-Test Splitting
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=2023)


"""
Fit model on training data
"""
model = RandomForestRegressor(random_state=2023)
model.fit(X_train, y_train)


"""
Evaluate model on test data
"""
y_pred = model.predict(X_test)

print("Mean Squared Error:", mse(y_test, y_pred))
print("Mean Absolute Error:", mae(y_test, y_pred))
print("Explained Variance Score:", evs(y_test, y_pred))
print("Max Error:", max_e(y_test, y_pred))
print("Median Absolute Error:", med_ae(y_test, y_pred))
print("R^2:", r2(y_test, y_pred))
print("Mean Squared Log Error:", msle(y_test, y_pred))
print("Mean Gamma Deviance:", mgd(y_test, y_pred))
print("Mean Poisson Deviance:", mpd(y_test, y_pred))
print("Mean Tweedie Deviance:", mtd(y_test, y_pred))
