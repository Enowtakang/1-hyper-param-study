import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data.csv')

    # Create a numpy array of all features
    X = data.drop('T2M', axis=1).values
    # Create a numpy array of all targets
    y = data['T2M'].values

    # Define model
    model = RandomForestRegressor(
        # The exact same process to be repeated every time
        random_state=2023)

    # Train-Test Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=2023)

    # Specify parameter grid
    parameter_grid = {
        "n_estimators": [25, 50, 100, 150, 200],
        "criterion": ["squared_error",
                      "absolute_error",
                      "friedman_mse",
                      "poisson"],
        "max_depth": [3, 5, 10, 15, 20, None],
        "min_samples_split": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}

    # Instantiate grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring='neg_mean_squared_error',
        n_jobs=-1, cv=5, refit=True)

    # Fit Grid Search
    grid_search.fit(X_train, y_train)

    # Print the best set of hyper-parameters
    print("Best Params:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
