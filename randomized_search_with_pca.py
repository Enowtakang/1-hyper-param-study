import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import decomposition, preprocessing, pipeline


if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data.csv')

    # Create a numpy array of all features
    X = data.drop('T2M', axis=1).values
    # Create a numpy array of all targets
    y = data['T2M'].values

    # Instantiate StandardScaler and PCA
    standard_scaler = preprocessing.StandardScaler()
    # Note that number of PCA components has
    # not been specified below
    pca = decomposition.PCA()

    # Define model
    model = RandomForestRegressor(random_state=2023)

    # Instantiate pipeline
    pipeLine = pipeline.Pipeline(
        [("scaling", standard_scaler),
         ("pca", pca),
         ("model", model)])

    # Train-Test Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=2023)

    # Specify parameter grid
    parameter_grid = {
        "pca__n_components": np.arange(2, 10),
        "model__n_estimators": np.arange(25, 200, 5),
        "model__criterion": ["squared_error",
                             "absolute_error",
                             "friedman_mse",
                             "poisson"],
        "model__max_depth": np.arange(1, 20),
        "model__min_samples_split": np.arange(1, 6, 1)}

    # Instantiate randomized search
    randomized_search = RandomizedSearchCV(
        estimator=pipeLine,
        param_distributions=parameter_grid,
        n_iter=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1, cv=5, refit=True)

    # Fit Grid Search
    randomized_search.fit(X_train, y_train)

    # Print the best set of hyper-parameters
    print("Best Params:", randomized_search.best_params_)
    print("Best Score:", randomized_search.best_score_)
