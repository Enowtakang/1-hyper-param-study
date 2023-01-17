import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, pipeline


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    X = data.drop('T2M', axis=1).values
    y = data['T2M'].values

    # Instantiate Pipeline
    standard_scaler = preprocessing.StandardScaler()
    model = RandomForestRegressor(random_state=2023)
    pipeLine = pipeline.Pipeline(
        [("scaling", standard_scaler), ("model", model)])

    # Train-Test Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=2023)

    # Specify parameter grid
    cats = ["mse", "mae"]
    parameter_grid = {
        "model__n_estimators": Integer(low=160, high=180),
        "model__criterion": Categorical(cats),
        "model__max_depth": Integer(low=9, high=14),
        "model__min_samples_split": Real(
            low=0, high=0.5,
            prior="uniform",)}

    # Initiate Bayesian Search class
    bayesian_search = BayesSearchCV(
        pipeLine,
        parameter_grid,
        n_iter=6,
        optimizer_kwargs={
            "base_estimator": "RF",
            "n_initial_points": 10,
            "initial_point_generator": "random",
            "acq_func": "LCB",
            "acq_optimizer": "auto",
            "n_jobs": -1,
            "random_state": 2023,
            "acq_func_kwargs": {"kappa": 1.96}
        },
        random_state=2023,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        refit=True, verbose=10)

    # Fit bayesian search class
    bayesian_search.fit(X_train, y_train)

    # Print the best set of hyper-parameters
    print("Best Params:", bayesian_search.best_params_)
    print("Best Score:", bayesian_search.best_score_)
