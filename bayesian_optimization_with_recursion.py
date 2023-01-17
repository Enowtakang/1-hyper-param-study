import pandas as pd
import numpy as np
from functools import partial
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from skopt import space
from skopt import gp_minimize


def optimize(params, param_names, a, b):
    """Dictionary of Parameters"""
    params = dict(zip(param_names, params))
    """Define model"""
    opt_model = RandomForestRegressor(**params)
    """metrics (mean square errors)"""
    mean_sq_err = []
    """Cross_validation"""
    kf = KFold(n_splits=5)
    """Iterate"""
    for idx in kf.split(X=a, y=b):
        train_idx, test_idx = idx[0], idx[1]
        x_train = a[train_idx]
        yy_train = b[train_idx]

        x_test = a[test_idx]
        yy_test = b[test_idx]

        opt_model.fit(x_train, yy_train)
        yy_pred = opt_model.predict(x_test)
        fold_mse = mse(yy_test, yy_pred)
        mean_sq_err.append(fold_mse)

    return -1.0 * np.mean(mean_sq_err)


if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data.csv')

    # Create a numpy array of all features
    X = data.drop('T2M', axis=1).values
    # Create a numpy array of all targets
    y = data['T2M'].values

    # Define param space
    cats = ["mse", "mae"]
    param_space = [
        space.Integer(160, 180, name="n_estimators"),
        space.Categorical(cats, name="criterion"),
        space.Integer(9, 14, name="max_depth"),
        space.Integer(2, 5, name="min_samples_split")]

    # Param names
    param_names = ["n_estimators", "criterion",
                   "max_depth", "min_samples_split"]

    # Optimization Function
    optimization_function = partial(
        optimize,
        param_names=param_names, a=X, b=y)

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=10,
        n_random_starts=10, random_state=2023)

    print(dict(zip(param_names, result.x)))
