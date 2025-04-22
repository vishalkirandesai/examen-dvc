import pandas as pd
from pathlib import Path
import logging
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import os
import joblib

def main():
    """ Runs grid search to search for the best parameters for the regression model.
    """
    logger = logging.getLogger(__name__)
    logger.info('Running grid search')

    data_dir = "data/processed_data"

    # Load the scaled data
    X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    logger.info(f'Loaded data from {data_dir}/X_train_scaled.csv')

    # Load the target data
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    logger.info(f'Loaded target data from {data_dir}/y_train.csv')

    model = Ridge()

    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 50, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga'],
        'max_iter': [1000, 5000, 10000]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=3
    )

    # Fit the grid search to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation R^2 Score: {best_score}")

    best_model = grid_search.best_estimator_

    joblib.dump(best_model, "models/ridge_linear_regression.pkl")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()