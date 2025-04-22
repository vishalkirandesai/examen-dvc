import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    """ Evaluates the trained model.
    """
    logger = logging.getLogger(__name__)
    logger.info('Evaluating the trained ridge regression model')

    data_dir = "data/processed_data"

    # Load the scaled data
    X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    X_test_scaled = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'))
    logger.info(f'Loaded data from {data_dir}/X_train_scaled.csv')

    # Load the target data
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    logger.info(f'Loaded target data from {data_dir}/y_train.csv')

    model = joblib.load("models/ridge_linear_regression_trained.pkl")
    logger.info('Loaded trained model from models/ridge_linear_regression_trained.pkl')

    # Evaluate the model
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    logger.info(f'Model evaluation metrics: MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2 (test): {r2_test}, R^2 (train): {r2_train}')

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_test': r2_test,
        'r2_train': r2_train
    }

    metrics_path = "metrics/metrics.json"

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
        
    logger.info(f'Metrics saved to {metrics_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()