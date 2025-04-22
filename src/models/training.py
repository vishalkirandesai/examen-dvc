import pandas as pd
from pathlib import Path
import logging
import os
import joblib

def main():
    """ Runs training for the regression model.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training the ridge regression model')

    data_dir = "data/processed_data"

    # Load the scaled data
    X_train_scaled = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    logger.info(f'Loaded data from {data_dir}/X_train_scaled.csv')

    # Load the target data
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    logger.info(f'Loaded target data from {data_dir}/y_train.csv')

    model = joblib.load("models/ridge_linear_regression.pkl")

    model.fit(X_train_scaled, y_train)
    logger.info('Model training completed')

    # Save the trained model
    joblib.dump(model, "models/ridge_linear_regression_trained.pkl")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()