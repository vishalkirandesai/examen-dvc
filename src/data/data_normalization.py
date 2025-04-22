import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import os

def main():
    """ Runs data normalization script to turn train and test variable data from (../processed_data) into scaled data.
    """
    logger = logging.getLogger(__name__)
    logger.info('Normalizing data')

    data_dir = "data/processed_data"

    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), parse_dates = ['date'])
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), parse_dates = ['date'])
    logger.info(f'Loaded data from {data_dir}/X_train.csv and {data_dir}/X_test.csv')

    print(X_train.info())

    X_train["year"] = X_train["date"].dt.year
    X_train["month"] = X_train["date"].dt.month
    X_train["day"] = X_train["date"].dt.day
    X_train["hour"] = X_train["date"].dt.hour

    X_test["year"] = X_test["date"].dt.year
    X_test["month"] = X_test["date"].dt.month
    X_test["day"] = X_test["date"].dt.day
    X_test["hour"] = X_test["date"].dt.hour

    # Drop the original date column
    X_train.drop(columns=['date'], inplace=True)
    X_test.drop(columns=['date'], inplace=True)
    logger.info('Date column processed and dropped')

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
    logger.info('Data normalization completed')

    X_train_scaled.to_csv(os.path.join(data_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(data_dir, 'X_test_scaled.csv'), index=False)

    print(X_train_scaled.info())
    print(X_train_scaled.head())
    logger.info('Normalized data saved to processed_data directory')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()