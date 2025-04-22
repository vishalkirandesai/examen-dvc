import pandas as pd
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import os

def main():
    """ Runs data splitting script to turn raw data from (../raw_data/raw.csv) into
        train and test for variables and target (saved in../processed_data).
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting data into train and test sets')

    input_dir = "data/raw_data"
    output_dir = "data/processed_data"
    target = "silica_concentrate"

    df = pd.read_csv(os.path.join(input_dir, 'raw.csv'))
    logger.info(f'Loaded data from {input_dir}/raw.csv')
    logger.info(f'Data shape: {df.shape}')

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logger.info(f'Created output directory: {output_dir}')

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    logger.info('Data splitting completed and saved to processed_data directory')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()