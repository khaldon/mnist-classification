from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mnist_classification.config import PROCESSED_DATA_DIR

DEFAULT_X_TRAIN_FILENAME = "x_train.parquet"
DEFAULT_X_TEST_FILENAME = "x_test.parquet"
DEFAULT_Y_TRAIN_FILENAME = "y_train.parquet"
DEFAULT_Y_TEST_FILENAME = "y_test.parquet"


# ====================
# 1.LOAD TRAIN DATA
# ====================
def load_train_data(X_train_file=None, y_train_file=None):
    """Load train set"""
    X_train_file = X_train_file or (PROCESSED_DATA_DIR / DEFAULT_X_TRAIN_FILENAME)
    logger.info(f"\n Loading X_train set from {X_train_file}.....")
    y_train_file = y_train_file or (PROCESSED_DATA_DIR / DEFAULT_Y_TRAIN_FILENAME)
    logger.info(f"\n Loading Y_train set from {y_train_file}.....")

    if not X_train_file.exists():
        raise FileNotFoundError(f"X_Train file not found :{X_train_file}")

    if not y_train_file.exists():
        raise FileNotFoundError(f"Y_Train file not found :{y_train_file}")

    X_train, y_train = pd.read_parquet(X_train_file), pd.read_parquet(y_train_file)
    logger.info("\n X_Train data is loaded")
    logger.info("\n y_train data is loaded")
    return X_train, y_train


# ====================
# 2.LOAD TEST DATA
# ====================
def load_test_data(X_test_file=None, y_test_file=None):
    """Load test set"""
    X_test_file = X_test_file or (PROCESSED_DATA_DIR / DEFAULT_X_TEST_FILENAME)
    logger.info(f"\n Loading X_test set from {X_test_file}.....")

    y_test_file = y_test_file or (PROCESSED_DATA_DIR / DEFAULT_Y_TEST_FILENAME)
    logger.info(f"\n Loading y_test set from {y_test_file}.....")

    if not X_test_file.exists():
        raise FileNotFoundError(f"X_Train file not found :{X_test_file}")

    if not y_test_file.exists():
        raise FileNotFoundError(f"Y_Train file not found :{y_test_file}")

    X_test, y_test = pd.read_parquet(X_test_file), pd.read_parquet(y_test_file)
    logger.info("\n X_test data is loaded")
    logger.info("\n y_test data is loaded")
    return X_test, y_test


app = typer.Typer()


@app.command()
def main():
    load_train_data()
    load_test_data()


if __name__ == "__main__":
    app()
