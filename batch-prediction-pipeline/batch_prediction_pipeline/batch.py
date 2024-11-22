from datetime import datetime
from pathlib import Path
from typing import Optional

import hopsworks
import pandas as pd

from batch_prediction_pipeline.modules import data
from batch_prediction_pipeline.modules import settings
from batch_prediction_pipeline.modules import utils
from datetime import datetime, timedelta


logger = utils.get_logger(__name__)


def predict(
    feature_view_version: Optional[int] = None,
    model_version: Optional[int] = None,
) -> None:
    """Main function used to do batch predictions.

    Args:
        fh (int, optional): forecast horizon. Defaults to 24.
        feature_view_version (Optional[int], optional): feature store feature view version. If None is provided, it will try to load it from the cached feature_view_metadata.json file.
        model_version (Optional[int], optional): model version to load from the model registry. If None is provided, it will try to load it from the cached train_metadata.json file.
        start_datetime (Optional[datetime], optional): start datetime used for extracting features for predictions. If None is provided, it will try to load it from the cached feature_pipeline_metadata.json file.
        end_datetime (Optional[datetime], optional): end datetime used for extracting features for predictions. If None is provided, it will try to load it from the cached feature_pipeline_metadata.json file.
    """

    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]
    if model_version is None:
        train_metadata = utils.load_json("train_metadata.json")
        model_version = train_metadata["model_version"]

    logger.info("Connecting to the feature store...")
    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"],
        project=settings.SETTINGS["FS_PROJECT_NAME"],
    )
    fs = project.get_feature_store()
    logger.info("Successfully connected to the feature store.")

    logger.info("Loading data from feature store...")
    X, y, prediction_time = data.load_data_from_feature_store(
        fs
    )
    logger.info("Successfully loaded data from feature store.")

    logger.info("Loading model from model registry...")
    model = load_model_from_model_registry(project, model_version)
    logger.info("Successfully loaded model from model registry.")

    logger.info("Making predictions...")
    predictions = forecast(model, X)
    logger.info("Successfully made predictions.")

    logger.info("Saving predictions...")
    predictions = pd.DataFrame(predictions)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X = reshape_x(prediction_time, X)
    predictions.index = [pd.to_datetime(prediction_time)]
    y.index = [pd.to_datetime(prediction_time)]
    save(X, y, predictions)
    logger.info("Successfully saved predictions.")



def load_model_from_model_registry(project, model_version: int):
    """
    This function loads a model from the Model Registry.
    The model is downloaded, saved locally, and loaded into memory.
    """

    mr = project.get_model_registry()
    model_registry_reference = mr.get_model(name="best_model", version=model_version)
    model_dir = model_registry_reference.download()
    model_path = Path(model_dir) / "best_model.pkl"

    model = utils.load_model(model_path)

    return model


def forecast(model, X: pd.DataFrame, fh: int = 24):
    """
    Get a forecast of the total load for the given areas and consumer types.

    Args:
        model (sklearn.base.BaseEstimator): Fitted model that implements the predict method.
        X (pd.DataFrame): Exogenous data with area, consumer_type, and datetime_utc as index.
        fh (int): Forecast horizon.

    Returns:
    """

    predictions = model.predict(X=X)
    predictions = pd.DataFrame(predictions)
    return predictions


def save(X: pd.DataFrame, y: pd.DataFrame, predictions: pd.DataFrame):
    """Save the input data, target data, and predictions to GCS."""

    # Ensure X, y, and predictions are DataFrames

    # Get the bucket object from the GCS client.
    bucket = utils.get_bucket()

    # Save the input data and target data to the bucket.
    for df, blob_name in zip(
        [X, y, predictions], ["X.parquet", "y.parquet", "predictions.parquet"]
    ):
        logger.info(f"Saving {blob_name} to bucket...")
        utils.write_blob_to(
            bucket=bucket,
            blob_name=blob_name,
            data=df,
        )
        logger.info(f"Successfully saved {blob_name} to bucket.")



def reshape_x(prediction_time: datetime, X: pd.DataFrame):
    # 確保 prediction_time 是 datetime 物件
    if isinstance(prediction_time, str):
        prediction_time = pd.to_datetime(prediction_time)
        
    time_index = [prediction_time - timedelta(hours=i) for i in range(1, 4)]
    time_index.reverse()
    
    columns = [0, 8, 16]
    X = X.iloc[:, columns]
    X = X.reset_index(drop=True)
    X = X.values.reshape(-1, 1)
    X = pd.DataFrame(X, index=time_index)

    return X

if __name__ == "__main__":
    predict()