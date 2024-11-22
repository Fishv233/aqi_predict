from datetime import datetime
from typing import Optional

import hopsworks
import hsfs

from feature_pipeline import utils
from feature_pipeline import settings

logger = utils.get_logger(__name__)


def create(
    config: dict = {}
) -> dict:
    """Create a new feature view version and training dataset
    based on the given feature group version and start and end datetimes.
    Returns:
        dict: The feature group version.

    """
    logger.info("start creating feature view")
    feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
    feature_group_version = feature_pipeline_metadata["feature_group_version"]
    logger.info(f"feature_pipeline_metadata: {feature_pipeline_metadata}")


    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"],
        project=settings.SETTINGS["FS_PROJECT_NAME"],
    )
    fs = project.get_feature_store()

    # delete old feature views
    try:
        feature_views = fs.get_feature_views(name=config["project_name"])
        logger.info(f"Feature views: {feature_views}")
    except hsfs.client.exceptions.RestAPIError:
        logger.info(f"No feature views found for {config['project_name']}.")

        feature_views = []

    for feature_view in feature_views:
        try:
            feature_view.delete_all_training_datasets()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete training datasets for feature view {feature_view.name} with version {feature_view.version}."
            )

        try:
            feature_view.delete()
        except hsfs.client.exceptions.RestAPIError:
            logger.error(
                f"Failed to delete feature view {feature_view.name} with version {feature_view.version}."
            )

    logger.info(f"feature view name: {config['project_name']}")
    # Create feature view in the given feature group version.
    feature_group = fs.get_feature_group(
        config["feature_group_description"]["name"], version=feature_group_version
    )
    ds_query = feature_group.select_all()
    feature_view = fs.create_feature_view(
        name=config["project_name"],
        description="aqi feature view.",
        query=ds_query,
        labels=[],
    )

    # Create training dataset.
    feature_view.create_training_data(
        description=f"{config['project_name']} training dataset",
        data_format="csv",
        write_options={"wait_for_job": True},
        coalesce=False,
    )

    # Save metadata.
    metadata = {
        "feature_view_version": feature_view.version,
        "training_dataset_version": 1,
        "project_name": config["project_name"],
    }
    utils.save_json(
        metadata,
        file_name="feature_view_metadata.json",
    )

    # 讀取並顯示訓練數據
    feature_view = fs.get_feature_view(
        name=config["project_name"],
        version=1,
    )
    df = feature_view.get_training_data(training_dataset_version=1)
    logger.info("Training data shape: %s", df)

    return metadata
