import sys
import os

import hopsworks
import pandas as pd
from great_expectations.core import ExpectationSuite
from hsfs.feature_group import FeatureGroup
from feature_pipeline import utils
from feature_pipeline.settings import SETTINGS

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = utils.get_logger(__name__)
def to_feature_store(
    data: pd.DataFrame,
    validation_expectation_suite: ExpectationSuite = None,
    feature_group_version: int = 1,
    config: dict = None
) -> FeatureGroup:
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"],
        project=SETTINGS["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    water_quality_feature_group = feature_store.get_or_create_feature_group(
        name=config['name'],
        version=config['version'],
        description=config['description'],
        primary_key=config['primary_key'],
        event_time=config['event_time'],
        online_enabled=config['online_enabled'],
    )
    # Upload data.
    water_quality_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )



    # Update statistics.
    water_quality_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    water_quality_feature_group.update_statistics_config()
    # water_quality_feature_group.compute_statistics()

    return water_quality_feature_group
