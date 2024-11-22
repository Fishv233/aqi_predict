import fire
import json

from feature_pipeline import utils
from feature_pipeline.elt.data_retrieve import DataRetrieve
from feature_pipeline.elt import data_retrieve, data_validation, data_loader
from feature_pipeline.elt.data_transform import transform
logger = utils.get_logger(__name__)


def run(
    config: dict = None
) -> dict:
    """
    Extract data from the API, transform it, and load it to the feature store.
    Returns:
          A dictionary containing metadata of the pipeline.
    """
    # data_retrieve
    logger.info(f"feature_group_version: {config['feature_group_description']['version']}")
    logger.info("retrieving data")
    retriever = DataRetrieve(ts_col='datacreationdate')
    data, metadata = retriever.run()
    logger.info("Successfully retrieve data.")

    # data_transform
    logger.info("Transforming data.")
    data = transform(data)
    logger.info(f'cleaned_data: {data.head()}')
    logger.info("Successfully transformed data.")


    # data_load
    logger.info("Validating data and loading it to the feature store.")
    data_loader.to_feature_store(
        data,
        config=config['feature_group_description']
    )
    metadata["feature_group_version"] = config['feature_group_description']['version']
    metadata["datetime_format"] = config['datetime_format']
    logger.info("Successfully validated data and loaded it to the feature store.")

    logger.info("Wrapping up the pipeline.")
    utils.save_json(metadata, file_name="feature_pipeline_metadata.json")
    logger.info("Done!")

    return metadata


if __name__ == "__main__":
    config = None
    fire.Fire(run(config=config))
    # create feature view
