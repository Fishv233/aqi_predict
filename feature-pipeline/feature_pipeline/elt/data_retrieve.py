import sys
import os
import pandas as pd
from feature_pipeline import utils
from feature_pipeline import settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests

logger = utils.get_logger(__name__)


class DataRetrieve:
    def __init__(self, ts_col: str = None):
        self.metadata = {}
        self.data = None
        self.ts_col = ts_col


    def _get_data(self) -> dict:
        """
        Extract data from api
        """
        basic_url = "https://data.moenv.gov.tw/api/v2/aqx_p_488"
        api_key = settings.SETTINGS["MINISTRY_OF_ENERGY_API_KEY"]
        response = requests.get(basic_url, params={"api_key": api_key})
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data from API: {response.status_code}")
        else:
            data = response.json()
            records = pd.DataFrame(data['records'])
            records_kaohsiung = records[records['county'] == '高雄市']
            logger.info(f"Successfully fetched data from API: {len(records_kaohsiung)} records")
            logger.info(records_kaohsiung.head())
            self.data = records_kaohsiung

    def _get_ts_range(self):
        """
        description:
            get ts range from data for metadata of feature group
        return:
            ts_min and ts_max
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call get_data() first.")
        self.data[self.ts_col] = pd.to_datetime(self.data[self.ts_col])
        ts_min = self.data[self.ts_col].min()
        ts_max = self.data[self.ts_col].max()
        self.metadata['ts_min'] = ts_min.strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['ts_max'] = ts_max.strftime('%Y-%m-%d %H:%M:%S')
        return self.metadata['ts_min'], self.metadata['ts_max']

    def run(self):
        """
        description:
            get data and ts range
        return:
            data and metadata
        """
        self._get_data()
        self._get_ts_range()
        return self.data, self.metadata
