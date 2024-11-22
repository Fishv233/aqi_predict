from datetime import datetime
from typing import Tuple
import numpy as np
import pandas as pd
from batch_prediction_pipeline.modules.utils import get_logger
from hsfs.feature_store import FeatureStore

logger = get_logger(__name__)

def load_data_from_feature_store(
    fs: FeatureStore,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data for a given time range from the feature store.

    Args:
        fs: Feature store.
        feature_view_version: Feature view version.
        start_datetime: Start datetime.
        end_datetime: End datetime.
        target: Name of the target feature.

    Returns:
        Tuple of exogenous variables and the time series to be forecasted.
    """
    get_feature_group = fs.get_feature_group(
        name="aqi_feature_group", 
        version=1
    )
    data = get_feature_group.read()
    X, y, prediction_time = prepare_data(data)
    return X, y, prediction_time


def prepare_data(raw_data, window_size=4) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    df = raw_data
    df.drop(columns=['pollutant'], inplace=True)
    df = df[df['sitename'] == '左營']
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values(['sitename', 'ts'])
    # 先將需要的欄位轉換為數值型
    
    numeric_cols = ['aqi', 'so2', 'co', 'o3', 'pm10', 'no2', 'windspeed', 'pm10_avg']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' 會將無法轉換的值設為 NaN

    for site in df['sitename'].unique():
        site_mask = df['sitename'] == site
        site_data = df[site_mask].copy()
        # 確保資料按時間排序
        site_data = site_data.sort_values('ts')
        # 對指定的數值欄位進行線性插值
        for col in numeric_cols:
            site_data[col] = site_data[col].interpolate(method='linear', limit_direction='both')
        # 更新原始資料框
        df.loc[site_mask] = site_data
    
    features = []
    targets = []
    
    # 排除不需要的欄位
    exclude_columns = ['ts', 'sitename']
    target_column = 'aqi'
    
    # 對每個測站分別處理
    for site in df['sitename'].unique():
        site_data = df[df['sitename'] == site]
        site_data.drop_duplicates(subset=['ts'], inplace=True)
        logger.info(f'site_data: {site_data}')
        #取出最新的預測時間點
        site_data['ts'] = pd.to_datetime(site_data['ts'])
        latest_time = site_data['ts'].max()
        next_hour = latest_time + pd.Timedelta(hours=1)
        prediction_time = next_hour.strftime('%Y-%m-%d %H:00:00')
        logger.info(f'prediction_time: {prediction_time}')

        site_data.drop(columns=exclude_columns, inplace=True)
        
        # 只取最後 window_size 筆資料
        if len(site_data) >= window_size:
            window_data = site_data.iloc[-window_size:]
            logger.info(f'window_data: {window_data}')

            
            # 收集前 window_size-1 個時間點的所有特徵
            feature = []
            for t in range(window_size-1):  # 收集前三筆資料的所有特徵
                feature.extend(window_data.iloc[t].values.tolist())
            
            # 目標是最後一個時間點的 AQI
            target = window_data.iloc[window_size-1][target_column]
            
            features.append(feature)
            targets.append(target)

    return np.array(features), np.array(targets), prediction_time

