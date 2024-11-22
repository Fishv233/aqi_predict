import pandas as pd
import numpy as np

def transform(data: pd.DataFrame) -> pd.DataFrame:
    data = filter_columns(data)
    data = rename_columns(data)
    data = cast_columns(data)
    data = interpolate_missing_values(data)
    return data

def rename_columns(data: pd.DataFrame) -> pd.DataFrame:
    data.rename(columns={'datacreationdate':'ts'}, inplace=True)
    return data

def filter_columns(data: pd.DataFrame) -> pd.DataFrame:
    data.drop(columns=['county'],inplace=True)
    required_feature = ['sitename','datacreationdate', 'aqi','pollutant','so2',
                        'co','o3','pm10',
                        'no2','windspeed','pm10_avg']
    data = data[required_feature]
    return data

def cast_columns(data: pd.DataFrame) -> pd.DataFrame:
    data['ts'] = pd.to_datetime(data['ts'])
    return data


def interpolate_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(['sitename', 'ts'])
    for site in data['sitename'].unique():
        site_mask = data['sitename'] == site
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data.loc[site_mask, col] = data.loc[site_mask, col].interpolate(method='linear')
    return data
