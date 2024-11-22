import requests

import streamlit as st
import logging
from settings import API_URL, TITLE
from components import build_data_plot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 添加 StreamHandler 來輸出到 stdout
    ]
)


st.set_page_config(page_title=TITLE)
st.title(TITLE)


# 定義數值對應的名稱
AREA_NAMES = {
    0: "左營",
}

# Create dropdown for area selection.
area_response = requests.get(API_URL / "area_values")
area_values = area_response.json().get("values", [])

# 創建選項列表，將數值轉換為可讀的名稱
area_options = [(value, AREA_NAMES.get(value, f"區域 {value}")) for value in area_values]

area = st.selectbox(
    label="AQI prediction area.",
    options=[value for value, _ in area_options],
    format_func=lambda x: AREA_NAMES.get(x, f"區域 {x}")
)


# Check if both area and consumer type have values listed, then create plot for data.
if area is not None:
    st.plotly_chart(build_data_plot(area))


logging.info(f"API URL: {area}")
