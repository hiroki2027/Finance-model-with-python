from airflow import DAG

# 対応する Airflow バージョンで import パスが異なるため、両対応させる
try:
    # Airflow 2.x 系
    from airflow.operators.python import PythonOperator
except ImportError:  # Airflow 1.10 系
    from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from fetch_daily import fetch_and_save

# UTC → 日本時間 JST +9h に合わせたい場合は pendulum を使う
import pendulum
local_tz = pendulum.timezone("Asia/Tokyo")

with DAG(
    dag_id="daily_stock_ingest",
    schedule_interval="0 6 * * 1-5",          # 平日 06:00 JST
    start_date=local_tz.datetime(2025, 6, 12),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
) as dag:

    PythonOperator(
        task_id="download_prices",
        python_callable=fetch_and_save,
    )
import pandas as pd, numpy as np, pyarrow.parquet as pq
from pathlib import Path

import pyarrow.dataset as ds

# 先頭 10 行だけ確認
preview = ds.dataset("data/raw/7203.T.parquet").head(10)
print(preview.to_pandas())

daily = pq.read_table("data/raw/7203.T.parquet").to_pandas()
daily['Date'] = pd.to_datetime(daily['Date'])
daily.set_index('Date', inplace=True)

# 月末終値
monthly = daily['Adj Close'].resample('M').last()

# 12-1 モメンタム
mom_12_1 = monthly.pct_change(12) - monthly.pct_change(1)

# 年率ボラ (252 日)
log_ret = np.log(daily['Adj Close']).diff()
ann_vol = log_ret.rolling(252).std() * np.sqrt(252)

# RSI
