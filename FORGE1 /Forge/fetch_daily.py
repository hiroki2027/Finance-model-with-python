# モジュールか
 
import numpy as np 
import pandas as pd          # データ加工
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf        # データ取得
import pyarrow.parquet as pq # 保存用
import pyarrow as pa        # テーブル結合用

### データ取得スクリプト
Tickers =  ["7203.T", "9984.T", "AAPL", "MSFT"]

raw_dir = Path("/Users/kondousatoshishi/Downloads/金融モデル/Forge_data/raw/daily")

def fetch_and_save():
    raw_dir.mkdir(parents=True, exist_ok=True)
    end = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # 毎日更新
    for tic in Tickers:
        path = raw_dir / f"{tic}.parquet"
        start = "2014-01-01"
        if path.exists():
            last = pq.read_table(path).to_pandas()["Date"].max()
            start = (pd.to_datetime(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
        df = yf.download(tic, start=start, end=end, auto_adjust=True)
        
        if df.empty:
            continue
        
        df.reset_index(inplace=True)
        
        if path.exists():
            old_table = pq.read_table(path)
            new_table = pa.Table.from_pandas(df)
            table = pa.concat_tables([old_table, new_table])
        else:
            table = pa.Table.from_pandas(df)
       
        pq.write_table(table, path)
        print("#2. データ取得成功")

if __name__ == "__main__":
  fetch_and_save()

