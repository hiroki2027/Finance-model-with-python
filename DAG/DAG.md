
#### DAGのメリット
１、完全自動化
２、失敗時のリトライ、通知
３、履歴の可視化
４、拡張が簡単
５、本番にそのままスケール
### Airflow DAG チートシート

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pendulum

local_tz = pendulum.timezone("Asia/Tokyo")

with DAG(
    dag_id="daily_stock_ingest",          # DAG の一意な ID
    schedule_interval="0 6 * * 1-5",      # 平日 6:00 (JST) に実行
    start_date=local_tz.datetime(2025, 6, 12),
    catchup=False,                        # 過去分を遡って実行しない
    default_args={
        "retries": 2,                     # 失敗時に 2 回まで再試行
        "retry_delay": timedelta(minutes=10),
    },
) as dag:
    PythonOperator(
        task_id="download_prices",        # タスク名
        python_callable=fetch_and_save,    # 実行する関数
    )
```

#### パラメータ早見表

| パラメータ          | 役割・意味                                                    |
|---------------------|---------------------------------------------------------------|
| `dag_id`            | DAG（ワークフロー）の一意な名前                               |
| `schedule_interval` | Cron 形式。`0 6 * * 1-5` は「平日の 6:00 JST」に相当           |
| `start_date`        | この日時以降にスケジューリングを開始                          |
| `catchup`           | `True` だと過去分を遡って一気に実行。ここでは `False` で無効化 |
| `PythonOperator`    | Python 関数を 1 つのタスクとして DAG に登録                   |

#### DAG を使うメリット

1. **完全自動化** — 指定時刻に処理が勝手に走る  
2. **失敗時のリトライ & 通知** — 再試行設定や Slack 連携が容易  
3. **履歴の可視化** — Web UI から実行状況を一覧確認  
4. **拡張が簡単** — 新タスクを追加するだけでワークフローを拡張  
5. **そのまま本番スケール** — ローカルで作った DAG をクラウド Airflow でも再利用可能  