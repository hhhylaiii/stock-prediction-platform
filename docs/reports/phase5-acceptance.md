# Phase 5 驗收報告

狀態：部分完成。SPEC §9 Phase 5 的驗收條件是「API 回傳最新交易日的預測排名、預測價格、價格區間、上漲機率與特徵；回應時間 500 毫秒內」。twp 目前還沒有任何模型與預測（Phase 4 的 4.7 尚未上線），所以預測類端點先用 seed 量。S5（回應時間）等 4.7 產出真預測、重量之後再勾。

## 驗收條件

| 條件 | 結果 | 證據 |
| --- | --- | --- |
| 11 個唯讀端點都回 200 | 通過 | 下面兩張表 |
| 資料類端點在 twp 真資料上 500 ms 內 | 通過 | 表一，最大值 31.4 ms |
| 預測、模型類端點 500 ms 內 | 暫定通過（seed） | 表二，最大值 141.8 ms；4.7 之後要重量，另見「已知風險」 |
| 三個視圖可用 | 通過（seed） | 截圖一到三 |
| S5 勾選 | 待辦 | 見最後一節 |

## 量測方法

- 指令：`manage.py measure_api --repeat 20`。在同一個 process 裡用 Django 的 test client 打每個端點：先打一次暖機，不計入；再連打 20 次，記中位數、p95、最大值。
- 量到的時間包含 view、查詢與序列化，不含網路與 gunicorn。
- 環境：本機 macOS、Docker 的 PostgreSQL 16。twp 那一輪是在 2026-11-05 15:54 量的，正好在 ETL 時段內。
- 表一以 head `44f8b5f` 量測（含 #64、#65）；表二與截圖在加深 seed 之後重量（含 #68）。

twp 那一輪用唯讀交易連線，任何寫入都會被資料庫拒絕。throttle 每個 process 各自計數，預設 300/min；`--repeat 20` 共 231 次，用不到調高，調高只是讓 `--repeat` 可以設得更大。指令會記下每一次呼叫的狀態碼，只要有一次不同，狀態欄就會寫成像 `200/429×3` 的樣子：

```bash
DATABASE_URL=<twp 的連線字串> PGOPTIONS='-c default_transaction_read_only=on' \
  API_THROTTLE_RATE=100000/min uv run python manage.py measure_api --repeat 20
```

seed 那一輪與截圖由一支 `slow` 測試產生。沒有設 `ACCEPTANCE_OUT` 時這支測試會略過，CI 不跑：

```bash
ACCEPTANCE_OUT=docs/reports/phase5 uv run pytest tests/e2e/test_acceptance_capture.py
```

## 表一：twp 真資料

twp 的資料量（2026-11-05 量測當下）：

- stock_master 3,065 檔。
- daily_price 7,349,318 列，到 2026-11-05；2330 有 4,126 列。
- etl_run 213 筆，data_quality_log 98 筆。
- feature_daily 6,736,842 列；2330 的 label_daily 有 8,250 列。
- model_run、prediction、backtest_metric 都是 0 筆。

| 端點 | 資料 | 狀態 | 中位數 ms | p95 ms | 最大 ms |
| --- | --- | --- | --- | --- | --- |
| `/api/stocks/2330` | 真資料 | 200 | 1.1 | 1.4 | 1.4 |
| `/api/stocks/2330/prices` | 真資料 | 200 | 5.6 | 11.1 | 31.4 |
| `/api/status` | 真資料，預測部分為空 | 200 | 4.5 | 7.4 | 8.0 |
| `/api/etl/runs` | 真資料 | 200 | 3.8 | 5.2 | 6.9 |
| `/api/data-quality` | 真資料 | 200 | 4.0 | 6.9 | 9.9 |
| `/api/stocks/2330/predictions` | 真資料，沒有預測 | 200 | 3.0 | 4.5 | 4.7 |
| `/api/stocks/2330/features` | 真資料，沒有預測 | 200 | 1.3 | 1.7 | 2.2 |
| `/api/stocks/2330/history` | 真資料，沒有預測 | 200 | 10.7 | 14.3 | 14.7 |
| `/api/predictions` | 真資料，沒有預測 | 200 | 1.1 | 1.3 | 1.8 |
| `/api/models` | 真資料，沒有模型 | 200 | 0.7 | 0.8 | 0.8 |
| `/api/models/ic-monthly` | 真資料，沒有研究結果 | 200 | 1.1 | 1.3 | 1.3 |

後六列只代表「沒有資料時」的速度，不能當驗收依據。驗收看表二。

## 表二：seed 資料

seed 的內容是 e2e 的基本資料再放大：

- 1,842 檔股票；6 個上線 run（排名 5／20 日、絕對報酬 5 日、分位數、波動率、上漲機率）。
- 70 個交易日裡，每一天每一檔都有 6 個 run 的預測，約 77 萬列，相當於上線三個多月沒換版。
- 2330 有 69 個交易日的 label。
- 每個 run 有 69 天的監控。
- 研究期有 26 個月的月 IC，2330 有特徵貢獻。

| 端點 | 資料 | 狀態 | 中位數 ms | p95 ms | 最大 ms |
| --- | --- | --- | --- | --- | --- |
| `/api/stocks/2330/predictions` | seed | 200 | 90.3 | 99.4 | 103.2 |
| `/api/stocks/2330/features` | seed | 200 | 36.2 | 37.8 | 38.0 |
| `/api/stocks/2330/history` | seed | 200 | 5.7 | 6.6 | 7.0 |
| `/api/status` | seed | 200 | 48.0 | 60.3 | 72.0 |
| `/api/predictions` | seed | 200 | 63.8 | 84.6 | 89.8 |
| `/api/models` | seed | 200 | 112.7 | 127.3 | 141.8 |
| `/api/models/ic-monthly` | seed | 200 | 32.7 | 33.9 | 34.6 |

表二列出會讀預測表的端點。同一輪的其他端點中位數都在 4 ms 內：stocks、prices、etl/runs、data-quality。測試另外斷言 11 個端點都是 200，p95 都在 500 ms 內。各次執行的數字會有小幅變動。

## 已知風險

`/api/models`、`/api/status`、`/api/stocks/{id}/predictions` 都會對服務中 run 的全部預測取 `Max(date)`。這在 `apps/api/model_runs.py` 的 `serving_models`，以及 `apps/api/serving.py` 的 `latest_served_date`。

- `prediction` 表沒有以 `run_id` 開頭、接著 `date` 的 index，成本會跟著服務中 run 的預測天數成長。
- seed 只有最新一天時，`/api/models` 中位數 18 ms；加深到 70 天後是 113 ms。
- 正常每月換版時，服務中的 run 只有一個月左右的預測。但新版沒過健全性檢查時會沿用舊版，那時天數會一直累積。
- 加 index（例如 `(run_id, horizon, date)`）需要 migration，已回報協調決定。

## 三個視圖（seed 資料）

圖一：今日預測（seed 資料）。

![今日預測，seed 資料](phase5/today.png)

圖二：個股頁 2330（seed 資料）。

![個股頁，seed 資料](phase5/stock.png)

圖三：模型與資料（seed 資料）。

![模型與資料，seed 資料](phase5/models.png)

## 待辦

- [ ] 4.7 在 twp 產出真預測後，用上面兩個指令重量，並重截三張圖，換成真資料的版本；之後勾 S5。
- [ ] 上線 6 個月後，在 twp 上用 `measure_api` 再量一次，確認預測表變深之後仍在 500 ms 內。
- [ ] web 容器還在跑 4 天前的映像。要等 Phase 2 同意，並避開 15:10–23:00 的 ETL 時段再重建；重建後在這裡記一筆。
