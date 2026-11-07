# Phase 5 驗收報告

狀態：部分完成。SPEC §9 Phase 5 的驗收條件是「API 回傳最新交易日的預測排名、預測價格、價格區間、上漲機率與特徵；回應時間 500 毫秒內」。twp 目前還沒有任何模型與預測，所以預測類端點先用 seed 量。Phase 4 的 4.7 正式 run 要等 4.9 holdout 之後才建（協調 2026-11-07），S5 在那之後用「待辦」一節的指令重量再勾。

## 驗收條件

| 條件 | 結果 | 證據 |
| --- | --- | --- |
| 11 個唯讀端點都回 200 | 通過 | 下面兩張表 |
| 資料類端點在 twp 真資料上 500 ms 內 | 通過 | 表一，最大值 31.4 ms |
| 預測、模型類端點 500 ms 內 | 暫定通過（seed） | 表二，最大值 40.8 ms；4.7 之後要在 twp 上重量 |
| 三個視圖可用 | 通過（seed） | 截圖一到三 |
| S5 勾選 | 待辦 | 見最後一節 |

## 量測方法

- 指令：`manage.py measure_api --repeat 20`。在同一個 process 裡用 Django 的 test client 打每個端點：先打一次暖機，不計入；再連打 20 次，記中位數、p95、最大值。
- 量到的時間包含 view、查詢與序列化，不含網路與 gunicorn。
- `manage.py phase5_report` 用同一套量法，另外透過 API 逐條檢查驗收條件、記下資料量，寫成一份 markdown。
- 環境：本機 macOS、Docker 的 PostgreSQL 16。twp 那一輪是在 2026-11-05 15:54 量的，正好在 ETL 時段內。
- 表一在 #72 的 commit「fix: keep negative monthly ic bars inside the chart」上量測，當時已含 #64、#65。截圖在加深 seed 之後截，當時已含 #68。表二在 #104 的分支上用 `phase5_report` 重量，分支以 develop 977c31b 為底，已含 #76。

twp 那一輪用唯讀交易連線，任何寫入都會被資料庫拒絕。throttle 每個 process 各自計數，預設 300/min；`--repeat 20` 共 273 次（11 個端點時是 231 次），用不到調高，調高只是讓 `--repeat` 可以設得更大。指令會記下每一次呼叫的狀態碼，只要有一次不同，狀態欄就會寫成像 `200/429×3` 的樣子：

```bash
DATABASE_URL=<twp 的連線字串> PGOPTIONS='-c default_transaction_read_only=on' \
  API_THROTTLE_RATE=100000/min uv run python manage.py measure_api --repeat 20
```

seed 那一輪與截圖由一支 `slow` 測試產生。沒有設 `ACCEPTANCE_OUT` 時這支測試會略過，CI 不跑。它在 seed 上跑 `phase5_report`（和下面的指令一樣把 `API_THROTTLE_RATE` 調到 100000/min：13 個端點各 21 次，加上逐頁讀預測，會超過預設的 300/min），每條檢查都要通過，報告寫到輸出目錄的 `seed.md`，表二就是從這份抄來的；這個檔不入庫。截圖和 twp 那一輪用同一個函式：

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
- 2330 的 label：5 日有 65 個交易日，20 日有 50 個交易日（最後 5、20 天還沒到期）。
- 每個 run 有 69 天的監控。
- 研究期有 26 個月的月 IC，2330 有特徵貢獻。

| 端點 | 資料 | 狀態 | 中位數 ms | p95 ms | 最大 ms |
| --- | --- | --- | --- | --- | --- |
| `/api/stocks/2330/predictions` | seed | 200 | 25.5 | 36.4 | 40.8 |
| `/api/stocks/2330/features` | seed | 200 | 8.6 | 9.9 | 10.1 |
| `/api/stocks/2330/history` | seed | 200 | 5.7 | 6.2 | 6.3 |
| `/api/status` | seed | 200 | 15.4 | 18.6 | 18.8 |
| `/api/predictions` | seed | 200 | 21.0 | 27.6 | 30.1 |
| `/api/models` | seed | 200 | 19.4 | 26.1 | 27.4 |
| `/api/models/ic-monthly` | seed | 200 | 4.4 | 5.5 | 5.6 |

表二列出會讀預測表的端點。同一輪的其他端點中位數都在 3 ms 內：stocks、prices、etl/runs、data-quality。同一輪的檢查確認 11 個端點都是 200，p95 都在 500 ms 內，其餘驗收檢查也都通過。各次執行的數字會有變動：中位數大致穩定，最大值受單次延遲影響，兩次執行之間可能差到一倍。

## 預測表變深之後

`/api/models`、`/api/status`、`/api/stocks/{id}/predictions` 都要找服務中 run 的最新預測日。

- 以前是對 run 的全部預測取 `Max(date)`，成本跟著預測天數成長。seed 只有最新一天時，`/api/models` 中位數 18 ms；加深到 70 天後是 113 ms。
- 正常每月換版時，服務中的 run 只有一個月左右的預測。但新版沒過健全性檢查時會沿用舊版，那時天數會一直累積。
- #76 加了 `(run_id, horizon, date)` 的 index，`apps/api/serving.py` 的 `latest_days` 現在每個 run 只查一次 index。

同一份 70 天的 seed，#76 之前與之後的中位數（ms）：

| 端點 | #76 之前 | #76 之後 |
| --- | --- | --- |
| `/api/models` | 112.7 | 19.4 |
| `/api/status` | 48.0 | 15.4 |
| `/api/stocks/2330/predictions` | 90.3 | 25.5 |

上線 6 個月後仍要在 twp 上再量一次，見「待辦」。

## 三個視圖（seed 資料）

圖一：今日預測（seed 資料）。

![今日預測，seed 資料](phase5/today.png)

圖二：個股頁 2330（seed 資料）。

![個股頁，seed 資料](phase5/stock.png)

圖三：模型與資料（seed 資料）。

![模型與資料，seed 資料](phase5/models.png)

## 待辦

- [ ] 4.9 之後、4.7 在 twp 產出真預測時，在主機上跑下面這個指令，把數字抄進本報告後勾 S5。
  - 先確認 web 映像含最新的頁面（見最後兩條）：`--views` 截的是正在跑的 web。
  - 它會量 13 個端點（2026-11-07 加上 `/api/data-quality/corrections` 與 `/api/stocks?q=`）、透過 API 逐條檢查驗收條件、記下資料量，寫成 `docs/reports/phase5/twp.md`（不入庫）。
  - 排名、價格、區間、機率逐頁看全市場 universe 內的股票；個股頁與特徵看 `--stock`（預設 2330），特徵至少要有一項原始值。
  - 每條都通過才截圖。每頁等資料區塊都載完（不剩「載入中」）才截，三張都截到才搬進 `docs/reports/phase5/`，蓋掉現在的 seed 截圖；中途失敗時一張都不動。
  - 有一條未通過時，報告照樣寫出，指令以非 0 結束並列出未通過的條目。
  - `--views` 用的 Playwright 只在 dev 依賴裡，容器內沒有，所以要在主機上跑。

  ```bash
  DATABASE_URL=<twp 的連線字串> PGOPTIONS='-c default_transaction_read_only=on' \
    API_THROTTLE_RATE=100000/min uv run python manage.py phase5_report \
    --out docs/reports/phase5/twp.md --views http://127.0.0.1:8000
  ```

- [ ] 上線 6 個月後，在 twp 上用 `phase5_report`（不加 `--views`）再量一次，確認預測表變深之後仍在 500 ms 內。
- [x] web 與 worker 在 2026-11-07 23:18 和 prod 設定一起重建（主 repo 33863dc，映像 `sha256:d6f3a4502c2b`）。這兩個映像都停在 33863dc，缺 #130–#133（含 #131 的晚間 60 日重算與 #133 的櫃買借券容忍值）。
- [ ] 2026-11-08 早上（協調排時段）把 web 與 worker 重建到當時最新的 develop，帶上 #130 之後的改動與 Phase 2 的新 migration，重建後在這裡記一筆。之後合併的頁面改動要等再下一次重建，跑 S5 的 `--views` 前先重建一次 web。
