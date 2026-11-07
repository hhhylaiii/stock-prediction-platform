# stock-prediction-platform

台股上市、上櫃普通股的每日預測平台：收盤後產出未來 5、20 個交易日的報酬排名、預測報酬與價格區間、上漲機率。規格見 [docs/SPEC.md](docs/SPEC.md)。個人研究用途，資料授權限制見規格第 3 節。

## 快速上手

已經有資料庫（`docker compose up -d db redis`，資料在 volume 裡）時，看畫面只要：

```bash
uv sync
uv run python manage.py migrate
uv run python manage.py runserver
```

打開 <http://127.0.0.1:8000/>。`.env` 會自動讀取，`DATABASE_URL` 預設連本機的 `twp`，特徵、模型、預測快取預設放在 `artifacts/`。第一次用先 `cp .env.example .env`，回補資料要填 `FINMIND_API_TOKEN`。

畫面是一頁三個視圖，用網址的 `#` 切換：

| 視圖 | 網址 | 內容 | 需要的資料 |
| --- | --- | --- | --- |
| 今日預測 | `/#today` | 全市場 5 日、20 日排名與預測報酬，20 日 80% 價格區間、漲幅超過 5% 的機率；可搜尋、只看追蹤 | 上線中的模型 |
| 個股 | `/#stock-2330` | K 線或收盤線、這次預測怎麼來的（特徵貢獻）、過去的預測與實際結果 | 行情；預測區塊要上線中的模型 |
| 模型與資料 | `/#models` | 上線中的模型、每月 rank IC 與 baseline、今日資料更新、資料品質與模型輸出檢查、資料更正紀錄 | ETL 紀錄；模型區塊要上線中的模型 |

還沒有上線中的模型時，預測相關區塊會顯示「預測資料尚未產出」，K 線、ETL 紀錄與品質檢查照常顯示。頂端的狀態列讀 `/api/status`：當天品質檢查有 error 時，當天不產出預測；`quality.warnings` 是有 warning 的檢查項數，`quality.warning_rows` 是 warning 的列數。

## 資料與模型的流程

```
來源（TWSE、TPEx、FinMind、MOPS、TDCC、Yahoo、data.gov.tw）
  → raw 表（原始回應）→ standard 表（日價量、籌碼、財報…，upsert、可重跑）
  → 品質檢查（data_quality_log）
  → label_daily、feature_daily（另存 Parquet）
  → walk-forward 訓練與評估（model_run、backtest_metric）
  → 上線模型每日預測（prediction、prediction_contrib）與事後監控（model_monitor）
  → 唯讀 API（/api/...）與頁面
```

每日排程由 `worker` 容器裡的 Celery beat 執行，時間表、閘門、重試與補抓見 [docs/runbook.md](docs/runbook.md)。

## 常用指令

都用 `uv run python manage.py <指令> -h` 看完整參數。

| 用途 | 指令 |
| --- | --- |
| 抓一天或一段日期 | `fetch --source twse --dataset daily_price --date 2026-11-06` |
| 逐檔回補（可續跑） | `backfill --source finmind --dataset <dataset> --from 2010-01-01 --resume` |
| 補抓漏掉的交易日 | `catchup --since 2026-11-01 --dry-run` |
| 跑一天的品質檢查 | `check_quality --date 2026-11-06` |
| 重建 label、特徵 | `build_labels --start 2010-01-01`、`build_features --start 2010-01-01`（不給範圍就重做最近 60 天） |
| 訓練與評估 | `train --model lightgbm --scope rank --run-key <key>`、`evaluate_rank --run-key <key>`、`ic_report --run-key <key>` |
| 事後監控 | `monitor --date 2026-11-06` |
| Phase 5 驗收報告 | `phase5_report --out <dir>` |

API 都是唯讀的 GET，回應格式是 `{"success", "data", "error", "meta"}`：`/api/status`、`/api/predictions`、`/api/stocks/<代號>`（加 `/prices`、`/predictions`、`/features`、`/history`）、`/api/models`、`/api/models/ic-monthly`、`/api/etl/runs`、`/api/data-quality`（加 `/corrections`）。

## 目錄

- `apps/etl/`：來源 adapter、raw 層、`etl_run`、品質檢查、排程任務。
- `apps/market/`：standard 表、交易日曆、還原價。
- `apps/research/`：label、特徵、walk-forward、模型、報表、監控。
- `apps/api/`：唯讀 API。`apps/web/`：頁面（`templates/web/index.html`、`static/web/`）。
- `docs/`：規格、runbook、各 Phase 的驗收報告（`docs/reports/`）。

## 需求

- macOS 或 Linux、[uv](https://docs.astral.sh/uv/)（會依 `.python-version` 自動安裝 Python 3.12）
- Docker Desktop（Compose v2.24 以上）
- macOS 跑 LightGBM 需要 OpenMP：`brew install libomp`

## 本機開發

```bash
cp .env.example .env
uv sync
uv run playwright install chromium
docker compose up -d db redis
uv run python manage.py migrate
uv run python manage.py check
uv run pytest -m "not e2e"
uv run pytest -m e2e
```

`-m e2e` 是 Playwright 的瀏覽器冒煙測試，要先跑過一次 `playwright install chromium`。

要平行跑就加 `-n auto`，同時設 `OMP_NUM_THREADS=1`：`OMP_NUM_THREADS=1 uv run pytest -m "not e2e" -n auto`。不設的話，每個 worker 裡的 LightGBM、XGBoost 都會吃滿全部核心、互相搶 CPU，反而比不平行還慢。

## 全部容器

```bash
docker compose up -d --build
docker compose ps
curl http://localhost:8000/healthz
```

四個容器：`web`（gunicorn）、`worker`（Celery worker 內含 beat）、`db`（PostgreSQL 16）、`redis`（broker）。每日排程、重試、手動補抓與防休眠見 [docs/runbook.md](docs/runbook.md)。所有容器與主機都要用 `TZ=Asia/Taipei`，否則排程會偏 8 小時。

- web、db、redis 的對外 port 只綁 `127.0.0.1`，區網連不到。
- Compose 預設用 `config.settings.dev`（DEBUG 開）。改成 `config.settings.prod` 時要在 `.env` 設 `DJANGO_SECRET_KEY` 與 `DJANGO_ALLOWED_HOSTS`，`DATABASE_URL` 也必填（Compose 會自動注入）。`DJANGO_ALLOWED_HOSTS` 要保留 `localhost`，web 的 healthcheck 靠它；只用 http 存取時設 `DJANGO_SECURE_COOKIES=false`，否則瀏覽器可能丟掉 session 與 CSRF cookie。
