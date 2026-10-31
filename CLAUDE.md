實作前先讀 docs/SPEC.md，第 8 節為硬性規則

# stock-prediction-platform

台股上市、上櫃普通股的每日預測平台：資料層（ETL、PostgreSQL）、研究層（特徵、訓練、walk-forward）、服務層（Django API 與頁面）。規格書是唯一規格來源，規格與程式碼衝突時以 `docs/SPEC.md` 為準並回報差異。

## 指令

| 用途 | 指令 |
| --- | --- |
| 安裝依賴 | `uv sync` |
| 起資料庫與 Redis | `docker compose up -d db redis` |
| 全部容器 | `docker compose up -d`（web、worker、db、redis） |
| 測試 | `uv run pytest` |
| Lint 與格式 | `uv run ruff check . && uv run ruff format --check .` |
| 型別 | `uv run mypy` |
| Django 檢查 | `uv run python manage.py check` |
| `#` 註解檢查 | `uv run python scripts/check_no_hash_comments.py origin/develop` |
| spec8 manifest | `uv run python scripts/check_spec8_manifest.py`（第 8 節的九個測試標 `@pytest.mark.spec8` 並列入 `tests/spec8_manifest.txt`） |

## 目錄

- `config/`：settings（`base`、`dev`、`test`、`prod`，全部讀環境變數）、Celery app、URL。
- `apps/market/`：standard 層 models、交易日曆、還原價。
- `apps/etl/`：來源 adapter、raw 層、`etl_run`、品質檢查、排程任務。
- `apps/research/`：特徵、label、模型、walk-forward、報表。
- `apps/api/`：唯讀 DRF 端點。`apps/web/`：頁面。
- `tests/`：pytest；adapter 測試用 `tests/fixtures/` 的錄製回應，不打真 API。

## 慣例

- Python 3.12、Django 5.x；所有函式寫型別註記，mypy `disallow_untyped_defs`。
- 不寫 `#` 註解；只放行 `#!`、`# noqa`、`# type:`、`# pragma:`、`# fmt:`。需要說明時寫一行 docstring，推理寫進 PR 說明。
- 時間一律 `Asia/Taipei`；日期運算一律走交易日曆模組，不用 `pandas.bdate_range`。
- 成交量存「股」、金額存「元」；寫入 standard 表一律 upsert，重跑冪等。
- 設定（API key、速率上限、排程時間）放環境變數或 settings，不寫死；範本見 `.env.example`。
- 分支從 `develop` 開 `<type>/<slug>`，PR 回 `develop`；`main` 只收 `develop` 測過的合併。
- commit 用英文 `type: description`。
