# stock-prediction-platform

A daily prediction platform for common stocks listed on the Taiwan Stock Exchange (TWSE) and the Taipei Exchange (TPEx). After each close it produces, for the next 5 and 20 trading days, a return ranking, predicted returns with price intervals, and the probability of beating the market. The specification is [docs/SPEC.md](docs/SPEC.md). For personal research use; see section 3 of the spec for the data licensing restrictions.

The spec, the runbook and the web pages are written in Traditional Chinese.

## Quick start

If the database already exists (`docker compose up -d db redis`, with the data kept in a volume), this is all it takes to see the pages:

```bash
uv sync
uv run python manage.py migrate
uv run python manage.py runserver
```

Open <http://127.0.0.1:8000/>. `.env` is loaded automatically, `DATABASE_URL` defaults to the local `twp` database, and features, models and the prediction cache go to `artifacts/` by default. On first use run `cp .env.example .env`; backfilling data requires `FINMIND_API_TOKEN`.

The UI is a single page with three views, switched by the `#` fragment of the URL:

| View | URL | Content | Data needed |
| --- | --- | --- | --- |
| Today's predictions (今日預測) | `/#today` | Market-wide 5-day and 20-day rankings and predicted returns, the 20-day 80% price interval and the probability that the 20-day return beats the market (the TAIEX total return index, dividends included) by more than 5 percentage points; supports search, a watchlist-only filter and sorting by a column header | A live model |
| Stock (個股) | `/#stock-2330` | Candlestick or closing-price chart, how this prediction came about (feature contributions), past predictions against actual outcomes | Prices; the prediction blocks need a live model |
| Models and data (模型與資料) | `/#models` | The live models, monthly rank IC against the baseline, today's data updates, data-quality and model-output checks, the data correction log | ETL records; the model blocks need a live model |

Until a model is live, the prediction blocks show 「預測資料尚未產出」 ("predictions have not been produced yet") while the candlestick chart, ETL records and quality checks are shown as usual; a model goes live only after its holdout verdict passes, or after the user releases a verdict that fell short. What each page notes about missing, released or force-swapped models, the 80% interval and the dividend-adjusted prices, and the API fields behind each note, are in [docs/api.md](docs/api.md).

## Data and model pipeline

```
Sources (TWSE, TPEx, FinMind, MOPS, TDCC, Yahoo, data.gov.tw)
  → raw tables (original responses) → standard tables (daily prices and volume, institutional and margin positions, financial statements…; upserted, safe to rerun)
  → quality checks (data_quality_log)
  → label_daily, feature_daily (also stored as Parquet)
  → walk-forward training and evaluation (model_run, backtest_metric)
  → daily predictions from the live model (prediction, prediction_contrib) and after-the-fact monitoring (model_monitor)
  → read-only API (/api/...) and pages
```

The daily schedule is run by Celery beat inside the `worker` container; see [docs/runbook.md](docs/runbook.md) for the timetable, gates, retries and catch-up.

## Common commands

Run `uv run python manage.py <command> -h` for the full options of any command.

| Purpose | Command |
| --- | --- |
| Fetch one day or a date range | `fetch --source twse --dataset daily_price --date 2026-11-06` |
| Backfill stock by stock (resumable) | `backfill --source finmind --dataset <dataset> --from 2010-01-01 --resume` |
| Catch up on missed trading days | `catchup --since 2026-11-01 --dry-run` |
| Run one day's quality checks | `check_quality --date 2026-11-06` |
| Rebuild labels and features | `build_labels --start 2010-01-01`, `build_features --start 2010-01-01` (without a range, the last 60 days are rebuilt) |
| Train and evaluate | `train --model lightgbm --scope rank --run-key <key>`, `evaluate_rank --run-key <key>`, `ic_report --run-key <key>` |
| After-the-fact monitoring | `monitor --date 2026-11-06` |
| Phase 5 acceptance report | `phase5_report --out <dir>` |

Every API endpoint is a read-only GET and responds with `{"success", "data", "error", "meta"}`: `/api/status`, `/api/predictions` (`?ordering=` sorts by a shown column), `/api/stocks?q=<code or name>` (at most 20 results), `/api/stocks/<code>` (plus `/prices`, `/predictions`, `/features`, `/history`), `/api/models`, `/api/models/ic-monthly`, `/api/etl/runs`, `/api/data-quality` (plus `/corrections`).

## Layout

- `apps/etl/`: source adapters, the raw layer, `etl_run`, quality checks, scheduled tasks.
- `apps/market/`: standard tables, the trading calendar, adjusted prices.
- `apps/research/`: labels, features, walk-forward, models, reports, monitoring.
- `apps/api/`: the read-only API. `apps/web/`: the pages (`templates/web/index.html`, `static/web/`).
- `docs/`: the spec, the runbook, the pages and API fields (`api.md`) and the acceptance report of each phase (`docs/reports/`).

## Requirements

- macOS or Linux, and [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 automatically from `.python-version`)
- Docker Desktop (Compose v2.24 or later)
- LightGBM on macOS needs OpenMP: `brew install libomp`

## Local development

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

`-m e2e` runs the Playwright browser smoke tests; `playwright install chromium` must have been run once beforehand.

To run the tests in parallel, add `-n auto` and set `OMP_NUM_THREADS=1`: `OMP_NUM_THREADS=1 uv run pytest -m "not e2e" -n auto`. Without it, LightGBM and XGBoost in every worker each take all the cores and fight over the CPU, which ends up slower than not running in parallel at all.

## All containers

```bash
docker compose up -d --build
docker compose ps
curl http://localhost:8000/healthz
```

Four containers: `web` (gunicorn), `worker` (a Celery worker with beat embedded), `db` (PostgreSQL 16) and `redis` (the broker). See [docs/runbook.md](docs/runbook.md) for the daily schedule, retries, manual catch-up and sleep prevention. Every container and the host must use `TZ=Asia/Taipei`; otherwise the schedule is off by 8 hours.

- The published ports of web, db and redis bind to `127.0.0.1` only, so they cannot be reached from the LAN.
- Compose uses `config.settings.dev` by default (DEBUG on). When switching to `config.settings.prod`, set `DJANGO_SECRET_KEY` and `DJANGO_ALLOWED_HOSTS` in `.env`; `DATABASE_URL` is required as well (Compose injects it automatically). Keep `localhost` in `DJANGO_ALLOWED_HOSTS` because the web healthcheck relies on it; when the site is served over plain http only, set `DJANGO_SECURE_COOKIES=false`, or the browser may drop the session and CSRF cookies.
