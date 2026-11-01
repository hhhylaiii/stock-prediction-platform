# stock-prediction-platform

台股上市、上櫃普通股的每日預測平台：收盤後產出未來 5、20 個交易日的報酬排名、預測報酬與價格區間、上漲機率。規格見 [docs/SPEC.md](docs/SPEC.md)。個人研究用途，資料授權限制見規格第 3 節。

## 需求

- macOS 或 Linux、[uv](https://docs.astral.sh/uv/)（會依 `.python-version` 自動安裝 Python 3.12）
- Docker Desktop（Compose v2.24 以上）
- macOS 跑 LightGBM 需要 OpenMP：`brew install libomp`

## 本機開發

```bash
cp .env.example .env
uv sync
docker compose up -d db redis
uv run python manage.py migrate
uv run python manage.py check
uv run pytest
```

## 全部容器

```bash
docker compose up -d --build
docker compose ps
curl http://localhost:8000/healthz
```

四個容器：`web`（gunicorn）、`worker`（Celery worker 內含 beat）、`db`（PostgreSQL 16）、`redis`（broker）。每日排程、重試、手動補抓與防休眠見 [docs/runbook.md](docs/runbook.md)。所有容器與主機都要用 `TZ=Asia/Taipei`，否則排程會偏 8 小時。

- web、db、redis 的對外 port 只綁 `127.0.0.1`，區網連不到。
- Compose 預設用 `config.settings.dev`（DEBUG 開）。改成 `config.settings.prod` 時要在 `.env` 設 `DJANGO_SECRET_KEY` 與 `DJANGO_ALLOWED_HOSTS`，`DATABASE_URL` 也必填（Compose 會自動注入）。`DJANGO_ALLOWED_HOSTS` 要保留 `localhost`，web 的 healthcheck 靠它；只用 http 存取時設 `DJANGO_SECURE_COOKIES=false`，否則瀏覽器可能丟掉 session 與 CSRF cookie。
