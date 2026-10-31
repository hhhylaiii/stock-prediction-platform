# Phase 0 驗收報告

Phase 0 於 2026-11-01 通過：四支 PR 合併進 `develop`，規格 §9 的驗收指令對證交所真 API 寫入 raw 與 standard 表。

## 驗收條件

| 條件（規格 §9 Phase 0、plan §6 S0） | 結果 | 證據 |
| --- | --- | --- |
| 測試框架可跑 | 通過，develop 上 266 條測試，coverage 100% | develop CI 的 test job；PR #4 說明的本機 `pytest --cov` |
| Docker Compose 可建置 | 通過 | develop CI 的 docker job（`docker compose build` 加 `manage.py check`）；`/healthz` 由 `tests/test_healthz.py` 驗 |
| CI 設好 | 通過，5 個 job：lint、typecheck、test、spec8、docker；第 6 個 e2e 依 plan 從 PR 5.4 起 | develop CI run 41218735721（`ccae21f`）全綠 |
| `manage.py fetch --source twse --dataset daily_price --date 2026-10-28` 寫入 raw 與 standard | 通過 | 見下節 |

## 驗收指令實測

2026-11-01 在本機（dev settings、本機 Postgres）執行，資料來自證交所 MI_INDEX：

```
$ uv run python manage.py load_calendar --file tests/fixtures/calendar/twse_2026_10.json
trading_calendar: 31 days 2026-10-01..2026-10-31
$ uv run python manage.py fetch --source twse --dataset daily_price --date 2026-10-28
etl_run 3 success: 1381 rows twse/daily_price 2026-10-28..2026-10-28
```

- `raw_twse` 1 列；重跑多次仍 1 列，`daily_price` 列數不變。
- `daily_price` 當天 1381 列：整張「每日收盤行情」1386 列，扣掉 5 檔外幣櫃位。
- 2330：開 2420、高 2430、低 2410、收 2425，量 21,866,835 股，額 52,940,678,675 元，與證交所 STOCK_DAY 一致。
- 非交易日 `--date 2026-10-25` 回 `CommandError: NotATradingDayError`，不發請求、不建 `etl_run`。

## 合併紀錄

| PR | 內容 | 合併 commit |
| --- | --- | --- |
| #1 | 專案骨架、settings、Compose、`/healthz` | `bcb0dac` |
| #2 | CI、`#` 註解檢查、spec8 manifest 檢查 | `364349e` |
| #3 | `stock_master`、`trading_calendar`、`daily_price`、交易日曆模組、還原價 helper | `3a422dd` |
| #4 | ETL 骨架、`raw_twse`、`etl_run`、限速、TWSE `daily_price`、`fetch` 與 `load_calendar` | `ccae21f` |
| #5 | 規格書 rev 112 同步 Phase 0 決定 | `383296c` |

每支程式 PR 都跑過兩輪 `/clean-review`，結果與 head sha 寫在各 PR 的「AI 已驗」。

## 已知限制

- private repo 免費方案不能設 required checks，靠流程只在 CI 綠時合併（2026-11-01 決定維持現狀）。
- Compose 目前跑 dev settings，最晚 PR 2.6 把 worker 切到 prod。
- 未驗：容器內執行 `fetch`；交易日資料公布前證交所的回應（plan PR 2.1 錄 fixture）。
