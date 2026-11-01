# 每日排程 runbook

排程機器是個人 Mac（規格 §5「執行環境」）。`worker` 容器裡的 Celery beat 依 `config/settings/base.py` 的 `CELERY_BEAT_SCHEDULE` 觸發工作；時間一律 `Asia/Taipei`，每個時間都可以用 `.env` 的 `ETL_AT_<名稱>=HH:MM` 覆寫。

## 啟動

```bash
docker compose up -d --build
docker compose logs -f worker
```

- worker 一次只跑一個 task（`CELERY_WORKER_CONCURRENCY=1`），同一來源的請求不會並行；`RateLimiter` 只在單一行程內保證間隔。
- 主機與所有容器都要 `TZ=Asia/Taipei`，否則排程會偏 8 小時。

## 排程表

每日工作每天都會觸發，非交易日自己略過（週末補行交易日照跑）。Mac 睡過排程時間、醒來後 beat 會補觸發一次：當日各批、`refresh_master`、`check_quality` 若還沒到當天的排定時間就略過（交給隔天 15:10 的補抓），其他工作照跑。

| 時間 | 工作 | 內容 | 與規格 §5 的差異 |
| --- | --- | --- | --- |
| 15:10 | `etl.prepare_day` | `refresh_calendar` 更正交易日曆（失敗只寫 log），再 `catch_up` 補抓前 14 天到前一交易日漏掉的日子；前一交易日若沒跑過品質檢查就補跑 | 規格「先補抓再跑當日」 |
| 15:30 | `etl.run_batch prices` | 證交所、櫃買日價量與指數 | — |
| 16:00 | `etl.refresh_master` | 用當天行情更新 `stock_master` | 新增（PR 2.5c） |
| 16:30 | `etl.run_batch afternoon` | 三大法人、本益比與淨值比、注意、處置、全額交割；證交所外資持股與發行股數 | 借券、當沖移到 21:30；櫃買外資持股與發行股數移到 18:30 |
| 18:30 | `etl.run_batch qfii` | 櫃買外資持股與發行股數 | 櫃買每日 18:00、22:00 更新 |
| 21:30 | `etl.run_batch night` | 融資融券、借券；當沖抓當天與前兩個交易日 | 借券約 20:30、22:30 更新；當沖晚間才公布、T+2 才定案 |
| 22:40 | `etl.run_batch notices` | 注意、處置再抓一次 | 名單可能晚於 16:30 公布，23:00 研究任務前補齊 |
| 22:50 | `etl.check_quality` | 五項品質檢查，另寫一列 `etl_runs` 列出當天沒完成的 dataset（warning 會告警、不擋預測），一起送告警 | — |
| 05:30 | `etl.us_close` | 前一個台股交易日的費半、TSM ADR、美元台幣，每個交易日只跑一次，寫一列 `data_quality_log`（`us_close`）：費半、TSM 早於那天、或美元台幣早於再前一個交易日，就是 warning | 美元台幣在倫敦 23:59 才收盤，05:30 本來就只有前一天 |
| 週六 10:00 | `etl.weekly_holding` | 集保股權分散表，抓最近一個已結束的週（週一才醒來也是抓上週），日期用那週最後一個交易日 | — |
| 每月 15 日 10:00 | `etl.business_signal` | data.gov.tw 景氣燈號 | 規格寫每月 1 日；資料在次次月 15 日才公布 |

MOPS 月營收與財報掃描（19:00）在 PR 2.4 上線時加入。某個 dataset 在當晚品質檢查之後才補進來時，若那天的檢查有 error，會自動重跑那天的檢查。研究層 23:00、06:00 的任務要先確認 `apps.etl.tasks.ready_for_research(day)`：當天每個每日 dataset 都有成功的 `etl_run`，且 `should_run_prediction(day)` 為 True。

## 重試與失敗

- 每個「來源／dataset／日期」是獨立的 task，一個失敗不擋其他。
- 來源還沒公布：30 分鐘後重試，最多 6 次（`ETL_NOT_READY_COUNTDOWN`、`ETL_NOT_READY_RETRIES`）。用完寫 error log `etl gave up …`。
- 其他錯誤不重試，寫 error log `etl failed …`；`etl_run` 記 failed，隔天 15:10 的 catchup 會再補。
- Redis 的 `visibility_timeout` 是 12 小時（`CELERY_VISIBILITY_TIMEOUT`），要大於最長的 task（補抓日加倍）與重試間隔，否則 `acks_late` 的 task 會被重送。

看失敗：

```bash
docker compose logs worker | grep -E "etl (failed|gave up)|catchup failed"
docker compose exec db psql -U twp -d twp -c \
  "select run_id, source, dataset, start_date, status, error from etl_run where status <> 'success' order by run_id desc limit 20"
```

## 手動補抓與重建

手動指令是另一個行程，限速不和 worker 共用；在 ETL 時段（15:10 到 02:00）以外跑，或先 `docker compose stop worker`。

```bash
uv run python manage.py catchup --since 2026-10-20 --dry-run
uv run python manage.py catchup --since 2026-10-20
uv run python manage.py reprocess --source twse --dataset daily_price --start 2026-10-27 --end 2026-10-28
uv run python manage.py check_quality --date 2026-10-28
```

- `catchup` 預設補到前一交易日；`--dry-run` 只列出每個 dataset 缺哪些日子。
- `reprocess` 只用 raw 表重建，不發請求；只支援證交所、櫃買。

## 告警

warning 與 error 一律寫 log；`.env` 設了 `ALERT_WEBHOOK_URL` 就另外 POST JSON：

```json
{"text": "2 data quality alert(s) for 2026-10-28", "alerts": [{"date": "2026-10-28", "check": "row_count", "severity": "warning", "detail": {}}]}
```

Slack 類的 incoming webhook 會顯示頂層的 `text`；完整內容在 `alerts`。webhook 失敗只寫 log，不影響 ETL。

## 交易日曆延伸

`trading_calendar` 只到證交所已公布開休市表的年底。每年 12 月證交所公布隔年的表之後要重跑一次，否則 1 月 1 日起每個排程工作都會因為日期在日曆範圍外而失敗：

```bash
uv run python manage.py load_calendar --from 2026-12-01
```

## 防休眠

ETL 時段（15:00 到隔天 02:00、清晨 05:15 到 06:45、週六 10:00）Mac 要醒著；螢幕可以關，不要合蓋。launchd 不會把睡著的 Mac 叫醒，所以以下兩項都要做：

1. 必要：接電源，並在「系統設定 > 電池（或能源）」開啟「顯示器關閉時避免自動進入睡眠」。這是週六集保、清晨美股收盤唯一可靠的保護。
2. 加上 launchd 的 caffeinate，讓沒接電源時的 ETL 時段也不會閒置睡眠：

```bash
cp ops/launchd/tw.stock-prediction-platform.caffeinate-*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/tw.stock-prediction-platform.caffeinate-evening.plist
launchctl load ~/Library/LaunchAgents/tw.stock-prediction-platform.caffeinate-morning.plist
```

- 晚上那支週一到週五 15:00 執行 `caffeinate -i -t 39600`（到 02:00，涵蓋 22:40 那批的重試）；清晨那支週二到週六 05:15 執行 `caffeinate -i -t 5400`。規格 §5 寫的是到 23:30（`-t 30600`），差異已列進規格同步待辦。
- launchd 不知道交易日：週末補行交易日當天要手動執行一次 `caffeinate -i -t 39600`，隔天清晨再執行 `caffeinate -i -t 5400`。
- 想讓 Mac 從睡眠中自己醒來，可以加 `sudo pmset repeat wakeorpoweron MTWRFS 05:10:00`（只能設一組時間）。
- 移除：`launchctl unload ~/Library/LaunchAgents/tw.stock-prediction-platform.caffeinate-*.plist` 後刪掉檔案。
