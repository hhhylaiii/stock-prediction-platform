# 每日排程 runbook

排程機器是個人 Mac（規格 §5「執行環境」）。`worker` 容器裡的 Celery beat 依 `config/settings/base.py` 的 `CELERY_BEAT_SCHEDULE` 觸發工作；時間一律 `Asia/Taipei`，每個時間都可以用 `.env` 的 `ETL_AT_<名稱>=HH:MM` 覆寫。

## 上線前

第一次開排程前做一次：

```bash
uv run python manage.py catchup --since $(date -v-14d +%F) --dry-run
uv run python manage.py refresh_stock_master
```

- 第一次 15:10 的 `prepare_day` 會補抓前 14 天缺的日子；靠 `etl_run` 判斷缺日的 dataset（股數、注意、處置、全額交割）在排程前沒有每日紀錄，會整段補抓，股數還會重建之後的日子，每個補到的日子都跑一次品質檢查並告警。先看 `--dry-run` 的清單，太長就先在 ETL 時段外手動跑 `catchup`。
- `refresh_stock_master` 先手動跑一次，確認第一次新增與下市的代號合理，再交給 16:00 的排程。

## 啟動

```bash
docker compose up -d --build
docker compose logs -f worker
```

- worker 一次只跑一個 task（`CELERY_WORKER_CONCURRENCY=1`），同一來源的請求不會並行；`RateLimiter` 只在單一行程內保證間隔。
- 主機與所有容器都要 `TZ=Asia/Taipei`，否則排程會偏 8 小時。

## 排程表

每日工作每天都會觸發，非交易日自己略過（週末補行交易日照跑）。Mac 睡過排程時間、醒來後 beat 會補觸發一次：當日各批、`refresh_master`、`check_quality`、`us_close` 若還沒到當天的排定時間就略過並寫 warning `etl skipped …`（交給隔天 15:10 的補抓），其他工作照跑。

| 時間 | 工作 | 內容 | 與規格 §5 的差異 |
| --- | --- | --- | --- |
| 15:10 | `etl.prepare_day` | `refresh_calendar` 更正交易日曆（失敗只寫 log），再 `catch_up` 補抓前 14 天到前一交易日漏掉的日子；排程開始後沒跑過品質檢查的交易日（至少前一交易日）補跑，跑過但還有告警的重新判斷 | 規格「先補抓再跑當日」 |
| 15:30 | `etl.run_batch prices` | 證交所、櫃買日價量與指數 | — |
| 16:00 | `etl.refresh_master` | 用當天行情更新 `stock_master` | 新增（PR 2.5c） |
| 16:30 | `etl.run_batch afternoon` | 三大法人、本益比與淨值比、注意、處置、全額交割；證交所外資持股與發行股數 | 借券、當沖移到 21:30；櫃買外資持股與發行股數移到 18:30 |
| 18:30 | `etl.run_batch qfii` | 櫃買外資持股與發行股數 | 櫃買每日 18:00、22:00 更新 |
| 19:00 | `etl.statement_scan STATEMENTS` | MOPS 財報彙總表（綜合損益、資產負債、現金流量，上市、上櫃），正在申報的季別與前一季，見下方「財報每日掃描」 | 規格寫「3/31、5/15、8/14、11/14 前後各 5 天」；這裡是季底隔天到保守截止日後 5 天 |
| 21:30 | `etl.run_batch night` | 融資融券、借券；當沖抓當天與前兩個交易日 | 借券約 20:30、22:30 更新；當沖晚間才公布、T+2 才定案 |
| 22:40 | `etl.run_batch notices` | 注意、處置再抓一次 | 名單可能晚於 16:30 公布，23:00 研究任務前補齊 |
| 22:40 | `etl.statement_scan STATEMENTS_LATE` | 財報彙總表再掃一次；時間固定等於 `FILING_VISIBLE_BEFORE` | 新增 |
| 22:50 | `etl.check_quality` | 五項品質檢查，另寫一列 `etl_runs` 列出當天沒完成的 dataset，一起送告警（見下方研究閘門） | — |
| 05:30 | `etl.us_close` | 前一個台股交易日的費半、TSM ADR、美元台幣，每個交易日只跑一次，寫一列 `data_quality_log`（`us_close`）：費半、TSM 早於那天、或美元台幣早於再前一個交易日，就是 warning | 美元台幣在倫敦 23:59 才收盤，05:30 本來就只有前一天 |
| 週六 10:00 | `etl.weekly_holding` | 集保股權分散表，抓最近一個已結束的週（週一才醒來也是抓上週），日期用那週最後一個交易日 | — |
| 每月 15 日 10:00 | `etl.business_signal` | data.gov.tw 景氣燈號 | 規格寫每月 1 日；資料在次次月 15 日才公布 |

MOPS 月營收掃描（19:00）在 PR 2.4b 加入。

### 研究閘門

研究層 23:00、06:00 的任務要先確認 `apps.etl.tasks.ready_for_research(day)`：`etl_complete(day)` 且 `should_run_prediction(day)`。

- `etl_complete`：當日快照都有成功的 `etl_run`，也就是兩個交易所的日價量、指數、三大法人、融資融券、借券、外資持股、本益比與淨值比（`GATE_DATASETS`）。這些缺一天，當天的特徵就是空的。
- 注意、處置、全額交割、股數變動、當沖不擋：前四個是事件或變動紀錄，漏一天只是晚一天反映，隔天 catchup 補；當沖是規格的第二批特徵，T+2 才定案。
- `etl_runs` 那一列跟閘門一致：缺快照是 error（規格「error 不執行當日預測」），只缺其他 dataset 是 warning，都到了是 info；`detail` 的 `missing` 列出全部缺的，`blocking` 列出擋閘門的。
- 品質檢查之後才補進來的資料：那天的 `etl_runs` 不是 info 就重算；五項檢查有 error 就重跑。重算只送之前沒送過的告警（同一個檢查、同一個 severity 不重送）。

## 重試與失敗

- 每個「來源／dataset／日期」是獨立的 task，一個失敗不擋其他。
- 來源還沒公布、或 limiter 自己退避完仍連不上（逾時、斷線、5xx）：30 分鐘後重試，最多 6 次（`ETL_NOT_READY_COUNTDOWN`、`ETL_NOT_READY_RETRIES`）。用完寫 error log `etl gave up …`。
- 格式改變、4xx 等其他錯誤不重試，寫 error log `etl failed …`；`etl_run` 記 failed，隔天 15:10 的 catchup 會再補。
- 被限流（429）時 limiter 會在同一個請求上冷卻 15 分鐘、最多 4 次，最壞讓唯一的 worker 卡住 1 小時，後面的 task 跟著延後；冷卻完仍被限流就當天放棄、不再重試（再打只會延長封鎖），隔天 catchup 補。22:40 之後才被限流，22:50 的檢查可能拖過午夜而被略過（`etl skipped`），由隔天 15:10 補跑。
- Redis 的 `visibility_timeout` 是 12 小時（`CELERY_VISIBILITY_TIMEOUT`），要大於最長的 task（補抓日加倍）與重試間隔，否則 `acks_late` 的 task 會被重送。

看失敗：

```bash
docker compose logs worker | grep -E "etl (failed|gave up|skipped)|catchup failed"
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

## 財報實際公告日回補

Phase 1 的 `financial_statement.announce_date` 是法定截止日的下一個交易日。證交所電子書（`doc.twse.com.tw` t57sb01）列出每份財報的上傳時間，一次性回補把它改成實際可用的交易日：

```bash
uv run python manage.py backfill --source mops --dataset statement_filing --from 2009-01-01
uv run python manage.py backfill --source mops --dataset statement_filing --from 2009-01-01 --resume
```

- 每檔只查 `financial_statement` 有資料的年度，每年一次請求，約 3.2 萬次，6 秒間隔約 53 小時。doc.twse 每分鐘大約只接受 12 次，超過就回「查詢過量，請稍後再查詢!」，client 會退避後重試（`MOPS_MIN_INTERVAL` 不能低於 5 秒）。拒收的代號用 `--retry-rejected <run_id>` 重跑。
- 上傳時間在交易日 22:40（`FILING_VISIBLE_BEFORE`，每日財報掃描的最後一次）之前算當天，之後或非交易日算下一個交易日；英文版不算。有「(重編)」重編報表的期別，公告日改成最後一次重編的上傳日，記一列 `statement_restated`（info）：FinMind 存的數字可能是重編後的，原始上傳日就看到會偷看。
- 規則（plan §8 K14）：沒有更正紀錄的財報，公告日就是實際上傳日，早於或晚於法定截止日都照實際；有更正紀錄的，上傳時間可能是更正後重傳，只拿來提早、不延後。晚於保守值（截止日的下一個交易日）的每檔記一列 `data_quality_log`：沒更正、已延後的是 `statement_filed_late`（info），有更正、沒延後的是 `statement_filed_later`（warning）。
- 每天 16:00 的 `refresh_master` 只要多一個代號，`--resume` 就會因為股票清單變了而拒絕。這時改用 `--stock` 只跑還沒跑到的代號（`etl_run.checkpoint.last_stock_id` 之後的），並把那次 run 已拒收的代號一起帶上；中斷的 run 沒有結束，不能用 `--retry-rejected`：

```bash
uv run python manage.py backfill --source mops --dataset statement_filing --from 2009-01-01 \
  $(docker compose exec -T db psql -U twp -d twp -At -c \
    "select distinct '--stock ' || stock_id from financial_statement where stock_id > '<last_stock_id>'
     union select '--stock ' || jsonb_object_keys(checkpoint->'rejected') from etl_run where run_id = <run_id>")
```

- 規則改了不用重抓：`reapply_filings` 用 `raw_mops` 裡每個「公司×年度」最新的一頁重套，幾分鐘內跑完（可加 `--stock`）。

```bash
uv run python manage.py reapply_filings
docker compose exec db psql -U twp -d twp -c \
  "select check_name, count(*), sum(jsonb_array_length(detail->'filings')) from data_quality_log where check_name in ('statement_filed_late', 'statement_filed_later', 'statement_restated') group by 1"
```

- 月營收沒有歷史申報日可查，維持 Phase 1 的保守值。

## 財報每日掃描

`etl.statement_scan` 在交易日 19:00 與 22:40 讀 MOPS 彙總表，把新申報的季報寫進 `financial_statement`。

- 掃描期間：季底隔天到保守截止日（`statement_deadline`，金融業 Q1–Q3 是 5/31、8/31、11/30）後 `STATEMENT_SCAN_DAYS_AFTER`（5）天。4 月初年報與 Q1 重疊時兩季都掃；沒有季別在申報時任務回 `no period open`，不寫 `etl_run`。
- 每季抓 6 頁（3 種報表 × 2 個市場），加上前一季 6 頁算單季值；全部先存 `raw_mops`（`statement_scan`）再解析。
- 單季值：年初至今減前一季；Q1 直接用。前後季報表種類不同（個體↔合併）的，流量欄位留空。三張表都出現的公司才寫，正在上傳中的等下一次掃描。
- 單季 EPS：彙總表只有累計 EPS。前後季隱含股數差 5% 內用相減，否則用單季淨利 × 累計 EPS ÷ 累計淨利，進位到分。2026-Q2 對 FinMind：1,980 家中 74 家差超過 0.02。
- 公告日：第一次看到的那次掃描，在當天 22:50 品質檢查之前開始就算當天，否則下一個交易日。已有的列（任何來源）不改值也不改日期；同一期已存另一種報表種類的不寫，記 warning `statement_type_mismatch`。
- 錯誤照每日 ETL 的規則：網路錯誤與 5xx 重試，格式變動失敗不重試，下一次掃描再來。不進研究閘門、不進 catchup：漏掉的掃描只會讓公告日晚一點，下一次掃描就補上。
- 手動掃一次（只能掃今天）：

```bash
uv run python manage.py fetch --source mops --dataset statement_scan --date $(date +%F)
```

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
