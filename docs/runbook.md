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

啟動 worker 之後，確認排程真的載入了：

```bash
docker compose up -d --build --no-deps worker
docker compose exec -T worker python manage.py shell -c "from django.conf import settings; print(len(settings.CELERY_BEAT_SCHEDULE), sorted(settings.CELERY_BEAT_SCHEDULE))"
```

- 要列出下方排程表的 15 個項目。舊映像的排程可能是空的：2026-11-01 起的 worker 用舊映像帶著空排程跑了 4 天，什麼都沒抓，直到 11/05 重建才上線。
- 第一個時段過後再看 `docker compose logs worker | grep "Sending due task"`，確認排程有送出任務。

worker 只掛主機 `artifacts/` 底下的 `features`、`models`、`predictions` 三個子目錄（研究指令在主機上跑，worker 讀寫同一份；`reports/` 不掛）。換掉 worker 容器之後，確認三個路徑都在掛載點下、讀得到主機的特徵分區：

```bash
docker compose exec -T worker python manage.py shell -c "from django.conf import settings as s; import os; print(s.FEATURE_PARQUET_DIR, s.MODEL_DIR, s.RESEARCH_CACHE_DIR, sorted(os.listdir(s.FEATURE_PARQUET_DIR))[:3])"
```

- 三個路徑都要是 `/app/artifacts/...`。worker 的這三個變數寫死在 `docker-compose.yml`，`.env` 裡的同名變數只影響主機上跑的指令。
- 不要刪 `artifacts/features`、`artifacts/models`、`artifacts/predictions` 這三個目錄本身（或整個 `artifacts/`），只清裡面的檔案：分開掛的目錄被刪掉再建，worker 看到的還是舊的空目錄，讀的時候會報「run build_features first」。刪了就跑 `docker compose up -d --force-recreate --no-deps worker`。
- 拉到這個 compose 之後，下一次碰到 worker 的 `docker compose up -d` 就會換掉 worker 容器（映像不重建），挑排程表上沒有任務的時段。

## 啟動

```bash
docker compose up -d --build
docker compose logs -f worker
```

- worker 一次只跑一個 task（`CELERY_WORKER_CONCURRENCY=1`），同一來源的請求不會並行；`RateLimiter` 只在單一行程內保證間隔。
- 主機與所有容器都要 `TZ=Asia/Taipei`，否則排程會偏 8 小時。

## 排程表

每日工作每天都會觸發，非交易日自己略過（週末補行交易日照跑）。Mac 睡過排程時間、醒來後 beat 會補觸發一次：當日各批、`refresh_master`、`check_quality`、`us_close` 若還沒到當天的排定時間就略過並寫 warning `etl skipped …`（交給隔天 15:10 的補抓），其他工作照跑。

| 時間 | 工作 | 內容 | 備註 |
| --- | --- | --- | --- |
| 15:10 | `etl.prepare_day` | `refresh_calendar` 更正交易日曆（失敗寫 log，當晚 22:50 的 `etl_runs` 列出），再 `catch_up` 補抓前 14 天到前一交易日漏掉的日子；排程開始後沒跑過品質檢查的交易日（至少前一交易日）補跑，跑過但還有告警的重新判斷 | 先補抓再跑當日 |
| 15:30 | `etl.run_batch prices` | 證交所、櫃買日價量與指數；證交所加權股價報酬指數，見下方「加權股價報酬指數」 | — |
| 16:00 | `etl.refresh_master` | 用當天行情更新 `stock_master` | — |
| 16:30 | `etl.run_batch afternoon` | 三大法人、本益比與淨值比、注意、處置、全額交割；證交所外資持股與發行股數 | 借券、當沖在 21:30；櫃買外資持股與發行股數在 18:30 |
| 18:30 | `etl.run_batch qfii` | 櫃買外資持股與發行股數 | 櫃買每日 18:00、22:00 更新 |
| 19:00 | `etl.statement_scan STATEMENTS` | MOPS 財報彙總表（綜合損益、資產負債、現金流量，上市、上櫃），正在申報的季別與前一季，見下方「財報每日掃描」 | 季底隔天到保守截止日後 45 天 |
| 19:00 | `etl.revenue_scan REVENUE` | MOPS 月營收彙總頁（上市、上櫃，國內與國外公司），正在申報的月份，見下方「月營收每日掃描」 | 次月 1 日到截止日後 20 天 |
| 21:30 | `etl.run_batch night` | 融資融券、借券；當沖抓當天與前兩個交易日 | 借券約 20:30、22:30 更新；當沖晚間才公布、T+2 才定案 |
| 22:40 | `etl.run_batch notices` | 注意、處置再抓一次 | 名單可能晚於 16:30 公布，23:00 研究任務前補齊 |
| 22:40 | `etl.statement_scan STATEMENTS_LATE` | 財報彙總表再掃一次；時間固定等於 `FILING_VISIBLE_BEFORE` | — |
| 22:40 | `etl.revenue_scan REVENUE_LATE` | 月營收彙總頁再掃一次，時間和財報的最後一次相同 | — |
| 22:50 | `etl.check_quality` | 五項品質檢查，另寫一列 `etl_runs`，列出當天沒完成的 dataset，以及這晚負責的排程工作哪些沒有成功的 run，一起送告警（見下方研究閘門） | — |
| 05:30 | `etl.us_close` | 前一個台股交易日的費半、TSM ADR、美元台幣，每個交易日只跑一次，寫一列 `data_quality_log`（`us_close`）：費半、TSM 早於那天、或美元台幣早於再前一個交易日，就是 warning | 美元台幣在倫敦 23:59 才收盤，05:30 本來就只有前一天 |
| 週六 10:00 | `etl.weekly_holding` | 集保股權分散表，抓最近一個已結束的週（週一才醒來也是抓上週），日期用那週最後一個交易日；漏跑時由週六之後第一個交易日（週六本身是交易日就是當天）的 `etl_runs` 列出 | — |
| 每月 15 日 10:00 | `etl.business_signal` | data.gov.tw 景氣燈號；晚跑時 run 記在當月的排程日；漏跑時由排程日當天或之後第一個交易日的 `etl_runs` 列出 | M 月的資料在 M+2 月 15 日寫入，並以那天為 announce_date；國發會約每月 27 日發布上月，年假會延後 |

### 研究閘門

研究層 23:00、06:00 的任務要先確認 `apps.etl.tasks.ready_for_research(day)`：`etl_complete(day)` 且 `should_run_prediction(day)`。

- `etl_complete`：當日快照都有成功的 `etl_run`，也就是兩個交易所的日價量、指數、三大法人、融資融券、借券、外資持股、本益比與淨值比（`GATE_DATASETS`）。這些缺一天，當天的特徵就是空的。
- 注意、處置、全額交割、股數變動、當沖不擋：前四個是事件或變動紀錄，漏一天只是晚一天反映，隔天 catchup 補；當沖是規格的第二批特徵，T+2 才定案。
- `etl_runs` 那一列跟閘門一致：缺快照是 error（規格「error 不執行當日預測」），只缺其他 dataset 是 warning，都到了是 info；`detail` 的 `missing` 列出全部缺的，`blocking` 列出擋閘門的。
- 品質檢查之後才補進來的資料：那天的 `etl_runs` 不是 info 就重算；五項檢查有 error 就重跑。重算只送之前沒送過的告警（同一個檢查、同一個 severity 不重送）。
- 排程工作（交易日曆更新、`refresh_master`、週持股、景氣燈號）只由負責那晚的 22:50 檢查列入：交易日曆更新與 `refresh_master` 是每個交易日，週持股與景氣燈號是到期日之後第一個交易日。補審沒審過的交易日時不列；已經列出的，之後重算時會保留，直到補上：
  - 交易日曆更新、週持股：要有 run 涵蓋那一天。週持股的檔案只放最新一週，下一週出來後就補不回來，那一列會一直是 warning，只能接受。
  - `refresh_master`、景氣燈號：之後任何一次成功的 run 都算補上，因為它們會把之前的變動一起帶進來。
  - MOPS 掃描照舊只在當晚列，重算時不列（掃描補不回來）。

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

- `catchup` 預設補到前一交易日；`--dry-run` 只列出每個 dataset 要抓的日子（`to fetch`：缺的日子，加上當沖要重抓的日子），不發請求。
- 當沖 T+2 才定案，每晚的批次會重抓前兩個交易日。所以 `catchup` 補當沖時，每個缺的日子連同它前兩個交易日一起重抓，不管那兩天有沒有列。例如連漏兩晚，漏掉之前的那個交易日只抓過初值，要靠這一步換成定案值。`--dry-run` 的清單也包含這些日子。
- `reprocess` 只用 raw 表重建，不發請求；只支援證交所、櫃買。

## 財報實際公告日回補

Phase 1 的 `financial_statement.announce_date` 是法定截止日的下一個交易日。證交所電子書（`doc.twse.com.tw` t57sb01）列出每份財報的上傳時間，一次性回補把它改成實際可用的交易日：

```bash
uv run python manage.py backfill --source mops --dataset statement_filing --from 2009-01-01
uv run python manage.py backfill --source mops --dataset statement_filing --from 2009-01-01 --resume
```

- 每檔只查 `financial_statement` 有資料的年度，每年一次請求（金控兩次，見下），約 3.2 萬次，6 秒間隔約 53 小時。doc.twse 每分鐘大約只接受 12 次，超過就回「查詢過量，請稍後再查詢!」，client 會退避後重試（`MOPS_MIN_INTERVAL` 不能低於 5 秒）。拒收的代號用 `--retry-rejected <run_id>` 重跑。
- 上傳時間在交易日 22:40（`FILING_VISIBLE_BEFORE`，每日財報掃描的最後一次）之前算當天，之後或非交易日算下一個交易日；英文版不算。有重編報表的期別（IFRS 的「(重編)」「(第二次重編)」，舊制的「母公司財報重編」「母公司財報第二次重編」「母子公司合併報表重編」），公告日改成最後一次重編的上傳日，記一列 `statement_restated`（info）：FinMind 存的數字可能是重編後的，原始上傳日就看到會偷看。
- 會讀的報表（`REPORT_TYPES`）：合併是「IFRSs合併財報」「母子公司合併報表」，個體是「IFRSs個體財報」「IFRSs個別財報」「母公司財報」。資料類型不是「財務報告書」、或細節說明不在這些名稱（含下面的重編與附件寫法）裡的，整檔拒收，要看頁面再決定怎麼收。
- 不算公告的附件：英文版、「關係企業合併報表」、銀行的「國際金融業務分行財報」。改過會計年度的公司，舊年制那一季的「舊會計年制--<上列任一報表>」照同種報表算（目前只見過 2718 民國 101 年 Q4 的母公司財報）；同一季同種報表新舊年制並存時，分不出 FinMind 存的是哪一份，整檔拒收（`lists old and current fiscal year reports`）。
- 金控（2880–2892 等）的查詢頁先回子公司清單；client 會帶 `check2858=Y` 再查一次，拿金控自己的報表，raw 存的是第二次的參數與頁面。
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

## 除權息日的公告日

`dividend_event.announce_date` 是除權息日第一次公開的那個 session，`fund_days_to_ex_date` 靠它判斷何時能用。

- 來源是 FinMind `TaiwanStockDividend` 的 `AnnouncementDate` 加 `AnnouncementTime`。驗證過它是除權息公告，不是董事會決議：2.35 萬筆有除權息日的列裡，96% 在除權息日前 30 天內，中位數 15 天。
- 換成 session 的規則和財報申報相同：交易日 `FILING_VISIBLE_BEFORE`（22:40）之前公告就算當天，否則算下一個交易日；沒有公告時間的算下一個交易日。同一個除權息日有多筆，取最早的。
- 公告晚於除權息日的（多半是事後更正）留空，資料表的約束也不允許；日誌會列出來。
- 只填空的：證交所、櫃買的除權息預告表（plan 2.7）上線後是主要來源，FinMind 只補它沒給的；已經有日期的列不改。FinMind 回補時一併填；現有的列用 raw 重套，不用重抓：

```bash
uv run python manage.py reapply_dividend_dates
uv run python manage.py reapply_dividend_dates --stock 2884 --stock 2885
```

- 第二行只重套指定的股票，例如修過某檔的 raw 之後。輸出 `N events dated across M stocks`；被拒收的股票逐行寫在 stderr（`rejected <代號>: <原因>`），例如 FinMind 少了 `AnnouncementDate` 欄位，或公告日落在交易日曆之外。拒收只影響那一檔。
- 重套會把同一檔存過的每一份 FinMind 回應合起來，同一個除權息日取最早的公告。所以清空再重套只能補回同樣的日期，或採用新回應裡更早的日期；FinMind 事後把公告日改晚，這樣是改不過來的，要人工更新那一列。要重填某檔（例如修過 raw 之後），先把那檔清空再重套；2.7 上線後清掉的也包含預告表給的日期，要一起重跑預告表：

  ```bash
  uv run python manage.py shell -v 0 -c "from apps.market.models import DividendEvent as D; print(D.objects.filter(stock_id='2884').update(announce_date=None))"
  uv run python manage.py reapply_dividend_dates --stock 2884
  ```

- 減資、分割這類只有參考價、沒有配息公告的事件，FinMind 沒有日期，等 2.7 的預告表。

## 財報每日掃描

`etl.statement_scan` 在交易日 19:00 與 22:40 讀 MOPS 彙總表，把新申報的季報寫進 `financial_statement`。

- 掃描期間：季底隔天到保守截止日（`statement_deadline`，金融業 Q1–Q3 是 5/31、8/31、11/30）後 `STATEMENT_SCAN_DAYS_AFTER`（45）天，晚交的公司也收得到。預設下全年每個交易日都有季別在掃（每天 24–36 個請求）；兩季重疊時（1 月上旬、4 月到 5 月中、7 月上旬、10 月上旬）兩季都掃。天數調小、出現沒有季別的日子時，任務回 `no period open`，不寫 `etl_run`。
- 每季抓 6 頁（3 種報表 × 2 個市場），加上前一季 6 頁算單季值；同一次掃描裡每季只抓一次，兩季重疊時共 18 頁。全部先存 `raw_mops`（`statement_scan`）再解析；頁面裡每次都不同的匯出檔名不算進去重，內容相同的頁只存一份。
- 單季值：年初至今減前一季；Q1 直接用（ROE 用前一年年報的權益）。前後季報表種類不同（個體↔合併）的，流量欄位留空。三張表都出現的公司才寫；正在上傳中的等下一次掃描，log 會寫 `scan waits for …`。
- 已過保守截止日的季別（前一季一定是，晚交期間的本季也是）頁面不該是空的：回「查詢無資料」就整次失敗，不寫入只有負債比的列。產業表少了 EPS、淨利、歸屬母公司淨利、資產、負債、權益、營業現金流任一欄（改名也算），或表外出現公司代號，也整次失敗。
- 單季 EPS：彙總表只有累計 EPS。前後季隱含股數差 5% 內用相減，否則用單季淨利 × 累計 EPS ÷ 累計淨利，進位到分；累計 EPS 不到 0.10 時進位誤差太大，不判斷股數，直接相減。對 FinMind：2026-Q2 1,980 家中 68 家差超過 0.02；2025-Q4 有 75 家，多半是 FinMind 錯（股票分割的 8422，FinMind 直接把年度減前三季）。
- 公告日：寫入的那一刻在當天 22:50 品質檢查之前就算當天，否則下一個交易日（掃描遇到退避、拖過 22:50 的就算隔天）。已有的列（任何來源）不改值也不改日期；同一期已存另一種報表種類的不寫，記 warning `statement_type_mismatch`；同一種但數字不同（EPS 差 0.02、金額差 1,000 元、比率差 1e-6 以上）的記 info `statement_source_mismatch`（規格 §8.5），列出各欄的舊值與新值。兩種 log 每天一列，重跑取代。
- `FILING_VISIBLE_BEFORE` 必須早於 `ETL_AT_CHECK_QUALITY`，設反了 settings 會拒絕載入：否則 22:40 掃描寫入的列都算隔天，回測卻把那段時間上傳的算當天。
- 錯誤照每日 ETL 的規則：網路錯誤與 5xx 重試，格式變動失敗不重試，下一次掃描再來。不進研究閘門、不進 catchup：漏掉的掃描只會讓公告日晚一點，下一次掃描就補上。有季別開放的交易日，當天沒有一次成功的掃描，22:50 的 `etl_runs` 會把 `mops/statement_scan` 列進 `missing`，記 warning 並告警，不擋研究。只有當晚的檢查會列；之後補審漏掉的交易日、或資料晚到重審時不列，因為掃描補不回來。
- 手動掃一次（只能掃今天）：

```bash
uv run python manage.py fetch --source mops --dataset statement_scan --date $(date +%F)
```

## 月營收每日掃描

`etl.revenue_scan` 在交易日 19:00 與 22:40 讀 MOPS 月營收彙總頁（`/nas/t21/{sii|otc}/t21sc03_{民國年}_{月}_{0|1}.html`，0 是國內公司、1 是國外公司），把新申報的月營收寫進 `monthly_revenue`。目前沒有其他每日來源寫月營收（FinMind 只做歷史回補），沒掃到的月份就沒有列。

- 掃描期間：營收月份的次月 1 日到法定截止日（`revenue_deadline`，2026 年起是次月 15 日）後 `REVENUE_SCAN_DAYS_AFTER`（20）天，晚交的公司也收得到。每天都有一個月份在掃（4 頁），月初幾天兩個月份重疊（8 頁）。
- 頁面大約每小時重新產生一次，頁面上的出表時間註解不算進 raw 去重。全部先存 `raw_mops`（`revenue_scan`）再解析。
- 營收由千元換成元。yoy、mom 用頁面上的去年當月、上月營收算，分母 ≤ 0 時為空，定義和 FinMind 相同；頁面上的是更正後的最新數字，所以和 FinMind 存的不一定一樣（2026-09：1,981 家中 110 家的 yoy 不同，全部是去年或當月營收被更正過）。研究層的成長率特徵自己由營收序列算，不用這兩欄。
- 格式檢查：標題要是請求的市場、年、月；每張產業表的單位都要是千元；頁面最後一列要是「全部國內／國外{市場}公司合計」（頁面被截斷就沒有這列）；產業表要有當月營收、上月營收、去年當月營收；表外出現公司代號、格子數不符、金額不是整數、同一代號出現兩次（含跨市場），都整次失敗。還沒有人申報時頁面只寫「查無資料」，是空答案；過了截止日某個市場國內加國外都沒有公司，也整次失敗。
- 公告日：寫入的那一刻在當天 22:50 品質檢查之前就算當天，否則下一個交易日，和財報掃描相同。已有的列不改值也不改日期；數字不同（營收不同，或 yoy、mom 差 1e-6 以上）記 info `revenue_source_mismatch`（規格 §8.5），每天一列，重跑取代。
- 錯誤、閘門、catchup、`etl_runs` 告警的處理和財報掃描相同（`mops/revenue_scan`）。掃描是唯一的線上來源，告警後要在掃描期間內修好；過了期間，晚交公司的那個月就沒有列。
- 不要對還在掃描期間的月份跑 FinMind `monthly_revenue` 回補：FinMind 的公告日一律是截止日後的交易日，比晚交公司的實際申報日早。這些列先寫進去的話，掃描「第一次看到的為準」會保留那個太早的日期，等於偷看。
- 手動掃一次（只能掃今天）：

```bash
uv run python manage.py fetch --source mops --dataset revenue_scan --date $(date +%F)
```

## MOPS 歷史回補（plan K17）

FinMind 回補缺的月份、季別，用 MOPS 的月營收頁（t21sc03）和財報彙總表（t163sb04／05／20）補一次：

```bash
for y in $(seq 2013 2025); do
  uv run python manage.py fetch --source mops --dataset revenue_history --start $y-01-01 --end $y-12-31
  uv run python manage.py fetch --source mops --dataset statement_history --start $y-01-01 --end $y-12-31
done
uv run python manage.py fetch --source mops --dataset revenue_history --start 2026-01-01 --end 2026-09-30
uv run python manage.py fetch --source mops --dataset statement_history --start 2026-01-01 --end 2026-06-30
```

- 一次執行逐個日曆年處理，每年記一次 checkpoint。區間裡沒有 2013 年以後的期別就拒絕；只要有每日掃描還在看的期別（截止日加上 `REVENUE_SCAN_DAYS_AFTER`／`STATEMENT_SCAN_DAYS_AFTER` 還沒過），整次就拒絕。那些期別的公告日由每日掃描用第一次看到的 session 決定。月營收從 2013-01 起，更早的頁面是另一種版面；財報從 2013-Q1（IFRS）起，這一季不抓前一季，所以 ROE 是空的。
- 只寫入 standard 表還沒有的（股票, 期別）；財報只要同一期已有任何一種報表種類就不寫。公告日用保守值，和 FinMind 回補相同：月營收是 `revenue_visible_from`，財報是保守截止日的下一個交易日。
- 每個年度區間記一列 info：`revenue_history_mismatch`、`statement_history_mismatch`。`date` 是區間的最後一天，重跑同一個區間（不論哪天跑）會取代那一列，沒有任何內容就不留。`detail` 有：
  - `added`：這個區間補進來的（股票, 期別），重跑會累積保留。standard 表沒有來源欄，這是哪些列由 MOPS 補進來的唯一紀錄。
  - `rows`：已有列的數字不同。月營收只比營收，容許差 1,000 元，因為頁面以千元為單位、FinMind 以元為單位；yoy、mom 的算法本來就不同，不比。財報比所有欄位，容忍值和財報掃描相同。已有的列不改，要不要覆寫看 plan K18。
  - `types`：已有的列是另一種報表種類。
  - `incomplete`：只出現在部分彙總表的公司，這些不寫入，要人工查。
- 財報補進來之後，對那些股票跑 `statement_filing` 回補，公告日就會換成實際上傳日：

  ```bash
  uv run python manage.py shell -v 0 -c "from apps.etl.models import DataQualityLog as L; print('\n'.join(sorted({a['stock_id'] for log in L.objects.filter(check_name='statement_history_mismatch') for a in log.detail.get('added', [])})), end='')" > added.txt
  test -s added.txt && sed 's/^/--stock /' added.txt | xargs uv run python manage.py backfill --source mops --dataset statement_filing --from 2013-01-01
  ```

  `test -s` 讓清單是空的時候不會變成全市場回補。

- 兩種頁面都是 MOPS 現在的狀態：彙總表只列現在還公開發行的公司，2013 年起的月營收頁也是 2026-11-04 重新產生的，所以下市公司不會出現在裡面。下市公司的財報要另外用 XBRL 檢視頁補（K17b）；月營收找不到來源（K19）。
- 每年約 48 個月營收請求、24 到 30 個財報請求，6 秒一次。避開 19:00 和 22:40 前後，那兩個時段 worker 的每日掃描也在打 mopsov。

### 下市公司的財報（plan K17b）

彙總表不列下市公司，改用 XBRL 檢視頁（t164sb01），一家公司一季一頁，下市公司也查得到：

```bash
uv run python manage.py shell -v 0 -c "from apps.market.models import StockMaster as M; print('\n'.join(M.objects.filter(delist_date__gte='2013-04-01', stock_id__regex=r'^[1-9][0-9]{3}$').exclude(stock_id__startswith='91').order_by('stock_id').values_list('stock_id', flat=True)), end='')" > delisted.txt
test -s delisted.txt && sed 's/^/--stock /' delisted.txt | xargs uv run python manage.py backfill --source mops --dataset statement_viewer --from 2013-01-01
```

- 只看 2013-Q1 起、在下市日之前結束、而且每日掃描已經不看的季別。先問合併報表（`REPORT_ID=C`），沒有就問個別報表（`A`）；同一家公司上一季回答的種類下一季先問。
- 每一季都抓（Q1 也抓前一年 Q4，ROE 才算得出來）：standard 表沒有的季別寫入，已存的季別拿來對照、不改，之後由 K18 的 `overwrite_statements` 從 raw 改成 MOPS 的數字。單季值和彙總表用同一套換算（`statement_rows`）。前一季沒申報的，MOPS 只換算得出負債比和 Q1 除 ROE 以外的各項；其他欄位換算不出來，不比，也不記進 `statement_viewer_mismatch`。
- 寫入和 log 跟彙總表歷史相同：保守公告日，`statement_viewer_mismatch` 每檔股票一列（`detail.stock_id`），區間結尾是最後一季的季底，所以重跑到哪一天都是同一列。另外 `absent` 列出 standard 表沒有、兩種報表也都查不到的季別（檢視頁回「檔案不存在」或「查無資料」的查詢表單），例如下市前停止申報的公司，或 2013 年後才上市的公司在上市前沒有申報的季別；已存的季別查不到則列在 `stored_absent`，這些列 K18 沒得改；`no_revenue` 列出認不得營收科目的季別（已存的季別列在 `stored_no_revenue`），毛利率、營益率也會跟著是空的，要人工看那家公司的科目名稱。證券商的「收益合計」不當營收，和 FinMind、彙總表一致（證券、銀行的 revenue 都是空的）。重跑時沒有重讀的季別，上次記下的 `rows`、`types` 照留。
- 保守公告日對晚申報的公司偏早，下市公司又常晚申報，所以補完一定要接著跑 `statement_filing`，換成實際上傳日（doc.twse 查得到下市公司）：

  ```bash
  uv run python manage.py shell -v 0 -c "from apps.etl.models import DataQualityLog as L; print('\n'.join(sorted({a['stock_id'] for log in L.objects.filter(check_name='statement_viewer_mismatch') for a in log.detail.get('added', [])})), end='')" > viewer_added.txt
  test -s viewer_added.txt && sed 's/^/--stock /' viewer_added.txt | xargs uv run python manage.py backfill --source mops --dataset statement_filing --from 2013-01-01
  ```
- 91 開頭的存託憑證是外國發行人，不用這張表申報，所以排除。
- 每家公司每季約 1 個請求，6 秒一次；165 檔下市公司共 4,498 季，約 4,500–5,000 個請求，約 8 小時。和每日掃描是同一台 mopsov，在 23:00 之後、早上或沒有 ETL 的週末跑，避開 19:00 和 22:40 前後。
- 每次抓到的頁面都把 raw 的 `fetched_at` 更新成這次，內容和舊頁相同也一樣；K18 從 raw 重套時讀的是最後一次抓到的頁，所以重讀遇到空白頁、`--retry-rejected` 再抓回正常頁之後，K18 讀到的是正常頁。
- 任何一季（包括只拿來對照的已存季別）的頁面讀不懂，整檔都會被拒收，缺的季別也不會寫入。檢視頁偶爾回只有標題的空白頁（例如 1333 的 2016-Q1，重抓就正常）也是這樣；修好格式或等一下，再用 `--retry-rejected <run_id>` 重跑。

## FinMind 財報改用 MOPS 原始數字（plan K18）

FinMind 的財報有些不是當季原始申報的數字。最明顯的是現金流：奇數年的季別很多是用隔年財報的比較數，2013–2025 年共 7,029 筆不同，偶數年只有 48 筆，這會偷看。所以已存的列，改用 MOPS 當季申報的數字（K17a 的彙總表、K17b 的檢視頁），從 raw 重套，不重抓：

```bash
uv run python manage.py overwrite_statements --dry-run
uv run python manage.py overwrite_statements
uv run python manage.py overwrite_statements --stock 2330
```

- 比對範圍是同一檔、同一季、同一種報表（合併或個體）。MOPS 的列和 K17a、K17b 一樣，用 `statement_rows` 換成單季值。前一季的頁面不在 raw 裡時，只比不需要前一季的數字：負債比，以及 Q1 除 ROE 以外的各項（Q1 的累計數就是單季數）；Q2–Q4 的流量、EPS 和各季的 ROE 換算不出來，是空的，不比。已申報季別的彙總表 raw 若是空的（沒有任何公司），指令會停下來，不會默默少比一個市場。
- `revenue`、`net_income`、`operating_cf`、`debt_ratio`、`roe`：超過門檻就覆寫（金額差超過 1,000 元，比率照 `figure_differences`），FinMind 是空的就補上。
- `eps` 只補空的，不覆寫：MOPS 的單季 EPS 是累計數相減，股數加權和財報上的單季數字不同。
- `gross_margin`、`op_margin` 不拿 MOPS 覆寫：兩邊都有值時，2013–2026 只有 4 筆 op_margin 不同。MOPS 是空的約 3,350 筆，多半是 Q2、Q4，而且前一季沒有申報，單季換算不出來；FinMind 這些列多半是累計數的比率（例如 1264 的 2013-Q2 等於上半年的毛利率），由下面的清空規則處理。
- MOPS 是空的欄位，一律保留 FinMind 的值。
- 每日掃描開始看的那一季起（第一次 `statement_scan` 的 `open_periods`，目前是 2026-Q3）不改：那些季別以掃描第一次看到的數字為準。
- 檢視頁 raw 讀不懂的股票整檔跳過，指令最後列出 `set aside …`；先修好格式、`--retry-rejected` 之後再跑一次。
- 每檔股票一列 `statement_overwritten`（info），日期是該檔最後一個被改季別的季底（和 K17b 一樣，不會全部落在執行當天），記 `[原值, 新值]`。重跑時同一欄位再改一次，原值仍是 FinMind 最早的那個；這次沒動到的紀錄照留。要還原就照這列把原值寫回。
- 2026-11-07 在 twp 的 dry run：92,792 個 MOPS 列，9,988 個財報列會改；1333、1469、1715、2847、6022 的檢視頁讀不懂，跳過。後四檔是 Q1 標「YYYY年第1季」、Q2 標「YYYY年上半年度」，K17b 重讀起認得；1333 是暫時性的空白頁，`--retry-rejected` 重抓。
- 沒得比的列：彙總表不列下市公司，K17b 第一輪只抓 standard 表缺的季別和它們的前一季，所以下市公司已存的 FinMind 季別大多沒有 MOPS 可比。2026-11-07 量到 3,045 列，其中奇數年、有現金流的 1,405 列；還在公開發行的公司只有 404 列，沒有奇數年的現金流。K17b 重讀改成每一季都抓，跑完、`--retry-rejected` 之後再跑一次本指令。
- 同一個指令也清掉不是單季的毛利率、營益率（`statement_margin_cleared`，info，每檔一列）：
  - FinMind 在前一季沒申報的季別，存的是累計或兩季合併的比率，同一個序列裡混著單季和累計，YoY 類特徵會失真。
  - 2013-Q1 起：用 MOPS 當季的累計數（彙總表，沒有就用檢視頁；同一種報表），試「最近 k 季」的毛利／營收、營業利益／營收，差在 0.0001 以內算對得上。對得上單季的保留；只對得上兩季以上（含年初至今）的清成 NULL，log 記 `[原值, null, 種類]`（`cumulative`、`2q`、`3q`），可以照這列還原。
  - 對不上的保留：前一季的頁面在，單季試過也對不上的列在 log 的 `unmatched`；前一季的頁面不在、單季沒得試的列在 `untried`。前一季不在時只要對得上累計就清，所以單季剛好等於累計的值也會被清（機率很低）。
  - 2013-Q1 起但當季沒有同一種報表的 MOPS 累計數：FinMind 自己把營收清掉的，毛利率、營益率一樣清成 NULL，種類記 `unjudged`；營收還在的照留（算「沒判斷」）。不這樣做的話，MOPS 彙總表不列下市公司，清不清會跟公司日後有沒有下市有關。
  - 有檢視頁讀不懂的股票整檔不判斷，指令最後列出 `margins of … not judged`。
  - 2013 年以前沒有 MOPS 頁面可比：FinMind 自己把營收清掉（同年前面有季沒申報）、毛利率或營益率還在的，一律清成 NULL，種類記 `unverified`。2013 年後同一種列約 96% 是累計或兩季合併的值（協調 2026-11-07 定案）。`unverified`、`unjudged` 是清掉的種類，`unmatched`、`untried` 是照留的清單。
  - 只看 Q2–Q4；每日掃描看過的季別不看；log 的日期是那一檔最後一個有紀錄的季底。重跑時之前清過的紀錄照留、同一欄不重複記，`unmatched`、`untried` 換成這次的結果。
  - FinMind 的寫入端不改：季報回補已跑完，2026-Q3 起的季別走 MOPS 每日掃描。之後若重跑 FinMind 季報回補，再跑一次本指令。
  - 2026-11-07 在 twp 的 dry run（K17b 每一季重讀進行中）：2013-Q1 起清 6,914 個（`cumulative` 4,160、`2q` 2,274、`unjudged` 480），2013 年前清 3,618 個（`unverified`）；14 個對不上、305 個單季沒得試，照留；3,332 個沒有 MOPS 累計數、營收也還在，沒判斷。K17b 讀完後再跑一次。
- 公告日不動。覆寫後要重建財報類特徵（Phase 3）。

## 當沖歷史回補

當沖量從 2014 年起回補，每個市場、每一年跑一個 run。兩個市場都從 2014-01-06 開始（現股當沖開放日，之前的日期來源沒有資料，run 會停在第一天），2026 年跑到每日排程接手的前一天（2026-10-21）。先確認 `daily_price` 的歷史回補已經跑完：

```bash
for source in twse tpex; do
  uv run python manage.py fetch --source $source --dataset day_trade --start 2014-01-06 --end 2014-12-31
  for year in $(seq 2015 2025); do
    uv run python manage.py fetch --source $source --dataset day_trade --start "$year-01-01" --end "$year-12-31"
  done
  uv run python manage.py fetch --source $source --dataset day_trade --start 2026-01-01 --end 2026-10-21
done
```

- 某一年中途失敗時，看那個 run 的 `etl_run.checkpoint.last_date`，從下一個交易日用 `--start` 接著跑到年底。
- 每個市場約 3,100 個請求，3 秒一次約 2.6 小時。手動指令不和 worker 共用限速，在 ETL 時段（15:10–02:00）以外跑，或先 `docker compose stop worker`。
- 「第一頁行情存檔」是從 raw 讀的，不是設定值：任何一次對更早日期抓交易所的 `daily_price`，界線就會移到那天，那天之後到 2026-10-21 的當沖都會 SourceNotReady。回補前先確認兩個市場的界線都是 `20261022`，回補期間不要對更早的日期抓交易所 `daily_price`：

  ```sql
  select min(payload->>'date') from raw_twse where dataset = 'daily_price';
  select min(payload->>'date') from raw_tpex where dataset = 'daily_price' and request_params->>'type' = 'EW';
  ```
- 籌碼表只寫當天行情表上有的代號（K6）。歷史日期沒有存交易所的原始行情頁，價量是 FinMind 回補的，所以當沖在該市場第一頁行情存檔（2026-10-22）之前的日子，改用 `daily_price` 當天有價的代號來篩；之後的日子照舊要等當天的行情頁。`daily_price` 不分市場，所以當天的當沖表如果一個代號都沒留下（例如本市場的價還沒回補、只有另一個市場有價），run 會 SourceNotReady，不會記成成功 0 列。其他籌碼表（三大法人、融資券、借券、外資持股）仍要等行情頁。
- 兩種篩法的差別（2018 年每月第一個交易日抽查）：`daily_price` 有價的代號全部都在行情表上；行情表多出 32–41 個代號。這些是 FinMind 沒有的 ETF、TDR，加上 5 檔 FinMind 整檔沒有價格的普通股（1589、4207、4804、5355、8176），以及沒成交那天的 4415。這些代號的當沖量會略過；它們沒有價格，本來也進不了研究母體。日後如果補了這些代號的 `daily_price`，要對同一段日期跑 `reprocess --source twse|tpex --dataset day_trade --start … --end …`（只讀 raw、不發請求），否則研究層會把它們的當沖量當成 0。
- 當沖表會列出量為 0 的標的。同市場當天的表有寫入、某檔不在表上，研究層算 0；同市場當天一列當沖都沒有的，才是空值。所以兩個市場每個交易日都要有成功、而且有列的 run。

## 加權股價報酬指數

`index_daily` 的 `TAIEX_TR` 是證交所的發行量加權股價報酬指數（MFI94U）：成分股的現金股利再投入，除息不會讓它下跌。`TAIEX` 是價格指數，除息時照樣扣掉股息，7、8 月除息旺季 20 日內約差 1 個百分點。研究層的 label 用還原價，要比大盤時用 `TAIEX_TR` 才同一個口徑。

- 每日 15:30 的 prices 批抓一次，一個月一個請求；不擋研究閘門，品質檢查的指數新鮮度也不含它，缺了只會在 `etl_run` 失敗時告警，隔天 catchup 補。
- 回補從 2010 年起，一個月一個請求，約 203 個、12 分鐘。先跑 `migrate`（`market.0003`）：

  ```bash
  uv run python manage.py migrate market
  uv run python manage.py fetch --source twse --dataset return_index --start 2010-01-04 --end 2026-11-06
  ```

- 中途失敗時，看那個 run 的 `etl_run.checkpoint.last_date`，從下一個交易日接著跑。

## 交易所逐日表補缺（plan K17c）

FinMind 的外資持股、流通股數、本益比在 2010–2020 有缺，下市股缺得特別多（約 21% 對現存股 0.4–0.7%），模型會把「缺值」學成「之後會下市」。交易所的逐日全市場表列出當天所有掛牌的股票，用它把缺的補上。範圍是 2010 年起到每日排程接手的前一天（2026-10-21）：只補到 2020 年的話，2021 年後才有第一筆股數、或完全沒有股數的股票，會把 2020 年最後一個股數一路帶到下市或下一筆。

兩個市場是不同主機，各開一個終端機平行跑；同一個主機一次只跑一個指令（每個程序各有一份限速，兩個同時打同一主機就會快過 3 秒一次）：

```bash
for fill in holdings valuation; do
  uv run python manage.py fill_from_exchange --source twse --fill $fill --start 2010-01-01 --end 2026-10-21
done
```

櫃買把 `--source` 換成 `tpex`。中斷了用 `--resume <run_id>` 接著跑。

- `holdings`：證交所 MI_QFIIS、櫃買 insti/qfii，一天一個請求，同一頁補外資持股比率（`daily_chip.fi_holding_pct`）和發行股數（`shares_outstanding`）。`valuation`：證交所 BWIBBU_d、櫃買 peQryDate，補 `daily_valuation`。
- 缺才補，不分股票之後有沒有下市：
  - `daily_chip`：沒有這一列就新增（只有這一欄）；有這一列、這一欄是空的才補；有值的不動。
  - `daily_valuation`：當天沒有這一列才新增，三個數字都空的不新增。
  - `shares_outstanding`：只補每檔第一筆已存股數之前的日子，照「變動才存」寫變動點；跑完時，補的最後一筆和第一筆已存股數相同的，刪掉那筆重複的已存列（生效中的股數不變），刪掉的列記在 log 的 `dropped`（`[代號, 日期, 股數]`），可以照著寫回。界線取第一次 `holdings` 補缺開跑時的第一筆已存股數（記在那個 `etl_run.checkpoint.windows`），之後的補缺、`--resume`、另一個市場都沿用，所以補進去的列不會被當成已存的界線。
- 只寫當天在 `daily_price` 有價的代號（不用歷史行情頁，K6）；當天交易所表上的代號一個都沒有價時 SourceNotReady，不會記成成功 0 列。
- 已有的值和交易所差到公布的最後一位（外資持股比、本益比、淨值比 0.01，殖利率 0.0001）、或界線那天的股數不同時，不改，記在 log 的 `differences`：每種表的筆數、分下市股和現存股的筆數，和前 20 筆 `[代號, 日期, 欄位, 已存值, 交易所值]`（SPEC §8.5）。
- `etl_run` 的 dataset 是 `fill:holdings`／`fill:valuation`，每天和寫入同一個交易記 checkpoint。跑完記一列 `exchange_fill`（info）：`filled` 是每檔補了哪段日期、幾天，`created` 是其中新增整列的部分，`totals` 分下市股和現存股的列數，`span` 是第一次開跑的起點到結尾。
- 先跑完 FinMind 的 `daily_price`、`shares_outstanding`、`daily_chip`、`daily_valuation` 回補（含 `--retry-rejected`）再跑補缺：`daily_price` 決定補哪些代號、`shares_outstanding` 決定股數的界線（第一次 `holdings` 補缺時就固定）；FinMind 寫 `daily_chip`、`daily_valuation` 時整列已存在就跳過，補缺新增的列會擋掉之後的 FinMind 回補。之後真的要再回補 FinMind，先刪掉補缺新增的列：`daily_chip` 是只有 `fi_holding_pct` 有值、其他欄位都是空的列；`daily_valuation` 的 `created` 只記每檔的日期範圍，範圍內可能夾著 FinMind 的列，要逐日分辨時從兩邊的 raw 重算。
- 每個請求約 3.5 秒；一個市場一個 fill 2010-01-04 到 2026-10-21 約 4,130 天、約 4 小時，兩個 fill 約 8 小時。可以跑的時段是平日 02:00–15:00 和週末，和每日 ETL（含 02:00 前的重試）錯開；平日 15:00 前沒跑完就停，之後 `--resume`。

## 交易所產業補缺（plan K17c）

`stock_master.industry` 是空的股票（多半是 FinMind 股票清單沒有的下市股），用交易所「依產業別」的收盤行情補上。依序跑，先證交所、再櫃買；`--start` 從 2009-10-01 起，2009-12-31 那次抽樣才看得到 2010 年第一季就下市的股票：

```bash
uv run python manage.py fill_industry --source twse --start 2009-10-01 --end 2026-09-30
uv run python manage.py fill_industry --source tpex --start 2009-10-01 --end 2026-09-30
```

每個市場約 2,500 個請求、約 2.5 小時。手動指令不和 worker 共用限速，在 ETL 時段（15:10–02:00）以外跑；同一主機一次只跑一個指令。`--end` 不能晚於今天。

- 每季取最後一個交易日（區間的結尾若在季中，就取結尾前最後一個交易日），一天一個產業別一個請求：證交所 `MI_INDEX`、櫃買 `afterTrading/otc`，產業別代碼 01–38，07、13 是彙總表，不抓。回應先存 raw（`industry_quotes`），再檢查：證交所「沒有符合條件的資料」當成那個代碼沒有股票，其他 stat、日期不符就停；櫃買 stat、日期、表名、`totalCount` 不符就停。一天 36 個代碼都沒有股票也停。
- 產業名稱先用頁面自己的（證交所表名括號裡、櫃買的 `category`），`stock_master` 有用這個名稱時就照用；沒有用的（例如當年的舊名稱）才改用那個代碼底下「產業已知的股票」最多數的產業。頁面名稱是當年的分類，不會把 2022–23 年才新增的類別往回套。
- 一檔出現在好幾季時，用最後一次看到的產業。只寫產業是空的股票，已有的不動；跨市場的股票由先跑的市場補，所以要依序跑（上櫃轉上市的以證交所為準）。
- 一次執行一列 `etl_run`（dataset `fill:industry`），每天記 checkpoint（`seen`、`types`），寫回 `stock_master` 留到最後一起做；中斷了就整段重跑（約 2 小時，raw 照存）。跑完記一列 `industry_fill`（info）：`types` 記每個抽樣日每個有股票的代碼 `[採用的產業（沒得採用是 null）, 頁面名稱, 多數產業, 票數, 有產業的股票數]`，`filled` 記真的寫進去的每檔 `[產業, 代碼, 這個產業第一次看到, 最後一次看到]`，`unfilled` 列出這個市場在 `--start` 時還沒下市、跑完仍是空的股票。
- 抽樣頁整頁留在 raw，所有股票都在，不只空的那些：之後要建依時間的產業表，從 raw 加 `types` 重建，不用重抓。
- `refresh_stock_master` 不更新 `industry`，補上的值會留著；但重跑 `load_stock_master` 會把 FinMind 沒有的股票洗回空的，之後要再跑一次本指令。每日的 `refresh_stock_master` 新增代號時產業也是空的，每季季底後兩個市場各跑一次（`--start` 季初、`--end` 季底）補上。
- 補完後產業類特徵的歷史值會變：`uv run python manage.py build_features --start 2010-01-01 --end <最新價日>`（不帶範圍只重算最近 60 個交易日）。中性化 IC 直接讀 `stock_master`，補完當下就會變，補完前產出的 Phase 4 報表要重算才能比。
- 2026-11-07 的 twp：3,065 檔裡 483 檔是空的，2010 年後下市或還在掛牌的 73 檔（櫃買 62、證交所 11），其餘 410 檔在 2010 年前下市，研究期間用不到。

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

- 晚上那支週一到週五 15:00 執行 `caffeinate -i -t 39600`（到 02:00，涵蓋 22:40 那批的重試）；清晨那支週二到週六 05:15 執行 `caffeinate -i -t 5400`。
- launchd 不知道交易日：週末補行交易日當天要手動執行一次 `caffeinate -i -t 39600`，隔天清晨再執行 `caffeinate -i -t 5400`。
- 想讓 Mac 從睡眠中自己醒來，可以加 `sudo pmset repeat wakeorpoweron MTWRFS 05:10:00`（只能設一組時間）。
- 移除：`launchctl unload ~/Library/LaunchAgents/tw.stock-prediction-platform.caffeinate-*.plist` 後刪掉檔案。
