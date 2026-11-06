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
| 19:00 | `etl.statement_scan STATEMENTS` | MOPS 財報彙總表（綜合損益、資產負債、現金流量，上市、上櫃），正在申報的季別與前一季，見下方「財報每日掃描」 | 規格寫「3/31、5/15、8/14、11/14 前後各 5 天」；這裡是季底隔天到保守截止日後 45 天 |
| 19:00 | `etl.revenue_scan REVENUE` | MOPS 月營收彙總頁（上市、上櫃，國內與國外公司），正在申報的月份，見下方「月營收每日掃描」 | 規格寫每月 1 日到 10 日；這裡是次月 1 日到截止日後 20 天 |
| 21:30 | `etl.run_batch night` | 融資融券、借券；當沖抓當天與前兩個交易日 | 借券約 20:30、22:30 更新；當沖晚間才公布、T+2 才定案 |
| 22:40 | `etl.run_batch notices` | 注意、處置再抓一次 | 名單可能晚於 16:30 公布，23:00 研究任務前補齊 |
| 22:40 | `etl.statement_scan STATEMENTS_LATE` | 財報彙總表再掃一次；時間固定等於 `FILING_VISIBLE_BEFORE` | 新增 |
| 22:40 | `etl.revenue_scan REVENUE_LATE` | 月營收彙總頁再掃一次，時間和財報的最後一次相同 | 新增 |
| 22:50 | `etl.check_quality` | 五項品質檢查，另寫一列 `etl_runs` 列出當天沒完成的 dataset，一起送告警（見下方研究閘門） | — |
| 05:30 | `etl.us_close` | 前一個台股交易日的費半、TSM ADR、美元台幣，每個交易日只跑一次，寫一列 `data_quality_log`（`us_close`）：費半、TSM 早於那天、或美元台幣早於再前一個交易日，就是 warning | 美元台幣在倫敦 23:59 才收盤，05:30 本來就只有前一天 |
| 週六 10:00 | `etl.weekly_holding` | 集保股權分散表，抓最近一個已結束的週（週一才醒來也是抓上週），日期用那週最後一個交易日 | — |
| 每月 15 日 10:00 | `etl.business_signal` | data.gov.tw 景氣燈號 | 規格寫每月 1 日；資料在次次月 15 日才公布 |

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
- 上傳時間在交易日 22:40（`FILING_VISIBLE_BEFORE`，每日財報掃描的最後一次）之前算當天，之後或非交易日算下一個交易日；英文版不算。有重編報表的期別（IFRS 的「(重編)」，舊制的「母公司財報重編」「母子公司合併報表重編」），公告日改成最後一次重編的上傳日，記一列 `statement_restated`（info）：FinMind 存的數字可能是重編後的，原始上傳日就看到會偷看。
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
- 只抓 standard 表沒有的季別，以及每一季的前一季（Q1 也抓前一年 Q4，ROE 才算得出來）。單季值和彙總表用同一套換算（`statement_rows`）。前一季也讀到的已存季別會拿來對照，不改。
- 寫入和 log 跟彙總表歷史相同：保守公告日，`statement_viewer_mismatch` 每檔股票一列（`detail.stock_id`），區間結尾是最後一季的季底，所以重跑到哪一天都是同一列。另外 `absent` 列出兩種報表都查不到的季別（檢視頁回「檔案不存在」或「查無資料」的查詢表單），例如下市前停止申報的公司，或 2013 年後才上市的公司在上市前沒有申報的季別；`no_revenue` 列出認不得營收科目的季別，毛利率、營益率也會跟著是空的，要人工看那家公司的科目名稱。證券商的「收益合計」不當營收，和 FinMind、彙總表一致（證券、銀行的 revenue 都是空的）。重跑時沒有重讀的季別，上次記下的 `rows`、`types` 照留。
- 保守公告日對晚申報的公司偏早，下市公司又常晚申報，所以補完一定要接著跑 `statement_filing`，換成實際上傳日（doc.twse 查得到下市公司）：

  ```bash
  uv run python manage.py shell -v 0 -c "from apps.etl.models import DataQualityLog as L; print('\n'.join(sorted({a['stock_id'] for log in L.objects.filter(check_name='statement_viewer_mismatch') for a in log.detail.get('added', [])})), end='')" > viewer_added.txt
  test -s viewer_added.txt && sed 's/^/--stock /' viewer_added.txt | xargs uv run python manage.py backfill --source mops --dataset statement_filing --from 2013-01-01
  ```
- 91 開頭的存託憑證是外國發行人，不用這張表申報，所以排除。
- 每家公司每季約 1 個請求，6 秒一次；全部下市公司約 1,500 個請求，約 2.5 小時。和每日掃描是同一台 mopsov，在 23:00 之後或早上跑，避開 19:00 和 22:40 前後。

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
