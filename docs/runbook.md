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

- 要列出下方排程表的 17 個項目（15 個 ETL 加 `research-evening`、`research-morning`）。舊映像的排程可能是空的：2026-11-01 起的 worker 用舊映像帶著空排程跑了 4 天，什麼都沒抓，直到 11/05 重建才上線。
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
| 15:30 | `etl.run_batch prices` | 證交所、櫃買日價量與指數；證交所加權股價報酬指數，見下方「加權股價報酬指數」；當天的除權息、減資、面額變更，見下方「每日除權息、減資、面額變更」 | 除權息、恢復買賣的結果表在當天開盤前就公布 |
| 16:00 | `etl.refresh_master` | 用當天行情更新 `stock_master` 與 `market_history`（見下方「市場歷史」） | — |
| 16:30 | `etl.run_batch afternoon` | 三大法人、本益比與淨值比、注意、處置、全額交割 | 借券、當沖、證交所外資持股與發行股數在 21:30；櫃買外資持股與發行股數在 18:30 |
| 18:30 | `etl.run_batch qfii` | 櫃買外資持股與發行股數 | 櫃買每日 18:00、22:00 更新 |
| 19:00 | `etl.statement_scan STATEMENTS` | MOPS 財報彙總表（綜合損益、資產負債、現金流量，上市、上櫃），正在申報的季別與前一季，見下方「財報每日掃描」 | 季底隔天到保守截止日後 45 天 |
| 19:00 | `etl.revenue_scan REVENUE` | MOPS 月營收彙總頁（上市、上櫃，國內與國外公司），正在申報的月份，見下方「月營收每日掃描」 | 次月 1 日到截止日後 20 天 |
| 21:30 | `etl.run_batch night` | 融資融券、借券；當沖抓當天與前兩個交易日；證交所外資持股與發行股數 | 借券約 20:30、22:30 更新；當沖晚間才公布、T+2 才定案；證交所 MI_QFIIS 2026-11-09 在 20:30 到 21:00 之間公布，排 16:30 時 30 分鐘重試 6 次到 19:33 都等不到 |
| 22:40 | `etl.run_batch notices` | 注意、處置再抓一次 | 名單可能晚於 16:30 公布，23:00 研究任務前補齊 |
| 22:40 | `etl.statement_scan STATEMENTS_LATE` | 財報彙總表再掃一次；時間固定等於 `FILING_VISIBLE_BEFORE` | — |
| 22:40 | `etl.revenue_scan REVENUE_LATE` | 月營收彙總頁再掃一次，時間和財報的最後一次相同 | — |
| 22:50 | `etl.check_quality` | 五項品質檢查，另寫一列 `etl_runs`，列出當天沒完成的 dataset，以及這晚負責的排程工作哪些沒有成功的 run，一起送告警（見下方研究閘門） | — |
| 23:00 | `research.evening` | 研究閘門通過後依序重算最近 60 個交易日的特徵、最近 81 個交易日的 label，見下方「研究排程」 | `RESEARCH_SCHEDULE_ENABLED` 打開才跑 |
| 05:30 | `etl.us_close` | 前一個台股交易日的費半、TSM ADR、美元台幣，每個交易日只跑一次，寫一列 `data_quality_log`（`us_close`）：費半、TSM 早於那天、或美元台幣早於再前一個交易日，就是 warning | 美元台幣在倫敦 23:59 才收盤，05:30 本來就只有前一天 |
| 06:00 | `research.morning` | 前一個交易日 D：等 D 的 `us_close`，再依序重算 D 的特徵、`predict`、`monitor`，見下方「研究排程」 | 同上 |
| 週六 10:00 | `etl.weekly_holding` | 集保股權分散表，抓最近一個已結束的週（週一才醒來也是抓上週），日期用那週最後一個交易日；漏跑時由週六之後第一個交易日（週六本身是交易日就是當天）的 `etl_runs` 列出 | — |
| 每月 15 日 10:00 | `etl.business_signal` | data.gov.tw 景氣燈號；晚跑時 run 記在當月的排程日；漏跑時由排程日當天或之後第一個交易日的 `etl_runs` 列出 | M 月的資料在 M+2 月 15 日寫入，並以那天為 announce_date；國發會約每月 27 日發布上月，年假會延後 |

### 研究閘門

研究層 23:00、06:00 的任務要先確認 `apps.etl.tasks.ready_for_research(day)`：`etl_complete(day)` 且 `should_run_prediction(day)`。

- `etl_complete`：當日快照都有成功的 `etl_run`，也就是兩個交易所的日價量、指數、三大法人、融資融券、借券、外資持股、本益比與淨值比，加上除權息事件（`GATE_DATASETS`）。前面幾項缺一天，當天的特徵就是空的；除權息事件缺了，除權息日的還原價會錯。
- 注意、處置、全額交割、股數變動、當沖、加權股價報酬指數不擋：前四個是事件或變動紀錄，漏一天只是晚一天反映，隔天 catchup 補；當沖是規格的第二批特徵，T+2 才定案；報酬指數只給第 7 節相對大盤的機率 label 用，停止更新時由指數新鮮度檢查告警（見下方「加權股價報酬指數」）。
- `etl_runs` 那一列跟閘門一致：缺快照是 error（規格「error 不執行當日預測」），只缺其他 dataset 是 warning，都到了是 info；`detail` 的 `missing` 列出全部缺的，`blocking` 列出擋閘門的。
- 品質檢查之後才補進來的資料：那天的 `etl_runs` 不是 info 就重算；五項檢查有 error 就重跑。重算只送之前沒送過的告警（同一個檢查、同一個 severity 不重送）。
- 排程工作（交易日曆更新、`refresh_master`、週持股、景氣燈號）只由負責那晚的 22:50 檢查列入：交易日曆更新與 `refresh_master` 是每個交易日，週持股與景氣燈號是到期日之後第一個交易日。補審沒審過的交易日時不列；已經列出的，之後重算時會保留，直到補上：
  - 交易日曆更新、週持股：要有 run 涵蓋那一天。週持股的檔案只放最新一週，下一週出來後就補不回來，那一列會一直是 warning，只能接受。
  - `refresh_master`、景氣燈號：之後任何一次成功的 run 都算補上，因為它們會把之前的變動一起帶進來。
  - MOPS 掃描照舊只在當晚列，重算時不列（掃描補不回來）。

### 研究排程

`.env` 設 `RESEARCH_SCHEDULE_ENABLED=true` 才跑；預設關閉，`research.evening`、`research.morning`、`research.retrain` 觸發了也直接結束（正式 run 上線後再打開）。

- 23:00（`RESEARCH_AT_EVENING`，要晚於 22:50 的品質檢查）：當天是交易日、`ready_for_research` 通過時，用一條 Celery chain 依序跑 `build_features --end D --recent 60`、`build_labels --end D --recent 81`（SPEC §6 步驟 6：每天重算最近 60 個交易日，catch-up 回補、來源更正與還原因子改寫都由這裡補上；label 多 21 天，涵蓋前看窗口跨進這 60 天的那些）。閘門沒過就 30 分鐘後重試，最多 6 次（同 ETL），用完寫 warning；重試帶著 D，過了午夜仍是 D 的工作。
- 06:00（`RESEARCH_AT_MORNING`，要晚於 05:30 的 `us_close`）：D 是今天之前最後一個交易日，閘門沒過就不預測、寫 warning。D 的 `us_close` 還沒寫就 5 分鐘後再看，最多 6 次（`RESEARCH_US_CLOSE_COUNTDOWN`、`RESEARCH_US_CLOSE_RETRIES`）；等不到照跑，市場特徵沿用前一個美股收盤，完成紀錄記 warning。chain 依序跑：`build_features --date D`（只算 D）→ `predict` → `monitor`，前一步成功才跑下一步；D 的 23:00 沒有成功時，改成補上 23:00 的兩步：先跑 81 天 label，`monitor` 之後再跑 60 天特徵（預測不用等它，也不受它失敗影響）。
- 每條 chain 跑完寫一列 `data_quality_log`（`research_evening`、`research_morning`，`date` 是 D）：成功是 info，`detail.steps` 依序列出每一步的輸出；有一步失敗是 warning，`detail` 寫哪一步、錯誤與已完成的步驟，後面的步驟不跑，並送告警。D 已經有 info 那列就不重跑，所以週六早上跑過週五後，週日、週一早上不會重複；失敗的 D 會在下一個早上（還是同一個 D 時，例如週末）重跑。
- 每月重訓：每月前 5 個交易日裡，第一條在 06:00 到 09:00（`RESEARCH_AT_MORNING` 到 `RESEARCH_RETRAIN_BEFORE`）之間跑完的早上 chain（D，info 或缺美股收盤的 warning 都算）之後排一個 `research.retrain(D)`；不在這段時間跑完的（例如中午補跑、深夜才醒來補觸發）留給下一個早上，以免重訓壓到 15:10 的 ETL 或隔天 06:00 的預測。`retrain` 每次開始（含記憶體不足後的重試）也只在這段時間內動工，否則不寫列直接結束，留給下一個早上重排。每月一次：那個月已有 `research_retrain` 列就不排，排重了也不會重跑。前 5 個交易日都沒排成時，第 6 個交易日起第一條跑完的 chain 寫一列 warning（三個 scope 都是 `no morning chain finished before the cutoff ...`）並告警，當月不再自動重訓，要的話手動 `build_run` 再 `promote`。月中第一次打開開關時也會記這一列。
  - 開始前看可用記憶體（worker 容器讀 `/proc/meminfo`）：低於 `RESEARCH_RETRAIN_MIN_FREE`（50%）就 30 分鐘後再看，最多 6 次（`RESEARCH_RETRAIN_COUNTDOWN`、`RESEARCH_RETRAIN_RETRIES`），用完記 warning、全部拒建；重試到 09:00 之後就照上一條停下。讀不到就照跑。
  - 每個 scope 各自處理：`production.DESIGNS[scope].research_run_key` 是 None（設計還沒凍結）就拒建；`label_daily` 還沒建到 D 就全部拒建；那個月已經手動建過（當月建立、`<scope>-` 開頭的正式 build，重訓自己的 `<scope>-YYYY-MM` 與它的 shadow 不算）就拒建（`already built this month`）；否則 `build_run`，run_key 是 `<scope>-YYYY-MM`。上次重訓中斷留下的 `<scope>-YYYY-MM` 會被重建。
  - 結果寫一列 `research_retrain`（`date` 是 D）：全部建好是 info，有拒建或失敗是 warning 並告警。`detail` 有 `built`、`refused`、`failed`、`free_percent`（開始時的可用記憶體 %）與 `runs`（每個 scope 的 `elapsed_s`，以及建完那個 scope 時 worker 行程到當時為止的最高 RSS `peak_rss_mb`，會跨 scope 累積）。
  - 重訓不取研究鎖，但會佔住唯一的 worker 好幾個小時（真資料上的時間還沒量過）。那段時間不要手動重建特徵或 label、不要停 worker（`acks_late`：中斷的重訓要等 12 小時的 `CELERY_VISIBILITY_TIMEOUT` 才重新投遞，那時已不在動工時段，會直接結束；下一個早上若還在前 5 個交易日內會重排、從頭重建），並讓 Mac 醒到這列寫完：清晨的 caffeinate 只到 08:00，重訓那天手動再跑一次 `caffeinate -i -t 36000`，或確認能源選項的「顯示器關閉時避免自動進入睡眠」已開。
- 換模：下一個早上起，06:00 的 chain 在 monitor 之後多一步 `promote`，把當月 `research_retrain` 建好、還沒試過的 run_key 一個個 `promote`，`--date` 取 D 與預設日（slot 已有列的最後一天）較晚的一天，所以第一次上線的 scope 也從 D 起服務。門檻沒過就舊模型續用並告警（`model_promotion` warning），之後由人決定要不要 `--force`。`promote` 報錯時其他 scope 照跑，chain 照常完成，另寫一列 `research_promote` warning（`detail.failed` 是 run_key 對錯誤、`detail.done` 是其他 scope 的結果）並告警；報錯的 run_key 不再自動重試，修好後先看它是不是已經 active（`model_promotion` 有沒有 `promoted: true` 那列），沒有再手動 `promote`。有 `model_promotion` 或 `research_promote` 列的 run_key 都算試過。設計凍結的那支 PR 要在 `DESIGNS` 填 `research_run_key`（不放環境變數）；填了之後，`build_run` 拒絕其他研究 run。
- 漏跑告警：beat 觸發的 06:00 若發現前一個交易日沒有任何 `research_morning` 列，而再往前 5 個交易日內有過（排程在跑），就替那天記一列 warning（`reason: missed`）並告警，不自動補跑。關掉開關不到 5 個交易日又打開時，第一個早上也會記這一列。
- 研究鎖：`predict`、`monitor`、`promote`、`restore_run` 與研究排程的每一步共用一把 Postgres advisory lock，同時只會有一個在跑（同一個工作內可重入，例如 `promote` 內部的 `predict`）。排程的步驟拿不到就 10 分鐘後重試，最多 6 次（`RESEARCH_LOCK_COUNTDOWN`、`RESEARCH_LOCK_RETRIES`），用完算失敗。手動指令拿不到會直接報錯，寫出持有者，例如 `the research lock is held by 'research 10-29 06:01 predict 2026-10-28' (pid 123)`（取得時間在前；Postgres 只留標籤前 63 個字元，太長的 run_id 尾巴會被截掉）；等它跑完再重跑。行程中斷時鎖隨連線釋放，不會卡住。
- 看狀態：

  ```bash
  docker compose logs worker | grep -E "research (skipped|.* failed)"
  docker compose exec db psql -U twp -d twp -c \
    "select date, check_name, severity, detail from data_quality_log where check_name like 'research_%' or check_name = 'adj_factor_changed' order by date desc limit 10"
  ```

- Mac 在 06:00 睡著、同一天晚一點醒來時，beat 會補觸發 `research.morning` 照跑（可能已經開盤）；睡過午夜才醒的那次早於排定時間，會略過。平日的 D 沒跑成（閘門沒過、睡過午夜）不會自動補：隔天早上的 D 已經是新的一天。閘門沒過會記 warning 並告警；睡過午夜而被略過的那次，隔天早上會補記一列 `missed` warning（見上）。等 D 的 ETL 補齊（15:10 的補抓）後手動跑 `build_features --date D`、`predict --date D`、`monitor --date D`。
- 手動長時間重建特徵或 label（`build_features`、`build_labels` 不取研究鎖）時先關掉開關，或避開 23:00–02:30 與 06:00–08:00，否則排程那一步可能和它搶同一批列而失敗（記 warning）。會改還原因子的手動操作（`fetch --dataset dividend_event`、`recompute_adj_factor`、FinMind 股利或價格回補）也避開這兩段：chain 開始之後才寫進來的改動，可能被那條 chain 記成已處理卻沒算到，事後照那天 `adj_factor_changed` 的 E 手動跑下面重建用的兩個指令。
- 排程的正確性假設 worker 只有一個（`CELERY_WORKER_CONCURRENCY=1`）：重訓和每日的 chain 靠排隊錯開，重訓本身不取研究鎖。`research.evening`、`research.morning`、`research.retrain` 開始時會讀 Celery 的 `worker_concurrency`，不是 1（包括設定成空值；環境變數沒設時預設是 1）就不跑，寫一列 warning（`reason` 寫明、`concurrency` 是讀到的值；重訓那列是每個 scope 都拒建，和記憶體不足一樣算用掉當月的重訓）並告警。只看設定值：`celery worker -c` 從指令列改的、`docker compose up --scale worker=2`、或在主機上另開一個連到同一個 redis 的 worker 都讀不到，不要那樣啟動。
- 還原因子改寫後的重建：`refresh_adj_factor` 改寫了某檔的歷史因子時，在當天的 `adj_factor_changed`（info，資料層的列，研究層只讀）記下那檔第一個「新舊因子的倍數和前一個價格列不同」的交易日 E（單一事件就是除權息日；多個事件一起改取最早的）。只是整段等比例縮放（例如公布未來的除權息）不記，那些日子的比值不變；新價格列取得因子（`upsert_daily_prices`、FinMind 價格回補）也不記。同一天合成一列，超過 200 檔時 `detail` 只留最早的 E 與 `count`（收合時是檔數，之後每記一次加 1，同一檔再記會多算）。
  - 23:00 與 06:00 的 chain 在有還沒處理的列時，最後多一步：E 在最近 60 個交易日內（D 與之前 59 個，同 `--recent 60`）時，23:00 那條（以及補跑 23:00 的 06:00）的 60 天特徵與 81 天 label 已經涵蓋，只記下來；一般的 06:00 那條只算 D 的特徵，所以另外跑 `build_features --start E --end <D 前一個交易日>` 與 `build_labels --start <E 前 21 個交易日> --end D`（前一晚之後才出現的改動不等到下一晚，當天排的重訓才吃得到）。E 晚於 D 的列留給之後的 chain。
  - 處理過的列記在研究層自己的 `research_rebuild`：成功是 info（`handled` 是 `covered` 或 `rebuilt`），`detail.notes` 是這次處理的列（E ≤ D 的那些，pk 對當時的內容）。E 在這 60 個交易日之前的列不自動重建，另寫一列 warning（`earliest`、`stocks`、`notes`、`limit_days`，`handled` 是 `by_hand`）並告警，照 E 手動跑上面兩個指令；同一批裡較新的列照樣自動重建。之後那天的 `adj_factor_changed` 又併進新股票時，內容和任何一次記下的都不同，會再處理一次。只看最近 90 天（`NOTE_WINDOW`）的列。
  - 重建這一步裡任何錯誤都只寫 `research_rebuild` warning（`error`、`limit_days`，`handled` 是 `failed`，沒有 `notes`）並告警，chain 照常完成（當天的預測與重訓排程不受影響），沒處理完的列留到下一條 chain 再試（同一條 chain 先寫了 `by_hand` 時，那批已處理，只有 60 個交易日內的留下）；一直失敗會每條 chain 告警一次，E 看 `adj_factor_changed`。
  - 開關關著時這些列照樣累積；打開後第一條 chain 可能對 90 天內較舊的改動告警一次（在最近 60 個交易日之前的）。告警的 `notes` 裡每一列的日期都早於開開關前那次手動重建時，才可略過；有較晚的列就照 `earliest` 再重建一次。
  - 早上的重建排在完成紀錄之前，所以算進重訓「06:00 到 09:00 之間跑完」的時間：重建拖過 09:00 時，那天不排重訓，月初 5 個交易日內的下一個早上再排（重訓因此一定吃到重建後的特徵）。
  - 特徵只用比值，沒有直接用還原價水準的（市值、本益比用原始收盤價），所以 E 之前的列不用重算。
- 打開開關前，先在真資料上手動跑一次 `build_run`、一次 `build_features --recent 60`，量時間與記憶體（上面手動操作要避開的時段照量到的時間調整：23:00 的閘門重試到 02:00 才開 chain 時，60 天特徵可能跑過 02:30）；重訓一旦跑過午夜，23:00 的 chain 與 ETL 的批次會因為早於排定時間被略過。
- 打開開關時延長 `ops/launchd` 的 caffeinate：傍晚那次（15:00 起 `-t 39600`，到 02:00）改成 `-t 46800`（到 04:00），清晨那次（05:15 起 `-t 5400`）改成 `-t 9900`（到 08:00）。23:00 的閘門最晚重試到 02:00 才開始 chain，06:00 的 chain 最多等美股收盤 30 分鐘，每一步搶不到鎖還能再等 60 分鐘。
- 上線前的日子要用 `predict --date D` 回填時（Phase 5 demo、驗收），先照「每日除權息、減資、面額變更」用 `fetch --source twse|tpex --dataset dividend_event` 補那段日子，否則研究閘門不會過。

## 重試與失敗

- 每個「來源／dataset／日期」是獨立的 task，一個失敗不擋其他。
- 來源還沒公布、或 limiter 自己退避完仍連不上（逾時、斷線、5xx）：30 分鐘後重試，最多 6 次（`ETL_NOT_READY_COUNTDOWN`、`ETL_NOT_READY_RETRIES`）。用完寫 error log `etl gave up …`。22:40 那次財報、月營收掃描的重試過了午夜會被略過（`etl skipped`）；那天新上傳的申報，下一個交易日 19:00 的掃描記同一個公告日（22:50 以後寫入的本來就算下一個交易日），只有掃描期間的最後一天會漏。
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
- 手動補完（`catchup`、`fetch`、`reprocess`）後，那天 22:50 稽核寫的 `etl_runs` error 不會自己清：`check_quality` 只重寫五項品質檢查，`catchup` 指令也不重審。要等下一次 15:10 的 `prepare_day` 在回看期（`ETL_CATCHUP_LOOKBACK_DAYS`）內重審；超出回看期或等不及，就手動重審那天：

  ```bash
  uv run python manage.py shell -c "from datetime import date; from apps.etl.tasks import _reaudit; _reaudit(date(2026, 10, 28))"
  ```

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
- 規則（plan §8 K14）：財報的公告日就是實際上傳日，早於或晚於法定截止日都照實際。有更正紀錄的同樣照頁面上的上傳時間：它可能是更正後重傳的時間，FinMind 存的也可能是更正後的數字，拿截止日當公告日會偷看。晚於保守值（截止日的下一個交易日）的每檔記一列 `data_quality_log`（都是 info）：沒更正的是 `statement_filed_late`，有更正的是 `statement_filed_later`。同一份報表列了好幾次上傳、又沒有重編時，取最早的一次；目前只有 2534 2019-Q4、6496 2014-Q4、6625 2019-Q4 的個體報表同時列了原始與更正上傳，`financial_statement` 這三期只有合併列，不影響公告日。未查證：有些更正可能只發補正公告、沒有重傳（例如 5483 2024-Q1～Q3 的更正列是一般申報的時間），這種期別頁面上的時間早於更正，電子書清單分辨不出來。
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
- 只填空的：證交所、櫃買的除權息預告表（plan 2.7b）上線後是主要來源，FinMind 只補它沒給的；已經有日期的列不改。FinMind 回補時一併填；現有的列用 raw 重套，不用重抓：

```bash
uv run python manage.py reapply_dividend_dates
uv run python manage.py reapply_dividend_dates --stock 2884 --stock 2885
```

- 第二行只重套指定的股票，例如修過某檔的 raw 之後。輸出 `N events dated across M stocks`；被拒收的股票逐行寫在 stderr（`rejected <代號>: <原因>`），例如 FinMind 少了 `AnnouncementDate` 欄位，或公告日落在交易日曆之外。拒收只影響那一檔。
- 重套會把同一檔存過的每一份 FinMind 回應合起來，同一個除權息日取最早的公告。所以清空再重套只能補回同樣的日期，或採用新回應裡更早的日期；FinMind 事後把公告日改晚，這樣是改不過來的，要人工更新那一列。要重填某檔（例如修過 raw 之後），先把那檔清空再重套；2.7b 上線後清掉的也包含預告表給的日期，要一起重跑預告表：

  ```bash
  uv run python manage.py shell -v 0 -c "from apps.market.models import DividendEvent as D; print(D.objects.filter(stock_id='2884').update(announce_date=None))"
  uv run python manage.py reapply_dividend_dates --stock 2884
  ```

- 減資、分割這類只有參考價、沒有配息公告的事件，FinMind 沒有日期，等 2.7b 的預告表。

## 每日除權息、減資、面額變更（plan 2.7）

`dividend_event` 的每日來源是證交所、櫃買，每個交易日 15:30 的 prices 批各跑一次；歷史仍是 FinMind 回補。

| 市場 | 除權息 | 減資 | 面額變更 |
| --- | --- | --- | --- |
| 證交所 | TWT49U 除權除息計算結果；`權/息` 不是「息」的列另查 TWT49UDetail | TWTAUU | TWTB8U |
| 櫃買 | exDailyQ（逐日查，區間查詢最多回 100 筆） | revivt | pvChgRslt |

- exDailyQ 一次最多回 100 筆，回滿 100 筆就整頁拒收。歷史上櫃買普通股單日最多 54 筆（2026-07-16）；哪天真的超過，那天的 `dividend_event` 會一直失敗、擋住研究閘門，要手動處理。

- ETF 的分割、反分割（證交所 TWTCAU，ETF 分割（反分割）恢復買賣參考價格表）不抓：ETF 不在研究母體，研究層也不讀 ETF 的價格；歷史上 FinMind 回補的 ETF 分割照舊留著。要接回來時放進 2.7b。

- 交易日 D 的一次執行只抓除權息（恢復買賣）日是 D 的列：結果表在 D 開盤前就公布（開盤競價基準要用），15:30 一定查得到，和 D 的行情同一批進來。不往後抓下一個交易日：先寫下的列不會被之後的執行更正，休市順延時會留下錯的那天。
- 因子：除權息 = 除權息參考價 ÷ 除權息前收盤價，和 FinMind 回補用的 `after_price` 同一個數；減資、面額變更 = 恢復買賣參考價 ÷ 停止買賣前收盤價。同一檔同一天有兩種以上時相乘。寫入或補上因子的股票會重算 `daily_price.adj_factor`；改到既有歷史時，研究排程會從受影響的交易日重建（見「研究排程」）。
- 金額：現金股利（元／股）；股票股利換成元／股（每仟股無償配股 ÷ 100，面額 10 元），和 FinMind 的單位相同。減資、面額變更的列金額是 0。退還股款的減資（例如 2026-10-12 的 2323，恢復買賣參考價 10.88、停止買賣前收盤 11.00）金額也存 0：每股退多少只在 TWTAUU 的詳細資料頁，這支不讀；因子照參考價算，還原價不受影響。
- 已有的列（FinMind 回補過的）只補空的 `adj_factor`；金額或因子不同時不覆蓋，記一列 `dividend_source_mismatch`（info，每個執行日、每個來源一列，重跑時取代）。新列的 `announce_date` 留空：結果表不是公告，公告日交給 FinMind 重套與之後的預告表（2.7b）。
- raw 各頁分開存：`ex_rights`、`ex_rights_detail`、`capital_reduction`、`par_value_change`。`reprocess --dataset dividend_event` 會照網址找回各自的頁，但和每日執行一樣只補空的因子，不改已存的列。要用改過的解析重套某段日期：先刪掉那段 `ex_date` 的列（2026-10-31 之後的都是交易所寫的，之前的是 FinMind 回補，不要刪），`reprocess` 那段日期，再 `reapply_dividend_dates` 補回 FinMind 的公告日；刪掉後沒有重寫回來的股票，用 `recompute_adj_factor --stock <代號>` 重算。
- `dividend_event` 擋研究閘門：除權息日當天的技術面報酬要用到當天的因子。閘門只認證交所、櫃買的 run（FinMind 回補不算），所以上線前的交易日不再通過 `ready_for_research`；要重算某段過去的預測，先用下面的 `fetch` 補那段的 `dividend_event`。
- 同一檔同一天有好幾張表時，因子相乘；其中一張的價格不能用、因子留空時，那天的因子整個留空，不寫只乘了一部分的數。
- 證交所在 200 回應裡回「系統忙碌中」時，client 原地退避重試，那份回應不存 raw；每個證交所端點都一樣。
- 上線時補 FinMind 回補之後的缺口（回補到 2026-10-30，catchup 只回看 14 天）：

  ```bash
  uv run python manage.py fetch --source twse --dataset dividend_event --start 2026-10-26 --end <今天>
  uv run python manage.py fetch --source tpex --dataset dividend_event --start 2026-10-26 --end <今天>
  ```

- 還沒做（plan 2.7b）：用預告表定 `announce_date`。在那之前，2026-10-30 之後新公告的除息日 `announce_date` 是空的，`fund_days_to_ex_date` 看不到。

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

- 一次執行逐個日曆年處理，每年記一次 checkpoint。區間裡沒有 2013 年以後的期別就拒絕；只要有每日掃描還在看的期別（截止日加上 `REVENUE_SCAN_DAYS_AFTER`／`STATEMENT_SCAN_DAYS_AFTER` 還沒過），整次就拒絕；區間裡有每日掃描接手之後的期別（第一次成功的 `revenue_scan`／`statement_scan` 那天還在看的最早一期起），也整次拒絕。那些期別的公告日由每日掃描用第一次看到的 session 決定；過了申報期還沒有列的，是晚交或沒交的公司，用保守公告日補會比實際申報日早。月營收從 2013-01 起，更早的頁面是另一種版面；財報從 2013-Q1（IFRS）起，這一季不抓前一季，所以 ROE 是空的。
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

- 兩種頁面都是 MOPS 現在的狀態：彙總表只列現在還公開發行的公司，2013 年起的月營收頁也是 2026-11-04 重新產生的，所以下市公司不會出現在裡面。下市公司的財報要另外用 XBRL 檢視頁補（K17b），月營收用新站的單一公司查詢補（見「缺的月營收」，2014-01 起）。
- 每年約 48 個月營收請求、24 到 30 個財報請求，6 秒一次。避開 19:00 和 22:40 前後，那兩個時段 worker 的每日掃描也在打 mopsov。

### 下市公司的財報（plan K17b）

彙總表不列下市公司，改用 XBRL 檢視頁（t164sb01），一家公司一季一頁，下市公司也查得到：

```bash
uv run python manage.py shell -v 0 -c "from apps.market.models import StockMaster as M; print('\n'.join(M.objects.filter(delist_date__gte='2013-04-01', stock_id__regex=r'^[1-9][0-9]{3}$').exclude(stock_id__startswith='91').order_by('stock_id').values_list('stock_id', flat=True)), end='')" > delisted.txt
test -s delisted.txt && sed 's/^/--stock /' delisted.txt | xargs uv run python manage.py backfill --source mops --dataset statement_viewer --from 2013-01-01
```

- 只看 2013-Q1 起、在下市日之前結束、每日掃描已經不看、而且在每日財報掃描接手之前的季別。先問合併報表（`REPORT_ID=C`），沒有就問個別報表（`A`）；同一家公司上一季回答的種類下一季先問。
- 每一季都抓（Q1 也抓前一年 Q4，ROE 才算得出來）：standard 表沒有的季別寫入，已存的季別拿來對照、不改，之後由 K18 的 `overwrite_statements` 從 raw 改成 MOPS 的數字。單季值和彙總表用同一套換算（`statement_rows`）。前一季沒申報的，MOPS 只換算得出負債比和 Q1 除 ROE 以外的各項；其他欄位換算不出來，不比，也不記進 `statement_viewer_mismatch`。
- 合併或個體看損益表有沒有歸屬母公司淨利。金控的格式把它寫成「淨利（淨損）歸屬」底下一列「母公司業主」（2888、5820，還在上市的 2881 也是），只認這個標題底下的那一列；「綜合損益總額歸屬」底下同名的那列是綜合損益，不算。2026-11-09 之前沒認得這個寫法，2888、5820 共 87 季被當成個體，記在 `types`，K18 沒套到。修正後先重跑這兩檔的檢視頁回補（`--stock 2888 --stock 5820`，約 90 個請求，`types` 換成 `rows` 的對照），再用 `overwrite_statements --stock 2888 --stock 5820` 從 raw 重套；補上的 roe、debt_ratio 空格是 Phase 3 遮罩月份的來源之一，套完要由 Phase 3 重建財報特徵、重新判定遮罩。
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

### 缺的月營收（單一公司查詢，plan K17）

彙總頁（t21sc03）不列下市公司；舊站的單一公司查詢（mopsov `ajax_t05st10_ifrs`）對下市公司只回「公開發行公司不繼續公開發行」。新站的 JSON API（`POST {MOPS_API_BASE_URL}/mops/api/t05st10_ifrs`）一家公司一個月一個請求，下市公司也查得到：

```bash
uv run python manage.py backfill --source mops --dataset revenue_company --from 2014-01-01
```

- 走過 `--from` 之後還在上市的每一檔，只問普通股（不含 91 開頭與存託憑證）上市期間 `monthly_revenue` 沒有的月份：從上市月（沒有 `list_date` 就用第一個收盤日的月份）到下市月，每日掃描已經不看，而且在每日月營收掃描接手之前（第一次成功的 `revenue_scan` 那天還在看的最早月份起都不問；twp 是 2026-11-05，所以到 2026-09）。接手之後過了申報期還沒有列的，是晚交或沒交的公司，用保守公告日補會比實際申報日早。沒有缺的股票不發請求。
- 從 2014-01 起，`--from` 再早也不問：新站 2013 年的月份回「查詢成功」但沒有 data（2026-11-08 測 5349、6238、2847 與 2330 的 2013 月份都一樣）。
- 每 6 秒一個請求（MOPS 的速率上限），避開交易日 19:00 和 22:40 前後的每日掃描。2026-11-08 在 twp 試算：走過 2,579 檔，159 檔有缺、4,177 個請求，約 7 小時。
- 數字取新台幣欄（KY 公司另有功能性貨幣欄），單位千元，乘 1,000 存成元；2026-11-08 抽 5347、4137（KY）、2823（下市）和 FinMind 已有的列比，逐位相同。yoy 用回應裡的去年同期；回應沒有上月，mom 只在前一個月也是這次查到的時候算，不拿 FinMind 的列當分母，否則留空（研究層的特徵只讀 `revenue`）。只寫 standard 表沒有的列，FinMind 已有的列不改；公告日用保守值 `revenue_visible_from`，和 FinMind 回補相同。
- raw 存在 `raw_mops`，dataset `revenue_company`；回應裡的查詢時間（`datetime`）不算，同樣的回答只存一份。
- log 和下市公司的財報相同：`revenue_company_mismatch` 每檔一列（`detail.stock_id`），區間結尾是最後一個可以問的月份的月底；下市股，以及每日掃描接手之後的上市股，這個月份不再移動，所以重跑到哪一天都是同一列。`added` 是補進來的月份；`absent` 是查了但沒有數字的月份（回「查詢成功」但沒有 data，或 code 406「查無相符資料」），例如下市前最後一個月（申報期在下市之後）或下市前停止申報；`not_public` 是 MOPS 回 code 500「公開發行公司不繼續公開發行」或「上市公司已下市」的月份，也就是那時公司已不再公開發行或已下市，只接受下市當月；更早的月份也這樣回，代表 MOPS 拒絕整家公司，那一檔記成 rejected。兩種都不寫。跑完看一下夾在兩個已有列中間的 `absent`、`not_public` 月份：MOPS 若整批改成這兩種回答，run 仍會 success，只會在這裡看出來。只問沒有列的月份，所以不會有 `rows`、`types`。
- 一個月的回答讀不懂（不是問的月份、沒有新台幣欄、本月沒有數字，或 code 不是 200，也不是 406「查無相符資料」或上面兩種 500），整檔不寫、記成 rejected，和檢視頁相同；raw 已存，但沒有從 raw 重套的指令，修好之後用 `--retry-rejected` 重問。
- 連續被拒的守門（`ETL_MAX_CONSECUTIVE_REJECTIONS`）在這裡幾乎不會觸發：沒有缺的股票不發請求、算成功，會把連續次數歸零。來源擋住時，HTTP 429 或 5xx 會讓整次停下，用 `--resume` 接著跑；回 200 但 code 不是 200（上面的 406 和兩種 500 除外）的，只會一檔一檔變成 rejected，跑完是 failed，用 `--retry-rejected` 重問。
- 中斷後用 `--resume` 從停下的股票之後接著跑；整個重跑會再問一次還缺的月份，包含上次 `absent`、`not_public` 的。

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

- 每日 15:30 的 prices 批抓一次，一個月一個請求；不擋研究閘門。品質檢查的指數新鮮度（`index_freshness`）含它，門檻和 `TAIEX` 一樣是當天要有收盤，沒有就記 warning，`detail.stale` 列出 `TAIEX_TR`。當天沒抓到時，22:50 的 `etl_runs` 檢查記 warning 並告警；隔天 15:10 的 catchup 補上後，會對那天重跑品質檢查，把那天的告警再送一次。
- 相對大盤版的機率模型在監控時要用預測日與到期日的 `TAIEX_TR` 收盤；到期日 D 沒抓到時，隔天 06:00 的監控把這個 run 記進 `monitor_unscored`（頁面「到期預測缺 label」），catchup 不會自動補評分。等 15:10 的 catchup 補上 D 的收盤後，手動跑 `monitor --date D`。
- 回補從 2010 年起，一個月一個請求，約 203 個、12 分鐘。先跑 `migrate`（`market.0003`），回補要在 worker 換成新程式後的第一次 15:10 之前做完；否則 `prepare_day` 的 catchup 會一天一個請求補回看期內的日子，並對那些日子重跑品質檢查、把舊告警再送一次：

  ```bash
  uv run python manage.py migrate market
  uv run python manage.py fetch --source twse --dataset return_index --start 2010-01-04 --end $(date -v-1d +%F)
  ```

- 中途失敗時，看那個 run 的 `etl_run.checkpoint.last_date`，從下一個交易日接著跑。

## 交易所逐日表補缺（plan K17c）

FinMind 的外資持股、流通股數、本益比在 2010–2020 有缺，下市股缺得特別多（約 21% 對現存股 0.4–0.7%），模型會把「缺值」學成「之後會下市」。融資券、借券賣出餘額在 2010–2026 都有缺（`chip_sbl_short_chg_5d` 在 57 檔下市股 100% 缺，現存股 1.96%）；融資券抽查 2017 年，下市股整段沒有融資資料的 2.3%、在有資料的期間內漏掉某幾天的 3.6%（現存股 0.27%、0%）。交易所的逐日全市場表列出當天所有掛牌的股票，用它把缺的補上。範圍是 2010 年起到每日排程接手的前一天（2026-10-21）：只補到 2020 年的話，2021 年後才有第一筆股數、或完全沒有股數的股票，會把 2020 年最後一個股數一路帶到下市或下一筆。

兩個市場是不同主機，各開一個終端機平行跑；同一個主機一次只跑一個指令（每個程序各有一份限速，兩個同時打同一主機就會快過 3 秒一次）：

```bash
for fill in holdings valuation margin sbl; do
  uv run python manage.py fill_from_exchange --source twse --fill $fill --start 2010-01-01 --end 2026-10-21
done
```

櫃買把 `--source` 換成 `tpex`，`sbl` 的 `--start` 改成 2012-10-02（櫃買借券表從那天才有資料）。中斷了用 `--resume <run_id>` 接著跑。某一天固定過不了檢查（`SchemaChangedError`，例如櫃買借券表有 4 列以上餘額對不上；raw 會先存，可以查）或固定回沒有資料（`SourceNotReadyError`，不存 raw）時，`--resume` 會一直停在那天。確定是來源本身的問題，就把那個 run 的 checkpoint 推到那一天再續跑，不要另開一個 run（另開的 run 從空的統計開始，卡住之前那段的 `filled`、`differences` 不會寫進 log，holdings 也不會刪掉前一段的重複股數）：

```bash
uv run python manage.py shell -c "from apps.etl.models import EtlRun; r = EtlRun.objects.get(run_id=<id>); r.checkpoint = {**r.checkpoint, 'last_date': '<卡住的那天>'}; r.save(update_fields=['checkpoint'])"
uv run python manage.py fill_from_exchange --source <twse|tpex> --fill <fill> --resume <id>
```

跳過的那天留成缺值，交給 SPEC §6 處理步驟 6 的殘餘缺口規則。`margin`、`sbl` 跳過的那天仍在 run 的 checkpoint 範圍內，Phase 3 的 `covered_days` 會把它算成交易所表已寫入，那天沒有列的股票會被當成交易所沒列它（PIT 可觀察），所以跳過前把那天和原因記下來交給 Phase 3，當成來源缺處理。櫃買借券表每列都檢查融券、借券賣出的前日餘額加減當日變動等於當日餘額；一頁最多 3 列對不上時照頁面的數字存、記 warning（例如 2013-07-31 的 6177：借券前日 124,000、當日沒有賣出或還券、當日餘額 0，FinMind 也存 0；2026-11-07 前抓過的 213 頁只有這一列），4 列以上才當成版面變更。每日抓取用的是同一個檢查：1 到 3 列對不上照存，研究閘門照常放行，只在 worker 的 log 留 warning（不寫 `data_quality_log`）；4 列以上那天的 tpex/sbl 失敗，閘門擋下當天。櫃買借券表沒有單位檢查（證交所 TWT93U 看「單位：股」、融資券表看「張」）：`sbl` 開跑幾分鐘後看一次 `etl_run.checkpoint->'differences'->'sbl'->'sample'`，已存值和交易所值差 1,000 倍就停下來查。

- `holdings`：證交所 MI_QFIIS、櫃買 insti/qfii，一天一個請求，同一頁補外資持股比率（`daily_chip.fi_holding_pct`）和發行股數（`shares_outstanding`）。`valuation`：證交所 BWIBBU_d、櫃買 peQryDate，補 `daily_valuation`。`margin`：證交所 MI_MARGN、櫃買 margin/balance，補 `daily_chip` 的融資餘額、融資限額、融券餘額（`margin_balance`、`margin_limit`、`short_balance`，張換成股）。頁面上有的股票才補，不在頁面上的（例如還沒有信用交易資格）不補；停止信用交易的股票有的照樣列在頁面上、數字是 0（例如證交所註記 `OX`），照存 0，和每日抓取相同。`sbl`：證交所 TWT93U、櫃買 margin/sbl，補 `daily_chip.sbl_short_balance`（借券賣出當日餘額，單位股），規則和 `margin` 相同。櫃買借券表 2012-10-02 才開始有資料（2026-11-07 從 2010-01-04 起二分搜尋找到的第一天有列的日子；空頁不存 raw，所以資料庫裡查不到之前的空回應），櫃買 `sbl` 從 2012-10-02 起跑。FinMind 在 2010-01-04～2012-10-01 有上櫃股的借券餘額，這段缺的沒有交易所來源，也沒有補缺的 run 涵蓋，留給 SPEC §6 處理步驟 6 當成來源缺。證交所 TWT93U 自 2012-03-19 起改用新公式（頁面註明「自101年3月19日起借券賣出餘額採用新公式計算及資訊揭露」），之前的頁面借券賣出只有前日餘額、賣出、庫存異動、今日餘額、可使用額度五欄；adapter 對 2012-03-19 以前的日子把「今日餘額」當成當日餘額讀，2012-03-19 起再出現舊欄名照樣當成版面變更。FinMind 存的同期值也是舊公式（2012-03-16、2012-03-19 兩天各 790 檔和頁面全同），但新舊公式前後不接續：2012-03-19 頁面的前日餘額和 2012-03-16 的今日餘額，805 檔裡有 212 檔不同。
- 缺才補，不分股票之後有沒有下市：
  - `daily_chip`：沒有這一列就新增（只有這次補的欄位）；有這一列時，每一欄分開看，空的才補；有值的不動。
  - `daily_valuation`：當天沒有這一列才新增，三個數字都空的不新增。
  - `shares_outstanding`：只補每檔第一筆已存股數之前的日子，照「變動才存」寫變動點；跑完時，補的最後一筆和第一筆已存股數相同的，刪掉那筆重複的已存列（生效中的股數不變），刪掉的列記在 log 的 `dropped`（`[代號, 日期, 股數]`），可以照著寫回。界線取第一次 `holdings` 補缺開跑時的第一筆已存股數（記在那個 `etl_run.checkpoint.windows`），之後的補缺、`--resume`、另一個市場都沿用，所以補進去的列不會被當成已存的界線。
- 只寫當天在 `daily_price` 有價的代號（不用歷史行情頁，K6）；當天交易所表上的代號一個都沒有價時 SourceNotReady，不會記成成功 0 列。
- 已有的值和交易所差到公布的最後一位（外資持股比、本益比、淨值比 0.01，殖利率 0.0001；融資券兩邊都是張乘 1,000、借券賣出餘額兩邊都是股，要完全相同）、或界線那天的股數不同時，不改，記在 log 的 `differences`：每種表的筆數、分下市股和現存股的筆數，和前 20 筆 `[代號, 日期, 欄位, 已存值, 交易所值]`（SPEC §8.5）。
- `etl_run` 的 dataset 是 `fill:holdings`／`fill:valuation`／`fill:margin`／`fill:sbl`，每天和寫入同一個交易記 checkpoint。跑完記一列 `exchange_fill`（info）：`filled` 是每檔補了哪段日期、幾天，`created` 是其中新增整列的部分，`totals` 分下市股和現存股的列數，`span` 是第一次開跑的起點到結尾。
- 先跑完 FinMind 的 `daily_price`、`shares_outstanding`、`daily_chip`、`daily_valuation` 回補（含 `--retry-rejected`）再跑補缺：`daily_price` 決定補哪些代號、`shares_outstanding` 決定股數的界線（第一次 `holdings` 補缺時就固定）；FinMind 寫 `daily_chip`、`daily_valuation` 時整列已存在就跳過，補缺新增的列會擋掉之後的 FinMind 回補。之後真的要再回補 FinMind，先刪掉補缺新增的列：`daily_chip` 是 `fi_holding_pct`、三個融資券欄和 `sbl_short_balance` 以外都是空的列（幾種補缺可能落在同一列：holdings 新增的列，margin 會當成已存的列補上融資券，所以不能只看其中一次的 `created`；這樣刪也會刪到 FinMind 只有外資持股比、只有融資券、只有借券或只有其中幾種的列，回補 FinMind 時會寫回來）。FinMind 回補完，對同一段日期再跑一次 `holdings`、`margin`、`sbl` 補缺，把補缺填過、FinMind 沒有的格子補回去；`daily_valuation` 的 `created` 只記每檔的日期範圍，範圍內可能夾著 FinMind 的列，要逐日分辨時從兩邊的 raw 重算。
- 補缺只替當天有價的代號寫列。之後回補某段日期的 `daily_price`，要對同一段日期重跑 `--fill margin` 和 `--fill sbl`（兩者都冪等）：否則新有價的那些股票日，交易所表有抓到、資料庫卻沒有列，會被 SPEC §6 處理步驟 6 算成 PIT 可觀察的缺，蓋掉真正的來源缺。
- 每個請求約 3.5 秒；一個市場一個 fill 2010-01-04 到 2026-10-21 約 4,130 天、約 4 小時，四個 fill 約 16 小時。可以跑的時段是平日 02:00–15:00 和週末，和每日 ETL（含 02:00 前的重試）錯開；平日 15:00 前沒跑完就停，之後 `--resume`。

## FinMind 外資持股 0 的覆寫（plan K21）

FinMind 的外資持股比有一部分存成 0，但交易所那天的頁面不是 0。依 0 連續的長度與頁面涵蓋分四類，只覆寫前兩類：

| 類別 | 條件 | 處理 |
| --- | --- | --- |
| 整段都錯 | 同一檔連續 20 天以上的 0，90% 以上的日子交易所頁面有這檔，其中 90% 以上不是 0 | 交易所不是 0 的那幾天改成交易所的值；交易所也是 0 的日子留 0 |
| 零星短段 | 連續不到 20 天的 0，交易所那天不是 0 | 改成交易所的值 |
| 長段混著幾天 | 連續 20 天以上的 0，頁面夠，但交易所不是 0 的日子不到 90% | 不動：改了會在一長段 0 裡做出 0 → 小值 → 0 的跳動 |
| 頁面不夠 | 連續 20 天以上的 0，交易所頁面有這檔的日子不到 90%（不看比例，先判這類） | 不動：只改有頁面的日子同樣會做出跳動；補齊頁面後重跑 |

- 只看普通股（`^[1-9][0-9]{3}$`，不含 91 開頭與產業是存託憑證的代號，和研究母體相同）；交易所的值讀已經存在 raw 的 `foreign_holding` 頁（證交所 MI_QFIIS、櫃買 insti/qfii，不發請求），同一天有好幾份時用最新的。
- 每覆寫一列記一列 `fi_holding_zero_replaced`（info，`date` 是那一天，`detail` 有 `stock_id`、`kind`、`old`、`new`、`source`、`raw_id`）；FinMind 和交易所的 raw 都不動。重跑只會碰到還是 0 的列，同一檔同一天不會記第二列 log。
- 要等兩個市場的持股補缺（plan K17c）都跑完、交易所的頁面都在 raw 裡再跑；先看 dry run 的數字：

  ```bash
  uv run python manage.py replace_zero_holdings --dry-run
  uv run python manage.py replace_zero_holdings
  ```

- 輸出的 `replaced` 是每個市場、每類的檔數與列數。`left` 是不動的列：`mixed_run`、`thin_run` 是兩類長段裡交易所不是 0 的檔數與列數；`unchecked` 是沒有交易所值的 0，分成 `no_page`（那天至少一個市場沒有頁面）、`unreadable`（那天至少一個市場的頁面讀不了，另記 warning）、`not_listed`（兩個市場的頁面都在、都沒列這檔）；原因是按整天判的，補缺跑到一半時兩個市場的數字會混在一起。補缺跑完後 `no_page`、`thin_run` 應該接近 0；不是的話先查補缺。
- 覆寫後外資持股類特徵的歷史值會變，要重算：`uv run python manage.py build_features --start 2010-01-01 --end <最新價日>`。

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

## 市場歷史（market_history）

`market_history` 記錄每檔股票每段時間在哪個市場，讓研究層用 t 日的市場，而不是 `stock_master.market` 的現值（SPEC §8.4；panel 改讀這張表的是 Phase 3 的 PR）。2010-01-04 到 2026-11-06 有 54 檔換過市場：52 檔上櫃轉上市、2 檔上市轉上櫃。用現值算的話，這些股票在上櫃時期的 beta 和市場類特徵會拿上市的資料來算。第一次建表：

```bash
uv run python manage.py migrate market
uv run python manage.py rebuild_market_history
```

- 資料來源只有已存的外資持股 raw 頁（證交所 MI_QFIIS、櫃買 insti/qfii），不發請求。兩個市場每天各一頁，通常列出所有上市櫃股票（例外見最後一條）；同一天有好幾頁時，取最晚抓的那頁。只收 `stock_master` 裡有的代號（不在 stock_master 的代號沒有段，2026-11-07 的 twp 是 4207、5355、8176 三檔、共 9 列 `daily_price`）。讀不了的頁跳過、記 warning，最後印出來。
- 重寫的範圍：頁上定得出市場的代號才重寫。兩個交易所的頁都在的日子以頁面為準；頁面看不到的日子（只有一邊有頁，或兩邊都沒有）上已存的界線（每日報價寫的）保留。頁上從沒出現過的代號（只有每日報價寫的段），以及每天都兩邊都列、定不出市場的代號，都不動。重跑結果相同。
  - 重寫前後的段不同時，記一列 `market_history_rebuilt`（info）：`changed` 是市場或界線不同的代號數，`sample` 是其中前 20 檔的 `[代號, 原本的段, 重建的段]`；只差第一段起日的另計 `start_shifted`（新股的頁面常比報價早一天列出，重建會把起日提早一天）。
- 什麼時候要重跑：外資持股頁回補過（例如 K21 重抓）之後。只有一個交易所有頁的日子不拿來判市場（換市場那天要看兩邊才分得出來），所以當天只抓到一邊（18:30 櫃買、21:30 證交所）時跑，那天保留報價寫的段；等兩邊都抓完再跑一次，就會以頁面為準。
- 區間的規則：
  - 一檔一段一列，start_date 到 end_date 頭尾都算在內。
  - 前一段的 end_date 是下一段 start_date 的前一個日曆日。
  - 最後一段的 end_date 是空的，代表到目前為止。
  - 第一段從第一個兩個交易所的頁都在、且列出這檔的日子起算；頁面看不到的日子上有更早的報價起點時，用那個起點。更早的日子表裡查不到；研究層把查不到的股票日排除在 universe 外（協調 2026-11-08，Phase 3 實作），不退回 `stock_master.market`。
  - 某天只要有一邊沒有頁，頁面就不拿那天判市場（市場照前一天，除非報價在那天寫了界線）；每個市場缺頁的交易日印在 `trading days without a page`。
- 轉換日：交易所在股票第一個交易日的前一天，就會把它列在外資持股頁上（新上市的股票也一樣），所以上櫃轉上市時會有一天兩邊的頁都列著它（上市轉上櫃的 2341、6423 沒有這一天）。
  - 這一天算舊市場。新市場從之後第一個只有新市場列出的日子起算，通常就是隔天。第一次出現就兩邊都列的，取之後第一個只有一邊列出的市場，起點仍是第一次出現那天。
  - 依據：2026-11-07 用融資券頁核對這 52 個轉換日。49 天的情形是當天還在櫃買的融資頁、不在證交所的，隔天才出現在證交所；另外 3 天沒有反例。
  - `stock_master.list_date` 存的是原本掛牌的日子，不能拿來當界線。
- 每日 16:00 的 `refresh_stock_master` 用報價接著寫：
  - 先補上次成功的 refresh 到今天之間漏掉的交易日（15:10 的補抓會把那幾天的報價 raw 補回來），再寫今天。
  - 代號在另一個市場報價時，把舊段的 end_date 設成前一天，從那天開新的一段。表裡沒有的代號，從那天開一段。
  - 開放段的 start_date 不早於那天時不改，所以重跑同一天結果不變。
  - 漏掉的交易日沒有報價 raw、或讀不了時，跳過那天、照常寫今天，那天記在 `stock_master` 那列的 `unquoted`，並升為 warning。那幾天換市場的界線會晚；等那幾天兩個交易所的外資持股頁都在，重跑 `rebuild_market_history` 就會改正。
- 指令印出：段數和股票數、讀到的頁數、換過市場的代號、兩邊都列的股票日數、讀不了的頁、定不出市場的代號、每個市場缺頁的交易日、和原本不同的代號。
- 研究層的 panel 讀這張表（Phase 3 的 PR）：每列的市場決定市場指數（`risk_beta_250d`、`risk_idio_vol_60d`）、當沖表與借券斷點，t 日沒有區間的股票日不在 universe。panel 開始讀這張表的那次部署（Phase 3 的 PR 合併後）、第一次建表、或重建的 `changed` 或 `start_shifted` 不是 0 時（第一段的起日移動，universe 就會多或少那幾個股票日），這些特徵、`in_universe` 與 label 的歷史值都會變，要重算：`uv run python manage.py build_features --start 2010-01-01 --end <最新價日>`，再跑同範圍的 `build_labels`（每晚的 chain 只重算最近 60 個交易日的特徵、81 個交易日的 label，舊的列不會自己更新）。
- 2026-11-07 在 twp 試算：兩個市場各 4,122 頁，0 頁讀不了，約 1 分鐘、峰值記憶體約 2.5 GB；結果是 2,484 檔、2,538 段，換過市場的 54 檔，兩邊都列的股票日 52 個，沒有定不出市場的代號。
  - 2026-10-22、10-27～10-30 這 5 個交易日，兩個市場都沒有外資持股頁。
  - `stock_master` 裡的普通股在 2010 年後每個有價的股票日都有對應的段（不在 stock_master 的 3 檔見第一條）。13 檔共 2,297 列 `daily_price`（收盤價非空的 2,014 天）早於該檔的第一段：其中 6 檔是 2026-10 下旬新上市、第一段從 11-02 起算，原因是上面那 5 天沒有頁；最多的是 6794，2020-12-25～2024-05-17 有 822 天不在外資持股頁上。

## 模型監控與換模

- `monitor --date D` 判的是 D 當天服務中的 run（D 在該 horizon 有 `prediction` 列的 run，依 API 的順序）。slot 當天沒有 `prediction` 列就不判，停機或 predict 失敗的那幾天監控會暫停；補跑時先補 `predict --date D`，再補 `monitor --date D`。
- `model_alert` 每筆帶 `mode`：`judged` 會把 active 的 run 降級（`reason` 是 `missing_reference` 時只告警）；`grace`（上線日之前補寫的日子，以及換模後 5 個交易日）與 `reset`（`restore_run` 之後、該 slot（目標 × horizon）的窗口還沒累積滿 60 列，之後換上的 run 沿用；h20 約要 80 個交易日、h5 約 65 個）只告警。被判 run 自己的恢復日晚於告警日時（恢復後補跑之前的日子），`reset_rows` 是 0。`mostly_predecessors` 為 true 表示窗口多為前任 run 的列，人工恢復前先看這個。
- `monitor`、`predict`、`promote`、`restore_run` 共用研究鎖，一次只跑一個（見「研究排程」）；手動指令拿不到鎖會報錯並寫出持有者，等它跑完再重跑。
- 換模日落在交易日曆最後 4 個交易日內時，寬限結束日算不出來（log 會有 warning）；12 月換模前先照「交易日曆延伸」延伸日曆。
- 換模：`promote --run-key K`，門檻沒過時舊模型續用並寫 `model_promotion` 告警；要強制換模用 `--force --reason "<原因>"`。門檻判斷後正常結束的 promote（換上或被擋）都寫一列 `model_promotion`；在那之前或之後任何一步報錯（例如非交易日、claim 不符、寫預測失敗），整筆回復、不寫。`detail.scope` 是 rank／return／interval（一次 promote 只處理一個 scope），`detail.promoted` 表示換了沒有：門檻全過是 info（不告警），沒過與強制換模是 warning。同一個 scope 依 id（寫入順序）最新的那一列，就是最後一次換模嘗試的結果；`promoted` 是 false 時服務中的仍是前一個 build，restore 與降級不寫這一列。`data_quality` 門檻的 reason 只寫摘要（幾天有 error、最近一天是哪些檢查），完整清單在 N 的 `params["gates"]` 那一項的 `errors`（新到舊，最多 200 筆）與 `errors_total`。晚幾天才換模時，`--date` 預設會取 slot 已有列的最後一天。換上之後，promote 會重算那天的 `output_width_ratio` 與 `output_odds_vs_q50`（換模重寫了那天的預測），寬度比超標照樣告警，機率矛盾只記 info、不告警；重算失敗只寫 error log，換模不回復，等下一次 `monitor --date D` 再算。
- 恢復：`restore_run --run-id R --date D --reason "<原因>"`，只接受 degraded 的 run，D 不早於它最後一個預測日、不晚於今天。

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
