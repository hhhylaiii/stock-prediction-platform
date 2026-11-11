# 台股預測平台 開發規格書

> 本檔是規格的唯一來源。最初匯出自 claude.ai 規格書《台股預測平台 開發規格書》rev 112（2026-11-01 匯出），原文：https://claude.ai/artifact/Nw8QAWMeFz5mB4SNVq8Twp ，保留為歷史版本，之後不再回寫。圖未保留，以表格與文字為準。

Oct 29, 2026 · @Owen Lai

## 1. 專案目標與範圍

本平台的目標：每個交易日收盤後，對全部台灣上市、上櫃普通股產出三種預測：未來 N 日的報酬排名、預測報酬（展示層換算成各股的預測價格）、報酬區間與上漲機率，並能用歷史資料回測驗證。第一版定位為個人研究工具，不對外提供服務。

**成功標準**

- 資料層：本地資料庫完整保存 2010 年起的日價量、籌碼、月營收、財報（下市公司 2013 年的月營收查不到，見第 3 節），股權分散表從開始接集保起每週累積、不回補歷史，每日自動更新，缺漏可被偵測並告警。
- 模型層：Ridge 迴歸為 baseline，LightGBM 多 seed bagging 為主模型，XGBoost 為可選對照；全部以 walk-forward 驗證（2014 到 2024 為研究期，2025 起為只跑一次的最終 holdout），排名模型的 rank IC 必須統計顯著且優於 baseline，絕對報酬模型的 MAE 必須優於「永遠預測 0」，區間模型的 pinball loss 必須優於波動率區間 baseline、實際覆蓋率接近宣稱的 80%。
- 服務層：Django 提供 API 與頁面，可查詢任一股票的最新預測排名、預測價格與價格區間、上漲機率、所用特徵、歷史預測準確度；訓練 universe 外的股票照常回傳預測並標示 in\_universe = 0。

**預測目標（label）**

- 相對目標（排名）：未來 5 與 20 個交易日的還原報酬率，轉成每日橫截面百分位排名；評估用 rank IC。
- 絕對目標（報酬）：5 日直接回歸未來 5 個交易日的還原報酬率，20 日取分位數模型的中位數 q50，不另訓模型。模型層只輸出報酬，展示層再依每檔股票自己的最新收盤價換算：預測價 = 收盤價 × (1 + 預測報酬)。
- 區間與機率目標：分位數回歸（10%、50%、90%）給出報酬區間並換算成價格區間，是展示用的主區間；波動率模型預測未來 20 日實現波動，只做 fallback 與 sanity check；分類模型給出 20 日漲幅超過 5% 的機率。5 日的價格區間只由 API 提供（波動率區間，見第 7 節輸出一致性），頁面不展示。
- 三類目標共用同一組特徵，分別評估、分別上線；輸出之間的一致性規則見第 7 節。

**範圍內**

- 上市（TWSE）與上櫃（TPEx）普通股；ETF 為可選。
- 日頻資料；收盤後批次計算，隔日開盤前產出預測。
- 六組特徵：技術面、規模與流動性、籌碼面、基本面、產業與風險、市場層級。

**範圍外（第一版不做）**

- 盤中即時行情、自動下單、分 K 與 tick 層級策略。
- 興櫃股票、權證、期貨選擇權。
- 美股以外的國際市場個股。

**授權前提**：FinMind 資料限教育與非商業用途，證交所行情對外再提供需另取得資訊使用授權。第一版只供個人使用，商業化前要重新評估資料來源（見第 3 節）。

## 2. 系統架構與技術選型

系統分三層：資料層（ETL 與 PostgreSQL）、研究層（特徵、訓練、回測，獨立於 Web 執行）、服務層（Django API 與頁面）。研究層只讀資料庫、只寫研究層自己的表（`feature_daily`、`label_daily`、`model_run`、`prediction`、`prediction_contrib`、`model_monitor`、`backtest_metric`、`holdout_run`），另把模型輸出檢查與告警寫進 `data_quality_log`（check 名稱以 `output_`、`monitor_`、`model_`、`research_` 開頭，severity 最高 warning）；Web 永遠不直接呼叫外部資料 API。

報酬換算成價格的工作放在服務層：每檔股票用自己當日的收盤價計算預測價與價格區間，模型層只輸出報酬、分位數、波動率與機率，不輸出價格。

| 元件 | 選型 | 說明 |
| --- | --- | --- |
| Web / API | Django 5 + Django REST Framework | 沿用既有技術棧；提供預測、特徵、模型版本查詢 |
| 主資料庫 | PostgreSQL 16 | 單一真相來源；日頻資料約千萬列等級，一般 Postgres 足夠 |
| 分析快取 | Parquet（pandas + pyarrow） | 特徵矩陣與訓練集快照，避免每次重算 |
| 排程 | Celery beat（或 cron 呼叫 management command） | 每日收盤後依資料公布時間分批執行 ETL |
| 資料擷取 | requests、FinMind SDK、官方 OpenAPI | 每個來源一個 adapter，輸出相同的 standard schema |
| 建模 | scikit-learn（Ridge）、LightGBM、XGBoost | 模型工件以版本號存檔，參數寫入 `model_run` |
| 回測與評估 | 自寫 walk-forward 框架（pandas、scipy） | 輸出 rank IC、ICIR、十分位組合報酬 |
| 容器 | Docker Compose（web、worker、db、redis） | 本機開發與部署同一份設定 |
| 執行環境 | 個人 Mac（macOS）+ Docker Desktop | 第一版不用雲端主機；ETL 時段用 caffeinate 防休眠、螢幕可關（見第 5 節執行環境） |

[圖：系統架構 · 三層、九個元件、資料單向流動（匯出未保留）]

讀法：上層由左到右是 ETL 流程，standard 表是資料層的出口；研究層每晚從它算特徵、訓練與回測，只把結果寫進預測結果表；服務層的 Django 只讀這張表回應使用者。

## 3. 資料來源總覽

原則：FinMind 做一次性歷史回補，證交所、櫃買中心、集保、MOPS 做每日增量，本地資料庫為唯一真相來源；券商 API 留到即時階段再接。所有限制與授權資訊以 2026-10-29 查到的為準，接入前要再確認。

| 來源 | 提供資料 | 存取方式 | 限制 | 授權 | 在系統中的角色 |
| --- | --- | --- | --- | --- | --- |
| [FinMind](https://pypi.org/project/FinMind/) | 日價量、PER/PBR、加權指數、三大法人、融資券、借券、外資持股、股權分散表、月營收、三大財報、股利與除權息、美股、匯率、利率 | Python SDK `FinMind`，或 REST `api.finmindtrade.com/api/v4/data` | 300 次/小時，註冊並驗證信箱後 600 次/小時；超限封鎖約 1 小時；週日 00:00–07:00 維護 | 教育與非商業用途 | 歷史回補（2010 起，一次性）；除權息歷史補洞；股權分散表限付費會員，不用 |
| [證交所 TWSE OpenAPI](https://glama.ai/mcp/servers/taux-io/twse-mcp) | 上市日成交、三大法人買賣超、融資融券、借券、處置與注意股、除權除息與減資、面額變更的參考價、當沖交易統計（第二批） | REST `openapi.twse.com.tw/v1`，端點多為當日快照；歷史用 `exchangeReport/STOCK_DAY?date=&stockNo=`，一次回傳整月 | 無公開配額，頻繁請求會封 IP；請求間隔至少 3 秒 | 政府資料開放授權；對外再提供行情須另取得資訊使用授權 | 上市股票每日增量 |
| [櫃買中心 TPEx OpenAPI](https://scidm.nchc.org.tw/en/dataset/best_wish11391) | 上櫃日成交、三大法人、融資融券、櫃買指數、除權除息與減資、面額變更的參考價、當沖交易統計（第二批） | REST，Swagger 文件 `tpex.org.tw/openapi/swagger.json` | 同證交所 | 政府資料開放授權 | 上櫃股票每日增量 |
| 公開資訊觀測站 MOPS | 財報（XBRL）、月營收、董監與內部人持股、重大訊息 | 網頁表單與檔案下載 `mops.twse.com.tw`，無正式 API | 需爬蟲，頁面格式會變；要記錄公告日 | 公開資料 | 月營收與財報每日掃描；取得 announce\_date |
| [集保 TDCC 股權分散表](https://scidm.nchc.org.tw/en/dataset/best_wish11452) | 每週各持股級距的人數、股數、占集保庫存比例 | 開放資料 CSV，每週最後一個營業日結算後發布 | CSV 只有最新一週，歷史要自己累積 | 政府資料開放授權 | 每週六增量累積；不回補歷史，`chip_*_1w` 在開始累積之前為空 |
| [TEJ Tool API](https://pypi.org/project/tej-tool-api) | PIT 資料庫 600 多個指標；財務數據依公告日對應 | Python `tej-tool-api`，需 API key | 付費；預設起始日 2013-01-01；僅台灣市場 | 商業授權；學校圖書館可能有 | 財報 PIT 對齊的對照組（可選） |
| [富果 Fugle Marketdata](https://developer.fugle.tw/docs/data/http-api/historical/candles) | 日 K、分 K、即時行情、技術指標 | REST + Python SDK，需 API key | 歷史每次最多 1 年，個股回溯到 2010、指數到 2015；分 K 只有近 30 日；速率依方案 | 免費方案有上限 | 即時報價（Phase 6） |
| [永豐 Shioaji](https://ithelp.ithome.com.tw/articles/10270639) | 分 K、tick、即時行情、下單 | Python SDK，需永豐帳號與憑證 | 官方不建議撈歷史；每 5 秒 500 次；濫用可能鎖帳號 | 券商條款 | 下單（Phase 6 之後，可選） |
| [yfinance](https://ithelp.ithome.com.tw/articles/10358317) | 美股、費半（^SOX）、TSM ADR；台股日 K 用 `.TW` / `.TWO` 後綴 | Python 套件 | 非官方，常失效；無籌碼資料 | 無正式授權 | 跨市場特徵，只用美股與指數 |
| 政府資料開放平臺 data.gov.tw | 景氣對策信號、出口統計、匯率與利率 | CSV / JSON 下載 | 更新頻率不一，多為月頻 | 政府資料開放授權 | 總經特徵（月頻） |

**多來源合併原則**

- 每類資料指定一個 primary source；其他來源只做比對驗證，不混用。例外只有第 8 節第 5 條列出的幾條。
- 價量 primary：證交所與櫃買中心；籌碼 primary：證交所與櫃買中心；財報與月營收 primary：MOPS；股權分散 primary：集保。
- FinMind 是回補與驗證來源，每日增量不依賴它，避免非商業授權與維護時間影響正式流程。
- FinMind 對 2016 到 2020 年下市的股票缺財報、月營收、股本與部分籌碼，這些缺口只補缺的列或空的欄位：財報與月營收用 MOPS 歷史彙總頁（下市公司不在彙總頁上，財報改讀 XBRL 檢視頁，月營收改用新站的單一公司查詢；plan K17），股本、籌碼與本益比用證交所、櫃買每日表，產業用交易所依產業別的行情（plan K17c；外資持股比的 0 另見 K21；規則都在第 8 節第 5 條）。單一公司查詢 2014-01 起才有資料，下市公司 2013 年的月營收和查不到的月份 standard 表維持缺值；特徵層的 `fund_rev_yoy`、`fund_rev_3m_yoy` 在月營收來源缺時改用季營收年增率（第 6 節處理步驟 6 的殘餘缺口規則）。

## 4. 資料模型與儲存

三層設計：raw 層原樣保存每個來源的回應，standard 層把所有來源統一成相同 key 與單位，feature 層是可直接餵模型的寬表。所有 standard 表的主鍵都是 `(stock_id, date)`，所有基本面資料都帶 `announce_date`。

**raw 層**

- 每個來源一張表 `raw_<source>`，例如 `raw_twse`、`raw_finmind`；以 `dataset` 欄區分資料集，唯一鍵 `(dataset, source_hash)`。
- `payload` 欄原樣存來源回應（JSON），外加 `dataset`、`fetched_at`、`request_params`、`source_hash`；不做任何清洗。
- 保留 raw 層的目的：來源改格式時可以重跑 normalize，不必重抓。

**standard 層核心表**

| 表名 | 主鍵 | 核心欄位 | primary 來源 | PIT 欄位 |
| --- | --- | --- | --- | --- |
| `stock_master` | stock\_id | name、market（TWSE/TPEx）、industry、list\_date、delist\_date | 證交所、櫃買中心 | — |
| `market_history` | id；唯一鍵 stock\_id, start\_date | end\_date（含當天；每檔最後一段留空，代表到目前）、market（TWSE/TPEx）；同一檔的區間連續、不重疊 | 證交所、櫃買中心（歷史：每日外資持股頁；每日：當天報價） | start\_date、end\_date |
| `trading_calendar` | date | is\_trading\_day、note（補班、颱風假） | 證交所 | — |
| `daily_price` | stock\_id, date | open、high、low、close、volume（股）、turnover（元）、adj\_factor | 證交所、櫃買中心 | date |
| `dividend_event` | stock\_id, ex\_date | cash\_dividend、stock\_dividend、adj\_factor、announce\_date（除權息日第一次公開的交易日，22:40 以後公告的算下一個交易日；沒有公告日的留空）；減資、分割、面額變更存成只帶 adj\_factor 的列（現金、股票股利為 0） | FinMind（歷史）；證交所、櫃買中心（每日） | ex\_date、announce\_date |
| `daily_chip` | stock\_id, date | fi\_net、it\_net、dealer\_self\_net、dealer\_hedge\_net（股）、margin\_balance、margin\_limit、short\_balance、sbl\_short\_balance、fi\_holding\_pct、day\_trade\_volume（股，第二批） | 證交所、櫃買中心 | date |
| `weekly_holding` | stock\_id, data\_date, level | level（1–15）、holders、shares、pct、publish\_date | 集保 | publish\_date（data\_date 隔日） |
| `monthly_revenue` | stock\_id, period（YYYY-MM） | revenue、yoy、mom | MOPS | announce\_date |
| `financial_statement` | stock\_id, period（YYYY-Qn）, fin\_type | eps、revenue、gross\_margin、op\_margin、net\_income、roe、debt\_ratio、operating\_cf | MOPS | announce\_date |
| `index_daily` | index\_id, date | close、volume；index\_id 含 TAIEX、TAIEX\_TR（發行量加權股價報酬指數，含息）、TPEX、SOX、TSM\_ADR、USDTWD | 證交所、yfinance | date |
| `macro_indicator` | indicator\_id, period | value | data.gov.tw | announce\_date |
| `shares_outstanding` | stock\_id, effective\_date | shares、reason（增資、減資、庫藏股註銷） | 證交所、櫃買中心 | effective\_date |
| `daily_valuation` | stock\_id, date | pe、pb、dividend\_yield（交易所每日公布的本益比、股價淨值比、殖利率） | 證交所、櫃買中心；歷史用 FinMind | date |
| `disposition_event` | stock\_id, start\_date, type | end\_date、type（注意、處置、全額交割；櫃買「屬管理股票」也存成全額交割） | 證交所、櫃買中心（含歷史） | start\_date |

**feature 與模型層**

| 表名 | 主鍵 | 核心欄位 | 說明 |
| --- | --- | --- | --- |
| `feature_daily` | stock\_id, date | 欄位命名 `<group>_<name>_<window>`，例如 `tech_ret_20d`、`chip_fi_net_5d`、`fund_rev_yoy`；另有 `in_universe` 旗標 | 寬表，只存原值與 in\_universe（原值是套過第 6 節殘餘缺口規則的值）；winsorize、z-score、填值在管線做，不落表；也同步輸出 Parquet 供訓練 |
| `label_daily` | stock\_id, date, horizon | fwd\_ret、fwd\_ret\_open、fwd\_rank\_pct、fwd\_vol\_20d、fwd\_up\_5pct、untradable\_t1 | 還原價計算；horizon 為 5 或 20；fwd\_ret\_open 以 t+1 還原開盤價進場，是訓練、評估與回測一律使用的版本，fwd\_ret 只供對照；fwd\_vol\_20d 為未來 20 日日報酬標準差；fwd\_up\_5pct 為 20 日漲幅是否超過 5%；horizon 內下市以最後成交價結算，列保留 |
| `model_run` | run\_id | run\_key、model\_type、objective（rank、regression、quantile、volatility、classification）、version、train\_start、train\_end、params（JSON）、feature\_list、seed、lib\_versions、feature\_hash、status、ic\_mean、icir、created\_at | 每次訓練一列；同一次 walk-forward 的各摺、各模型共用一個 run\_key |
| `prediction` | run\_id, stock\_id, date, horizon | score、rank、pct、ret\_pred、q10、q50、q90、vol\_pred、prob\_up\_5pct、in\_universe | API 讀這張表；每列只存該列 run 產出的欄位，其他為 NULL，API 依 (stock\_id, date) 合併各角色當天服務的 run，波動率兩個 horizon 都讀 20 日 run；同類有多個 run 時依 active、degraded、retired 的順序，再取最新的；預測價與價格區間在 API 層用當日收盤價換算，不存進表 |
| `prediction_contrib` | run\_id, stock\_id, date, horizon, position | feature、contrib、n\_features | 排名預測的逐檔特徵貢獻（LightGBM pred\_contrib，5 seed 平均）：前 5 名各一列，最後一列 feature = `_rest` 是其餘特徵的合計，n\_features 為合計的特徵數；同一筆預測內 feature 不重複；服務層「這次預測怎麼來的」讀這張表 |
| `model_monitor` | run\_id, date, horizon | ic、coverage、brier、direction\_acc、mae、width、auc、n\_stocks | 事後評估結果；date 是預測日；每日對 horizon 已到期的預測計算，服務層的歷史預測準確度讀這張表 |
| `backtest_metric` | id；唯一鍵 run\_key, model\_type, objective, horizon, metric, grain, period\_start, period\_end, segment | value、created\_at | walk-forward 指標的長表；grain 為 total、fold、year、month，segment 標子群（例如成交金額三分位）；模型頁的 IC 與 baseline 對比讀這張表，`model_monitor` 只存上線後的事後評估 |
| `holdout_run` | id；唯一鍵 (scope, attempt)、run\_key | scope（rank、return、interval）、attempt、run\_key、last\_date、reason、claimed\_at | holdout 只跑一次的紀錄：每個 scope 第一次是 attempt 1，之後重跑每次加一列並必須寫 reason，舊結果保留；每次 claim 換一個 run\_key，晚於 last\_date 的 holdout 摺拒跑 |
| `etl_run` | run\_id | source、dataset、date\_range、status、rows、error | 每次 ETL 一列 |
| `data_quality_log` | id | date、check\_name、severity、detail | 品質檢查結果 |

**規則**

- 成交量統一為「股」，金額統一為「元」；來源若為張（1 張 = 1,000 股）在 normalize 時轉換。
- 成交量與成交金額：證交所含一般、零股、盤後定價、鉅額交易，與 `MI_INDEX`、`STOCK_DAY` 相同；櫃買不含鉅額（含零股、盤後定價），與 FinMind 回補同定義；開高低收只反映一般交易，所以判斷當日有無成交看收盤價是否為空，不看成交量。
- 價量表照收來源的整張表（含 ETF、ETN、特別股、TDR），只研究普通股由 universe 篩選；外幣計價的證券（雙幣 ETF 的外幣櫃位，代號第 6 碼為 K、M、S、C）不進 standard 表。
- 時間戳一律 `Asia/Taipei`，日期欄位用 `DATE` 型別。
- `daily_price` 存原始價與 `adj_factor`，還原價在查詢時相乘，不另存一份。
- 索引：所有 `(stock_id, date)` 表建複合索引，`prediction` 再加 `(date, horizon, rank)` 與 `(run_id, horizon, date)`。
- 大量寫入用 `bulk_create` 或 PostgreSQL `COPY`，並以 upsert 保證重跑冪等。

## 5. ETL 與排程

兩條路徑：一次性歷史回補走 FinMind，每日增量走官方來源；每個來源一個 adapter，輸出相同的 standard schema，重跑不會重複寫入。

**Adapter 介面**

- `fetch(dataset, start, end, stock_id=None) -> DataFrame`：呼叫來源，原樣寫入 raw 層。
- `normalize(dataset, raw_df) -> DataFrame`：轉成 standard schema，統一欄名、單位、日期。
- `upsert(dataset, df)`：以主鍵 upsert 進 standard 表。
- 每個 adapter 內建 sleep 與指數退避，速率上限寫在設定檔，不寫死在程式。
- 每次執行寫一列 `etl_run`，含 checkpoint（最後處理的 stock\_id 或日期；歷史回補被拒收的股票另記在 `rejected`），中斷後可續跑。

**歷史回補（Phase 1）**

1. 從證交所與櫃買中心取得全部上市、上櫃清單，含已下市股票，建立 `stock_master`。
2. 依 `stock_master` 逐檔向 FinMind 拉 2010-01-01 至今的日價量、三大法人、融資券、借券、外資持股、月營收、財報、股利。股權分散表不回補（FinMind 限付費會員），由集保每週累積；注意、處置、全額交割的歷史不用 FinMind，改用證交所、櫃買的官方查詢回補。
3. 某一檔的來源資料不能用（格式改變、日期不在 `trading_calendar`、缺欄位）時只拒收那一檔，記進 checkpoint 的 `rejected`，其他股票照常寫入，整次跑完標 failed；連續 `ETL_MAX_CONSECUTIVE_REJECTIONS`（預設 20）檔被拒收時停下，checkpoint 倒回這串拒收之前，續跑時重抓。連線錯誤與配額用盡仍整批停下。
4. 月營收與財報的 `announce_date`：FinMind 沒有公告日，回補時先存法定截止日的下一個交易日（保守值，只會晚不會早）。月營收的截止日是次月 10 日（截止月份在 2026 年起保險業可到 15 日，這段期間的保守值不分產業一律用 15 日），月初遇休市的平日順延；財報 Q1、Q2、Q3 是 5/31、8/31、11/30，年報是隔年 3/31（2010 年度以前是 4/30）。財報再以證交所電子書（t57sb01）的實際上傳時間覆寫，規則見第 6 節處理步驟 2；月營收查不到歷史申報日，研究期維持保守值，上線後由每日掃描記錄實際申報日。
5. 約 2,000 檔乘以 8 個 dataset，在 600 次/小時的上限下需要分批跑數小時；排在週一到週六執行，避開週日維護時間。
6. 回補完成後抽樣比對：至少 20 檔、每檔 3 個年度，與證交所 `STOCK_DAY` 的收盤價與成交量逐日核對。

**每日增量（台北時間，依資料公布時間排序）**

| 時間 | 工作 | 來源 | 頻率 |
| --- | --- | --- | --- |
| 15:10 | 更正交易日曆；補抓前 14 天到前一交易日漏掉的日子，並補跑那些日子的品質檢查 | 證交所、櫃買中心 | 每交易日 |
| 15:30 | 日成交價量、加權與櫃買指數、加權股價報酬指數；當日的除權除息、減資、面額變更計算結果 | 證交所、櫃買中心 | 每交易日 |
| 16:00 | 以當日行情更新 `stock_master` 與 `market_history` | 證交所、櫃買中心 | 每交易日 |
| 16:30 | 三大法人買賣超、本益比與淨值比、注意與處置名單、全額交割名單 | 證交所、櫃買中心 | 每交易日 |
| 18:30 | 櫃買的外資持股與流通股數變動（櫃買每日 18:00、22:00 更新） | 櫃買中心 | 每交易日 |
| 19:00、22:40 | 月營收掃描 | MOPS | 每交易日，營收月份的次月 1 日到法定截止日後 20 天 |
| 19:00、22:40 | 財報掃描 | MOPS | 每交易日，季底隔天到保守截止日後 45 天（等於全年每天掃，每天 24 到 36 個請求） |
| 21:30 | 融資融券餘額、借券（約 20:30、22:30 更新）；當沖交易統計（第二批）抓當日與前兩個交易日（T、T+1、T+2 各更新一次，以 T+2 為準）；證交所的外資持股與流通股數變動（2026-11-09 觀測在 20:30 到 21:00 之間公布） | 證交所、櫃買中心 | 每交易日 |
| 22:40 | 注意與處置名單再抓一次（名單可能晚於 16:30 公布） | 證交所、櫃買中心 | 每交易日 |
| 22:50 | 資料品質檢查，並列出當日該跑卻沒有成功的工作 | — | 每交易日 |
| 23:00 | 個股層級特徵重算最近 60 個交易日、label 重算最近 81 個交易日（第 6 節處理步驟 6）；還原因子改寫了歷史時，受影響日在這 60 個交易日內的由這兩步涵蓋，更早的告警、手動重建；這一步出錯只告警，不擋當天 | 研究層 | 每交易日，研究閘門通過後（見資料品質檢查） |
| 05:30（t+1） | 美股、費半、TSM ADR 的美國 t 日收盤，匯率的倫敦 t−1 日收盤（倫敦 t 日的收盤在台北 t+1 07:00 後才收完）；抓不到不寫補值列，由特徵端 asof 沿用前一個收盤，並寫 data\_quality\_log | yfinance | 每交易日 |
| 06:00（t+1） | 等 05:30 的美股收盤寫完，依序重算特徵（補上美股、匯率）、產出預測、模型監控，前一步成功才跑下一步。23:00 沒跑成時補上它的兩步：先算 81 天 label，監控之後再算 60 天特徵，還原因子改寫照 23:00 處理；其他早上只算 D 的特徵，還沒處理的還原因子改寫，受影響日在最近 60 個交易日內的在最後從受影響日起重建，更早的告警。這一步出錯只告警，不擋當天 | 研究層 | 每交易日，開盤前 |
| 每月重訓 | 每月前 5 個交易日裡，第一條在 06:00 到 09:00 之間跑完的早上 chain（含缺美股收盤而記 warning 的）之後，每個設計已凍結的類別（`production.DESIGNS` 的 `research_run_key` 有值）訓 N 與 N′（訓練到 D），可用記憶體不到一半就延後；下一個早上的 chain 在監控之後換模（第 7 節的門檻，沒過就告警）；前 5 個交易日都沒排成時，之後第一條跑完的早上告警 | 研究層 | 每月 |
| 週六 10:00 | 股權分散表 | 集保 | 每週 |
| 每月 15 日 10:00 | 景氣燈號；M 月的資料在 M+2 月 15 日寫入，以那天為 announce\_date（國發會約每月 27 日發布上月，年假會延後） | data.gov.tw | 每月 |
| 每月 1 日 | 出口、匯率利率（尚未實作） | data.gov.tw | 每月 |

公布時間會變動，排程只是起跑時間；每個工作要先檢查來源是否已有當日資料，沒有就 30 分鐘後重試，最多 6 次。網路錯誤（逾時、斷線、5xx）比照重試；被限流時只在 adapter 內冷卻，冷卻完仍被限流就當天放棄，隔日補抓。05:30 美股收盤只抓一次、不重試，缺的由 asof 沿用；15:10 本身就是補抓，不另外重試。

研究層會改變服務內容的工作（產出預測、模型監控、換模、恢復，以及排程中的特徵與 label 重算）共用一把鎖，同時只跑一個：排程拿不到就延後重試，手動執行拿不到就報錯並說明誰在跑。研究層的排程跑完記一筆完成紀錄，失敗只寫 warning 並告警，不擋隔天。

月營收與財報掃描：寫入時間在當天 22:50 品質檢查之前的算當天公告，否則算下一個交易日；已有的列只新增不改，同一期數字不同時記 `revenue_source_mismatch`、`statement_source_mismatch`（info），不覆蓋。

**資料品質檢查（每日 ETL 之後）**

- 當日有資料的股票數與前一交易日相比差異超過 5% 就告警；這項算全部代號（含 ETF 等非普通股）。
- 單日漲跌幅超過 10% 且無除權息紀錄的普通股標記為待查。
- daily\_price、daily\_chip、daily\_valuation 的當日缺值比例超過 1% 就告警（收盤價用 5%，因為有停牌與無成交）；缺值只算普通股。
- 日期必須存在於 `trading_calendar`，否則整批拒收；歷史回補只拒收該檔（見歷史回補）。
- 指數新鮮度：TAIEX、TAIEX\_TR、TPEX 當天要有收盤，SOX、TSM\_ADR、USDTWD 最多落後 4 個日曆日，超過就告警（warning）。
- 結果寫入 `data_quality_log`，severity 為 error 時不執行當日預測。
- 研究閘門：研究層的每日任務要等兩個交易所的日價量、指數、三大法人、融資融券、借券、外資持股、本益比與淨值比這些當日快照，以及除權息事件（除權息日的還原價要用到當天的因子），都有成功的 `etl_run`，且當日品質檢查沒有 error；注意、處置、全額交割、流通股數變動、當沖、加權股價報酬指數不擋，漏一天只是晚一天反映，隔日補抓；報酬指數只給第 7 節相對大盤的機率 label 用，停止更新時由指數新鮮度檢查告警。

**失敗處理**

- 單一來源失敗不阻塞其他來源；該來源的 `etl_run` 標 failed，隔日自動補抓缺的日期。
- 來源回傳格式改變時 normalize 會失敗，raw 層已保存，修好 normalize 後重跑即可。
- 月營收與財報掃描不進補抓：漏掉的掃描只讓公告日晚一點，下一次掃描就補上；有月份或季別在掃描期間的交易日，當天沒有一次成功的掃描時，22:50 的檢查記 warning 並告警，不擋研究閘門。

**執行環境（第一版）**

- 排程機器是個人 Mac：Docker Desktop 跑 Compose 的 web、worker、db、redis 四個容器（redis 是 Celery 的 broker），Celery beat 在 worker 容器內觸發所有排程；第一版不用雲端主機。worker 只開一個行程（`CELERY_WORKER_CONCURRENCY=1`）；研究排程設定不是 1 時不跑並告警。
- 防休眠：ETL 時段（15:00 到隔天 02:00、清晨 05:15 到 06:45、週六 10:00）要讓 Mac 保持醒著。螢幕可以關，但不要合蓋，MacBook 合蓋會直接睡眠。指令：`caffeinate -i -t 39600`（防止閒置睡眠 11 小時，涵蓋 22:40 那批的重試，顯示器仍可自行熄滅）；接電源時可改用 `caffeinate -s`。
- 自動化：用 launchd 在週一到週五 15:00、週二到週六 05:15 各執行一次 caffeinate 指令，清晨那次用 -t 5400；launchd 不知道交易日，週末補行交易日要手動執行。並在系統設定的能源選項開啟「顯示器關閉時避免自動進入睡眠」（接電源時有效）；launchd 不會叫醒已睡著的 Mac，這一項是週六集保與清晨美股收盤唯一可靠的防護。
- 漏跑處理：Mac 當天睡著或沒開機時，隔日 ETL 先補抓缺的日期再跑當日（見失敗處理）；補抓那天的執行時間會加倍，排程之間要留餘裕。研究排程不補跑：前一個交易日沒有 06:00 的紀錄時，下一個早上替它記 warning（`missed`）並告警。
- 時區：macOS 與所有容器都設 `TZ=Asia/Taipei`，否則排程時間會偏 8 小時。
- 升級條件：之後需要全天在線（例如 Phase 6 即時報價）再評估 1 到 2 GB RAM 的 VPS。

## 6. 特徵工程

六組特徵。台股資料以「t 日收盤後可知」為準，美股用美國 t 日收盤（台北 t+1 約 05:00 可得），美元兌台幣用倫敦日 ≤ t−1 的收盤（倫敦 t 日的收盤在台北 t+1 06:00 時還沒收完），全部早於 t+1 開盤進場；每個特徵在表中註明可用時點。前五組是個股層級；第六組「市場層級」對全市場同值，原本只給絕對報酬、區間、波動、機率模型用，排名模型不用；Phase 4 比較後這四類也不加（見第 7 節模型表）。四種 label：未來 5 日與 20 日還原報酬（以 t+1 還原開盤價進場）、其當日橫截面百分位、未來 20 日實現波動、20 日漲幅是否超過 5%（機率模型改用相對大盤版，見第 7 節三類預測目標）。第一版 63 個特徵；表中標「第二批」的 14 個照常計算並存進 `feature_daily`，訓練預設只讀第一批，在 Phase 4 主模型過關後依序加入，每加一組重跑第 7 步的篩選與 walk-forward，總體 OOS rank IC 沒有提高就不留（「提高」的定義見第 7 步）。表中標「不用」的也照常計算並存進 `feature_daily`，但不進任何一批，篩選、訓練與第二批的逐組比較都不讀。先求正確再求多。

| 類別 | 特徵（欄位名） | 來源表 | 可用時點 | 備註 |
| --- | --- | --- | --- | --- |
| 技術面 | `tech_ret_1d` `tech_ret_5d` `tech_ret_20d` `tech_ret_60d` | daily\_price | t 日晚間 | 還原價報酬 |
| 技術面 | `tech_mom_250_20` | daily\_price | t 日晚間 | 12-1 月動能：t−250 到 t−20 的還原報酬，跳過近 20 日 |
| 技術面 | `tech_ma_gap_5d` `tech_ma_gap_20d` `tech_ma_gap_60d` | daily\_price | t 日晚間 | 收盤 / 均線 − 1 |
| 技術面 | `tech_vol_20d` `tech_vol_60d` | daily\_price | t 日晚間 | 日報酬標準差，年化 ×√252，與 fwd\_vol\_20d 同單位 |
| 技術面 | `tech_volume_ratio_5_20` `tech_turnover_rate_5d` | daily\_price、shares\_outstanding | t 日晚間 | 量能相對變化；週轉率 = 5 日平均成交股數 / 流通股數 |
| 技術面 | `tech_rsi_14` `tech_kd_k` `tech_kd_d` `tech_macd_hist_pct` | daily\_price | t 日晚間 | 標準參數 14 / 9 / 12-26-9；MACD 柱除以收盤價，讓橫截面可比 |
| 技術面 | `tech_high_52w_gap` `tech_low_52w_gap` | daily\_price | t 日晚間 | 距 52 週高低點；上市未滿 52 週用現有區間 |
| 技術面 | `tech_limit_up_t` `tech_limit_up_cnt_20d` `tech_limit_down_cnt_20d` `tech_attention_t` | daily\_price、disposition\_event | t 日晚間 | t 日是否收漲停；過去 20 日漲停、跌停次數；t 日是否在注意期間。漲停價 = 前收 × (1 + 限幅) 按升降單位無條件捨去，跌停價無條件進位；限幅 2015-06-01 前 7%、之後 10%；新上市前 5 個交易日無限制，不計 |
| 規模與流動性 | `size_log_mcap` `liq_dollar_vol_med_20d` `liq_amihud_20d` | daily\_price、shares\_outstanding | t 日晚間 | log(收盤 × 流通股數)；20 日成交金額中位數；Amihud = 20 日平均(日報酬絕對值 / 成交金額)，成交金額為 0 的日子跳過 |
| 籌碼面 | `chip_fi_net_1d` `chip_fi_net_5d` `chip_fi_net_20d` | daily\_chip、shares\_outstanding | t 日晚間 | 外資買賣超股數 / 流通股數 |
| 籌碼面 | `chip_it_net_5d` `chip_dealer_self_net_5d` `chip_dealer_hedge_net_5d` | daily\_chip、shares\_outstanding | t 日晚間 | 投信；自營商分自行買賣與避險；皆除以流通股數 |
| 籌碼面 | `chip_margin_chg_5d` `chip_margin_usage` `chip_short_ratio` | daily\_chip、shares\_outstanding | t 日晚間 | 融資餘額 5 日變化 / 流通股數；融資使用率 = 融資餘額 / 融資限額；券資比 |
| 籌碼面 | `chip_fi_holding_pct` `chip_sbl_short_chg_5d` | daily\_chip、shares\_outstanding | t 日晚間 | 外資持股比；借券賣出餘額 5 日變化 / 流通股數，用借券賣出餘額而非借券餘額。證交所 2012-03-19 起改了借券賣出餘額的計算方式並公告，上市股票的 5 日變化讀到的兩筆借券賣出餘額分屬這天前後時（停牌日沿用前一筆餘額，沒有停牌時就是 t 在 3/19 到其後第 4 個交易日），`chip_sbl_short_chg_5d` 為 NaN，算 PIT 可觀察（補上該有卻沒有的借券列後就不跨這天的，照來源缺算）；上櫃沒有這個斷點。市場依 t 日的市場歷史（第 6 節處理步驟 1） |
| 籌碼面 | `chip_big_holder_pct_chg_1w` `chip_holder_count_chg_1w` | weekly\_holding | publish\_date，即 data\_date 隔日 | 400 張以上級距占比週變化、總股東人數週變化；取 publish\_date ≤ t 的最近一期；level 16、17 排除 |
| 基本面 | `fund_rev_yoy` `fund_rev_3m_yoy` | monthly\_revenue；來源缺時 financial\_statement | announce\_date | 單月 yoy、近 3 個月合計 yoy。月營收來源缺（最近一期月營收可見超過 62 天（日曆日）沒有新一期，或要讀的月份沒有紀錄）時改用同一天的 `fund_rev_yoy_q`；月份都有紀錄但讀到 2013-01 兩側或去年同期營收 ≤ 0 時照舊為 NaN，不補 |
| 基本面 | `fund_rev_mom` | monthly\_revenue | announce\_date | 不用。單月 mom；月營收來源缺、讀到 2013-01 兩側或上月營收 ≤ 0 時為 NaN。下市股票 2013 年的月營收和單一公司查詢查不到的月份沒有來源（第 3 節），mom 沒有季資料可替代，進模型時缺值會透露之後下市 |
| 基本面 | `fund_ep` `fund_eps_yoy` | financial\_statement、daily\_price | announce\_date | E/P = EPS TTM / 收盤價，虧損為負值，不用本益比；EPS TTM 為 PIT 最近四季合計，Q4 由年報減前三季；EPS yoy 為 TTM 年增，分母取去年 TTM 絕對值，去年為負或零時 NaN；最近一期財報可見超過 200 天（日曆日）沒有新一期時為 NaN |
| 基本面 | `fund_bp` `fund_dividend_yield` | daily\_valuation | t 日晚間 | 證交所、櫃買每日公布的股價淨值比與殖利率；B/P = 1 / 股價淨值比 |
| 基本面 | `fund_roe_ttm` `fund_gross_margin` `fund_op_margin` `fund_debt_ratio` | financial\_statement | announce\_date | 以 announce\_date 對齊；過期上限 200 天，同 `fund_ep` |
| 基本面 | `fund_gm_yoy_chg` `fund_opm_yoy_chg` `fund_cfo_to_ni` | financial\_statement | announce\_date | 毛利率、營益率較去年同季的變化；營業現金流 / 淨利，TTM；過期上限 200 天，同 `fund_ep` |
| 基本面 | `fund_days_since_fs` | financial\_statement | announce\_date | 距最近一期財報公告的交易日數；不套 200 天上限，最近一期公告超過 936 天（日曆日，往回讀財報的範圍）時為 NaN |
| 基本面 | `fund_rev_yoy_q` | financial\_statement | announce\_date | 單季營收對 4 季前的年增率，4 季前 ≤ 0 時 NaN；下市股缺月營收時仍有營收訊號；過期上限 200 天，同 `fund_ep` |
| 產業與風險 | `risk_beta_250d` `risk_idio_vol_60d` | daily\_price、index\_daily | t 日晚間 | 250 日回歸的 beta，上市股對加權指數、上櫃股對櫃買指數（市場依 t 日的市場歷史，轉換市場前後的報酬各對各自的指數）；60 日殘差報酬標準差 |
| 產業與風險 | `ind_sector_ret_20d` `ind_rel_ret_20d` | daily\_price、stock\_master | t 日晚間 | 個股所屬產業 20 日平均報酬；個股 20 日報酬減產業平均 |
| 市場層級 | `mkt_taiex_ret_5d` `mkt_taiex_ret_20d` `mkt_sox_ret_5d` | index\_daily | 台股 t 日晚間；美股用美國 t 日收盤，台北 t+1 約 05:00 可得 | 大盤、費半；全市場同值，排名模型不用、不做橫截面 z-score |
| 市場層級 | `mkt_tsm_adr_gap` `mkt_usdtwd_chg_20d` | index\_daily | TSM ADR 用美國 t 日收盤，台北 t+1 約 05:00 可得；美元兌台幣用倫敦日 ≤ t−1 的收盤 | ADR 溢價 = TSM ADR 收盤 × 美元兌台幣 / 5 / 2330 台股 t 日收盤 − 1，1 ADR 為 5 股；匯率 20 日變化；同上 |
| 市場層級 | `mkt_business_signal` | macro\_indicator | announce\_date | 景氣燈號分數；同上 |
| 技術面 | `tech_parkinson_vol_20d` `tech_gap_1d` `tech_close_loc_5d` | daily\_price | t 日晚間 | 第二批。Parkinson 波動 = √(20 日平均 ln(高/低)² / (4 ln 2))，年化；跳空 = 開盤 / 前收 − 1；收盤在 5 日高低區間的位置，0 到 1 |
| 技術面 | `tech_max_ret_20d` `tech_skew_60d` | daily\_price | t 日晚間 | 第二批。過去 20 日最大單日報酬，樂透效應；60 日日報酬偏態 |
| 籌碼面 | `chip_day_trade_ratio_5d` | daily\_chip | t−6 到 t−2 的 T+2 定案值 | 第二批。5 日當沖成交股數 / 5 日成交股數，5 日是 t−6 到 t−2（當沖表在 T、T+1、T+2 各更新一次，以 T+2 為準）；同市場當天有當沖表、該檔不在表上算 0，整個市場當天沒資料才是空值；證交所與櫃買每日公布，存在 daily\_chip 的 day\_trade\_volume |
| 籌碼面 | `chip_fi_holding_chg_20d` `chip_it_net_20d` `chip_fi_buy_streak` | daily\_chip、shares\_outstanding | t 日晚間 | 第二批。外資持股比 20 日變化，百分點；投信 20 日淨買 / 流通股數；外資連續淨買超交易日數 |
| 產業與風險 | `ind_sector_id` `ind_mom_neutral` `ind_ep_neutral` | stock\_master、daily\_price、financial\_statement | t 日晚間 | 第二批。產業代碼，LightGBM 以 categorical 處理，Ridge 不用；`tech_mom_250_20` 與 `fund_ep` 減同日同產業中位數 |
| 基本面 | `fund_days_to_ex_date` | dividend\_event | t 日晚間 | 第二批。距下一個已公告、有現金股利（`cash_dividend > 0`）的除息日的交易日數，無公告為 NaN；填息效應 |
| 市場層級 | `mkt_days_to_quarter_end` | trading\_calendar | t 日晚間 | 第二批。距季末的交易日數，投信作帳；全市場同值，排名模型不用 |

**處理步驟（每日對全市場執行）**

1. 定義 universe：特徵與預測對全部上市、上櫃普通股計算；每檔股票在 t 日屬於哪個市場，依 `market_history` 在 t 日的區間（由交易所外資持股頁重建），決定市場指數、當沖表與借券斷點；t 日沒有區間的股票日不在訓練 universe，但和其他 universe 外的列一樣照常算特徵、推論，市場取之前最近的一段，沒有就取之後的第一段，整檔都沒有區間的才用 `stock_master` 現在的市場；訓練 universe 另外排除停牌、上市未滿 60 個交易日、處置期間、全額交割、當日無成交、20 日成交金額中位數低於 1,000 萬元的股票；注意股不排除，以 `tech_attention_t` 當特徵；處置與全額交割期間來自 `disposition_event`。以下所有橫截面統計與 label 的百分位都只在當日訓練 universe 內計算；universe 外的股票套用同一天的轉換參數照常推論，`feature_daily` 與 `prediction` 都帶 `in_universe` 旗標，API 回傳時標示，事後評估不計入。
2. 可用時點對齊：每個特徵依表中「可用時點」以 merge\_asof 對齊到 t；基本面與總經用 announce\_date，weekly\_holding 用 publish\_date，美股用美國日曆日 ≤ t 的最近一個收盤，美國休市日自動取前一日；美元兌台幣用倫敦日 ≤ t−1 的最近一個收盤，週五與連假前也一樣。財報的 announce\_date：照實際上傳日，晚於法定截止日也延後；有更正紀錄的同樣照頁面上的上傳時間（可能是更正後重傳的時間，FinMind 存的也可能是更正後的數字）；同一種報表列了多次上傳、又沒有重編的取最早一次；有重編報表的期別用最後一次重編的上傳日；交易日 22:40 以後或非交易日上傳的算下一個交易日。財報數字：2013-Q1 起，revenue、net\_income、operating\_cf、debt\_ratio、roe 用該期報表當時公布的 MOPS 原值（彙總表、XBRL 檢視頁）覆寫 FinMind 重分類過的比較期數字，eps 只補 FinMind 的空值；gross\_margin、op\_margin 在 FinMind 已有的列保留 FinMind 值，清空與 MOPS 補列的規則見第 8 節第 5 條。前後期報表種類（個體、合併）不同時，revenue、gross\_margin、op\_margin、operating\_cf 不跨種類取前期，相關的年增與 TTM 為 NaN；月營收訊號讀到 2013-01 兩側的月份時為 NaN（2013 年起改為 IFRS 合併營收）。
3. 極值處理：個股層級特徵以當日 universe 內 1% / 99% 分位 winsorize；某一側的截斷界會讓 universe 全部變成同值時（稀疏的 0/1 與計數特徵），那一側改用 universe 的極值當界；市場層級特徵與 `ind_sector_id` 不做。
4. 標準化：個股層級特徵做當日橫截面 z-score，均值與標準差只用 universe 內的股票算。Ridge 與 logistic baseline 一律用 z-score 值；排名模型的 LightGBM 也用 z-score 值，原值跨年度有漂移，樹模型看不出相對當日市場的高低；絕對報酬、區間、波動、機率模型的 LightGBM 用原值；市場層級特徵進模型前先在訓練窗內依分位切成 5 級，Phase 4 比較後這四類都不加，見第 7 節模型表。市場層級特徵與 `ind_sector_id` 不做橫截面 z-score，市場層級特徵排名模型不使用。當日 universe 沒有離散度時 z 為 0，不是 NaN。
5. 缺值：Ridge 與 logistic 填當日中位數並加 `_missing` 旗標；LightGBM 保留 NaN。
6. 寫入前先套殘餘缺口規則：回補後仍只缺在之後會下市的股票上的欄位，對所有股票一律處理，不看個股之後是否下市。C：有第二來源的欄位，主來源缺時改用第二來源；目前是 `fund_rev_yoy`、`fund_rev_3m_yoy` 在月營收來源缺時改用財報算出的單季營收年增率 `fund_rev_yoy_q`。A：沒有第二來源的欄位，以及 C 補完仍分得出下市的欄位，在受影響月份對所有股票遮成 NaN；受影響月份是單看那個月，下市組的缺值比例或有值比例高於現存組的 1.5 倍、而且多出超過 5 個百分點的月份（兩個方向都算：缺得多會透露下市，C 只補到缺月營收的股票時，有值也會透露；剛好 1.5 倍或剛好 5 個百分點不算；C 的欄位另看補值：那個月下市組被第二來源補到的比例高於現存組 1.5 倍、而且多出超過 5 個百分點，或只有下市組被補到，也算，因為補到的列和第二來源同值，看得出是補的），用 `residual_coverage` 在遮罩前的特徵上判定一次後寫死，任一欄被遮的月份超過所查期間的 20% 時先回報再定；被遮的月份超過整段資料期（2010-01～2024-12，含研究期之前的訓練資料，共 180 個月）一半的欄位整欄不用，分母不用篩選窗，篩選窗內有值天數不到一半的照第 7 步不判定、保留。缺值分兩種：「來源缺」是特徵要讀的期別或欄位不在資料庫（FinMind 漏列、K18 清掉的毛利、2013 年以前沒有 XBRL；過了法定截止日還沒申報的也算，偏保守）；「PIT 可觀察的缺」是 t 日當下看得到原因：要讀的期別與欄位都在、由公式留空（去年同期 ≤ 0、淨利 ≤ 0、前後期報表種類不同），或金融業（`stock_master` 的產業別）沒有毛利率與營益率，涵蓋 `fund_gross_margin`、`fund_op_margin`、`fund_gm_yoy_chg`、`fund_opm_yoy_chg`。目前分類的是財報類特徵、由它算出的 `fund_ep` 與 `ind_ep_neutral`，以及月營收兩欄（月營收或補值用的季營收，其中一個的輸入都在就算 PIT 可觀察；月營收來源缺、季營收又因前後期報表種類不同而留空的，算來源缺），以及融資券、借券的 `chip_margin_chg_5d`、`chip_margin_usage`、`chip_short_ratio`、`chip_sbl_short_chg_5d`（融資、借券各自判斷，餘額欄是 NULL 就當作沒有那一列；停牌日照特徵沿用前一列，不另外判斷。要讀的交易日沒有那一列時，若上市、上櫃兩邊當天的交易所融資券表或借券表都已由補缺或每日更新成功寫入（`etl_run` 涵蓋那一天；股票的市場可能換過，所以兩邊都要有），代表交易所沒有列它（沒有資格、失去資格或停止信用交易），算 PIT 可觀察；有一邊沒有寫入的日子，從該股在資料庫裡那個來源的第一筆餘額起，或該股整段都沒有時，算來源缺，第一筆之前算 PIT 可觀察。股本缺算來源缺；列都在、由公式留空的，例如限額或融券餘額缺、限額為 0、融資餘額為 0，算 PIT 可觀察）；技術面的 `tech_mom_250_20` 與由它算出的 `ind_mom_neutral`，在上市未滿 250 個交易日、視窗比上市天數長時的缺，算 PIT 可觀察（上市天數與「上市未滿 60 個交易日」同一個算法）；其他欄位的缺值一律算來源缺。受影響月份與 Phase 3 的缺值驗收只算來源缺（受影響月份的有值方向只看真的有值，PIT 可觀察的缺不算有值），PIT 可觀察的缺當作真實訊號保留，不遮也不補；A 遮罩月份的缺值對所有股票一樣，驗收時也不算來源缺。前後期報表種類不同而留空的，`residual_coverage` 另外依下市組與現存組列出比例；下市組高於現存組 1.5 倍而且多出超過 1 個百分點時，可能藏著漏了合併報表的來源缺，先回報再定。Phase 3 報表用當下的資料庫分類缺值，partition 存的缺值和資料庫重算的不一致時（例如重建之後又回補），報表拒跑，要先重建。C、A 都是特徵層的衍生值，套用後的值當原值寫入；standard 表仍各自只有一個來源，第二來源不寫回 standard 表，不違反第 8 節第 5 點。Phase 3 報表列出每個 C 欄位下市組與現存組的補值比例、A 的遮罩月份，以及每個欄位 PIT 可觀察的缺值比例；`residual_coverage` 也列出每個 C 欄位在受影響月份以外的補值比例。兩處的補值比例，下市組高於現存組 3 倍或只有下市組被補到時，先回報再定。「被補到」是指主來源要讀的月份不在錄、值來自第二來源；在錄的值剛好等於第二來源不算（例如季末月的三個月營收合計剛好等於季營收）。寫入 `feature_daily`，只存 winsorize 前的原值與 `in_universe`，同時輸出 Parquet 分區檔（依年份）。步驟 3 到 5 的 winsorize、z-score、填值與 `_missing` 旗標在訓練與推論管線按日重算，不落表，參數一律由同一天的訓練 universe 算出。每日重算最近 60 個交易日的特徵與已可結算的 label（label 多算 21 個交易日，涵蓋往後看的窗口跨進這 60 天的那些），以 upsert 覆蓋，處理除權息與財報延遲入庫造成的歷史錯值。
7. 特徵驗收：Phase 3 產出每個特徵的單因子 rank IC，只用調參摺（36 個月窗下測試期落在 2014 到 2015 年的 8 個季摺；訓練窗改成 60 個月後也沿用這一組，不重篩）的訓練窗計算，篩選窗從第一個調參摺的訓練起點到最後一個調參摺的訓練終點，之後不再用新資料重篩；單因子 rank IC 對 5 日與 20 日的 `fwd_rank_pct` 各算一次；成對相關取日橫截面 Spearman 的平均，特徵依兩個 horizon 中較大的單因子 IC 絕對值由高到低保留，與已保留特徵的相關絕對值高於 0.95 的剔除；訊號型特徵在兩個 horizon 的 rank IC 絕對值都低於 0.005 才剔除，任一 horizon 的 IC 有值天數不到篩選窗一半時不判定、保留並記錄；條件型特徵（`size_*`、`liq_*`、`tech_vol_*`、`risk_*`、`fund_days_since_fs`、`ind_sector_id`、`*_days_to_*`）不受 IC 門檻約束。市場層級特徵不篩選，記在同一列 `model_run` 的 params.unscreened，非排名模型照第 4 步加回。保留清單寫入 `model_run.feature_list`；剔除清單在 Phase 4 定案後不得依 2025 年起的 holdout 結果再改。第二批特徵每加一組重跑同一套篩選與 walk-forward，總體 OOS rank IC 沒有提高就不留；「提高」指 seed 0、3、6 的平均至少高 0.001，而且 3 個 seed 裡至少 2 個較高（使用者 2026-11-10 拍板）。

**Label 定義**

- 所有 t+N 都是 `trading_calendar` 上的交易日，不是日曆日。
- `fwd_ret_5d_open`、`fwd_ret_20d_open` = 還原收盤價(t+N) / 還原開盤價(t+1) − 1，是訓練、評估與回測一律使用的版本；t+1 無法以開盤價進場的列標記 `untradable_t1 = 1`，包括 t+1 開盤即漲停（漲停判定同特徵表）、t+1 停牌或無成交、上市已在 t 結束，訓練與回測都跳過。5 日由絕對報酬模型直接回歸，20 日取分位數模型的 q50。
- `fwd_ret_5d`、`fwd_ret_20d` = 還原收盤價(t+N) / 還原收盤價(t) − 1，收盤對收盤版本，只存 `label_daily` 供對照與事後分析，不進訓練。
- `fwd_rank_pct` = 當日訓練 universe 內、`untradable_t1 = 0` 的列的 `fwd_ret_open` 橫截面百分位（0 到 1），等當日的 t+horizon 進到資料內、全部列都到期後才計算；排名模型預測這個。
- `fwd_vol_20d` = t+1 到 t+20 共 20 個交易日還原日報酬的標準差，年化 ×√252；波動率模型的目標。
- `fwd_up_5pct` = `fwd_ret_20d_open` 是否大於 5%（0 或 1）；原本是分類模型的目標，Phase 4 比較後改用相對大盤版（見第 7 節三類預測目標）。
- horizon 內下市的股票：以最後一個成交日的還原收盤價當作 t+N 的價格計算，該列保留，不得設為 NaN 丟棄。

## 7. 建模與驗證

同一組特徵訓練三類目標：排名（相對強弱）、絕對報酬（回歸）、區間與機率（分位數、波動率、分類）。每類目標都以 Ridge 或 logistic 為 baseline、LightGBM 多 seed bagging 為主模型、XGBoost 為可選對照；全部用 walk-forward 驗證，摺分研究期（2014 到 2024）與最終 holdout（2025 起），所有設計選擇在研究期決定、holdout 只跑一次；各類目標有自己的關卡與目標值，任何一類沒過關卡就不上線。各類輸出必須互相一致：20 日預測報酬取分位數模型的 q50，價格區間以分位數區間為主，波動率區間只做 fallback 與 sanity check。所有 label 以 t+1 還原開盤價進場（fwd\_ret\_\*\_open），訓練、評估、回測一致，close-to-close 版只存 label\_daily 供對照；untradable\_t1 = 1 的列訓練與回測都跳過。

| 模型 | 角色 | 輸入 | 起始參數 | 輸出 |
| --- | --- | --- | --- | --- |
| Ridge、logistic | baseline，可解釋 | 當日橫截面 z-score 特徵，缺值已填並帶 `_missing` 旗標 | Ridge 的 alpha、logistic 的 C 都由訓練窗內的時間序列 CV 在 0.1 到 100 之間選 | 預測分數、係數表（每個特徵的權重與符號） |
| LightGBM | 主模型 | 排名目標用 z-score 值；絕對報酬、區間、波動率、機率目標用原值，保留 NaN（見第 6 節處理步驟）。市場層級特徵每日全市場同值，樹會靠它認出日期、等於 in-sample 看到大盤走勢，所以先在訓練窗內依分位切成 5 級再進模型；Phase 4 比較有、無市場特徵兩版，OOS 指標沒有提高就不用；實驗另加「逐摺差 NW t 達 2」才算提高。研究期結果（篩選 `p3-final-20261109`）：絕對報酬 MAE 0.0378 對 0.0377（t +1.39）；區間 pinball 0.0811 對 0.0806（t +2.11，加了反而變差）；波動率 MAE 0.0978 對 0.0981、QLIKE 0.3398 對 0.3446 略好但不顯著（t −0.34、−0.81）；機率 AUC、Brier 變差。四類都不加 | objective 依目標（見下表）；num\_leaves 31、learning\_rate 0.02、feature\_fraction 0.7、bagging\_fraction 0.8、min\_data\_in\_leaf 200、lambda\_l2 1；early stopping 100 輪，驗證集與停止指標見 walk-forward 一節 | 預測分數、feature importance（gain）與各摺前 10 名的變動 |
| LightGBM seed bagging | 正式輸出 | 同 LightGBM | 同一組參數、同一輪數訓 5 個 seed（`seed`、`bagging_seed`、`feature_fraction_seed` 不同），排名與回歸平均原始分數，機率模型平均各 seed 的機率，校準套在平均後的輸出；研究階段的 walk-forward 用單 seed，只有最終驗證跑與正式上線用 5 seed | `prediction` 表的 score、ret\_pred、q10/q50/q90、vol\_pred、prob\_up\_5pct |
| XGBoost（可選） | 對照模型，只用來抓實作錯誤 | 同 LightGBM | tree\_method hist；max\_depth 6、eta 0.02、subsample 0.8、colsample\_bytree 0.7、min\_child\_weight 200（對應 LightGBM 的 min\_data\_in\_leaf 200） | 與 LightGBM 的 rank IC 差異超過 20% 時要追查實作；不進 ensemble，不是上線條件 |
| Ensemble（可選） | 排名目標的最終分數 | Ridge 與 LightGBM bagging 的分數 | 兩者各自轉當日 rank 後平均；walk-forward 總體 rank IC 不高於 LightGBM 單獨就不用 | `prediction` 表的 score |

調參範圍只有四個：num\_leaves 15 到 63、min\_data\_in\_leaf 200 到 2,000、lambda\_l2 0 到 10、feature\_fraction 0.5 到 0.8；learning\_rate 固定 0.02，輪數由 early stopping 決定。

**三類預測目標**

| 目標 | label | LightGBM objective | 輸出欄位 | 展示層換算 |
| --- | --- | --- | --- | --- |
| 排名 | `fwd_rank_pct`，5 日與 20 日各訓一個 | regression（L2）。lambdarank（`fwd_rank_pct` 分成 10 級整數相關度、以日期為 group）是備選，Phase 4 比較後不用：研究期 rank IC −0.056 對 0.133（篩選 `p3-final-20261108b`，2026-11-09 財報修正前的資料）；NDCG 截斷在前 30 名、增益 2^級−1，主要獎勵把漲幅前段的股票排上前面，模型因此把高波動股排到前面 | score, rank, pct | 不換算 |
| 絕對報酬 5 日 | `fwd_ret_5d`\_open | regression（Huber，對極端報酬較穩）；LightGBM 的 alpha 即 Huber delta，預設 0.9 對 5 日報酬約等於 20 個標準差、實際上就是 L2；設為訓練窗內 fwd\_ret\_5d\_open MAD 的 1.5 倍，約 0.05 | ret\_pred（horizon 5） | 預測價 = 當日收盤價 × (1 + ret\_pred)，每檔用自己的收盤價 |
| 絕對報酬 20 日 | 不另訓模型 | 取報酬區間模型的 q50 | ret\_pred（horizon 20）= q50 | 同上；預測價必然落在價格區間內 |
| 報酬區間 | `fwd_ret_20d`\_open | quantile，alpha = 0.1、0.5、0.9 各訓一個模型；預測後逐列排序，保證 q10 ≤ q50 ≤ q90 | q10, q50, q90 | 價格區間 = 收盤價 × (1 + q10) 到 收盤價 × (1 + q90)；這是展示用的主區間 |
| 波動率 | log(`fwd_vol_20d`) | regression（L2）對 log 值，預測取 exp 還原；最重要的輸入是 `tech_vol_20d`、`risk_idio_vol_60d`（`tech_vol_60d` 和 `risk_idio_vol_60d` 的日均 Spearman 是 0.963，Phase 3 篩選時剔除） | vol\_pred（年化） | 80% 區間 = 收盤價 × exp(±1.28 × vol\_pred × √(20/252))，做 sanity check；fallback 改以預測報酬為中心（見輸出一致性）；不預設展示 |
| 上漲機率 | 相對大盤版：`fwd_ret_20d`\_open 減 `TAIEX_TR`（加權股價報酬指數，含息，和還原報酬同一口徑）收盤 t 到收盤 t+20 的報酬後是否大於 5%；個股從 t+1 開盤起算、大盤從 t 收盤起算，差一段隔夜跳空，屬已知近似（Phase 4 比較後由 `fwd_up_5pct` 改用，見評估指標的機率一條）。logistic baseline 也用這個 label。以下跟著改的地方在設計凍結時決定：頁面與 API 的說法、輸出一致性的 q50 檢查、`label_daily` 要不要存這一欄、上線後事後評估與告警（`model_monitor` 的 Brier、AUC）和每月門檻 (3) 讀哪個 label | binary | prob\_up\_5pct | 直接顯示機率；ECE 高於 0.05 時以 early-stopping 驗證集做 isotonic 校準再上線 |

分類目標的 baseline 用 logistic regression；分位數目標的 baseline 用波動率區間（q50 = 0，q10 與 q90 = ±1.28 × tech\_vol\_20d × √(20/252)），訓練窗內的無條件歷史分位數只當 sanity 下限，打不過它代表實作有錯；波動率目標的 baseline 是直接拿 `tech_vol_20d` 當預測。絕對報酬與 q50 的輸出會明顯集中在 0 附近，這是個股報酬可預測性低的正常現象，不要為了拉開分布去調整。

**輸出一致性**

- 20 日的預測報酬與價格區間來自同一個分位數模型，不會出現預測價落在區間外的情況。
- 波動率區間與分位數區間的寬度比落在 0.5 到 2 之外的股票，每日寫入 `data_quality_log`，severity 為 warning；比例超過 10% 要追查。
- 分位數模型的覆蓋率告警成立（見模型管理與上線後監控）的期間，價格區間改用以預測報酬為中心的波動率區間：收盤價 × (1 + ret\_pred ± 1.28 × vol\_pred × √(h/252))，預測價仍落在區間內；缺 vol\_pred 時用 `tech_vol_20d`；API 回應附 `interval_source` 欄位標明來源。
- 5 日沒有分位數模型：API 用同一式（h = 5，以 √(5/252) 縮放）給波動率區間並標 `interval_source = volatility`；缺 vol\_pred 時改用 tech\_vol\_20d 並標 `tech_vol_20d`，缺 ret\_pred 時以 0 為中心；頁面依「波動率區間不預設展示」，5 日不畫區間。
- 上漲機率與 q50 矛盾的列（prob\_up\_5pct 大於 0.5 但 q50 小於 5%，或 prob\_up\_5pct 小於 0.5 但 q50 大於 5%）每日計數，比例超過 10% 要追查。
- `ret_pred` 與 q10/q50/q90 以 t+1 還原開盤價為基準，展示層仍以最新收盤價換算，差額是隔夜跳空，屬已知近似，不另修正。
- 訓練 universe 外的股票照常輸出，API 附 `in_universe` 欄位，事後評估不計入。

**Walk-forward 驗證**

1. 訓練窗 60 個月（Phase 4 依第 3 條比較後選定；比較前的預設是 36 個月），之後留 20 個交易日 embargo（訓練資料的 label 視窗不得跨進測試期），再測試 3 個月；每 3 個月往前滾一摺，所有測試期串起來算總體指標，並另報每摺指標。摺的編號依測試期起點從 2014 年第一季起算；資料從 2010 年起，60 個月窗的第一摺是第 5 摺（測試期 2015 年第二季），到 2026 年 11 月共 47 摺：調參 3（第 5–7 摺）、研究期 36（第 8–43 摺）、holdout 8。
2. 研究期與最終 holdout：測試期落在 2014 到 2024 年的摺是研究期，窗長、取樣、ensemble、lambdarank、label 變體、市場特徵有無等所有設計選擇只在研究期上比較與決定；2025 年起的摺是 holdout，設計定案前不得執行，定案後只跑一次，其結果是對外報告的數字。holdout 結果不佳只能記錄，不得回頭改設計再跑，否則 holdout 失效。holdout 由 `holdout` 指令執行，正式跑之前先在一個研究摺上用同一支擬合函式演練（不 claim）；claim 之前先擋掉跑不完的情況（沒有摺、某一摺還沒有已到期的 label、沒有特徵篩選紀錄），中斷時用同一個 run_key 續跑沒寫入的摺，設計或特徵和開跑時不同就拒絕續跑；正式 run 要等 holdout 每一摺都寫入，且只建在 holdout 跑過的那個設計與特徵上，`build_run` 比對 holdout 的 `model_run` 記下的設計，不同就拒建。每摺的指標只算測試期起點到 `eval_end`：時期交界（調參摺接研究期、研究期接 holdout）前的摺，`eval_end` 是下一個時期第一個測試日往前 21 個交易日，讓 label 視窗不跨進下一個時期；其他摺的 `eval_end` 就是測試期終點。
3. 訓練窗長度是參數，不是定案：Phase 4 在研究期同一組摺上比較 36 個月、60 個月、expanding 加時間衰減權重（半衰期 24 個月），以總體 OOS rank IC 選定後寫回本節。60 個月窗的第一摺只能從 2015 年起，三種窗長一律只比 2015 年起的摺。研究期結果（每 6 個月一摺的粗摺 4–21，seed 0、3、6篩選 `p3-final-20261108b`，2026-11-09 財報修正前的資料）：36 個月 0.1231、60 個月 0.1325、expanding 0.1284，選定 60 個月。
4. 訓練列取樣：20 日 label 每檔每天一列，相鄰 20 列的 label 高度重疊，有效樣本遠低於列數。Phase 4 粗跑比較「全部列」與「每 5 個交易日取一列」兩版，後者訓練快 5 倍、減少對重疊列的過擬合，OOS rank IC 不輸就採用；預測與評估仍每日對全市場做。研究期結果（篩選 `p3-final-20261108b`，2026-11-09 財報修正前的資料）：每 5 日一列 0.1315、全部列 0.1325，輸了，維持全部列；差距和 seed 間標準差同一量級，在 `p3-final-20261109` 上重跑後以重跑為準。
5. Early stopping 的驗證集固定是訓練窗最後 6 個月，與前段訓練資料之間同樣留 20 個交易日 embargo；排名目標用每日 rank IC 當停止指標（自訂 feval，每 10 輪算一次），其他目標用各自的 loss。取得最佳輪數後，以整個訓練窗重訓固定輪數再產出測試期預測；排名目標的重訓輪數取最佳輪數與 200 的較大者（Phase 4 比較「最佳輪數」「至少 200 輪」「調參摺的固定輪數」三版：0.1325、0.1335、0.1325（篩選 `p3-final-20261108b`，2026-11-09 財報修正前的資料）；20 日模型 54 個摺×seed 裡有 10 個最佳輪數不到 100；差距和 seed 間標準差同一量級，在 `p3-final-20261109` 上重跑後以重跑為準）。測試期永遠不得用於 early stopping 或任何選擇。
6. 超參數只在研究期中測試期落在 2014 到 2015 年的摺（調參摺；36 個月窗下季摺是前 8 摺、每 6 個月一摺的粗跑是前 4 摺，60 個月窗下只剩季摺第 5–7 摺、粗跑第 3 摺）聯合調一次，之後凍結；調參摺的指標已不是純 OOS，不計入總體指標，另行標註。正式上線後每年重調一次，只用重調時點之前的資料。單摺的 IC 噪音太大，不逐摺調。
7. 永遠不用隨機 K-fold；任何以隨機切分得到的指標一律視為無效。
8. 算力：完整 walk-forward 約 47 摺（60 個月窗：調參 3、研究期 36、holdout 8），每摺 8 個 GBDT 模型（排名 2、絕對報酬 1、分位數 3、波動率 1、機率 1），單 seed、全部列粗估一到兩天，每 5 日取樣約五分之一。調參與窗長比較先用每 6 個月一摺粗跑，定案後再跑完整版；每摺的預測結果快取成 Parquet，指標重算不重訓。

**評估指標**

- 排名：日頻 rank IC（每日預測分數與 `fwd_rank_pct` 的 Spearman 相關）的均值、標準差與 ICIR（均值 / 標準差）。20 日 label 重疊使日 IC 序列自相關，顯著性用 Newey-West t 值，lag 等於 horizon。關卡（研究期，不含調參摺）：OOS rank IC 均值大於 0 且 NW t 值大於 2；逐摺比較 LightGBM 的 IC 高於 Ridge 的摺數達 60% 以上。目標值（日頻，初始設定，Phase 4 第一輪結果後校正）：20 日 IC 0.03、ICIR 0.3；5 日 IC 0.02、ICIR 0.2。
- 中性化 IC：每日把預測分數對產業 dummy（`stock_master` 產業代碼）、`size_log_mcap`、`risk_beta_250d`、`tech_vol_20d` 做橫截面迴歸取殘差，再算殘差與 `fwd_rank_pct` 的 rank IC。中性化 IC 接近 0 代表訊號只是產業、規模或 beta 曝險，十分位組合會變成產業賭注；Phase 4 據此決定排名 label 是否改為殘差化排名（第 10 節待決）。
- 排名穩定性：報每摺與每年的 IC、IC 大於 0 的摺數比例、前 10 名特徵重要度在各摺的變動；另依 `liq_dollar_vol_med_20d` 當日三分位分組各報 IC，台股小型股的 IC 常不可交易，大型股那一組的 IC 才是可用訊號的下限。
- 十分位組合：每日在 universe 內依分數分 10 組、等權；持有期等於 horizon，用 overlapping tranches（20 日 horizon 即 20 組各持有 20 日，每日只換其中一組），不每日全換；進場價用 `fwd_ret_*_open`，`untradable_t1 = 1` 的列跳過。主報表是第 10 組減 universe 等權的月化超額報酬與最大回撤；多空（第 10 組減第 1 組）為輔，因台股小型股放空不可行。另報只取成交金額前三分之一股票重做的十分位版本。
- 交易成本：手續費 0.1425% 買賣各一次、證交稅 0.3% 賣出時，只對實際換手的部位扣；十分位報酬一律報扣成本後的版本並附年化換手率。
- 絕對報酬（5 日）：MAE 與 RMSE 對上「永遠預測 0」與「預測過去 60 日平均報酬」兩個 baseline。關卡不是單純較低：逐摺算模型與 baseline 的 MAE 差，差值序列的 Newey-West t 值大於 2 才算過，因為差距會在小數第四位、單純比大小是噪音。另報方向準確率（預測符號與實際符號一致的比例，label 為 0 的列不計）及其對 50% 的 NW t 值。20 日的 ret\_pred 就是 q50，隨區間目標評估。
- 區間：主指標是 pinball loss（三個 alpha 加總）必須低於波動率區間 baseline（q50 = 0，q10 與 q90 = ±1.28 × `tech_vol_20d` × √(20/252)），同樣用逐摺配對差的 NW t 值大於 2 判定；無條件歷史分位數只當 sanity 下限。總體覆蓋率在 75% 到 85% 之間，且每一摺都在 70% 到 90% 之間，過窄或過寬都算失敗；另報依 `tech_vol_20d` 五分位分組的覆蓋率與區間平均寬度。
- 波動率：MAE 與 QLIKE，對上「用 tech\_vol\_20d 當預測」的 baseline。
- 機率：每日 AUC 的平均（不用全期混合 AUC，會被市場漲跌主導）、Brier score 對上 logistic baseline。校準看全區 reliability 圖（10 格）並報 ECE；ECE 高於 0.05 時以 early-stopping 驗證集擬合 isotonic 校準，套到測試期重驗，不得用測試期擬合。Phase 4 另比較相對大盤的標籤（`fwd_ret_20d_open` 減 TAIEX 同期報酬是否大於 5%），日均 AUC 較高就改用。研究期結果（篩選 `p3-final-20261109`，當時減的是價格指數 `TAIEX`）：相對版 0.5977、原版 0.5837（逐摺差 NW t +2.44），改用相對版；兩版各對自己的 label 評估。價格指數除息時會跌，和含息的還原報酬口徑不同，所以改減 `TAIEX_TR` 後重跑，以重跑結果為準。
- 洩漏警訊：兩個條件任一成立，就先當成洩漏追查，不當成好消息。(1) 研究期（不含調參摺）任一摺 LightGBM 的 OOS 日 IC 均值比同摺 Ridge 高 0.05 以上（在兩者都有分數的同一批列上算），5 日、20 日都適用；沒有同摺 Ridge 可比的摺標「未判定」，不算沒響。(2) 第 8 節的標籤打亂測試（20 日 Ridge），OOS rank IC 的絕對值達 0.02；報表這次沒跑打亂測試，或算不出 IC 時，也標「未判定」。IC 均值本身不設上限。

**模型管理與上線後監控**

- 每次訓練寫一列 `model_run`，含參數、特徵清單、訓練區間、指標、seed、套件版本、特徵 Parquet 的 hash；模型檔以 `run_id` 命名存檔。`status` 取值 candidate、active、degraded、retired。
- 每月重訓：新模型 N 以當下為訓練窗終點；同時訓一個設定相同的影子模型 N′，訓練窗截止在 6 個月加 20 個交易日之前，用來在最近 6 個月上做 OOS 檢查。上線門檻全部是對 N 的健全性檢查，不拿 N′ 跟舊模型比 IC（兩個同配方模型比 6 個月 IC 是噪音，且 N′ 看不到 N 新加入的最近 6 個月資料）：(1) N 與現役模型在最新交易日的預測 Spearman 相關 ≥ 0.7；(2) N 與現役模型 gain 前 10 名特徵重疊 ≥ 6 個；(3) N′ 最近 6 個月 OOS 達絕對門檻：排名 rank IC > 0、區間覆蓋率在 70% 到 90%、機率 Brier 不高於 logistic baseline、波動率 QLIKE 不高於 `tech_vol_20d` baseline、5 日 MAE 不高於零預測；(4) N 訓練窗起點到 as_of（N 最後幾列訓練資料的 label 讀到 as_of 的價格）之間，`data_quality_log` 無 severity = error 的紀錄。任一不過，舊模型續用、告警並記錄原因，是否強制換模由人工決定：強制換模只能跳過這四項的不過（模型組沒建完整、holdout 尚未 claim 都不能跳過），必須寫原因，原因與沒過的門檻記進 N 的 `model_run.params` 並告警。換模把 N 改 active、該類目標原本服務中的版本改 retired，並記下上線日。
- 線上預測只用 `model_run` 中 `status = active` 的版本，每類目標各一個。
- 事後評估：每日對 horizon 已到期的預測，從 `prediction` 與 `label_daily` 算實際 rank IC、覆蓋率、區間寬度、Brier、AUC、MAE（報酬或波動率，依 run 的目標）、方向準確率（同樣不計 label 為 0 的列），只算 universe 內且 `untradable_t1 = 0` 的列，寫入 `model_monitor` 表；服務層的「歷史預測準確度」讀這張表。
- 告警：60 個交易日滾動 rank IC 低於門檻、覆蓋率出 70% 到 90%、Brier 高於 logistic baseline，任一成立就告警並把該模型 `status` 改 degraded，API 回應附警示。被判的是當天服務中的版本，當天沒有預測列就不判；60 個交易日的窗口串接該目標歷來服務過的版本，每列只算各自版本的樣本外日子。換模後有 5 個交易日的寬限期，從 max(樣本外界線, 上線日) 起算，期間超標只告警、不降級；上線日之前補寫的日子也不判降級。是否停用由人工決定；degraded 改回 active 只能人工恢復（`restore_run`），必須寫原因，降級期間記成一段警示期；恢復後該目標同一 horizon 的窗口從恢復日重新起算（之後換上的版本沿用），累積滿 60 列才重新自動降級，重設期間串接的 60 列仍超標時只告警、不降級；之後補跑恢復日之前的監控，也不會把恢復的版本降回去。滾動 IC 的門檻不用固定 0：20 日 horizon 下 60 個交易日只有約 3 個獨立樣本，健康模型（IC 0.03）也會有約三成時間滾動均值小於 0；門檻取研究期 walk-forward 全部 60 日滾動 IC 的 5% 分位，寫進 `model_run.params`，每次重訓更新。
- 第 8 節的標籤打亂測試在每次完整 walk-forward 前執行一次。

## 8. 必須遵守的規範與已知陷阱

這一節是硬性規則。寫任何 ETL、特徵或回測程式之前先讀一遍；違反其中任何一條，回測結果都不可信。

1. **Point-in-time 對齊**：基本面特徵一律用 `announce_date` 對齊。查 t 日特徵時，只能取 `announce_date <= t` 的最新一期財報或月營收。Q1 財報在 5/15 前公布，不能在 4 月就讓模型看到。月營收在每月 10 日前公布。做法：特徵計算用 `merge_asof(by=stock_id, on=date, right_on=announce_date)`。
2. **還原權息**：label 一律用還原價，否則除息日會出現假跌幅，模型會學到「除息前賣出」這種假訊號。除權息資料來自 `dividend_event`，`adj_factor` 回溯調整。
3. **存活者偏差**：`stock_master` 必須包含已下市股票，訓練集不得過濾掉它們。只用現存股票訓練，模型只學到「活下來的公司長什麼樣」。
4. **不看未來**：台股特徵只能用 `<= t` 日收盤的資料，含移動平均的視窗也是；美股只能用美國日曆日 ≤ t 的收盤，匯率只能用倫敦日 ≤ t−1 的收盤，兩者都早於 t+1 開盤進場。回測進場價用 t+1 還原開盤價，不用 t 日收盤價。股票在 t 日屬於哪個市場、有沒有在上市櫃交易，讀 `market_history` 那天的區間，不用 `stock_master.market` 的現值（現值是之後才知道的事，例如之後才上櫃轉上市）；那天沒有區間的不在訓練 universe，算特徵用的市場照第 6 節處理步驟 1 取（整檔都沒有區間的才用現值）。
5. **來源衝突**：每類資料只有一個 primary source，secondary 只做比對。同一欄位在兩個來源數字不同時寫 `data_quality_log`，不得取平均或混用。例外只有下面幾條，其他欄位、其他情況一律照前一句。這幾條都不取平均：每個欄位一次只採用一個來源，兩邊的 raw 都保留；除了下面註明的，兩邊數字不同照樣寫 `data_quality_log`。
   - **財報（FinMind 回補的季別，也就是每日財報掃描接手的那一季之前；plan K18）**：2013-Q1 起，同一檔、同一季、同一種報表（個體或合併）的 `revenue`、`net_income`、`operating_cf`、`debt_ratio`、`roe` 以 MOPS 當期原值為 primary，MOPS 空或只有另一種報表時用 FinMind；`eps`、`gross_margin`、`op_margin` 以 FinMind 為 primary，`eps` 空時用 MOPS。FinMind 已有的列，兩個利益率不用 MOPS 補；只判 Q2 到 Q4（Q1 的累計就是單季）：FinMind 的值不是那一季自己的（跨季累計算出的）就清空，查不到同一種報表的 MOPS 數字而 FinMind 的 `revenue` 也是空的時同樣清空（2013 年前沒有 MOPS 頁面，`revenue` 空就清），判斷不了的留著。掃描接手那一季起，財報由 MOPS 直接寫入，照第 5 節「已有的列只新增不改」。
   - **MOPS 補 FinMind 沒有的月份與季（plan K17，2013-Q1 起）**：FinMind 沒有某檔某期的月營收或財報列時，整列用 MOPS 歷史彙總頁、XBRL 檢視頁或單一公司月營收查詢（新站 JSON API，2014-01 起）新增，財報的兩個利益率由 MOPS 的損益表算；這一步不改 FinMind 已有的列（已有列的覆寫與清空照上一條），數字不同記 `revenue_history_mismatch`、`statement_history_mismatch`、`statement_viewer_mismatch`；單一公司查詢只問沒有列的月份，`revenue_company_mismatch` 記補進來的與查不到的月份。每日掃描接手之後的期別（第一次成功的每日掃描那天還在看的最早一期起）由掃描以第一次看到的 session 記公告日，MOPS 的這幾種回補都不補。
   - **交易所逐日表補歷史缺值（plan K17c）**：FinMind 歷史回補缺的部分，用證交所、櫃買的逐日全市場表補，只寫當天在 `daily_price` 有價的代號，已有的值不改：`daily_chip` 的外資持股比、融資餘額、融資限額、融券餘額與借券賣出餘額是那天沒有列就新增、欄位空的才補（各欄分開看）；`daily_valuation` 的本益比、股價淨值比、殖利率是那天整列沒有才新增；`shares_outstanding` 只補每檔第一筆已存股數之前的日子（界線取第一次補缺時的值，沒有已存股數的股票整段都補），只寫股數變動的那天，股數只和界線那天的已存股數比對，跑完刪掉和補上的股數重複的已存列（生效的股數不變，刪掉的列記在 log）。`stock_master` 的產業以 FinMind 股票清單為 primary；空的用交易所依產業別的季底收盤行情補（多季看到時取最後一次，新上市的代號每季再補），不比對已有的，也不記差異。
   - **外資持股比的 0（plan K21）**：只看普通股（不含存託憑證）；交易所的值讀已存的 raw 頁、不重抓，同一天有幾份時取最新的。FinMind 存 0、交易所當天的值大於 0 時，只有兩種情況改成交易所的值：同一檔依日期排序的已存非空值裡連續 20 天以上的 0，90% 以上的日子交易所頁面有這檔，且其中 90% 以上大於 0（整段裡大於 0 的日子都改）；連續不到 20 天的 0（大於 0 的那幾天改）。其餘長段的 0 不改。每改一列寫一列 `fi_holding_zero_replaced`，記舊值、新值與交易所 raw。
   - **除權息事件（plan 2.7）**：FinMind 回補的列 `adj_factor` 是空的時，用交易所除權除息、恢復買賣結果表的因子補；已有的金額、因子和交易所不同時不改，記 `dividend_source_mismatch`，空的因子照樣補。交易所寫的列 `announce_date` 留空；已存的 FinMind 回應裡有公告日的由 `reapply_dividend_dates` 補（只補空的），其餘等 plan 2.7b 的預告表。FinMind 股利回補只到 2026-10-30：它對已存的列會覆寫金額與因子，之後的日子不用 FinMind 重跑。
   - **市場歷史（`market_history`）**：歷史以每日外資持股頁為 primary，每日以當天報價接著寫；兩個交易所的頁都在的日子以頁面為準，頁面看不到的日子（缺一邊或兩邊都沒有頁）保留報價寫的界線。重建改到既有的段時記 `market_history_rebuilt`；只差第一段起日的（新股的頁面常比報價早一天列出）另計 `start_shifted`。
6. **上市、上櫃分流**：證交所與櫃買中心格式不同，adapter 分開寫，normalize 後才合併。興櫃股票無漲跌幅限制，第一版排除。
7. **交易日曆**：以證交所公告為準，含補班日與颱風假。所有日期運算走 `trading_calendar`，不用 `pandas.bdate_range`。
8. **單位**：成交量以「股」入庫，金額以「元」入庫。證交所與 FinMind 的成交量都是股；部分網頁與 CSV 以「張」顯示，normalize 時乘以 1,000。
9. **時區**：所有時間戳用 `Asia/Taipei`。yfinance 回傳的美股日期是美國日曆日；特徵列 date = t 的美股欄位取美國日曆日 ≤ t 的最近一個收盤，於台北 t+1 清晨取得，不往前推也不往後推。
10. **速率限制**：每個 adapter 內建 sleep 與指數退避。FinMind 超限會封鎖約 1 小時，證交所會封 IP；大量回補一律分批並記 checkpoint。
11. **隨機切分無效**：任何以隨機 K-fold 得到的指標都不得寫入 `model_run`；驗證只能用 walk-forward。
12. **授權**：FinMind 限非商業用途；證交所行情對外再提供需資訊使用授權。第一版只供個人研究，對外開放前要重新評估第 3 節的來源。

**對應的自動化測試（必須在 CI 內）**

- PIT 洩漏測試：故意把某檔財報的 `announce_date` 設到未來，確認該期數字不會出現在之前任何一天的特徵中。
- 集保 PIT 測試：週五 t 的 `chip_*_1w` 特徵不得用到 data\_date = t 的那一期，只能用 publish\_date ≤ t 的最近一期。
- 還原價測試：取一檔有大額配股的股票，確認除權日前後的還原報酬接近 0。
- 存活者測試：確認訓練集中存在 `delist_date` 非空的股票。
- 下市 label 測試：取一檔 horizon 內下市的股票，確認其 `fwd_ret` 非 NaN 且以最後成交價結算。
- 交易日測試：確認 `feature_daily` 的日期集合是 `trading_calendar` 的子集。
- 標籤打亂測試：把 label 在每個日期內跨股票隨機置換後重跑一摺 walk-forward，OOS rank IC 必須接近 0；明顯大於 0 代表特徵或切分有洩漏。
- 漲跌停測試：2015-06-01 前後各取一檔有漲停的股票，確認 `tech_limit_up_t` 依 7% 與 10% 限幅及升降單位捨去的判定一致；取一列 t+1 開盤漲停的資料確認 `untradable_t1 = 1`。
- 美股對齊測試：date = t 的 `mkt_sox_ret_5d` 用到美國日曆日 t 的收盤、不得用到美國 t+1 的收盤；美國休市日取前一個收盤。

## 9. 開發階段與里程碑

七個階段依序進行，每個階段有明確驗收條件，未通過不進下一階段。Phase 0 到 2 先把資料做對，Phase 3 才開始建模；Phase 6 為可選。工時為粗估，依可投入時間調整。

[圖：開發路線圖 · 7 個階段、6 個驗收關卡（匯出未保留）]

讀法：由上往下依序執行，菱形是驗收關卡，右側字是通過條件；Phase 3 是資料轉建模的分界點。

| 階段 | 交付物 | 驗收條件 | 預估 |
| --- | --- | --- | --- |
| Phase 0 專案骨架 | Django 專案、PostgreSQL、Docker Compose、`apps/etl` adapter 介面、`stock_master` 與 `trading_calendar` | `manage.py fetch --source twse --dataset daily_price --date 2026-10-28` 可寫入 raw 與 standard 表；測試框架可跑 | 1 週 |
| Phase 1 歷史回補 | FinMind adapter；2010 年起的日價量、籌碼、月營收、財報、股利全部入庫（FinMind 缺的下市股資料見第 3 節；股權分散表不回補歷史，Phase 2 起由集保每週累積）；checkpoint 續跑 | 抽樣 20 檔、每檔 3 個年度與證交所逐日核對無誤差；表中存在已下市股票 | 1 到 2 週 |
| Phase 2 每日增量與品質 | 證交所、櫃買中心、集保、MOPS、yfinance adapter；Celery 排程；`data_quality_log`；告警 | 連續 5 個交易日自動更新，無人工介入；品質檢查能攔下人為注入的錯誤資料 | 1 到 2 週 |
| Phase 3 特徵與 baseline | `feature_daily`、`label_daily`、PIT 對齊、三類目標的 label 與 baseline（Ridge、logistic、零預測）、walk-forward、IC 報表 | 第 8 節的九個自動化測試全部通過；IC 報表產出，含每個特徵的單因子 rank IC；成對相關高於 0.95 的特徵只留一個，低 IC 的訊號型特徵依第 6 節處理步驟 7 的規則剔除並記錄 | 1 到 2 週 |
| Phase 4 主模型 | 三類目標（排名、絕對報酬、區間與機率）的 LightGBM 訓練與 5 seed bagging、XGBoost 對照（可選）、walk-forward 含窗長比較、ensemble、`model_run` 與 `prediction` 與 model\_monitor 表、每月重訓與影子模型門檻、事後評估工作、第二批特徵逐組加入並比較、最終 holdout 一次性執行 | 研究期（2014 到 2024，不含調參摺）LightGBM 的 OOS rank IC 大於 0 且 Newey-West t 值大於 2，逐摺贏過 Ridge 的摺數達 60%；中性化 IC 與依成交金額分三組的 IC 報表產出；十分位報酬（持有期等於 horizon、只扣實際換手成本）報表產出；5 日絕對報酬 MAE 以逐摺配對 NW t 大於 2 優於零預測；區間 pinball loss 優於波動率區間 baseline，總體覆蓋率在 75% 到 85% 之間且每摺在 70% 到 90% 之間；分位數不交叉；機率 ECE 低於 0.05；設計定案後 2025 年起的 holdout 只跑一次並記錄；model\_monitor 有資料 | 2 到 3 週，含 walk-forward 計算時間 |
| Phase 5 服務層 | DRF API：`/predictions`、`/stocks?q=`（代號前綴或名稱包含，最多 20 筆，下市股也列並標示）、`/stocks/{id}/features`、`/stocks/{id}/history`、`/models`、`/data-quality/corrections`（依套用、重建或資料本身日期記下的資料更正紀錄，依 check 彙總，不算進每日檢查）；報酬到價格與區間的換算（每檔用自己的最新收盤價）；簡單查詢頁面 | API 回傳最新交易日的預測排名、預測價格、價格區間、上漲機率與特徵；回應時間 500 毫秒內 | 1 到 2 週 |
| Phase 6 即時（可選） | Fugle 即時報價、Shioaji 下單介面、盤中監控 | 盤中可取得追蹤清單的即時價；下單走模擬帳戶驗證 | 視需求 |

## 10. 給 Claude Code 的指引

本文件是專案的單一規格來源。開始任何 Phase 之前先讀第 8 節，再依第 9 節的交付物與驗收條件擬 plan；plan 中每個步驟都要能對應到本文件的某一節。

**如何使用本文件**

- `docs/SPEC.md` 是唯一的規格來源（最初由 claude.ai 原文匯出），`CLAUDE.md` 第一行寫「實作前先讀 docs/SPEC.md，第 8 節為硬性規則」。
- 文件中的圖只是輔助，匯出後可能不保留；表格與文字才是規格本體。
- 規格與程式碼衝突時，以本文件為準並回報差異；需要偏離時先在 plan 中說明理由。
- 本文件的數值（參數起點、工時、閾值）是初始設定，可在 plan 中提議調整，但第 8 節的規則不可調。

**程式碼慣例**

- Python 3.12、Django 5.x；套件管理用 `uv`；型別註記必寫，`ruff` 做 lint 與格式。
- 目錄結構：`apps/market/`（standard 層 models）、`apps/etl/`（adapters、排程、品質檢查）、`apps/research/`（特徵、label、模型、回測）、`apps/api/`（DRF）、`docs/`、`tests/`。
- 每個 adapter 一個模組，繼承 `BaseAdapter`，三個方法 `fetch`、`normalize`、`upsert`；測試用錄製的 JSON fixture，不打真實 API。
- 資料表一律走 Django ORM migration；大量寫入用 `bulk_create` 或 `COPY`，以主鍵 upsert。
- 研究層以 Django management command 入口（`manage.py build_features`、`manage.py train --model lightgbm`），內部用 pandas，不依賴 request 生命週期。
- 設定（API key、速率上限、排程時間）全部放環境變數或 `settings`，不寫死。
- 每個 PR 必須附帶測試；第 8 節的九個自動化測試在 CI 內必跑。

**待決事項**

- [ ] 是否取得 TEJ 授權，決定財報 PIT 對齊的對照來源。
- [ ] 主要預測 horizon 以 5 日還是 20 日為準，影響 embargo 與重訓頻率。
- [ ] 是否納入 ETF。
- [ ] 部署環境：已決定第一版用個人 Mac（見第 5 節執行環境）；需要全天在線時再評估 VPS。
- [ ] 模型版本管理用自寫 `model_run` 表還是導入 MLflow。
- [ ] 月營收與財報是否從 MOPS 爬取，或先以 FinMind 為 primary、MOPS 只補 `announce_date`。

- [ ] 上漲機率的門檻（5%）與價格區間的信賴水準（80%）是否調整。

- [ ] 排名 label 是否改為對產業與規模殘差化後的排名：看 Phase 4 中性化 IC，接近 0 才改；影響展示的是全市場排名還是產業內相對強弱。
