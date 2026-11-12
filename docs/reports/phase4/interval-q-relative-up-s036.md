# 實驗：relative-up（interval-q）

> 財報修正後資料（篩選 p3-final-20261109，develop eba50db），窗長 w60，seed 0、3、6；兩組都不加市場特徵

- w60-nomarket：只用篩選後的個股特徵，不加市場特徵
- w60-relative-nomarket：同 relative，但兩組都不加市場特徵
- 訓練窗：rolling 60 個月，驗證段 6 個月
- 比較 18 個研究摺（摺 4–21），seed 0, 3, 6；每組的快取 run_key 為 `interval-q-<組名>`，seed 不是 0 時再加 `-s<seed>`
- 判定規則：w60-relative-nomarket 的 AUC 要高於 w60-nomarket，而且逐摺差的 NW t 要高於 2 才採用（SPEC §7：「日均 AUC 較高就改用」；逐摺差的 NW t 門檻是實驗另加的）；其他欄只列參考。各 seed 先逐日平均，再看研究期與逐摺差
- 日均 AUC 兩組各對自己的 label 評估：第一組看 20 日報酬是否大於 5%，第二組看減掉 TAIEX_TR 同期報酬（收盤 t 到收盤 t+20）後是否大於 5%。TAIEX_TR 是含息的報酬指數，和個股的還原報酬同一口徑；個股從 t+1 開盤起算、大盤從 t 收盤起算，差一段隔夜跳空
- 第一組沿用 market-interval 的 w60-nomarket 組；market-interval 判定採用市場特徵時，拿掉 --no-market 重跑

| 組 | AUC ↑ |
| --- | --- |
| w60-nomarket | 0.5837 |
| w60-relative-nomarket | 0.6025 |
| 差（w60-relative-nomarket − w60-nomarket） | +0.0188 |
| 逐摺差 NW t | +3.2810 |

**判定：採用 w60-relative-nomarket**
- 參考欄（w60-relative-nomarket 對 w60-nomarket）：較好 無；較差 無

## 各 seed（AUC）

| 組 | seed 0 | seed 3 | seed 6 |
| --- | --- | --- | --- |
| w60-nomarket | 0.5839 | 0.5835 | 0.5837 |
| w60-relative-nomarket | 0.6027 | 0.6026 | 0.6023 |

## 每摺（AUC）

| 摺 | w60-nomarket | w60-relative-nomarket |
| --- | --- | --- |
| 4 | 0.5560 | 0.6062 |
| 5 | 0.5748 | 0.6113 |
| 6 | 0.5954 | 0.6340 |
| 7 | 0.6050 | 0.6311 |
| 8 | 0.6181 | 0.6311 |
| 9 | 0.5553 | 0.5540 |
| 10 | 0.5874 | 0.6184 |
| 11 | 0.6107 | 0.6331 |
| 12 | 0.6162 | 0.6140 |
| 13 | 0.5745 | 0.6111 |
| 14 | 0.5913 | 0.6168 |
| 15 | 0.5551 | 0.5699 |
| 16 | 0.5498 | 0.5054 |
| 17 | 0.5786 | 0.5712 |
| 18 | 0.6051 | 0.6300 |
| 19 | 0.5822 | 0.5958 |
| 20 | 0.5756 | 0.6222 |
| 21 | 0.5797 | 0.5955 |
