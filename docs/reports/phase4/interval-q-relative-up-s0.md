# 實驗：relative-up（interval-q）

> 財報修正後資料（篩選 p3-final-20261109，develop eba50db），窗長 w60，seed 0；兩組都不加市場特徵

- w60-nomarket：只用篩選後的個股特徵，不加市場特徵
- w60-relative-nomarket：同 relative，但兩組都不加市場特徵
- 訓練窗：rolling 60 個月，驗證段 6 個月
- 比較 18 個研究摺（摺 4–21），seed 0；每組的快取 run_key 為 `interval-q-<組名>`，seed 不是 0 時再加 `-s<seed>`
- 判定規則：w60-relative-nomarket 的 AUC 要高於 w60-nomarket，而且逐摺差的 NW t 要高於 2 才採用（SPEC §7：「日均 AUC 較高就改用」；逐摺差的 NW t 門檻是實驗另加的）；其他欄只列參考。各 seed 先逐日平均，再看研究期與逐摺差
- 日均 AUC 兩組各對自己的 label 評估：第一組看 20 日報酬是否大於 5%，第二組看減掉 TAIEX_TR 同期報酬（收盤 t 到收盤 t+20）後是否大於 5%。TAIEX_TR 是含息的報酬指數，和個股的還原報酬同一口徑；個股從 t+1 開盤起算、大盤從 t 收盤起算，差一段隔夜跳空
- 第一組沿用 market-interval 的 w60-nomarket 組；market-interval 判定採用市場特徵時，拿掉 --no-market 重跑

| 組 | AUC ↑ |
| --- | --- |
| w60-nomarket | 0.5839 |
| w60-relative-nomarket | 0.6027 |
| 差（w60-relative-nomarket − w60-nomarket） | +0.0188 |
| 逐摺差 NW t | +3.2414 |

**判定（暫定）：採用 w60-relative-nomarket**
- 只跑了 1 個 seed：要用 3 個 seed（`--seeds 0,3,6`）重跑再定案
- 參考欄（w60-relative-nomarket 對 w60-nomarket）：較好 無；較差 無

## 每摺（AUC）

| 摺 | w60-nomarket | w60-relative-nomarket |
| --- | --- | --- |
| 4 | 0.5592 | 0.6052 |
| 5 | 0.5743 | 0.6115 |
| 6 | 0.5939 | 0.6339 |
| 7 | 0.6032 | 0.6310 |
| 8 | 0.6170 | 0.6313 |
| 9 | 0.5548 | 0.5554 |
| 10 | 0.5863 | 0.6197 |
| 11 | 0.6116 | 0.6335 |
| 12 | 0.6185 | 0.6152 |
| 13 | 0.5732 | 0.6105 |
| 14 | 0.5923 | 0.6177 |
| 15 | 0.5565 | 0.5690 |
| 16 | 0.5505 | 0.5036 |
| 17 | 0.5780 | 0.5730 |
| 18 | 0.6054 | 0.6291 |
| 19 | 0.5846 | 0.5968 |
| 20 | 0.5748 | 0.6228 |
| 21 | 0.5814 | 0.5950 |
