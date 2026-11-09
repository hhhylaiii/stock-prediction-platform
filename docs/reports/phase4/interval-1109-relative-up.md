# 實驗：relative-up（interval-1109）

> 財報修正後資料（篩選 p3-final-20261109，develop 7c7a83b），窗長 w60，seed 0、3、6，兩組都不加市場特徵（market-interval 判定 nomarket）；窗長沿用 p3-final-20261108b 上 rank 窗長實驗的結論

- w60-nomarket：只用篩選後的個股特徵，不加市場特徵
- w60-relative-nomarket：同 relative，但兩組都不加市場特徵
- 訓練窗：rolling 60 個月，驗證段 6 個月
- 比較 18 個研究摺（摺 4–21），seed 0, 3, 6；每組的快取 run_key 為 `interval-1109-<組名>`，seed 不是 0 時再加 `-s<seed>`
- 判定規則：w60-relative-nomarket 的 AUC 要高於 w60-nomarket，而且逐摺差的 NW t 要高於 2 才採用（SPEC §7：「日均 AUC 較高就改用」；逐摺差的 NW t 門檻是實驗另加的）；其他欄只列參考。各 seed 先逐日平均，再看研究期與逐摺差
- 日均 AUC 兩組各對自己的 label 評估：第一組看 20 日報酬是否大於 5%，第二組看減掉 TAIEX 同期報酬（收盤 t 到收盤 t+20）後是否大於 5%。TAIEX 是價格指數、個股是含息的還原報酬，除息旺季的 TAIEX 報酬偏低，第二組的正例會偏多
- 第一組沿用 market-interval 的 w60-nomarket 組；market-interval 判定採用市場特徵時，拿掉 --no-market 重跑

| 組 | AUC ↑ |
| --- | --- |
| w60-nomarket | 0.5837 |
| w60-relative-nomarket | 0.5977 |
| 差（w60-relative-nomarket − w60-nomarket） | +0.0140 |
| 逐摺差 NW t | +2.4427 |

**判定：採用 w60-relative-nomarket**
- 參考欄（w60-relative-nomarket 對 w60-nomarket）：較好 無；較差 無

## 各 seed（AUC）

| 組 | seed 0 | seed 3 | seed 6 |
| --- | --- | --- | --- |
| w60-nomarket | 0.5839 | 0.5835 | 0.5837 |
| w60-relative-nomarket | 0.5976 | 0.5978 | 0.5977 |

## 每摺（AUC）

| 摺 | w60-nomarket | w60-relative-nomarket |
| --- | --- | --- |
| 4 | 0.5560 | 0.6025 |
| 5 | 0.5748 | 0.6056 |
| 6 | 0.5954 | 0.6265 |
| 7 | 0.6050 | 0.6234 |
| 8 | 0.6181 | 0.6251 |
| 9 | 0.5553 | 0.5510 |
| 10 | 0.5874 | 0.6154 |
| 11 | 0.6107 | 0.6216 |
| 12 | 0.6162 | 0.6089 |
| 13 | 0.5745 | 0.6085 |
| 14 | 0.5913 | 0.6163 |
| 15 | 0.5551 | 0.5634 |
| 16 | 0.5498 | 0.5004 |
| 17 | 0.5786 | 0.5692 |
| 18 | 0.6051 | 0.6238 |
| 19 | 0.5822 | 0.5897 |
| 20 | 0.5756 | 0.6201 |
| 21 | 0.5797 | 0.5935 |
