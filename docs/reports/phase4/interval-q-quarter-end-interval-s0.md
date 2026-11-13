# 實驗：quarter-end-interval（interval-q）

> 財報修正後資料（篩選 p3-final-20261109，develop eba50db），窗長 w60，seed 0；第一組沿用 relative-up 的快取

- w60-relative-nomarket：同 relative，但兩組都不加市場特徵
- w60-relative-nomarket-qend：同 relative-nomarket，再加 `mkt_days_to_quarter_end`
- 訓練窗：rolling 60 個月，驗證段 6 個月
- 比較 18 個研究摺（摺 4–21），seed 0；每組的快取 run_key 為 `interval-q-<組名>`，seed 不是 0 時再加 `-s<seed>`
- 判定規則：w60-relative-nomarket-qend 的 pinball 要低於 w60-relative-nomarket，而且逐摺差的 NW t 要低於 −2 才採用（SPEC §6：「總體 OOS rank IC 沒有提高就不留」；逐摺差的 NW t 門檻是實驗另加的）；其他欄只列參考。各 seed 先逐日平均，再看研究期與逐摺差
- `mkt_days_to_quarter_end` 是市場層級特徵，排名模型不用，沒有 rank IC 可比；照 §7 市場特徵的比較，改看這個目標的主指標，並加逐摺差 NW t 門檻。兩組都不加其他市場特徵，只差這一欄，進模型前和其他市場特徵一樣在訓練窗內切成 5 級

| 組 | pinball ↓ | 覆蓋率（目標 80%） | 波動率 MAE ↓ | QLIKE ↓ | AUC ↑ | Brier ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| w60-relative-nomarket | 0.0805 | 0.7757 | 0.0981 | 0.3445 | 0.6027 | 0.1618 |
| w60-relative-nomarket-qend | 0.0805 | 0.7711 | 0.0980 | 0.3448 | 0.6021 | 0.1619 |
| 差（w60-relative-nomarket-qend − w60-relative-nomarket） | -0.0001 | -0.0046 | -0.0001 | +0.0003 | -0.0007 | +0.0001 |
| 逐摺差 NW t | -0.1794 | -2.0216 | -0.7948 | +0.1059 | -0.7203 | +0.6617 |

**判定（暫定）：採用 w60-relative-nomarket**
- w60-relative-nomarket-qend 的 pinball 較低，但逐摺差 NW t -0.18 沒有低於 −2：照規則留 w60-relative-nomarket
- 只跑了 1 個 seed：要用 3 個 seed（`--seeds 0,3,6`）重跑再定案
- 參考欄（w60-relative-nomarket-qend 對 w60-relative-nomarket）：較好 波動率 MAE；較差 覆蓋率、QLIKE、AUC、Brier

## 每摺（pinball）

| 摺 | w60-relative-nomarket | w60-relative-nomarket-qend |
| --- | --- | --- |
| 4 | 0.0770 | 0.0761 |
| 5 | 0.0641 | 0.0638 |
| 6 | 0.0651 | 0.0647 |
| 7 | 0.0745 | 0.0744 |
| 8 | 0.0771 | 0.0765 |
| 9 | 0.0826 | 0.0828 |
| 10 | 0.0749 | 0.0741 |
| 11 | 0.0611 | 0.0609 |
| 12 | 0.1264 | 0.1313 |
| 13 | 0.0833 | 0.0829 |
| 14 | 0.1079 | 0.1070 |
| 15 | 0.0791 | 0.0792 |
| 16 | 0.0800 | 0.0796 |
| 17 | 0.0834 | 0.0836 |
| 18 | 0.0815 | 0.0815 |
| 19 | 0.0700 | 0.0695 |
| 20 | 0.0843 | 0.0844 |
| 21 | 0.0816 | 0.0810 |
