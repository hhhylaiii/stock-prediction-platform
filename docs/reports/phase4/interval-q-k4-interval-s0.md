# 實驗：k4-interval（interval-q）

> 財報修正後資料（篩選 p3-final-20261109），本機實驗分支 exp/p4-k4-target-features（6aa171d，develop 93302e0 加一個 commit，未 push），窗長 w60，seed 0；第一組沿用快取；新組多的五個特徵和排名候選設計 cand60 多的五個逐一相同；K4 的採用規則照使用者 2026-11-11 同意的規則另判，不看下面工具的判定

- w60-relative-nomarket：同 relative，但兩組都不加市場特徵
- w60-relative-nomarket-k4：同 relative-nomarket，再加排名第二批加入的 chip 4 個與 `fund_days_to_ex_date`
- w60-relative-nomarket 讀 55 個特徵，清單雜湊 `15823336d71e`
- w60-relative-nomarket-k4 讀 60 個特徵，清單雜湊 `54d4336bc87f`
- w60-relative-nomarket-k4 比 w60-relative-nomarket 多 5 個：`chip_day_trade_ratio_5d`、`chip_fi_holding_chg_20d`、`chip_it_net_20d`、`chip_fi_buy_streak`、`fund_days_to_ex_date`
- 訓練窗：rolling 60 個月，驗證段 6 個月
- 比較 18 個研究摺（摺 4–21），seed 0；每組的快取 run_key 為 `interval-q-<組名>`，seed 不是 0 時再加 `-s<seed>`
- 判定規則：w60-relative-nomarket-k4 的 pinball 要低於 w60-relative-nomarket，而且逐摺差的 NW t 要低於 −2 才採用（SPEC §7：「同一組特徵訓練三類目標」；逐摺差的 NW t 門檻是實驗另加的）；其他欄只列參考。各 seed 先逐日平均，再看研究期與逐摺差
- 新組多的五個個股特徵就是排名第二批加入的那五個，兩組都不加市場特徵；下面的判定是工具的通用規則，K4 的採用規則另訂

| 組 | pinball ↓ | 覆蓋率（目標 80%） | 波動率 MAE ↓ | QLIKE ↓ | AUC ↑ | Brier ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| w60-relative-nomarket | 0.0805 | 0.7757 | 0.0981 | 0.3445 | 0.6027 | 0.1618 |
| w60-relative-nomarket-k4 | 0.0806 | 0.7742 | 0.0980 | 0.3458 | 0.6031 | 0.1619 |
| 差（w60-relative-nomarket-k4 − w60-relative-nomarket） | +0.0001 | -0.0015 | -0.0001 | +0.0012 | +0.0003 | +0.0000 |
| 逐摺差 NW t | +0.5041 | -0.5956 | -0.7816 | +0.3847 | +0.7243 | +0.4088 |

**判定（暫定）：採用 w60-relative-nomarket**
- 只跑了 1 個 seed：要用 3 個 seed（`--seeds 0,3,6`）重跑再定案
- 參考欄（w60-relative-nomarket-k4 對 w60-relative-nomarket）：較好 波動率 MAE、AUC；較差 覆蓋率、QLIKE、Brier

## 每摺（pinball）

| 摺 | w60-relative-nomarket | w60-relative-nomarket-k4 |
| --- | --- | --- |
| 4 | 0.0770 | 0.0765 |
| 5 | 0.0641 | 0.0644 |
| 6 | 0.0651 | 0.0652 |
| 7 | 0.0745 | 0.0740 |
| 8 | 0.0771 | 0.0770 |
| 9 | 0.0826 | 0.0849 |
| 10 | 0.0749 | 0.0743 |
| 11 | 0.0611 | 0.0611 |
| 12 | 0.1264 | 0.1268 |
| 13 | 0.0833 | 0.0834 |
| 14 | 0.1079 | 0.1080 |
| 15 | 0.0791 | 0.0790 |
| 16 | 0.0800 | 0.0800 |
| 17 | 0.0834 | 0.0836 |
| 18 | 0.0815 | 0.0815 |
| 19 | 0.0700 | 0.0699 |
| 20 | 0.0843 | 0.0843 |
| 21 | 0.0816 | 0.0814 |
