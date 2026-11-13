# 手動檢查的原始輸出（PR 4.4）

以下檢查用一次性的腳本跑出，腳本不在 repo 裡；這裡原樣保留它們印出的結果，供彙整報告 `docs/reports/phase4-experiments.md` 對照。K20、打亂檢查與 lambdarank 用篩選 `p3-final-20261108b`（develop `fce950c`）；market-return、market-interval、relative-up 用 `p3-final-20261109`（develop `7c7a83b`）。

## K20 三個響的摺：打亂 label 檢查

訓練窗的 label 欄在每天之內打亂，再對真的 label 算測試期的日 rank IC。Ridge 是規格第 8 節的 20 日 Ridge 檢查換成響的那個 horizon；LightGBM 用預設設計（36 個月窗），seed 0。

```
fold 25 h5 2020-04-01..2020-06-30: shuffled ridge 0.0000 lightgbm -0.0254 (81 rounds) 35s
fold 32 h20 2022-01-03..2022-03-31: shuffled ridge 0.0015 lightgbm 0.0203 (41 rounds) 35s
fold 34 h20 2022-07-01..2022-09-30: shuffled ridge -0.0026 lightgbm -0.0128 (11 rounds) 34s
worst |IC| 0.0254: STOP, report
      105.38 real        93.10 user        16.05 sys
          7051476992  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             6278596  page reclaims
                4445  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                 262  messages sent
               81886  messages received
                   1  signals received
                3358  voluntary context switches
             1250972  involuntary context switches
           244669418  instructions retired
           128320656  cycles elapsed
            11600304  peak memory footprint
```

LightGBM 單一 seed 超過 0.02，所以每摺再跑 10 個 seed 看虛無分布：

```
       0
fold 25 h5 seed 0: -0.0254 (81 rounds) 17s
fold 25 h5 seed 1: 0.0107 (41 rounds) 15s
fold 25 h5 seed 2: -0.0308 (41 rounds) 14s
fold 25 h5 seed 3: -0.0304 (221 rounds) 17s
fold 25 h5 seed 4: 0.0219 (21 rounds) 14s
fold 25 h5 seed 5: -0.0141 (81 rounds) 15s
fold 25 h5 seed 6: 0.0007 (31 rounds) 14s
fold 25 h5 seed 7: -0.0009 (71 rounds) 15s
fold 25 h5 seed 8: 0.0400 (11 rounds) 14s
fold 25 h5 seed 9: 0.0250 (181 rounds) 16s
fold 25 h5: mean -0.0003 sd 0.0248 t -0.04 max|IC| 0.0400
fold 32 h20 seed 0: 0.0203 (41 rounds) 18s
fold 32 h20 seed 1: -0.0232 (11 rounds) 15s
fold 32 h20 seed 2: -0.0587 (141 rounds) 18s
fold 32 h20 seed 3: -0.0285 (11 rounds) 16s
fold 32 h20 seed 4: 0.0304 (21 rounds) 15s
fold 32 h20 seed 5: -0.0207 (1 rounds) 15s
fold 32 h20 seed 6: 0.0305 (131 rounds) 17s
fold 32 h20 seed 7: -0.0164 (1 rounds) 15s
fold 32 h20 seed 8: 0.0288 (1 rounds) 15s
fold 32 h20 seed 9: 0.0611 (11 rounds) 15s
fold 32 h20: mean 0.0024 sd 0.0369 t 0.20 max|IC| 0.0611
fold 34 h20 seed 0: -0.0128 (11 rounds) 16s
fold 34 h20 seed 1: -0.0346 (21 rounds) 16s
fold 34 h20 seed 2: -0.0107 (171 rounds) 18s
fold 34 h20 seed 3: 0.0161 (1 rounds) 16s
fold 34 h20 seed 4: -0.0112 (11 rounds) 16s
fold 34 h20 seed 5: 0.0557 (101 rounds) 17s
fold 34 h20 seed 6: -0.0157 (31 rounds) 16s
fold 34 h20 seed 7: -0.0393 (1 rounds) 15s
fold 34 h20 seed 8: 0.0341 (81 rounds) 17s
fold 34 h20 seed 9: 0.0179 (11 rounds) 16s
fold 34 h20: mean -0.0000 sd 0.0303 t -0.01 max|IC| 0.0557
pooled 30: mean 0.0007 sd 0.0300 t 0.12
      475.77 real       502.99 user       103.06 sys
          6276268032  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
            26970049  page reclaims
                1031  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                1249  messages sent
              403352  messages received
                   1  signals received
                2176  voluntary context switches
            12561839  involuntary context switches
           155220197  instructions retired
            64334923  cycles elapsed
            11682248  peak memory footprint
```

## lambdarank：合成資料上的方向

120 天 × 400 檔，一個有訊號的特徵 x（y = x + 雜訊，雜訊標準差 2）和一個雜訊特徵 z；前 100 天訓練、後 20 天測試，同一支 `gbdt.boost`。

```
regression: best 1 valid 0.4369 test IC 0.4073
lambdarank: best 1 valid 0.4177 test IC 0.3906
lambdarank-notrunc: best 1 valid 0.4177 test IC 0.3906
```

## 4.8b 排名警訊摺：打亂 label 檢查

凍結設計（60 個月窗、重訓輪數 max(最佳輪數, 200)）的標準摺上，`rank-std1109` 報告的洩漏警訊摺。Ridge 用規格的線性特徵，LightGBM 用凍結設計各跑 3 個 seed。第一段在 `4e69871` 跑（2026-11-09）：

```
fold 25 h5 seed 0: lightgbm -0.0174 (200 rounds)
fold 25 h5 seed 1: lightgbm -0.0065 (200 rounds)
fold 25 h5 seed 2: lightgbm -0.0128 (200 rounds)
fold 25 h5 2020-04-01..2020-06-30: shuffled ridge 0.0039 lightgbm mean -0.0123 max|IC| 0.0174 60s
fold 19 h20 seed 0: lightgbm 0.0333 (200 rounds)
fold 19 h20 seed 1: lightgbm -0.0092 (200 rounds)
fold 19 h20 seed 2: lightgbm -0.0333 (200 rounds)
fold 19 h20 2018-10-01..2018-12-28: shuffled ridge 0.0127 lightgbm mean -0.0031 max|IC| 0.0333 71s
fold 25 h20 seed 0: lightgbm -0.0015 (200 rounds)
fold 25 h20 seed 1: lightgbm -0.0125 (200 rounds)
fold 25 h20 seed 2: lightgbm -0.0161 (200 rounds)
fold 25 h20 2020-04-01..2020-06-30: shuffled ridge 0.0027 lightgbm mean -0.0100 max|IC| 0.0161 70s
fold 29 h20 seed 0: lightgbm 0.0007 (200 rounds)
fold 29 h20 seed 1: lightgbm -0.0421 (200 rounds)
fold 29 h20 seed 2: lightgbm 0.0379 (200 rounds)
fold 29 h20 2021-04-01..2021-06-30: shuffled ridge 0.0025 lightgbm mean -0.0012 max|IC| 0.0421 67s
pooled 12: mean -0.0066 sd 0.0232 t -0.99
      269.80 real       305.23 user        61.65 sys
```

第二段在 `eba50db` 跑（2026-11-11）：補摺 24 的 5 日，並用第二批後的 60 個特徵在摺 8 跑 LightGBM（Ridge 固定用第一批特徵，不涵蓋新特徵，所以印 nan）：

```
frozen 55 features; rank-f2-w60-floor-chip-exdt 60; added ['chip_day_trade_ratio_5d', 'chip_fi_buy_streak', 'chip_fi_holding_chg_20d', 'chip_it_net_20d', 'fund_days_to_ex_date']; dropped []
frozen fold 24 h5 seed 0: lightgbm 0.0210 (200 rounds)
frozen fold 24 h5 seed 1: lightgbm -0.0462 (200 rounds)
frozen fold 24 h5 seed 2: lightgbm -0.0316 (200 rounds)
frozen fold 24 h5 2020-01-02..2020-03-31 (55 features): shuffled ridge -0.0039 lightgbm mean -0.0189 max|IC| 0.0462 73s
f2 fold 8 h5 seed 0: lightgbm -0.0203 (391 rounds)
f2 fold 8 h5 seed 1: lightgbm 0.0303 (200 rounds)
f2 fold 8 h5 seed 2: lightgbm 0.0186 (200 rounds)
f2 fold 8 h5 2016-01-04..2016-03-31 (60 features): shuffled ridge nan lightgbm mean 0.0095 max|IC| 0.0303 43s
f2 fold 8 h20 seed 0: lightgbm -0.0011 (200 rounds)
f2 fold 8 h20 seed 1: lightgbm -0.0038 (200 rounds)
f2 fold 8 h20 seed 2: lightgbm -0.0110 (200 rounds)
f2 fold 8 h20 2016-01-04..2016-03-31 (60 features): shuffled ridge nan lightgbm mean -0.0053 max|IC| 0.0110 36s
pooled 9: mean -0.0049 sd 0.0254 t -0.58
      153.56 real       213.61 user        55.13 sys
```

## 60 個特徵的排名設計：警訊摺的打亂 label 檢查

`rank-cand60` 的洩漏警訊摺和 55 個特徵的設計相同。用它的 60 個特徵（取自 `model_run.feature_list`）在這五組上各跑 3 個 seed 的 LightGBM 打亂，重訓輪數照凍結設計。Ridge 沒有重跑：它固定用第一批特徵，不涵蓋新加的 5 個。在 `eba50db` 跑（2026-11-11 14:44–14:48）：

```
rank-cand60: 60 features; variant Variant(every=1, half_life=None, objective='regression', min_rounds=200, fixed_rounds=(), masked=(), features=())
fold 24 h5 seed 0: lightgbm 0.0117 (200 rounds)
fold 24 h5 seed 1: lightgbm -0.0571 (200 rounds)
fold 24 h5 seed 2: lightgbm -0.0339 (200 rounds)
fold 24 h5 2020-01-02..2020-03-31: lightgbm mean -0.0264 max|IC| 0.0571 43s
fold 25 h5 seed 0: lightgbm -0.0287 (200 rounds)
fold 25 h5 seed 1: lightgbm -0.0093 (200 rounds)
fold 25 h5 seed 2: lightgbm 0.0021 (200 rounds)
fold 25 h5 2020-04-01..2020-06-30: lightgbm mean -0.0120 max|IC| 0.0287 49s
fold 19 h20 seed 0: lightgbm 0.0383 (200 rounds)
fold 19 h20 seed 1: lightgbm -0.0143 (200 rounds)
fold 19 h20 seed 2: lightgbm -0.0311 (261 rounds)
fold 19 h20 2018-10-01..2018-12-28: lightgbm mean -0.0024 max|IC| 0.0383 47s
fold 25 h20 seed 0: lightgbm -0.0033 (200 rounds)
fold 25 h20 seed 1: lightgbm -0.0237 (200 rounds)
fold 25 h20 seed 2: lightgbm -0.0240 (200 rounds)
fold 25 h20 2020-04-01..2020-06-30: lightgbm mean -0.0170 max|IC| 0.0240 42s
fold 29 h20 seed 0: lightgbm -0.0105 (200 rounds)
fold 29 h20 seed 1: lightgbm -0.0348 (341 rounds)
fold 29 h20 seed 2: lightgbm 0.0396 (200 rounds)
fold 29 h20 2021-04-01..2021-06-30: lightgbm mean -0.0019 max|IC| 0.0396 46s
pooled 15: mean -0.0119 sd 0.0267 t -1.73 max|IC| 0.0571 (null max 0.0611)
verdict: no sign of leakage (|t| < 2 and max|IC| <= 0.0611)
```

- 判定規則是協調 2026-11-11 在結果出來前寫死的：合併平均的 |t| < 2，且單一 seed 最大 |IC| 不超過 0.0611。兩條都成立。
- 印出的 variant 沒有 `features`，是因為腳本直接把 60 個特徵傳給擬合函式；輪數下限與其他設定和凍結設計相同。
- 三批打亂檢查（上一節的 12 次與 9 次，加這 15 次）用的 seed 都是 0、1、2，摺也重疊，不是獨立樣本。合併平均偏負的事記在 `phase4-experiments.md` 的「已知但未解釋」。

## 4.8b 區間覆蓋率：波動率 baseline 與漏接方向

從 `interval-std1109` 的快取算，不重訓（`eba50db`）。列是 universe 內 t+1 可交易、模型與波動率區間 baseline 都有值的列；覆蓋率與漏接比例都是逐日平均再逐摺平均。欄位：`model_*` 是分位數模型的覆蓋率，`volband_shared` 是 baseline 的覆蓋率，`below_q10`、`above_q90` 是模型漏接在下、上的比例，`band_*` 是 baseline 的，`label_mean`、`q50_mean` 是 20 日報酬與 q50 的平均。

```
rows tradable 1757592, with model q10/q90 1757592, shared with volband 1757592
total coverage (research): model as reported 0.7768, model on shared rows 0.7768, volband on shared rows 0.7918
research folds outside 70-90%: model [24, 33, 34], volband [24]
      tuning  model_reported  model_shared  volband_shared  below_q10  above_q90  band_below  band_above  label_mean  q50_mean
fold                                                                                                                          
5       True          0.6231        0.6231          0.7322     0.3128     0.0641      0.2128      0.0549     -0.0427   -0.0010
6       True          0.6560        0.6560          0.7443     0.1566     0.1874      0.1511      0.1046      0.0034   -0.0145
7       True          0.7464        0.7464          0.8226     0.0788     0.1748      0.0824      0.0950     -0.0016   -0.0091
8      False          0.7566        0.7566          0.8005     0.0351     0.2083      0.0561      0.1433      0.0256   -0.0322
9      False          0.8309        0.8309          0.7928     0.0754     0.0937      0.0838      0.1234      0.0095   -0.0064
10     False          0.8560        0.8560          0.8044     0.0376     0.1064      0.0706      0.1250      0.0052   -0.0120
11     False          0.8087        0.8087          0.8121     0.0999     0.0915      0.0851      0.1028      0.0038   -0.0015
12     False          0.7947        0.7947          0.7416     0.0704     0.1350      0.0724      0.1860      0.0194   -0.0057
13     False          0.8451        0.8451          0.8330     0.0645     0.0905      0.0434      0.1236      0.0113   -0.0029
14     False          0.7871        0.7871          0.7829     0.0470     0.1658      0.0642      0.1530      0.0205   -0.0033
15     False          0.8206        0.8206          0.8070     0.0435     0.1358      0.0603      0.1327      0.0101   -0.0088
16     False          0.7994        0.7994          0.7992     0.1089     0.0917      0.1207      0.0800     -0.0108   -0.0055
17     False          0.7999        0.7999          0.8068     0.0918     0.1083      0.0874      0.1058      0.0025   -0.0040
18     False          0.7015        0.7015          0.7214     0.2406     0.0579      0.2094      0.0691     -0.0528   -0.0038
19     False          0.7878        0.7878          0.8497     0.0815     0.1308      0.0706      0.0797      0.0116   -0.0175
20     False          0.7549        0.7549          0.7713     0.0146     0.2305      0.0197      0.2091      0.0464   -0.0170
21     False          0.7577        0.7577          0.7615     0.1135     0.1288      0.1141      0.1244      0.0023   -0.0057
22     False          0.8319        0.8319          0.7933     0.0677     0.1005      0.0852      0.1215      0.0115   -0.0059
23     False          0.8368        0.8368          0.7914     0.0727     0.0904      0.0869      0.1217      0.0036   -0.0045
24     False          0.4906        0.4906          0.5532     0.3611     0.1483      0.3341      0.1127     -0.0360   -0.0036
25     False          0.7214        0.7214          0.8191     0.0131     0.2655      0.0066      0.1743      0.0732   -0.0074
26     False          0.8284        0.8284          0.8574     0.0715     0.1001      0.0425      0.1001      0.0075   -0.0134
27     False          0.7990        0.7990          0.7676     0.0513     0.1497      0.0420      0.1905      0.0330   -0.0083
28     False          0.7438        0.7438          0.7349     0.0619     0.1943      0.0243      0.2409      0.0665   -0.0003
29     False          0.7039        0.7039          0.7618     0.1868     0.1093      0.1322      0.1060      0.0133    0.0002
30     False          0.7577        0.7577          0.8475     0.1790     0.0633      0.0837      0.0688     -0.0258   -0.0058
31     False          0.8276        0.8276          0.8337     0.0396     0.1328      0.0362      0.1300      0.0262   -0.0161
32     False          0.8124        0.8124          0.8265     0.1195     0.0681      0.0840      0.0895     -0.0152   -0.0038
33     False          0.6630        0.6630          0.7528     0.2838     0.0531      0.2002      0.0470     -0.0419    0.0031
34     False          0.6807        0.6807          0.7665     0.2118     0.1075      0.1617      0.0718     -0.0162   -0.0014
35     False          0.7613        0.7613          0.8377     0.0356     0.2031      0.0247      0.1376      0.0394   -0.0095
36     False          0.7944        0.7944          0.7739     0.0270     0.1787      0.0278      0.1982      0.0407   -0.0093
37     False          0.8146        0.8146          0.7787     0.0463     0.1390      0.0567      0.1646      0.0289   -0.0156
38     False          0.7681        0.7681          0.8280     0.1655     0.0663      0.0894      0.0826     -0.0092   -0.0000
39     False          0.8672        0.8672          0.8270     0.0304     0.1024      0.0365      0.1364      0.0219   -0.0105
40     False          0.7839        0.7839          0.7555     0.0673     0.1489      0.0521      0.1924      0.0302    0.0002
41     False          0.8141        0.8141          0.8170     0.0388     0.1470      0.0324      0.1505      0.0328   -0.0080
42     False          0.7528        0.7528          0.8248     0.1808     0.0664      0.1076      0.0675     -0.0111    0.0011
43     False          0.7840        0.7840          0.8395     0.1570     0.0589      0.0912      0.0693     -0.0206   -0.0025
fold 18 2018-07-02..2018-09-28: TAIEX_TR +0.0475
fold 24 2020-01-02..2020-03-31: TAIEX_TR -0.1888
fold 29 2021-04-01..2021-06-30: TAIEX_TR +0.0842
fold 33 2022-04-01..2022-06-30: TAIEX_TR -0.1483
fold 34 2022-07-01..2022-09-30: TAIEX_TR -0.0700
       20.80 real         9.55 user         2.35 sys
```

## 耗時與峰值記憶體

每一步用 `/usr/bin/time -l` 跑，取 real 與 maximum resident set size：

| 步驟 | real | 峰值 RSS |
| --- | --- | --- |
| K20 Ridge | 23.9 分 | 4.96 GB |
| K20 LightGBM | 33.4 分 | 4.90 GB |
| K20 評估 | 2.9 分 | 5.18 GB |
| 窗長 | 152.6 分 | 7.19 GB |
| 停止輪數 | 130.5 分 | 6.77 GB |
| 取樣 | 46.0 分 | 6.65 GB |
| objective | 69.6 分 | 6.62 GB |
| 缺值遮蔽 | 49.4 分 | 6.08 GB |
| market-return | 71.1 分 | 7.28 GB |
| market-interval | 219.2 分 | 7.99 GB |
| relative-up | 118.6 分 | 7.33 GB |
| 停止輪數重跑（seed 0） | 46.0 分 | 5.97 GB |
| 停止輪數重跑（seed 0、3） | 51.4 分 | 6.33 GB |
| 取樣重跑（seed 0） | 16.0 分 | 6.30 GB |
| 取樣補 seed 3、6（seed 0、3、6） | 55.1 分 | 6.31 GB |
| tech 組（seed 0） | 28.2 分 | 6.35 GB |
| tech 組（seed 0、3、6；14:55 停在第 84 個 fit） | 63.0 分 | 未記錄 |
| tech 組（seed 0、3、6；接續剩下的 24 個 fit） | 23.1 分 | 6.46 GB |
| chip 組（seed 0） | 28.0 分 | 5.97 GB |
| chip 組（seed 0、3、6） | 56.4 分 | 6.00 GB |
| ind 組（seed 0） | 28.5 分 | 5.99 GB |
| ind 組（seed 0、3、6） | 55.1 分 | 6.47 GB |
| exdt 組（seed 0） | 29.1 分 | 6.78 GB |
| exdt 組（seed 0、3、6） | 55.8 分 | 6.26 GB |
| relative-up 改減 TAIEX_TR（seed 0） | 81.0 分 | 7.22 GB |
| 4.8b interval LightGBM | 74.0 分 | 7.12 GB |
| 4.8b interval baseline | 37.5 分 | 8.14 GB |
| 4.8b interval 評估 | 2.0 分 | 6.98 GB |
| 季底天數 return（seed 0） | 12.8 分 | 7.72 GB |
| 季底天數 interval（seed 0） | 45.5 分 | 8.44 GB |
| 4.8b 打亂（摺 24 h5；60 個特徵摺 8） | 2.6 分 | 7.45 GB |

## Holdout（2026-11-12）：label 重建的比對

`build_labels --end 2026-11-11 --recent 81` 印出 `label_daily: 319496 rows for 2026-07-15..2026-11-11`（15.01 秒）。重建前先在記憶體裡算一次、不寫入，重建後逐格比對寫入的結果與那次預演（容差 1e-9，實際最大差都是 0）：

```

fwd_ret cells differing 0 max abs gap 0.0
fwd_ret_open cells differing 0 max abs gap 0.0
fwd_rank_pct cells differing 0 max abs gap 0.0
fwd_vol_20d cells differing 0 max abs gap 0.0
fwd_up_5pct cells differing 0
untradable_t1 cells differing 0
h 5 last row day 2026-11-11 last settled day 2026-11-04 ranked per day {datetime.date(2026, 10, 30): 959, datetime.date(2026, 11, 2): 953, datetime.date(2026, 11, 3): 955, datetime.date(2026, 11, 4): 962}
h 20 last row day 2026-11-11 last settled day 2026-10-12 ranked per day {datetime.date(2026, 10, 6): 915, datetime.date(2026, 10, 7): 916, datetime.date(2026, 10, 8): 908, datetime.date(2026, 10, 12): 913}
rows per day, last 6: {datetime.date(2026, 11, 4): 3968, datetime.date(2026, 11, 5): 3968, datetime.date(2026, 11, 6): 3968, datetime.date(2026, 11, 9): 3968, datetime.date(2026, 11, 10): 3968, datetime.date(2026, 11, 11): 3968}
unbuilt rank []
unbuilt return []
unbuilt interval []
MATCH
```

重建後對重建前的備份（307,532 列，逐位元比較，不給容差）：

```
backup rows 307532 stored rows 319496 both 307532 only backup 0 only stored 11964
fwd_ret had 254711 lost value 0 bitwise changed 58 max abs gap 2.220446049250313e-16 over 1e-12: 0
  stocks {'8390': 58} days 2026-07-16 .. 2026-10-30
fwd_ret_open had 254682 lost value 0 bitwise changed 40 max abs gap 2.220446049250313e-16 over 1e-12: 0
  stocks {'8390': 40} days 2026-07-16 .. 2026-10-30
fwd_rank_pct had 127087 lost value 0 bitwise changed 0
fwd_vol_20d had 112750 lost value 0 bitwise changed 86876 max abs gap 8.847089727481716e-15 over 1e-12: 0
fwd_up_5pct had 112740 changed 0
untradable_t1 had 303570 changed 0
new rows 11964 of which before 11-09: 60
  4207 20 rows 2026-10-22 .. 2026-11-06 ranked 0 with fwd_ret_open 2
  5355 20 rows 2026-10-22 .. 2026-11-06 ranked 0 with fwd_ret_open 2
  8176 20 rows 2026-10-22 .. 2026-11-06 ranked 0 with fwd_ret_open 2
```

新增的 11,964 列是 2026-11-09 到 11-11 各 3,968 列，加上 4207、5355、8176 三檔各 20 列；這三檔在 2026 年的特徵分區裡沒有列，不會進評估。

## Holdout（2026-11-12）：執行紀錄

```
rank-holdout-20261112: holdout attempt 1 of rank through 2026-11-11; fitted folds [44, 45, 46, 47, 48, 49, 50, 51], already written []
rank-holdout-20261112: 316 backtest_metric rows
     2002.71 real      6479.62 user       546.57 sys
interval-holdout-20261112: holdout attempt 1 of interval through 2026-11-11; fitted folds [44, 45, 46, 47, 48, 49, 50, 51], already written []
     3365.90 real     12382.21 user      1300.32 sys
return-holdout-20261112: holdout attempt 1 of return through 2026-11-11; fitted folds [44, 45, 46, 47, 48, 49, 50, 51], already written []
      740.16 real      2403.60 user       279.15 sys
```

跑完後的核對（報告檔與報告標記的 sha256、寫入的摺、label 的列數與最大日期）：

```
rank status 明顯退化 failed [] flagged ['排名 5 日 IC'] report file matches marker True sha256 e19230e33abb5257 folds written [44, 45, 46, 47, 48, 49, 50, 51] model_run rows 32
interval status 明顯退化 failed [] flagged ['20 日覆蓋率'] report file matches marker True sha256 2e327d95d2049bf1 folds written [44, 45, 46, 47, 48, 49, 50, 51] model_run rows 40
return status 達標 failed [] flagged [] report file matches marker True sha256 371232b625638ea1 folds written [44, 45, 46, 47, 48, 49, 50, 51] model_run rows 8
label_daily rows, max date (13489076, datetime.date(2026, 11, 11))
label digest since 2021 (5180568, Decimal('2986485676490873462376704'))
```

label 的列數、最大日期與 2021 年以後的指紋（列數與逐列 md5 的加總）在開跑前（11:44）與跑完後（13:33）各查一次，三個值都相同。

2026 年特徵分區檔的 sha256，每個目標開跑前與跑完後各記一次（檔案修改時間都是 2026-11-09 04:23:17）：

```
rank: features before f2b07d36703eeabea8a1de258d2816055085fd0fd2174d71d19bb00b0c04fbef
rank: exit 0, features after f2b07d36703eeabea8a1de258d2816055085fd0fd2174d71d19bb00b0c04fbef
interval: features before f2b07d36703eeabea8a1de258d2816055085fd0fd2174d71d19bb00b0c04fbef
interval: exit 0, features after f2b07d36703eeabea8a1de258d2816055085fd0fd2174d71d19bb00b0c04fbef
return: features before f2b07d36703eeabea8a1de258d2816055085fd0fd2174d71d19bb00b0c04fbef
return: exit 0, features after f2b07d36703eeabea8a1de258d2816055085fd0fd2174d71d19bb00b0c04fbef
```

演練（研究摺 43，整支指令的時間）：bags 1 是 rank 254 秒、return 100 秒、interval 249 秒；bags 5 是 rank 306 秒、return 135 秒、interval 337 秒。bags 5 三次的報告是 `rank-rehearsal-b5-11120230.md`、`interval-rehearsal-b5-11120230.md`、`return-rehearsal-b5-11120230.md`（bags 5 不和研究 run 對帳）。

## Holdout（2026-11-12）：排名的對照數字（讀 `backtest_metric`）

```

research folds 8 .. 43 n 36
research per-fold lightgbm 5d: [0.1349, 0.1503, 0.1682, 0.1505, 0.1394, 0.212, 0.129, 0.1684, 0.1728, 0.1601, 0.1963, 0.1203, 0.0995, 0.1632, 0.1084, 0.161, 0.1518, 0.1143, 0.1472, 0.1219, 0.1229, 0.1473, 0.1911, 0.1136, 0.1591, 0.129, 0.0932, 0.1029, 0.098, 0.1125, 0.1416, 0.1385, 0.1132, 0.1278, 0.155, 0.1418]
research per-fold lightgbm 20d: [0.127, 0.1346, 0.1887, 0.0709, 0.1211, 0.204, 0.0662, 0.1533, 0.1836, 0.1538, 0.289, 0.1234, -0.0002, 0.1741, 0.069, 0.1472, 0.1384, 0.0608, 0.0971, 0.1102, 0.0858, 0.2094, 0.2065, 0.0941, 0.1715, 0.169, 0.1891, 0.028, 0.0752, 0.1398, 0.1882, 0.109, 0.1535, 0.0851, 0.1451, 0.168]
research mean of 36 folds: 5d 0.14047 20d 0.13415 | ridge 5d 0.11889 20d 0.11915
window 34-41 5d 0.11598 20d 0.12098
window 20-27 5d 0.13342 20d 0.09959
window 36-43 (latest) 5d 0.12856 20d 0.13299
5d 8-fold windows n 29 min 0.11598 at folds ending 41 second 0.11613 median 0.13876 max 0.16605 sd of fold values 0.02843
   lowest five windows: [(34, 41, 0.11598), (33, 40, 0.11613), (31, 38, 0.11874), (32, 39, 0.12187), (35, 42, 0.12371)]
20d 8-fold windows n 29 min 0.09959 at folds ending 27 second 0.10124 median 0.13371 max 0.1618 sd of fold values 0.05811
   lowest five windows: [(20, 27, 0.09959), (19, 26, 0.10124), (21, 28, 0.11034), (22, 29, 0.11475), (35, 42, 0.11549)]
holdout 5d lightgbm: [0.1658, 0.16179, 0.08999, 0.07923, 0.10037, 0.08038, 0.0338, 0.21235] days [55, 61, 65, 62, 55, 61, 63, 22]
holdout 5d ridge:    [0.14943, 0.1106, 0.08426, 0.06815, 0.08313, 0.05276, 0.0785, 0.25278]
   lightgbm mean8 0.11546 mean without 51 0.10162 days-weighted 0.10580
   ridge    mean8 0.10995 mean without 51 0.08955 days-weighted 0.09677
   lightgbm minus ridge per fold [0.01637, 0.05119, 0.00573, 0.01108, 0.01724, 0.02761, -0.0447, -0.04043] folds won 6
holdout 20d lightgbm: [0.22307, 0.15348, 0.06973, 0.03056, 0.16906, 0.02015, 0.04304, 0.40191] days [55, 61, 65, 62, 55, 61, 63, 7]
holdout 20d ridge:    [0.1947, 0.08889, 0.07423, 0.03361, 0.07266, 0.10861, 0.08397, 0.43956]
   lightgbm mean8 0.13888 mean without 51 0.10130 days-weighted 0.10282
   ridge    mean8 0.13703 mean without 51 0.09381 days-weighted 0.09797
   lightgbm minus ridge per fold [0.02837, 0.06459, -0.0045, -0.00305, 0.0964, -0.08846, -0.04093, -0.03765] folds won 3
total lightgbm 5 beat_baseline_folds 8.0
total lightgbm 5 beat_baseline_share 0.75
total lightgbm 5 neutral_rank_ic 0.05674
total lightgbm 5 neutral_rank_ic_nw_t 10.28374
total lightgbm 5 positive_fold_share 1.0
total lightgbm 5 rank_ic 0.1058
total lightgbm 5 rank_icir 0.55816
total lightgbm 5 rank_ic_nw_t 6.80823
total lightgbm 20 beat_baseline_folds 8.0
total lightgbm 20 beat_baseline_share 0.375
total lightgbm 20 neutral_rank_ic 0.06254
total lightgbm 20 neutral_rank_ic_nw_t 5.76073
total lightgbm 20 positive_fold_share 1.0
total lightgbm 20 rank_ic 0.10282
total lightgbm 20 rank_icir 0.59437
total lightgbm 20 rank_ic_nw_t 3.69924
total ridge 5 beat_baseline_folds None
total ridge 5 beat_baseline_share None
total ridge 5 neutral_rank_ic 0.04282
total ridge 5 neutral_rank_ic_nw_t 8.58118
total ridge 5 positive_fold_share 1.0
total ridge 5 rank_ic 0.09677
total ridge 5 rank_icir 0.52715
total ridge 5 rank_ic_nw_t 5.94502
total ridge 20 beat_baseline_folds None
total ridge 20 beat_baseline_share None
total ridge 20 neutral_rank_ic 0.05418
total ridge 20 neutral_rank_ic_nw_t 5.56284
total ridge 20 positive_fold_share 1.0
total ridge 20 rank_ic 0.09797
total ridge 20 rank_icir 0.64525
total ridge 20 rank_ic_nw_t 3.91537
```

## Holdout（2026-11-12）：區間與絕對報酬的研究期對照（讀研究 run 的預測快取）

```

== interval research run interval-std1109 folds 8 .. 43
pinball 差（模型減波動率區間）
  per fold: [0.00372, 1e-05, -0.00047, -0.00322, -0.00157, -0.00274, -0.0004, -0.00059, -0.00234, -0.00077, -0.00103, -0.0002, 0.00445, -0.00132, -0.00135, -0.00175, 0.00656, 0.00191, -0.0014, -0.00253, -0.00621, -0.00305, -0.00286, -0.00089, -0.00117, -6e-05, -0.00275, -0.00062, -0.00043, -0.00182, -0.00058, -0.00237, -0.00447, -0.00209, -0.00374, -0.00147]
  mean of 36 folds -0.0011 sd of fold values 0.00237
  8-fold windows: n 29 min -0.00244 ending 34 max 0.00091 ending 25 median -0.00121 latest (36-43) -0.00212
  lowest three: [(27, 34, -0.00244), (26, 33, -0.00227), (28, 35, -0.0022)] highest three: [(17, 24, 0.00057), (19, 26, 0.00086), (18, 25, 0.00091)]
Brier 差（模型減 logistic）
  per fold: [0.00039, -0.01859, -0.0134, -0.00378, -0.00073, -0.00398, -0.0005, -0.00111, -0.00138, -0.00266, 0.00329, -0.01309, 0.00037, 0.00102, 0.00109, -0.00215, -0.02621, -0.00721, -0.00117, -0.00099, -0.01092, -0.00391, 0.00421, -0.00227, -0.00521, 0.00041, -0.00203, 0.00027, 0.00436, -0.00134, 0.00117, -0.00288, 0.00123, -0.00062, -0.00181, -0.00159]
  mean of 36 folds -0.0031 sd of fold values 0.00627
  8-fold windows: n 29 min -0.00643 ending 29 max 0.00015 ending 40 median -0.00248 latest (36-43) -0.00019
  lowest three: [(22, 29, -0.00643), (24, 31, -0.00606), (23, 30, -0.00604)] highest three: [(34, 41, 2e-05), (35, 42, 5e-05), (33, 40, 0.00015)]
20 日覆蓋率
  per fold: [0.75659, 0.83089, 0.85596, 0.80867, 0.79467, 0.84506, 0.78714, 0.82065, 0.79938, 0.79985, 0.7015, 0.78775, 0.75493, 0.75767, 0.83188, 0.83683, 0.49065, 0.72143, 0.82837, 0.79897, 0.74376, 0.70391, 0.75771, 0.82756, 0.81239, 0.66302, 0.68072, 0.76133, 0.79436, 0.81463, 0.76813, 0.8672, 0.78388, 0.81414, 0.7528, 0.78405]
  mean of 36 folds 0.77607 sd of fold values 0.06868
  8-fold windows: n 29 min 0.73404 ending 31 max 0.8178 ending 16 median 0.76696 latest (36-43) 0.7974
  lowest three: [(24, 31, 0.73404), (23, 30, 0.7352), (18, 25, 0.73533)] highest three: [(8, 15, 0.81245), (10, 17, 0.81392), (9, 16, 0.8178)]
== return research run return-std1109 folds 8 .. 43
MAE 差（模型減零預測）
  per fold: [0.00027, -0.0001, 0.00024, -0.0003, 4e-05, -0.00042, 8e-05, -0.00019, -0.00034, -0.00069, -0.00054, -0.00075, 0.00128, -7e-05, -8e-05, -0.00016, -0.0002, 0.00064, -0.00049, -6e-05, 0.0001, -0.00221, -0.00112, 0.00046, -0.0008, -0.0009, -0.00037, -0.00036, 0.00146, -0.00022, -0.0002, -0.00092, -0.00038, 0.00015, -0.00109, -0.0005]
  mean of 36 folds -0.00024 sd of fold values 0.00065
  8-fold windows: n 29 min -0.00065 ending 35 max 0.00011 ending 27 median -0.0002 latest (36-43) -0.00021
  lowest three: [(28, 35, -0.00065), (26, 33, -0.00063), (27, 34, -0.00061)] highest three: [(18, 25, 1e-05), (19, 26, 2e-05), (20, 27, 0.00011)]
```

## Holdout（2026-11-12）：區間覆蓋率在影子模型評估期那幾個月的粗估

```

claim last_date 2026-11-11 settled cutoff 2026-10-12
days in window 104 first 2026-05-13 last 2026-10-12
coverage in window (mean of daily) 0.744722
             min         max  nunique
fold                                 
49    2026-05-13  2026-06-30       34
50    2026-07-01  2026-09-30       63
51    2026-10-01  2026-10-12        7
           mean  size
2026-05  0.7143    13
2026-06  0.6994    21
2026-07  0.7361    22
2026-08  0.8668    21
2026-09  0.8127    20
2026-10  0.4039     7
whole holdout (mean of daily) 0.749898 days 429
```

## Holdout（2026-11-12）：用 holdout 的預測照監控的 60 日窗算

```
monitor window 60 thresholds from rank-cand60 {5: 0.09233, 20: 0.03824}
h5: IC days 444 mean 0.10580; rolling 60-day windows 385, min 0.02918 on 2026-05-18, below threshold 217 (56.4%) first below 2025-08-13 last below 2026-10-12 latest window 0.10277
h20: IC days 429 mean 0.10282; rolling 60-day windows 370, min -0.03427 on 2026-07-15, below threshold 116 (31.4%) first below 2025-08-05 last below 2026-09-23 latest window 0.12016
monitor window 60 research logistic Brier (the monitor's baseline) 0.16504
coverage: days 429 mean 0.74990; rolling 60-day windows 370, min 0.61008 on 2025-04-09, max 0.84821, outside 70-90%: 56 (15.1%), below 70%: 56, latest window 0.78721
Brier model: days 429 mean 0.16491; rolling windows 370, min 0.12817, max 0.22112, above the research baseline: 118 (31.9%), latest window 0.18070
Brier logistic in holdout: days 429 mean 0.16493; rolling windows 370, min 0.12605, max 0.22297, above the research baseline: 116 (31.4%), latest window 0.18187
share of rows with the up label true, holdout mean of daily 0.2122
```

## Holdout（2026-11-12）：絕對報酬各摺 MAE 差的未四捨五入值

用報告同一條評估路徑（`_frame`、`_settled`、`evaluate_return`、`fold_gap`）重算；每列是摺、MAE 差、模型的 MAE、零預測的 MAE：

```
44 -0.00019709157778124986 model 0.04750882809724165 zero 0.0477059196750229
45 -0.001758740170557882 model 0.04239749032227938 zero 0.04415623049283726
46 -0.0003463411693883159 model 0.03932989012417342 zero 0.03967623129356174
47 -0.00017504449667247318 model 0.041138600761618356 zero 0.04131364525829083
48 -0.0001221000879866735 model 0.051062822324564036 zero 0.05118492241255071
49 3.7840083728785112e-06 model 0.06057873891590465 zero 0.06057495490753177
50 -0.0003833587037966671 model 0.05312567538805395 zero 0.053509034091850614
51 -0.0017755612833179543 model 0.05852898822568541 zero 0.060304549509003366
mean of folds -0.0005943066851410422 pooled model 0.048298608451424835 pooled zero 0.048796776804773116
```

## Holdout（2026-11-12）：門檻 (3) 的合併值，8 摺與不計摺 51

用報告同一條評估路徑重算；「all 8 folds」那一段和三份報告相同。最後一行是波動率區間（頁面的備援）的覆蓋率。

```
== all 8 folds (must match the reports)
  rank h5 pooled IC: 0.10580 (NW t 6.81); ridge 0.09677
  rank h20 pooled IC: 0.10282 (NW t 3.70); ridge 0.09797
  return pooled MAE: 0.048299 vs zero 0.048797 (gap -0.000498, fold-gap NW t -2.55)
  interval pooled coverage: 0.74990
  Brier: 0.164910 vs logistic 0.164928 (gap -0.000017, fold-gap NW t -0.90)
  QLIKE: 0.46734 vs tech_vol 0.65396 (fold-gap NW t -3.35)
  pinball: 0.102653 vs vol band 0.105213 (fold-gap NW t -2.31); folds where the model is better 5 of 8
  AUC (mean of daily): 0.61061 vs logistic 0.60287 (fold-gap NW t 1.25)
  vol-band coverage (mean of daily): 0.78694 over 429 days
== without fold 51
  rank h5 pooled IC: 0.10024 (NW t 6.34); ridge 0.08863
  rank h20 pooled IC: 0.09786 (NW t 3.52); ridge 0.09230
  return pooled MAE: 0.047765 vs zero 0.048197 (gap -0.000432, fold-gap NW t -2.10)
  interval pooled coverage: 0.75564
  Brier: 0.163547 vs logistic 0.163419 (gap +0.000128, fold-gap NW t 0.32)
  QLIKE: 0.46981 vs tech_vol 0.65208 (fold-gap NW t -2.73)
  pinball: 0.101895 vs vol band 0.104339 (fold-gap NW t -1.85); folds where the model is better 4 of 7
  AUC (mean of daily): 0.61478 vs logistic 0.60839 (fold-gap NW t 0.70)
  vol-band coverage (mean of daily): 0.79279 over 422 days
```

## Holdout（2026-11-12）：label 重建多出來的三檔在特徵分區裡沒有列

2026 年的特徵分區（建到 2026-11-06）裡，4207、5355、8176 沒有任何列（同一查法 2330 有 203 列），所以三個目標的預測快取都沒有這三檔，label 是接在預測列上的，不會進評估：

```
stock_id type string date type date32[day]
2330 rows 203 first 2026-01-02 last 2026-11-06 in_universe days 203
4207 rows 0 first None last None in_universe days 0
5355 rows 0 first None last None in_universe days 0
8176 rows 0 first None last None in_universe days 0
partition rows 397489 stocks 1989 last day 2026-11-06 rows on last day 1981 share in_universe 0.5111
```

## 集保兩欄填值後模型輸出變不變（2026-11-12）

讀三支 holdout 與三支研究 run 已 fit 的模型檔；LightGBM 看每個 booster 的分裂次數與 gain 並把兩欄填成亂數直接比，線性模型看係數並把兩個缺值旗標由 1 改成 0 直接比：

```
feature rows on 2026-11-04 1981 | stored values in the two columns: {'chip_big_holder_pct_chg_1w': 0, 'chip_holder_count_chg_1w': 0}
interval-holdout-20261112:histq:no model file 8
interval-holdout-20261112:lightgbm:boosters 200
interval-holdout-20261112:lightgbm:checked 24
interval-holdout-20261112:lightgbm:predicted both ways 3
interval-holdout-20261112:logistic:checked 8
interval-std1109:histq:no model file 39
interval-std1109:lightgbm:boosters 195
interval-std1109:lightgbm:checked 117
interval-std1109:lightgbm:predicted both ways 3
interval-std1109:logistic:checked 39
rank-cand60:lightgbm:boosters 78
rank-cand60:lightgbm:checked 78
rank-cand60:lightgbm:predicted both ways 2
rank-cand60:ridge:checked 78
rank-holdout-20261112:lightgbm:boosters 80
rank-holdout-20261112:lightgbm:checked 16
rank-holdout-20261112:lightgbm:predicted both ways 2
rank-holdout-20261112:ridge:checked 16
return-holdout-20261112:lightgbm:boosters 40
return-holdout-20261112:lightgbm:checked 8
return-holdout-20261112:lightgbm:predicted both ways 1
return-std1109:lightgbm:boosters 39
return-std1109:lightgbm:checked 39
return-std1109:lightgbm:predicted both ways 1
worst over every model: {'gain': 0.0, 'split': 0, 'coef_ridge': 0.0, 'coef_logistic': 0.5142943355447955, 'lgbm_diff': 0.0, 'linear_diff': 0.19740894390315236}
```

logistic 的兩個缺值旗標的係數（47 個模型裡的前後各三個），以及旗標由 1 變 0 時機率的變化（其他輸入都在 0）：

```
interval-holdout-20261112.logistic.h20.f044 {'chip_big_holder_pct_chg_1w': 0.0, 'chip_holder_count_chg_1w': 0.0, 'chip_big_holder_pct_chg_1w_missing': -0.42761, 'chip_holder_count_chg_1w_missing': -0.42761} intercept -0.4335 prob shift 0.17722
interval-holdout-20261112.logistic.h20.f045 {'chip_big_holder_pct_chg_1w': 0.0, 'chip_holder_count_chg_1w': 0.0, 'chip_big_holder_pct_chg_1w_missing': -0.41727, 'chip_holder_count_chg_1w_missing': -0.41727} intercept -0.4229 prob shift 0.1744
interval-holdout-20261112.logistic.h20.f046 {'chip_big_holder_pct_chg_1w': 0.0, 'chip_holder_count_chg_1w': 0.0, 'chip_big_holder_pct_chg_1w_missing': -0.43283, 'chip_holder_count_chg_1w_missing': -0.43283} intercept -0.4384 prob shift 0.17864
interval-std1109.logistic.h20.f041 {'chip_big_holder_pct_chg_1w': 0.0, 'chip_holder_count_chg_1w': 0.0, 'chip_big_holder_pct_chg_1w_missing': -0.42822, 'chip_holder_count_chg_1w_missing': -0.42822} intercept -0.4346 prob shift 0.17735
interval-std1109.logistic.h20.f042 {'chip_big_holder_pct_chg_1w': 0.0, 'chip_holder_count_chg_1w': 0.0, 'chip_big_holder_pct_chg_1w_missing': -0.42955, 'chip_holder_count_chg_1w_missing': -0.42955} intercept -0.4361 prob shift 0.1777
interval-std1109.logistic.h20.f043 {'chip_big_holder_pct_chg_1w': 0.0, 'chip_holder_count_chg_1w': 0.0, 'chip_big_holder_pct_chg_1w_missing': -0.42908, 'chip_holder_count_chg_1w_missing': -0.42908} intercept -0.4312 prob shift 0.17788
models 47 max |coef| per name: 兩欄本身 0.0；兩個缺值旗標各 0.51429
prob shift when both flags flip (all other inputs at 0): min 0.11698 max 0.19741 mean abs 0.18197
```

## Holdout（2026-11-12）：波動率區間（頁面備援）的覆蓋率與缺波動率的列

和模型的覆蓋率同一個算法（每天 universe 內可交易列落在區間內的比例，再對天數取平均）；另印把所有列合起來算的值，以及缺 `tech_vol_20d` 的列數：

```

== all 8 folds | tradable rows with a label 404544 | tech_vol_20d missing 0 (0.0000%) | zero-width vol band 0 | both 0 | days 429
   vol band, every row: mean of daily 0.78694 over 429 days; pooled rows 0.78696
   vol band, rows with tech_vol_20d: mean of daily 0.78694 over 429 days; pooled rows 0.78696
   model, every row: mean of daily 0.74990 over 429 days; pooled rows 0.74744
   model, rows with tech_vol_20d: mean of daily 0.74990 over 429 days; pooled rows 0.74744
   missing rows per day: mean 0.0 max 0 days with any 0
== without fold 51 | tradable rows with a label 398174 | tech_vol_20d missing 0 (0.0000%) | zero-width vol band 0 | both 0 | days 422
   vol band, every row: mean of daily 0.79279 over 422 days; pooled rows 0.79261
   vol band, rows with tech_vol_20d: mean of daily 0.79279 over 422 days; pooled rows 0.79261
   model, every row: mean of daily 0.75564 over 422 days; pooled rows 0.75294
   model, rows with tech_vol_20d: mean of daily 0.75564 over 422 days; pooled rows 0.75294
   missing rows per day: mean 0.0 max 0 days with any 0
```

## Holdout（2026-11-12）：波動率區間（頁面備援）逐摺的覆蓋率

2026-11-13 補算，只讀：讀 `interval-holdout-20261112` 已經存下的預測快取與 `label_daily`，沒有重新訓練、沒有重跑 holdout、沒有寫資料庫。算法和上一節相同（每天 universe 內可交易、20 日 label 在 2026-10-12 以前已到期的列落在 q10 到 q90 之內的比例，再對天數取平均；label 是 `fwd_ret_open`），改成一摺一摺讀。模型那一欄是對帳用的：8 摺的值和 `interval-holdout-20261112.md`「每摺覆蓋率」那張表逐一相同，合併值、不計摺 51 的值、列數與天數和上一節相同。「below q10」「above q90」是每天低於下界、高於上界的列的比例，再對天數取平均。摺 51 的 7 天就是 holdout 最後 7 個 label 已到期的交易日。

```
run interval-holdout-20261112; claim last_date 2026-11-11; labels settled through 2026-10-12; label fwd_ret_open
band columns volband_q10_h20, volband_q90_h20; model columns lightgbm_q10_h20, lightgbm_q90_h20
fold | first day | last day | days | rows | band coverage | model coverage | band: below q10 | band: above q90 | tech_vol_20d missing
44 | 2025-01-02 | 2025-03-31 | 55 | 51188 | 0.62809 | 0.59403 | 0.28184 | 0.09007 | 0
45 | 2025-04-01 | 2025-06-30 | 61 | 54018 | 0.89404 | 0.84270 | 0.02866 | 0.07729 | 0
46 | 2025-07-01 | 2025-09-30 | 65 | 57592 | 0.79429 | 0.78305 | 0.04725 | 0.15846 | 0
47 | 2025-10-01 | 2025-12-31 | 62 | 55819 | 0.79390 | 0.79311 | 0.07187 | 0.13423 | 0
48 | 2026-01-02 | 2026-03-31 | 55 | 53520 | 0.78644 | 0.75727 | 0.07307 | 0.14049 | 0
49 | 2026-04-01 | 2026-06-30 | 61 | 63371 | 0.77665 | 0.69559 | 0.04817 | 0.17517 | 0
50 | 2026-07-01 | 2026-09-30 | 63 | 62666 | 0.85706 | 0.80398 | 0.04453 | 0.09841 | 0
51 | 2026-10-01 | 2026-10-12 | 7 | 6370 | 0.43449 | 0.40393 | 0.52828 | 0.03723 | 0
all folds: days 429, rows 404544
  band: mean of daily 0.78694; all rows together 0.78696
  model: mean of daily 0.74990; all rows together 0.74744
  band: mean of the 8 fold values 0.74562; folds under 0.70: 2
  model: mean of the 8 fold values 0.70921; folds under 0.70: 3
without fold 51: days 422
  band: mean of daily 0.79279
  model: mean of daily 0.75564
fold 51 day by day (band, model):
  2026-10-01 | 0.55458 | 0.53583
  2026-10-02 | 0.50721 | 0.49390
  2026-10-05 | 0.38022 | 0.35385
  2026-10-06 | 0.31366 | 0.28087
  2026-10-07 | 0.38974 | 0.34934
  2026-10-08 | 0.41630 | 0.38877
  2026-10-12 | 0.47974 | 0.42497
band folds under 0.70: [0.62809, 0.43449]; over 0.90: []
```
