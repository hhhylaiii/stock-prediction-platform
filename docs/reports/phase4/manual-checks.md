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
