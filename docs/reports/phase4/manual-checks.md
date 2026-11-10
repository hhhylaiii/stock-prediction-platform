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
| tech 組（seed 0） | 28.2 分 | 6.35 GB |
| tech 組（seed 0、3、6；14:55 停在第 84 個 fit） | 63.0 分 | 未記錄 |
| tech 組（seed 0、3、6；接續剩下的 24 個 fit） | 23.1 分 | 6.46 GB |
| chip 組（seed 0） | 28.0 分 | 5.97 GB |
| chip 組（seed 0、3、6） | 56.4 分 | 6.00 GB |
| ind 組（seed 0） | 28.5 分 | 5.99 GB |
| ind 組（seed 0、3、6） | 55.1 分 | 6.47 GB |
