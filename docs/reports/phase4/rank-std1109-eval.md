# 評估報表：rank-std1109（5 日）

> 手動附註：篩選 `p3-final-20261109`（develop `4e69871`），凍結設計（60 個月窗、重訓輪數 max(最佳輪數, 200)、每日一列、regression，第一批篩選後的 55 個特徵），seed 0

## 排名 5 日

- 研究期 IC 均值 0.1383，標準差 0.1335，NW t 27.66，ICIR 1.04，2175 天
- IC 為正的研究摺比例：100%
- 中性化 IC：0.0870，NW t 41.29，2175 天
- 贏 baseline 的研究摺比例：94%（36／36 摺可比）
- 🔴 洩漏警訊：第 24、25 摺比同摺 baseline 的 IC 高 0.05 以上，先當成洩漏追查
- 打亂 label：這次沒跑，未判定

### 每摺

| 摺 | 類別 | 測試期 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 | 共同列 IC | baseline IC | 贏 baseline | 洩漏 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 調參 | 2015-04-01–2015-06-30 | 0.1482 | 0.1184 | 1.25 | 5.41 | 89% | 61 | 0.1482 | 0.1186 | 是 | — |
| 6 | 調參 | 2015-07-01–2015-09-30 | 0.1209 | 0.1122 | 1.08 | 4.28 | 83% | 63 | 0.1209 | 0.0985 | 是 | — |
| 7 | 調參 | 2015-10-01–2015-12-03 | 0.1031 | 0.1287 | 0.80 | 4.06 | 82% | 45 | 0.1031 | 0.0809 | 是 | — |
| 8 | 研究 | 2016-01-04–2016-03-31 | 0.1271 | 0.0967 | 1.31 | 5.62 | 91% | 57 | 0.1271 | 0.1155 | 是 | 否 |
| 9 | 研究 | 2016-04-01–2016-06-30 | 0.1457 | 0.0985 | 1.48 | 6.17 | 89% | 61 | 0.1457 | 0.1295 | 是 | 否 |
| 10 | 研究 | 2016-07-01–2016-09-30 | 0.1638 | 0.0927 | 1.77 | 9.43 | 97% | 62 | 0.1638 | 0.1313 | 是 | 否 |
| 11 | 研究 | 2016-10-03–2016-12-30 | 0.1499 | 0.1133 | 1.32 | 6.15 | 91% | 64 | 0.1499 | 0.1309 | 是 | 否 |
| 12 | 研究 | 2017-01-03–2017-03-31 | 0.1358 | 0.1133 | 1.20 | 5.67 | 88% | 57 | 0.1358 | 0.1206 | 是 | 否 |
| 13 | 研究 | 2017-04-05–2017-06-30 | 0.2104 | 0.1025 | 2.05 | 9.88 | 97% | 61 | 0.2104 | 0.1910 | 是 | 否 |
| 14 | 研究 | 2017-07-03–2017-09-30 | 0.1223 | 0.1156 | 1.06 | 4.45 | 86% | 66 | 0.1223 | 0.0964 | 是 | 否 |
| 15 | 研究 | 2017-10-02–2017-12-29 | 0.1650 | 0.0868 | 1.90 | 9.15 | 95% | 62 | 0.1650 | 0.1378 | 是 | 否 |
| 16 | 研究 | 2018-01-02–2018-03-31 | 0.1706 | 0.1211 | 1.41 | 6.69 | 93% | 58 | 0.1706 | 0.1522 | 是 | 否 |
| 17 | 研究 | 2018-04-02–2018-06-29 | 0.1576 | 0.1115 | 1.41 | 5.83 | 92% | 60 | 0.1576 | 0.1455 | 是 | 否 |
| 18 | 研究 | 2018-07-02–2018-09-28 | 0.1923 | 0.1151 | 1.67 | 9.40 | 95% | 64 | 0.1923 | 0.1813 | 是 | 否 |
| 19 | 研究 | 2018-10-01–2018-12-28 | 0.1168 | 0.2114 | 0.55 | 2.56 | 72% | 65 | 0.1168 | 0.0982 | 是 | 否 |
| 20 | 研究 | 2019-01-02–2019-03-29 | 0.0958 | 0.1285 | 0.75 | 3.12 | 74% | 54 | 0.0958 | 0.0855 | 是 | 否 |
| 21 | 研究 | 2019-04-01–2019-06-28 | 0.1660 | 0.1504 | 1.10 | 4.69 | 87% | 61 | 0.1660 | 0.1492 | 是 | 否 |
| 22 | 研究 | 2019-07-01–2019-09-27 | 0.1032 | 0.1237 | 0.83 | 4.00 | 83% | 63 | 0.1032 | 0.0793 | 是 | 否 |
| 23 | 研究 | 2019-10-01–2019-12-31 | 0.1562 | 0.1047 | 1.49 | 7.27 | 95% | 64 | 0.1562 | 0.1296 | 是 | 否 |
| 24 | 研究 | 2020-01-02–2020-03-31 | 0.1545 | 0.1420 | 1.09 | 6.62 | 88% | 56 | 0.1545 | 0.0948 | 是 | 是 |
| 25 | 研究 | 2020-04-01–2020-06-30 | 0.1077 | 0.1200 | 0.90 | 4.39 | 77% | 60 | 0.1077 | 0.0425 | 是 | 是 |
| 26 | 研究 | 2020-07-01–2020-09-30 | 0.1435 | 0.1440 | 1.00 | 4.84 | 79% | 66 | 0.1435 | 0.1116 | 是 | 否 |
| 27 | 研究 | 2020-10-05–2020-12-31 | 0.1218 | 0.1173 | 1.04 | 4.76 | 83% | 63 | 0.1218 | 0.1100 | 是 | 否 |
| 28 | 研究 | 2021-01-04–2021-03-31 | 0.1223 | 0.1526 | 0.80 | 3.28 | 69% | 55 | 0.1223 | 0.0917 | 是 | 否 |
| 29 | 研究 | 2021-04-01–2021-06-30 | 0.1514 | 0.1544 | 0.98 | 5.08 | 87% | 61 | 0.1514 | 0.1302 | 是 | 否 |
| 30 | 研究 | 2021-07-01–2021-09-30 | 0.1919 | 0.1264 | 1.52 | 8.78 | 95% | 64 | 0.1919 | 0.1746 | 是 | 否 |
| 31 | 研究 | 2021-10-01–2021-12-30 | 0.1115 | 0.1111 | 1.00 | 4.54 | 84% | 64 | 0.1115 | 0.0925 | 是 | 否 |
| 32 | 研究 | 2022-01-03–2022-03-31 | 0.1564 | 0.1283 | 1.22 | 5.22 | 89% | 56 | 0.1564 | 0.1558 | 是 | 否 |
| 33 | 研究 | 2022-04-01–2022-06-30 | 0.1331 | 0.1125 | 1.18 | 5.99 | 92% | 61 | 0.1331 | 0.1299 | 是 | 否 |
| 34 | 研究 | 2022-07-01–2022-09-30 | 0.0937 | 0.1747 | 0.54 | 2.51 | 69% | 65 | 0.0937 | 0.0715 | 是 | 否 |
| 35 | 研究 | 2022-10-03–2022-12-30 | 0.1003 | 0.1896 | 0.53 | 2.43 | 72% | 64 | 0.1003 | 0.1118 | 否 | 否 |
| 36 | 研究 | 2023-01-03–2023-03-31 | 0.0943 | 0.1559 | 0.60 | 2.76 | 74% | 54 | 0.0943 | 0.0824 | 是 | 否 |
| 37 | 研究 | 2023-04-06–2023-06-30 | 0.1196 | 0.1286 | 0.93 | 4.58 | 83% | 59 | 0.1196 | 0.0937 | 是 | 否 |
| 38 | 研究 | 2023-07-03–2023-09-28 | 0.1383 | 0.1380 | 1.00 | 4.30 | 83% | 63 | 0.1383 | 0.1278 | 是 | 否 |
| 39 | 研究 | 2023-10-02–2023-12-29 | 0.1343 | 0.1245 | 1.08 | 4.95 | 83% | 63 | 0.1343 | 0.1266 | 是 | 否 |
| 40 | 研究 | 2024-01-02–2024-03-29 | 0.1122 | 0.1225 | 0.92 | 3.69 | 82% | 56 | 0.1122 | 0.1159 | 否 | 否 |
| 41 | 研究 | 2024-04-01–2024-06-28 | 0.1143 | 0.1330 | 0.86 | 3.61 | 79% | 61 | 0.1143 | 0.1061 | 是 | 否 |
| 42 | 研究 | 2024-07-01–2024-09-30 | 0.1486 | 0.1222 | 1.22 | 5.93 | 92% | 63 | 0.1486 | 0.1151 | 是 | 否 |
| 43 | 研究 | 2024-10-01–2024-12-03 | 0.1412 | 0.1558 | 0.91 | 4.02 | 81% | 42 | 0.1412 | 0.1218 | 是 | 否 |

### 每年

| 年 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0.1470 | 0.1010 | 1.46 | 12.98 | 92% | 244 |
| 2017 | 0.1580 | 0.1099 | 1.44 | 12.51 | 91% | 246 |
| 2018 | 0.1589 | 0.1490 | 1.07 | 9.55 | 88% | 247 |
| 2019 | 0.1314 | 0.1305 | 1.01 | 8.69 | 85% | 242 |
| 2020 | 0.1317 | 0.1318 | 1.00 | 9.67 | 81% | 245 |
| 2021 | 0.1450 | 0.1393 | 1.04 | 9.61 | 84% | 244 |
| 2022 | 0.1195 | 0.1570 | 0.76 | 6.96 | 80% | 246 |
| 2023 | 0.1227 | 0.1369 | 0.90 | 8.12 | 81% | 239 |
| 2024 | 0.1286 | 0.1323 | 0.97 | 8.30 | 84% | 222 |

### 成交金額三分位

| 組 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 低 | 0.1432 | 0.1371 | 1.04 | 28.48 | 86% | 2175 |
| 中 | 0.1436 | 0.1376 | 1.04 | 28.03 | 86% | 2175 |
| 高 | 0.1266 | 0.1632 | 0.78 | 20.96 | 78% | 2175 |

### 十分位（扣成本）

| 樣本 | 組合 | 月報酬 | 月化超額 | 超額最大回撤 | 年化換手率（倍） | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | D1 | -4.63% | -4.59% | -99.40% | 34.57 | 2169 |
| 全部 | D2 | -3.22% | -3.17% | -97.01% | 41.32 | 2169 |
| 全部 | D3 | -2.62% | -2.58% | -94.18% | 42.75 | 2169 |
| 全部 | D4 | -2.09% | -2.05% | -89.53% | 43.35 | 2169 |
| 全部 | D5 | -1.74% | -1.69% | -84.47% | 43.53 | 2169 |
| 全部 | D6 | -1.54% | -1.50% | -80.49% | 43.28 | 2169 |
| 全部 | D7 | -1.21% | -1.17% | -72.02% | 42.60 | 2169 |
| 全部 | D8 | -0.84% | -0.79% | -58.23% | 41.28 | 2169 |
| 全部 | D9 | -0.30% | -0.25% | -30.03% | 38.19 | 2169 |
| 全部 | D10 | 0.83% | 0.88% | -9.10% | 27.07 | 2169 |
| 全部 | universe | -0.04% | 0.00% | 0.00% | 2.92 | 2169 |
| 全部 | D10-D1 | 2.37% | 2.37% | -11.95% | 61.64 | 2169 |
| 成交金額前三分之一 | D1 | -4.74% | -4.57% | -99.41% | 35.18 | 2169 |
| 成交金額前三分之一 | D2 | -3.51% | -3.34% | -97.56% | 41.96 | 2169 |
| 成交金額前三分之一 | D3 | -2.68% | -2.52% | -93.77% | 43.18 | 2169 |
| 成交金額前三分之一 | D4 | -2.16% | -2.00% | -89.29% | 43.92 | 2169 |
| 成交金額前三分之一 | D5 | -1.84% | -1.67% | -84.29% | 43.91 | 2169 |
| 成交金額前三分之一 | D6 | -1.60% | -1.43% | -79.22% | 43.61 | 2169 |
| 成交金額前三分之一 | D7 | -1.37% | -1.20% | -73.31% | 42.92 | 2169 |
| 成交金額前三分之一 | D8 | -0.82% | -0.65% | -52.35% | 41.52 | 2169 |
| 成交金額前三分之一 | D9 | -0.23% | -0.06% | -27.85% | 37.69 | 2169 |
| 成交金額前三分之一 | D10 | 0.87% | 1.04% | -12.29% | 27.13 | 2169 |
| 成交金額前三分之一 | universe | -0.17% | 0.00% | 0.00% | 4.16 | 2169 |
| 成交金額前三分之一 | D10-D1 | 2.45% | 2.45% | -16.47% | 62.31 | 2169 |

## 特徵重要度前 10 名

| 摺 | 前 10 名（gain） | 和前一摺重疊 |
| --- | --- | --- |
| 8 | tech_ma_gap_5d, risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_ret_1d, tech_kd_d, tech_turnover_rate_5d, tech_limit_up_t, tech_mom_250_20 | — |
| 9 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_turnover_rate_5d, tech_mom_250_20, tech_kd_d, tech_ret_1d, tech_limit_up_cnt_20d | 9 |
| 10 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ret_1d, fund_dividend_yield, tech_turnover_rate_5d, tech_kd_d, chip_it_net_5d, tech_limit_up_cnt_20d | 9 |
| 11 | tech_ma_gap_5d, risk_idio_vol_60d, chip_margin_usage, tech_limit_down_cnt_20d, tech_turnover_rate_5d, tech_ret_1d, fund_dividend_yield, chip_it_net_5d, chip_fi_net_1d, tech_kd_d | 9 |
| 12 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_limit_down_cnt_20d, tech_turnover_rate_5d, chip_fi_net_1d, chip_it_net_5d, tech_kd_d, tech_high_52w_gap | 9 |
| 13 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ret_1d, tech_turnover_rate_5d, chip_it_net_5d, tech_kd_d, fund_dividend_yield, chip_fi_net_1d | 9 |
| 14 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ret_1d, tech_turnover_rate_5d, chip_it_net_5d, chip_fi_net_1d, tech_ret_5d, tech_kd_d | 9 |
| 15 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_1d, chip_it_net_5d, tech_turnover_rate_5d, tech_ret_5d, tech_kd_d, chip_fi_net_1d | 10 |
| 16 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_1d, chip_it_net_5d, tech_ret_5d, tech_kd_d, chip_fi_net_1d, tech_turnover_rate_5d | 10 |
| 17 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_1d, tech_kd_d, chip_it_net_5d, tech_ret_5d, tech_turnover_rate_5d, chip_fi_net_1d | 10 |
| 18 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_1d, tech_ret_5d, tech_kd_d, chip_fi_net_1d, chip_it_net_5d, tech_vol_20d | 9 |
| 19 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_kd_d, tech_vol_20d, tech_ret_5d, tech_limit_down_cnt_20d, tech_ret_1d, chip_fi_net_1d, chip_it_net_5d | 10 |
| 20 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_ret_5d, tech_kd_d, tech_vol_20d, tech_limit_down_cnt_20d, tech_ret_1d, chip_fi_net_1d, chip_it_net_5d | 10 |
| 21 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_5d, tech_vol_20d, tech_kd_d, tech_turnover_rate_5d, chip_fi_net_1d, tech_limit_up_cnt_20d | 8 |
| 22 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, tech_vol_20d, tech_ret_5d, tech_limit_up_cnt_20d, tech_turnover_rate_5d, tech_kd_d, tech_high_52w_gap | 9 |
| 23 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, tech_ma_gap_5d, tech_vol_20d, tech_turnover_rate_5d, fund_dividend_yield, tech_kd_d, tech_ret_5d | 9 |
| 24 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_limit_up_cnt_20d, fund_dividend_yield, fund_roe_ttm, tech_vol_20d, fund_ep, fund_rev_3m_yoy, fund_rev_yoy | 6 |
| 25 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, tech_ma_gap_5d, fund_dividend_yield, tech_vol_20d, tech_ret_5d, tech_turnover_rate_5d, tech_high_52w_gap | 6 |
| 26 | risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, fund_dividend_yield, tech_vol_20d, tech_ret_5d, tech_high_52w_gap, tech_turnover_rate_5d | 10 |
| 27 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, tech_vol_20d, tech_high_52w_gap, tech_turnover_rate_5d, tech_ret_5d | 10 |
| 28 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, tech_vol_20d, tech_turnover_rate_5d, tech_high_52w_gap, tech_ret_5d | 10 |
| 29 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_vol_20d, tech_turnover_rate_5d, fund_dividend_yield, tech_high_52w_gap, tech_ret_5d | 10 |
| 30 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_vol_20d, tech_ret_1d, tech_high_52w_gap, fund_dividend_yield, tech_turnover_rate_5d | 9 |
| 31 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_vol_20d, tech_ret_1d, tech_high_52w_gap, tech_attention_t, tech_ret_5d | 8 |
| 32 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, tech_vol_20d, chip_margin_usage, tech_high_52w_gap, tech_ret_1d, tech_attention_t, tech_ret_5d | 10 |
| 33 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_vol_20d, tech_ret_1d, tech_high_52w_gap, tech_attention_t, tech_turnover_rate_5d | 9 |
| 34 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_ma_gap_5d, tech_limit_up_cnt_20d, chip_margin_usage, tech_vol_20d, tech_ret_1d, tech_high_52w_gap, tech_turnover_rate_5d, fund_dividend_yield | 9 |
| 35 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_vol_20d, tech_high_52w_gap, tech_attention_t, tech_ret_1d, tech_limit_up_t | 8 |
| 36 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_attention_t, tech_vol_20d, tech_high_52w_gap, risk_beta_250d, fund_dividend_yield | 8 |
| 37 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, chip_margin_usage, risk_beta_250d, tech_ma_gap_5d, tech_attention_t, fund_dividend_yield, tech_vol_20d, tech_high_52w_gap | 10 |
| 38 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_attention_t, tech_high_52w_gap, risk_beta_250d, tech_vol_20d, fund_dividend_yield | 10 |
| 39 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, fund_dividend_yield, tech_high_52w_gap, tech_attention_t, tech_limit_up_t, risk_beta_250d | 9 |
| 40 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, tech_high_52w_gap, tech_turnover_rate_5d, tech_limit_up_t, tech_vol_20d | 8 |
| 41 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, tech_high_52w_gap, tech_turnover_rate_5d, tech_limit_up_t, tech_ret_1d | 9 |
| 42 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, fund_dividend_yield, chip_margin_usage, risk_beta_250d, tech_limit_up_t, tech_high_52w_gap, tech_attention_t, tech_ma_gap_5d | 8 |
| 43 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_turnover_rate_5d, fund_dividend_yield, tech_high_52w_gap, chip_margin_usage, tech_ma_gap_5d, tech_limit_up_t, risk_beta_250d | 9 |

- 進前 10 名的摺數：chip_margin_usage 36、risk_idio_vol_60d 36、tech_limit_down_cnt_20d 36、tech_ma_gap_5d 35、tech_limit_up_cnt_20d 25、tech_turnover_rate_5d 24、fund_dividend_yield 22、tech_vol_20d 22、tech_high_52w_gap 21、tech_ret_1d 20、tech_ret_5d 17、tech_kd_d 16、chip_fi_net_1d 11、chip_it_net_5d 11、tech_attention_t 9、tech_limit_up_t 7、risk_beta_250d 6、tech_mom_250_20 2、fund_ep 1、fund_rev_3m_yoy 1、fund_rev_yoy 1、fund_roe_ttm 1

## 關卡：排名 5 日

- 研究期 IC 均值 0.1383 大於 0 且 NW t 27.66 大於 2：通過
- 贏 Ridge 的研究摺 34／36 摺 達 60%：通過
- Ridge 研究期 IC 均值 0.1191，NW t 21.64，2175 天
- 重要度變動只列研究摺，調參摺不計

# 評估報表：rank-std1109（20 日）

## 排名 20 日

- 研究期 IC 均值 0.1317，標準差 0.1400，NW t 13.08，ICIR 0.94，2175 天
- IC 為正的研究摺比例：97%
- 中性化 IC：0.0799，NW t 20.80，2175 天
- 贏 baseline 的研究摺比例：61%（36／36 摺可比）
- 🔴 洩漏警訊：第 19、25、29 摺比同摺 baseline 的 IC 高 0.05 以上，先當成洩漏追查
- 打亂 label：這次沒跑，未判定

### 每摺

| 摺 | 類別 | 測試期 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 | 共同列 IC | baseline IC | 贏 baseline | 洩漏 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 調參 | 2015-04-01–2015-06-30 | 0.1346 | 0.0819 | 1.64 | 4.09 | 95% | 61 | 0.1346 | 0.1966 | 否 | — |
| 6 | 調參 | 2015-07-01–2015-09-30 | 0.0732 | 0.1897 | 0.39 | 0.86 | 57% | 63 | 0.0732 | 0.0790 | 否 | — |
| 7 | 調參 | 2015-10-01–2015-12-03 | 0.0693 | 0.0860 | 0.81 | 2.87 | 80% | 45 | 0.0693 | 0.0321 | 是 | — |
| 8 | 研究 | 2016-01-04–2016-03-31 | 0.1211 | 0.1137 | 1.07 | 2.25 | 93% | 57 | 0.1211 | 0.1161 | 是 | 否 |
| 9 | 研究 | 2016-04-01–2016-06-30 | 0.1406 | 0.1316 | 1.07 | 2.46 | 84% | 61 | 0.1406 | 0.1271 | 是 | 否 |
| 10 | 研究 | 2016-07-01–2016-09-30 | 0.1785 | 0.0808 | 2.21 | 7.69 | 97% | 62 | 0.1785 | 0.1547 | 是 | 否 |
| 11 | 研究 | 2016-10-03–2016-12-30 | 0.0693 | 0.0901 | 0.77 | 2.08 | 73% | 64 | 0.0693 | 0.0465 | 是 | 否 |
| 12 | 研究 | 2017-01-03–2017-03-31 | 0.1192 | 0.1192 | 1.00 | 2.23 | 81% | 57 | 0.1192 | 0.1225 | 否 | 否 |
| 13 | 研究 | 2017-04-05–2017-06-30 | 0.1970 | 0.1002 | 1.97 | 7.00 | 97% | 61 | 0.1970 | 0.1848 | 是 | 否 |
| 14 | 研究 | 2017-07-03–2017-09-30 | 0.0752 | 0.1156 | 0.65 | 2.22 | 80% | 66 | 0.0752 | 0.0520 | 是 | 否 |
| 15 | 研究 | 2017-10-02–2017-12-29 | 0.1475 | 0.0759 | 1.94 | 5.09 | 95% | 62 | 0.1475 | 0.1436 | 是 | 否 |
| 16 | 研究 | 2018-01-02–2018-03-31 | 0.1721 | 0.1162 | 1.48 | 4.30 | 90% | 58 | 0.1721 | 0.1784 | 否 | 否 |
| 17 | 研究 | 2018-04-02–2018-06-29 | 0.1585 | 0.1533 | 1.03 | 2.47 | 77% | 60 | 0.1585 | 0.1393 | 是 | 否 |
| 18 | 研究 | 2018-07-02–2018-09-28 | 0.3090 | 0.0817 | 3.78 | 11.57 | 100% | 64 | 0.3090 | 0.2703 | 是 | 否 |
| 19 | 研究 | 2018-10-01–2018-12-28 | 0.1211 | 0.1900 | 0.64 | 2.86 | 80% | 65 | 0.1211 | 0.0275 | 是 | 是 |
| 20 | 研究 | 2019-01-02–2019-03-29 | -0.0001 | 0.1432 | -0.00 | -0.00 | 43% | 54 | -0.0001 | 0.0011 | 否 | 否 |
| 21 | 研究 | 2019-04-01–2019-06-28 | 0.1692 | 0.2261 | 0.75 | 1.55 | 62% | 61 | 0.1692 | 0.1657 | 是 | 否 |
| 22 | 研究 | 2019-07-01–2019-09-27 | 0.0516 | 0.0973 | 0.53 | 2.11 | 78% | 63 | 0.0516 | 0.0527 | 否 | 否 |
| 23 | 研究 | 2019-10-01–2019-12-31 | 0.1445 | 0.0952 | 1.52 | 4.47 | 92% | 64 | 0.1445 | 0.1640 | 否 | 否 |
| 24 | 研究 | 2020-01-02–2020-03-31 | 0.1353 | 0.2069 | 0.65 | 1.95 | 77% | 56 | 0.1353 | 0.1218 | 是 | 否 |
| 25 | 研究 | 2020-04-01–2020-06-30 | 0.0583 | 0.1008 | 0.58 | 2.64 | 70% | 60 | 0.0583 | -0.0165 | 是 | 是 |
| 26 | 研究 | 2020-07-01–2020-09-30 | 0.0896 | 0.1143 | 0.78 | 2.21 | 77% | 66 | 0.0896 | 0.1028 | 否 | 否 |
| 27 | 研究 | 2020-10-05–2020-12-31 | 0.1214 | 0.0898 | 1.35 | 4.14 | 89% | 63 | 0.1214 | 0.1372 | 否 | 否 |
| 28 | 研究 | 2021-01-04–2021-03-31 | 0.0787 | 0.1229 | 0.64 | 1.98 | 75% | 55 | 0.0787 | 0.0501 | 是 | 否 |
| 29 | 研究 | 2021-04-01–2021-06-30 | 0.2106 | 0.0978 | 2.15 | 8.98 | 98% | 61 | 0.2106 | 0.1379 | 是 | 是 |
| 30 | 研究 | 2021-07-01–2021-09-30 | 0.2193 | 0.0875 | 2.51 | 11.68 | 100% | 64 | 0.2193 | 0.2248 | 否 | 否 |
| 31 | 研究 | 2021-10-01–2021-12-30 | 0.0901 | 0.1857 | 0.49 | 1.13 | 53% | 64 | 0.0901 | 0.0979 | 否 | 否 |
| 32 | 研究 | 2022-01-03–2022-03-31 | 0.1699 | 0.0869 | 1.95 | 6.29 | 98% | 56 | 0.1699 | 0.1559 | 是 | 否 |
| 33 | 研究 | 2022-04-01–2022-06-30 | 0.1619 | 0.1037 | 1.56 | 5.44 | 92% | 61 | 0.1619 | 0.1614 | 是 | 否 |
| 34 | 研究 | 2022-07-01–2022-09-30 | 0.1897 | 0.1749 | 1.08 | 2.62 | 86% | 65 | 0.1897 | 0.1497 | 是 | 否 |
| 35 | 研究 | 2022-10-03–2022-12-30 | 0.0195 | 0.1986 | 0.10 | 0.24 | 50% | 64 | 0.0195 | 0.0440 | 否 | 否 |
| 36 | 研究 | 2023-01-03–2023-03-31 | 0.0662 | 0.1227 | 0.54 | 1.67 | 70% | 54 | 0.0662 | 0.0520 | 是 | 否 |
| 37 | 研究 | 2023-04-06–2023-06-30 | 0.1409 | 0.0715 | 1.97 | 6.41 | 100% | 59 | 0.1409 | 0.1097 | 是 | 否 |
| 38 | 研究 | 2023-07-03–2023-09-28 | 0.1857 | 0.1019 | 1.82 | 4.76 | 98% | 63 | 0.1857 | 0.1992 | 否 | 否 |
| 39 | 研究 | 2023-10-02–2023-12-29 | 0.1083 | 0.1091 | 0.99 | 2.98 | 79% | 63 | 0.1083 | 0.1065 | 是 | 否 |
| 40 | 研究 | 2024-01-02–2024-03-29 | 0.1469 | 0.0933 | 1.57 | 3.69 | 93% | 56 | 0.1469 | 0.1387 | 是 | 否 |
| 41 | 研究 | 2024-04-01–2024-06-28 | 0.0649 | 0.1272 | 0.51 | 1.34 | 66% | 61 | 0.0649 | 0.0650 | 否 | 否 |
| 42 | 研究 | 2024-07-01–2024-09-30 | 0.1382 | 0.1357 | 1.02 | 2.56 | 94% | 63 | 0.1382 | 0.1396 | 否 | 否 |
| 43 | 研究 | 2024-10-01–2024-12-03 | 0.1617 | 0.0690 | 2.34 | 7.14 | 100% | 42 | 0.1617 | 0.1651 | 否 | 否 |

### 每年

| 年 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0.1270 | 0.1122 | 1.13 | 4.80 | 86% | 244 |
| 2017 | 0.1338 | 0.1127 | 1.19 | 5.92 | 88% | 246 |
| 2018 | 0.1908 | 0.1585 | 1.20 | 6.03 | 87% | 247 |
| 2019 | 0.0943 | 0.1631 | 0.58 | 2.60 | 70% | 242 |
| 2020 | 0.1005 | 0.1360 | 0.74 | 4.59 | 78% | 245 |
| 2021 | 0.1516 | 0.1446 | 1.05 | 4.88 | 82% | 244 |
| 2022 | 0.1340 | 0.1649 | 0.81 | 3.75 | 81% | 246 |
| 2023 | 0.1272 | 0.1109 | 1.15 | 5.91 | 87% | 239 |
| 2024 | 0.1247 | 0.1184 | 1.05 | 4.74 | 87% | 222 |

### 成交金額三分位

| 組 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 低 | 0.1369 | 0.1465 | 0.93 | 13.33 | 82% | 2175 |
| 中 | 0.1427 | 0.1360 | 1.05 | 15.24 | 86% | 2175 |
| 高 | 0.1143 | 0.1718 | 0.67 | 9.03 | 75% | 2175 |

### 十分位（扣成本）

| 樣本 | 組合 | 月報酬 | 月化超額 | 超額最大回撤 | 年化換手率（倍） | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | D1 | -0.69% | -1.56% | -81.70% | 8.10 | 2154 |
| 全部 | D2 | -0.08% | -0.94% | -63.88% | 10.15 | 2154 |
| 全部 | D3 | 0.07% | -0.79% | -57.65% | 10.58 | 2154 |
| 全部 | D4 | 0.35% | -0.52% | -42.79% | 10.78 | 2154 |
| 全部 | D5 | 0.48% | -0.38% | -34.04% | 10.82 | 2154 |
| 全部 | D6 | 0.57% | -0.30% | -27.57% | 10.74 | 2154 |
| 全部 | D7 | 0.67% | -0.20% | -20.44% | 10.65 | 2154 |
| 全部 | D8 | 0.86% | -0.01% | -10.46% | 10.33 | 2154 |
| 全部 | D9 | 1.14% | 0.27% | -8.98% | 9.57 | 2154 |
| 全部 | D10 | 1.53% | 0.66% | -12.80% | 6.76 | 2154 |
| 全部 | universe | 0.87% | 0.00% | 0.00% | 1.87 | 2154 |
| 全部 | D10-D1 | 1.44% | 1.44% | -25.07% | 14.86 | 2154 |
| 成交金額前三分之一 | D1 | -0.89% | -1.71% | -85.01% | 8.83 | 2154 |
| 成交金額前三分之一 | D2 | -0.03% | -0.85% | -61.65% | 10.45 | 2154 |
| 成交金額前三分之一 | D3 | 0.23% | -0.59% | -48.65% | 10.87 | 2154 |
| 成交金額前三分之一 | D4 | 0.38% | -0.43% | -39.06% | 10.98 | 2154 |
| 成交金額前三分之一 | D5 | 0.56% | -0.26% | -26.51% | 10.98 | 2154 |
| 成交金額前三分之一 | D6 | 0.65% | -0.16% | -19.58% | 10.88 | 2154 |
| 成交金額前三分之一 | D7 | 0.68% | -0.13% | -16.54% | 10.73 | 2154 |
| 成交金額前三分之一 | D8 | 0.79% | -0.02% | -16.69% | 10.43 | 2154 |
| 成交金額前三分之一 | D9 | 0.92% | 0.10% | -15.95% | 9.66 | 2154 |
| 成交金額前三分之一 | D10 | 1.39% | 0.57% | -18.22% | 6.78 | 2154 |
| 成交金額前三分之一 | universe | 0.82% | 0.00% | 0.00% | 2.59 | 2154 |
| 成交金額前三分之一 | D10-D1 | 1.43% | 1.43% | -34.39% | 15.61 | 2154 |

## 特徵重要度前 10 名

| 摺 | 前 10 名（gain） | 和前一摺重疊 |
| --- | --- | --- |
| 8 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, risk_idio_vol_60d, tech_mom_250_20, tech_high_52w_gap, tech_ma_gap_60d, fund_roe_ttm, tech_turnover_rate_5d | — |
| 9 | chip_margin_usage, fund_dividend_yield, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_mom_250_20, tech_high_52w_gap, tech_ma_gap_60d, risk_idio_vol_60d, chip_short_ratio, fund_roe_ttm | 9 |
| 10 | tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, tech_ma_gap_60d, tech_mom_250_20, fund_roe_ttm, risk_idio_vol_60d, tech_turnover_rate_5d | 9 |
| 11 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_roe_ttm, tech_ma_gap_60d, tech_turnover_rate_5d, tech_mom_250_20, chip_it_net_5d | 9 |
| 12 | tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_roe_ttm, tech_turnover_rate_5d, risk_idio_vol_60d, tech_ma_gap_60d, tech_mom_250_20 | 9 |
| 13 | chip_margin_usage, tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_dividend_yield, tech_mom_250_20, tech_turnover_rate_5d, tech_ma_gap_60d, fund_roe_ttm | 10 |
| 14 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, fund_dividend_yield, tech_high_52w_gap, risk_idio_vol_60d, fund_ep, tech_ma_gap_60d, tech_turnover_rate_5d | 9 |
| 15 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_roe_ttm, fund_dividend_yield, tech_ma_gap_60d, tech_turnover_rate_5d, tech_mom_250_20, risk_idio_vol_60d | 9 |
| 16 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, tech_high_52w_gap, fund_dividend_yield, fund_gross_margin, risk_idio_vol_60d, fund_ep, fund_op_margin | 7 |
| 17 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, fund_dividend_yield, tech_high_52w_gap, risk_idio_vol_60d, fund_debt_ratio, fund_ep, fund_gross_margin | 9 |
| 18 | chip_margin_usage, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, fund_roe_ttm, tech_high_52w_gap, fund_op_margin, tech_ma_gap_60d, fund_ep | 8 |
| 19 | chip_margin_usage, risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, tech_high_52w_gap, fund_roe_ttm, fund_rev_yoy, fund_gm_yoy_chg, fund_op_margin | 8 |
| 20 | risk_idio_vol_60d, chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, fund_dividend_yield, tech_high_52w_gap, fund_rev_yoy, fund_ep, fund_gm_yoy_chg | 9 |
| 21 | chip_margin_usage, tech_limit_down_cnt_20d, risk_idio_vol_60d, fund_dividend_yield, fund_roe_ttm, tech_limit_up_cnt_20d, fund_gm_yoy_chg, tech_high_52w_gap, fund_rev_yoy, fund_opm_yoy_chg | 9 |
| 22 | chip_margin_usage, risk_idio_vol_60d, tech_limit_down_cnt_20d, fund_dividend_yield, tech_high_52w_gap, tech_limit_up_cnt_20d, fund_roe_ttm, fund_gm_yoy_chg, fund_rev_yoy, fund_rev_3m_yoy | 9 |
| 23 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, fund_roe_ttm, fund_gm_yoy_chg, risk_idio_vol_60d, fund_rev_3m_yoy, fund_debt_ratio, fund_gross_margin | 8 |
| 24 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, fund_gm_yoy_chg, fund_roe_ttm, fund_rev_3m_yoy, tech_limit_up_cnt_20d, fund_rev_yoy, fund_ep | 8 |
| 25 | chip_margin_usage, fund_dividend_yield, risk_idio_vol_60d, tech_limit_down_cnt_20d, fund_rev_3m_yoy, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_gm_yoy_chg, fund_roe_ttm, fund_rev_yoy | 9 |
| 26 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, risk_idio_vol_60d, fund_roe_ttm, fund_gm_yoy_chg, tech_high_52w_gap, risk_beta_250d, fund_rev_3m_yoy | 9 |
| 27 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, fund_roe_ttm, fund_gm_yoy_chg, risk_idio_vol_60d, risk_beta_250d, tech_limit_up_cnt_20d, fund_rev_yoy_q, fund_rev_3m_yoy | 9 |
| 28 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, tech_high_52w_gap, fund_gm_yoy_chg, tech_limit_up_cnt_20d, fund_roe_ttm, risk_beta_250d, fund_rev_3m_yoy | 9 |
| 29 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, fund_gm_yoy_chg, tech_limit_up_cnt_20d, tech_high_52w_gap, risk_beta_250d, fund_roe_ttm, fund_rev_3m_yoy | 10 |
| 30 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_gm_yoy_chg, risk_beta_250d, risk_idio_vol_60d, fund_roe_ttm, fund_rev_yoy | 9 |
| 31 | tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, risk_idio_vol_60d, fund_dividend_yield, risk_beta_250d, tech_high_52w_gap, fund_gm_yoy_chg, fund_rev_3m_yoy, fund_roe_ttm | 9 |
| 32 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, fund_dividend_yield, risk_idio_vol_60d, risk_beta_250d, tech_high_52w_gap, fund_gm_yoy_chg, fund_rev_3m_yoy, fund_rev_yoy | 9 |
| 33 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, fund_dividend_yield, tech_limit_up_cnt_20d, risk_beta_250d, tech_high_52w_gap, fund_gm_yoy_chg, fund_rev_3m_yoy, fund_rev_yoy | 10 |
| 34 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_dividend_yield, risk_beta_250d, fund_gm_yoy_chg, fund_roe_ttm, fund_rev_3m_yoy | 9 |
| 35 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, tech_high_52w_gap, risk_beta_250d, fund_gm_yoy_chg, fund_roe_ttm, fund_dividend_yield, fund_eps_yoy | 9 |
| 36 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_high_52w_gap, tech_limit_up_cnt_20d, risk_beta_250d, fund_dividend_yield, fund_eps_yoy, fund_gm_yoy_chg, fund_roe_ttm | 10 |
| 37 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_limit_up_cnt_20d, risk_beta_250d, fund_dividend_yield, tech_high_52w_gap, fund_eps_yoy, fund_gm_yoy_chg, fund_roe_ttm | 10 |
| 38 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_limit_up_cnt_20d, fund_dividend_yield, risk_beta_250d, tech_high_52w_gap, fund_eps_yoy, fund_gm_yoy_chg, fund_roe_ttm | 10 |
| 39 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, chip_margin_usage, tech_high_52w_gap, tech_limit_up_cnt_20d, risk_beta_250d, fund_eps_yoy, fund_gm_yoy_chg, chip_short_ratio | 9 |
| 40 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, tech_high_52w_gap, chip_margin_usage, tech_limit_up_cnt_20d, risk_beta_250d, fund_eps_yoy, fund_gm_yoy_chg, tech_ret_20d | 9 |
| 41 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, tech_high_52w_gap, tech_limit_up_cnt_20d, chip_margin_usage, risk_beta_250d, fund_eps_yoy, fund_gm_yoy_chg, chip_short_ratio | 9 |
| 42 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_high_52w_gap, chip_margin_usage, risk_beta_250d, fund_eps_yoy, chip_short_ratio, fund_gm_yoy_chg | 10 |
| 43 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_beta_250d, risk_idio_vol_60d, tech_high_52w_gap, chip_margin_usage, fund_eps_yoy, tech_limit_up_cnt_20d, fund_gm_yoy_chg, fund_rev_yoy_q | 9 |

- 進前 10 名的摺數：chip_margin_usage 36、fund_dividend_yield 36、tech_limit_down_cnt_20d 36、tech_limit_up_cnt_20d 36、risk_idio_vol_60d 35、tech_high_52w_gap 33、fund_roe_ttm 29、fund_gm_yoy_chg 25、risk_beta_250d 18、fund_rev_3m_yoy 12、fund_eps_yoy 9、fund_rev_yoy 9、tech_ma_gap_60d 9、tech_mom_250_20 7、tech_turnover_rate_5d 7、fund_ep 6、chip_short_ratio 4、fund_gross_margin 3、fund_op_margin 3、fund_debt_ratio 2、fund_rev_yoy_q 2、chip_it_net_5d 1、fund_opm_yoy_chg 1、tech_ret_20d 1

## 關卡：排名 20 日

- 研究期 IC 均值 0.1317 大於 0 且 NW t 13.08 大於 2：通過
- 贏 Ridge 的研究摺 22／36 摺 達 60%：通過
- Ridge 研究期 IC 均值 0.1193，NW t 11.35，2175 天
- 重要度變動只列研究摺，調參摺不計
