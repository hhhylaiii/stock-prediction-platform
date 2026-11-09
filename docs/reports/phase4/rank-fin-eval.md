# 評估報表：rank-fin（5 日）

## 排名 5 日

- 研究期 IC 均值 0.1303，標準差 0.1363，NW t 25.74，ICIR 0.96，2175 天
- IC 為正的研究摺比例：100%
- 中性化 IC：0.0809，NW t 37.28，2175 天
- 贏 baseline 的研究摺比例：86%（36／36 摺可比）
- 🔴 洩漏警訊：第 25 摺比同摺 baseline 的 IC 高 0.05 以上，先當成洩漏追查
- 打亂 label（20 日 Ridge）：第 8 摺的 OOS rank IC -0.0070，絕對值低於 0.02

### 每摺

| 摺 | 類別 | 測試期 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 | 共同列 IC | baseline IC | 贏 baseline | 洩漏 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 調參 | 2014-01-02–2014-03-31 | 0.1052 | 0.1043 | 1.01 | 5.36 | 82% | 56 | 0.1052 | 0.0883 | 是 | — |
| 1 | 調參 | 2014-04-01–2014-06-30 | 0.1347 | 0.0751 | 1.79 | 8.31 | 98% | 62 | 0.1347 | 0.1008 | 是 | — |
| 2 | 調參 | 2014-07-01–2014-09-30 | 0.1604 | 0.0768 | 2.09 | 11.50 | 97% | 64 | 0.1604 | 0.1370 | 是 | — |
| 3 | 調參 | 2014-10-01–2014-12-31 | 0.1214 | 0.1487 | 0.82 | 3.80 | 80% | 66 | 0.1214 | 0.0964 | 是 | — |
| 4 | 調參 | 2015-01-05–2015-03-31 | 0.1696 | 0.1346 | 1.26 | 4.74 | 85% | 55 | 0.1696 | 0.1489 | 是 | — |
| 5 | 調參 | 2015-04-01–2015-06-30 | 0.1436 | 0.1011 | 1.42 | 6.11 | 95% | 61 | 0.1436 | 0.1284 | 是 | — |
| 6 | 調參 | 2015-07-01–2015-09-30 | 0.1166 | 0.1218 | 0.96 | 3.69 | 76% | 63 | 0.1166 | 0.0915 | 是 | — |
| 7 | 調參 | 2015-10-01–2015-12-03 | 0.0949 | 0.1268 | 0.75 | 3.66 | 76% | 45 | 0.0949 | 0.0853 | 是 | — |
| 8 | 研究 | 2016-01-04–2016-03-31 | 0.1250 | 0.0979 | 1.28 | 5.50 | 89% | 57 | 0.1250 | 0.1136 | 是 | 否 |
| 9 | 研究 | 2016-04-01–2016-06-30 | 0.1394 | 0.1159 | 1.20 | 5.45 | 85% | 61 | 0.1394 | 0.1254 | 是 | 否 |
| 10 | 研究 | 2016-07-01–2016-09-30 | 0.1597 | 0.0887 | 1.80 | 9.51 | 98% | 62 | 0.1597 | 0.1326 | 是 | 否 |
| 11 | 研究 | 2016-10-03–2016-12-30 | 0.1414 | 0.1209 | 1.17 | 5.40 | 89% | 64 | 0.1414 | 0.1311 | 是 | 否 |
| 12 | 研究 | 2017-01-03–2017-03-31 | 0.1356 | 0.1095 | 1.24 | 5.89 | 86% | 57 | 0.1356 | 0.1219 | 是 | 否 |
| 13 | 研究 | 2017-04-05–2017-06-30 | 0.2084 | 0.1000 | 2.08 | 10.33 | 97% | 61 | 0.2084 | 0.1935 | 是 | 否 |
| 14 | 研究 | 2017-07-03–2017-09-30 | 0.1165 | 0.1153 | 1.01 | 4.30 | 85% | 66 | 0.1165 | 0.0956 | 是 | 否 |
| 15 | 研究 | 2017-10-02–2017-12-29 | 0.1635 | 0.0857 | 1.91 | 9.18 | 95% | 62 | 0.1635 | 0.1366 | 是 | 否 |
| 16 | 研究 | 2018-01-02–2018-03-31 | 0.1661 | 0.1234 | 1.35 | 6.38 | 93% | 58 | 0.1661 | 0.1508 | 是 | 否 |
| 17 | 研究 | 2018-04-02–2018-06-29 | 0.1577 | 0.1168 | 1.35 | 5.51 | 92% | 60 | 0.1577 | 0.1481 | 是 | 否 |
| 18 | 研究 | 2018-07-02–2018-09-28 | 0.1879 | 0.1233 | 1.52 | 7.95 | 98% | 64 | 0.1879 | 0.1871 | 是 | 否 |
| 19 | 研究 | 2018-10-01–2018-12-28 | 0.1168 | 0.1512 | 0.77 | 4.17 | 77% | 65 | 0.1168 | 0.0930 | 是 | 否 |
| 20 | 研究 | 2019-01-02–2019-03-29 | 0.0850 | 0.1345 | 0.63 | 2.59 | 70% | 54 | 0.0850 | 0.0823 | 是 | 否 |
| 21 | 研究 | 2019-04-01–2019-06-28 | 0.1330 | 0.1860 | 0.72 | 3.10 | 75% | 61 | 0.1330 | 0.1508 | 否 | 否 |
| 22 | 研究 | 2019-07-01–2019-09-27 | 0.0842 | 0.1347 | 0.63 | 2.99 | 70% | 63 | 0.0842 | 0.0789 | 是 | 否 |
| 23 | 研究 | 2019-10-01–2019-12-31 | 0.1447 | 0.0828 | 1.75 | 8.18 | 100% | 64 | 0.1447 | 0.1295 | 是 | 否 |
| 24 | 研究 | 2020-01-02–2020-03-31 | 0.1273 | 0.1489 | 0.86 | 5.17 | 86% | 56 | 0.1273 | 0.0915 | 是 | 否 |
| 25 | 研究 | 2020-04-01–2020-06-30 | 0.0972 | 0.1140 | 0.85 | 4.16 | 78% | 60 | 0.0972 | 0.0415 | 是 | 是 |
| 26 | 研究 | 2020-07-01–2020-09-30 | 0.1344 | 0.1339 | 1.00 | 5.11 | 82% | 66 | 0.1344 | 0.1109 | 是 | 否 |
| 27 | 研究 | 2020-10-05–2020-12-31 | 0.1147 | 0.1150 | 1.00 | 4.58 | 81% | 63 | 0.1147 | 0.1067 | 是 | 否 |
| 28 | 研究 | 2021-01-04–2021-03-31 | 0.1021 | 0.1796 | 0.57 | 2.31 | 69% | 55 | 0.1021 | 0.0896 | 是 | 否 |
| 29 | 研究 | 2021-04-01–2021-06-30 | 0.1607 | 0.1453 | 1.11 | 6.88 | 90% | 61 | 0.1607 | 0.1330 | 是 | 否 |
| 30 | 研究 | 2021-07-01–2021-09-30 | 0.1854 | 0.1323 | 1.40 | 7.24 | 95% | 64 | 0.1854 | 0.1742 | 是 | 否 |
| 31 | 研究 | 2021-10-01–2021-12-30 | 0.1017 | 0.1188 | 0.86 | 3.76 | 84% | 64 | 0.1017 | 0.0970 | 是 | 否 |
| 32 | 研究 | 2022-01-03–2022-03-31 | 0.1529 | 0.1234 | 1.24 | 5.55 | 89% | 56 | 0.1529 | 0.1444 | 是 | 否 |
| 33 | 研究 | 2022-04-01–2022-06-30 | 0.1415 | 0.1225 | 1.16 | 6.10 | 89% | 61 | 0.1415 | 0.1288 | 是 | 否 |
| 34 | 研究 | 2022-07-01–2022-09-30 | 0.0839 | 0.1789 | 0.47 | 2.24 | 68% | 65 | 0.0839 | 0.0765 | 是 | 否 |
| 35 | 研究 | 2022-10-03–2022-12-30 | 0.0995 | 0.1801 | 0.55 | 2.58 | 69% | 64 | 0.0995 | 0.0975 | 是 | 否 |
| 36 | 研究 | 2023-01-03–2023-03-31 | 0.0724 | 0.1644 | 0.44 | 2.04 | 67% | 54 | 0.0724 | 0.0760 | 否 | 否 |
| 37 | 研究 | 2023-04-06–2023-06-30 | 0.0798 | 0.1666 | 0.48 | 2.50 | 66% | 59 | 0.0798 | 0.0928 | 否 | 否 |
| 38 | 研究 | 2023-07-03–2023-09-28 | 0.1297 | 0.1201 | 1.08 | 4.98 | 86% | 63 | 0.1297 | 0.1287 | 是 | 否 |
| 39 | 研究 | 2023-10-02–2023-12-29 | 0.1255 | 0.1259 | 1.00 | 4.59 | 81% | 63 | 0.1255 | 0.1268 | 否 | 否 |
| 40 | 研究 | 2024-01-02–2024-03-29 | 0.1120 | 0.1303 | 0.86 | 3.45 | 80% | 56 | 0.1120 | 0.1226 | 否 | 否 |
| 41 | 研究 | 2024-04-01–2024-06-28 | 0.1163 | 0.1322 | 0.88 | 3.58 | 77% | 61 | 0.1163 | 0.1054 | 是 | 否 |
| 42 | 研究 | 2024-07-01–2024-09-30 | 0.1357 | 0.1581 | 0.86 | 3.74 | 81% | 63 | 0.1357 | 0.1145 | 是 | 否 |
| 43 | 研究 | 2024-10-01–2024-12-03 | 0.1452 | 0.1533 | 0.95 | 4.16 | 81% | 42 | 0.1452 | 0.1247 | 是 | 否 |

### 每年

| 年 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0.1417 | 0.1070 | 1.32 | 12.04 | 91% | 244 |
| 2017 | 0.1556 | 0.1085 | 1.43 | 12.53 | 91% | 246 |
| 2018 | 0.1568 | 0.1317 | 1.19 | 11.16 | 90% | 247 |
| 2019 | 0.1127 | 0.1407 | 0.80 | 6.92 | 79% | 242 |
| 2020 | 0.1186 | 0.1283 | 0.92 | 9.21 | 82% | 245 |
| 2021 | 0.1385 | 0.1481 | 0.94 | 8.50 | 85% | 244 |
| 2022 | 0.1179 | 0.1569 | 0.75 | 7.08 | 78% | 246 |
| 2023 | 0.1033 | 0.1461 | 0.71 | 6.68 | 75% | 239 |
| 2024 | 0.1262 | 0.1432 | 0.88 | 7.22 | 80% | 222 |

### 成交金額三分位

| 組 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 低 | 0.1342 | 0.1403 | 0.96 | 26.29 | 84% | 2175 |
| 中 | 0.1365 | 0.1406 | 0.97 | 26.11 | 83% | 2175 |
| 高 | 0.1192 | 0.1646 | 0.72 | 19.76 | 77% | 2175 |

### 十分位（扣成本）

| 樣本 | 組合 | 月報酬 | 月化超額 | 超額最大回撤 | 年化換手率（倍） | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | D1 | -4.21% | -4.17% | -99.03% | 33.62 | 2169 |
| 全部 | D2 | -2.98% | -2.94% | -96.13% | 40.49 | 2169 |
| 全部 | D3 | -2.51% | -2.47% | -93.44% | 42.00 | 2169 |
| 全部 | D4 | -2.06% | -2.01% | -89.15% | 42.78 | 2169 |
| 全部 | D5 | -1.91% | -1.87% | -87.21% | 42.84 | 2169 |
| 全部 | D6 | -1.41% | -1.37% | -77.67% | 42.43 | 2169 |
| 全部 | D7 | -1.16% | -1.12% | -70.50% | 41.72 | 2169 |
| 全部 | D8 | -0.90% | -0.86% | -61.27% | 40.26 | 2169 |
| 全部 | D9 | -0.48% | -0.44% | -40.09% | 37.35 | 2169 |
| 全部 | D10 | 0.61% | 0.66% | -10.09% | 26.91 | 2169 |
| 全部 | universe | -0.04% | 0.00% | 0.00% | 2.92 | 2169 |
| 全部 | D10-D1 | 1.79% | 1.79% | -16.42% | 60.53 | 2169 |
| 成交金額前三分之一 | D1 | -4.24% | -4.07% | -98.98% | 34.08 | 2169 |
| 成交金額前三分之一 | D2 | -3.24% | -3.07% | -96.70% | 41.27 | 2169 |
| 成交金額前三分之一 | D3 | -2.74% | -2.58% | -94.18% | 42.71 | 2169 |
| 成交金額前三分之一 | D4 | -2.10% | -1.93% | -88.41% | 43.23 | 2169 |
| 成交金額前三分之一 | D5 | -1.93% | -1.76% | -85.71% | 43.37 | 2169 |
| 成交金額前三分之一 | D6 | -1.69% | -1.52% | -81.46% | 43.07 | 2169 |
| 成交金額前三分之一 | D7 | -1.18% | -1.01% | -67.45% | 42.28 | 2169 |
| 成交金額前三分之一 | D8 | -0.98% | -0.81% | -60.22% | 40.47 | 2169 |
| 成交金額前三分之一 | D9 | -0.39% | -0.22% | -37.38% | 37.12 | 2169 |
| 成交金額前三分之一 | D10 | 0.70% | 0.87% | -12.23% | 26.93 | 2169 |
| 成交金額前三分之一 | universe | -0.17% | 0.00% | 0.00% | 4.16 | 2169 |
| 成交金額前三分之一 | D10-D1 | 1.88% | 1.88% | -23.82% | 61.01 | 2169 |

## 特徵重要度前 10 名

| 摺 | 前 10 名（gain） | 和前一摺重疊 |
| --- | --- | --- |
| 8 | tech_limit_down_cnt_20d, tech_ma_gap_5d, risk_idio_vol_60d, tech_ret_1d, chip_margin_usage, tech_limit_up_cnt_20d, chip_fi_net_1d, tech_high_52w_gap, chip_it_net_5d, tech_ret_5d | — |
| 9 | tech_ma_gap_5d, risk_idio_vol_60d, tech_ret_1d, chip_margin_usage, chip_fi_net_1d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_it_net_5d, tech_kd_d, tech_ret_5d | 9 |
| 10 | tech_ma_gap_5d, risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ret_1d, tech_limit_up_cnt_20d, chip_it_net_5d, chip_fi_net_1d, tech_ret_5d, tech_ma_gap_60d | 9 |
| 11 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ret_1d, tech_limit_up_cnt_20d, chip_fi_net_1d, fund_roe_ttm, tech_ret_5d, chip_it_net_5d | 9 |
| 12 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_limit_up_cnt_20d, fund_roe_ttm, tech_ma_gap_60d, tech_ret_1d, chip_it_net_5d, chip_fi_net_1d | 9 |
| 13 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, fund_roe_ttm, tech_limit_up_cnt_20d, tech_ma_gap_60d, tech_ret_1d, chip_it_net_5d, tech_vol_20d | 9 |
| 14 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_limit_up_cnt_20d, tech_ret_1d, tech_vol_20d, tech_ret_5d, chip_it_net_5d, tech_ma_gap_60d | 9 |
| 15 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_limit_up_cnt_20d, tech_ret_1d, tech_kd_d, chip_it_net_5d, tech_vol_20d, chip_fi_net_1d | 8 |
| 16 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_vol_20d, tech_ret_1d, tech_limit_up_cnt_20d, chip_it_net_5d, tech_kd_d, chip_fi_net_1d | 10 |
| 17 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_limit_up_cnt_20d, tech_kd_d, tech_vol_20d, tech_ret_5d, chip_it_net_5d, tech_ret_1d | 9 |
| 18 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, tech_ret_5d, tech_limit_up_cnt_20d, chip_margin_usage, tech_kd_d, tech_vol_20d, tech_ret_1d, chip_it_net_5d | 10 |
| 19 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_vol_20d, tech_ret_5d, tech_limit_up_cnt_20d, tech_kd_d, tech_turnover_rate_5d, tech_ret_1d | 9 |
| 20 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_ret_5d, tech_vol_20d, tech_turnover_rate_5d, tech_limit_down_cnt_20d, tech_kd_d, fund_debt_ratio, tech_limit_up_cnt_20d | 9 |
| 21 | risk_idio_vol_60d, tech_ma_gap_5d, tech_vol_20d, chip_margin_usage, tech_ret_5d, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, chip_it_net_5d, tech_turnover_rate_5d, fund_debt_ratio | 9 |
| 22 | risk_idio_vol_60d, tech_vol_20d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ret_5d, tech_turnover_rate_5d, tech_limit_up_cnt_20d, tech_kd_d, fund_dividend_yield | 8 |
| 23 | risk_idio_vol_60d, tech_vol_20d, tech_limit_down_cnt_20d, tech_ma_gap_5d, tech_limit_up_cnt_20d, chip_margin_usage, tech_turnover_rate_5d, tech_ret_5d, tech_kd_d, fund_roe_ttm | 9 |
| 24 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_vol_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, fund_debt_ratio, fund_roe_ttm, tech_ret_5d | 8 |
| 25 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, tech_vol_20d, chip_margin_usage, fund_dividend_yield, tech_ret_5d, tech_high_52w_gap, fund_debt_ratio | 9 |
| 26 | tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, fund_debt_ratio, fund_dividend_yield, fund_roe_ttm, tech_ma_gap_5d, risk_beta_250d, tech_vol_20d | 8 |
| 27 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, risk_idio_vol_60d, chip_margin_usage, fund_dividend_yield, fund_debt_ratio, tech_ma_gap_5d, risk_beta_250d, tech_vol_20d, fund_roe_ttm | 10 |
| 28 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, risk_idio_vol_60d, tech_ma_gap_5d, fund_dividend_yield, chip_margin_usage, tech_high_52w_gap, risk_beta_250d, tech_vol_20d, fund_roe_ttm | 9 |
| 29 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, risk_idio_vol_60d, chip_margin_usage, fund_dividend_yield, tech_ma_gap_5d, tech_vol_20d, tech_high_52w_gap, risk_beta_250d, tech_turnover_rate_5d | 9 |
| 30 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, risk_idio_vol_60d, tech_attention_t, fund_dividend_yield, risk_beta_250d, tech_ma_gap_5d, tech_vol_20d, fund_roe_ttm | 8 |
| 31 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_attention_t, risk_beta_250d, tech_ma_gap_5d, fund_dividend_yield, tech_high_52w_gap, tech_ret_1d | 8 |
| 32 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, risk_idio_vol_60d, tech_ma_gap_5d, risk_beta_250d, tech_ret_1d, tech_high_52w_gap, chip_margin_usage, fund_dividend_yield, tech_vol_20d | 9 |
| 33 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_ma_gap_5d, tech_ret_1d, chip_margin_usage, tech_vol_20d, fund_dividend_yield, tech_high_52w_gap, risk_beta_250d | 10 |
| 34 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, tech_ret_1d, chip_margin_usage, tech_vol_20d, tech_limit_up_t, fund_dividend_yield, tech_high_52w_gap | 9 |
| 35 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, chip_margin_usage, tech_high_52w_gap, tech_ret_1d, tech_vol_20d, tech_ma_gap_5d, risk_beta_250d, tech_ret_20d | 8 |
| 36 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, chip_margin_usage, risk_beta_250d, tech_high_52w_gap, tech_vol_20d, tech_ma_gap_5d, tech_limit_up_t, tech_attention_t | 8 |
| 37 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_vol_20d, chip_margin_usage, tech_ma_gap_5d, tech_limit_up_cnt_20d, tech_ret_20d, tech_attention_t, tech_ret_1d, tech_ma_gap_20d | 7 |
| 38 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, risk_beta_250d, tech_high_52w_gap, tech_attention_t, tech_ma_gap_5d, tech_vol_20d, fund_eps_yoy | 7 |
| 39 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, risk_beta_250d, tech_high_52w_gap, tech_limit_up_t, tech_vol_20d, tech_ma_gap_5d, tech_attention_t | 9 |
| 40 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_high_52w_gap, tech_limit_up_t, tech_attention_t, tech_turnover_rate_5d, risk_beta_250d | 8 |
| 41 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, chip_margin_usage, fund_dividend_yield, risk_beta_250d, tech_limit_up_t, tech_attention_t, tech_high_52w_gap, fund_eps_yoy | 9 |
| 42 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, risk_beta_250d, fund_dividend_yield, fund_debt_ratio, fund_eps_yoy, fund_roe_ttm, tech_attention_t | 8 |
| 43 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_turnover_rate_5d, chip_margin_usage, fund_dividend_yield, risk_beta_250d, tech_attention_t, tech_limit_up_t, tech_high_52w_gap | 7 |

- 進前 10 名的摺數：chip_margin_usage 36、risk_idio_vol_60d 36、tech_limit_down_cnt_20d 36、tech_limit_up_cnt_20d 36、tech_ma_gap_5d 32、tech_vol_20d 26、tech_ret_1d 18、fund_dividend_yield 16、risk_beta_250d 16、tech_high_52w_gap 15、tech_ret_5d 14、chip_it_net_5d 12、fund_roe_ttm 10、tech_attention_t 10、tech_kd_d 9、tech_turnover_rate_5d 8、chip_fi_net_1d 7、fund_debt_ratio 7、tech_limit_up_t 6、tech_ma_gap_60d 4、fund_eps_yoy 3、tech_ret_20d 2、tech_ma_gap_20d 1

## 關卡：排名 5 日

- 研究期 IC 均值 0.1303 大於 0 且 NW t 25.74 大於 2：通過
- 贏 Ridge 的研究摺 31／36 摺 達 60%：通過
- Ridge 研究期 IC 均值 0.1184，NW t 21.57，2175 天
- 重要度變動只列研究摺，調參摺不計

# 評估報表：rank-fin（20 日）

## 排名 20 日

- 研究期 IC 均值 0.1129，標準差 0.1475，NW t 10.17，ICIR 0.77，2175 天
- IC 為正的研究摺比例：97%
- 中性化 IC：0.0644，NW t 15.41，2175 天
- 贏 baseline 的研究摺比例：58%（36／36 摺可比）
- 🔴 洩漏警訊：第 32、34 摺比同摺 baseline 的 IC 高 0.05 以上，先當成洩漏追查
- 打亂 label（20 日 Ridge）：第 8 摺的 OOS rank IC -0.0070，絕對值低於 0.02

### 每摺

| 摺 | 類別 | 測試期 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 | 共同列 IC | baseline IC | 贏 baseline | 洩漏 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 調參 | 2014-01-02–2014-03-31 | 0.0664 | 0.0750 | 0.88 | 2.79 | 79% | 56 | 0.0664 | 0.0407 | 是 | — |
| 1 | 調參 | 2014-04-01–2014-06-30 | 0.0579 | 0.0527 | 1.10 | 3.16 | 82% | 62 | 0.0579 | 0.0548 | 是 | — |
| 2 | 調參 | 2014-07-01–2014-09-30 | 0.0890 | 0.0464 | 1.92 | 6.32 | 98% | 64 | 0.0890 | 0.0752 | 是 | — |
| 3 | 調參 | 2014-10-01–2014-12-31 | 0.0621 | 0.1906 | 0.33 | 0.78 | 59% | 66 | 0.0621 | 0.0225 | 是 | — |
| 4 | 調參 | 2015-01-05–2015-03-31 | 0.2080 | 0.1333 | 1.56 | 3.87 | 93% | 55 | 0.2080 | 0.1876 | 是 | — |
| 5 | 調參 | 2015-04-01–2015-06-30 | 0.1212 | 0.1069 | 1.13 | 2.58 | 80% | 61 | 0.1212 | 0.1885 | 否 | — |
| 6 | 調參 | 2015-07-01–2015-09-30 | 0.0293 | 0.2147 | 0.14 | 0.29 | 49% | 63 | 0.0293 | 0.0618 | 否 | — |
| 7 | 調參 | 2015-10-01–2015-12-03 | 0.0662 | 0.0852 | 0.78 | 2.80 | 80% | 45 | 0.0662 | 0.0464 | 是 | — |
| 8 | 研究 | 2016-01-04–2016-03-31 | 0.1223 | 0.0875 | 1.40 | 3.13 | 100% | 57 | 0.1223 | 0.1115 | 是 | 否 |
| 9 | 研究 | 2016-04-01–2016-06-30 | 0.1304 | 0.1153 | 1.13 | 2.62 | 84% | 61 | 0.1304 | 0.1219 | 是 | 否 |
| 10 | 研究 | 2016-07-01–2016-09-30 | 0.1796 | 0.0906 | 1.98 | 6.52 | 97% | 62 | 0.1796 | 0.1591 | 是 | 否 |
| 11 | 研究 | 2016-10-03–2016-12-30 | 0.0356 | 0.1011 | 0.35 | 0.95 | 66% | 64 | 0.0356 | 0.0446 | 否 | 否 |
| 12 | 研究 | 2017-01-03–2017-03-31 | 0.1206 | 0.1123 | 1.07 | 2.41 | 81% | 57 | 0.1206 | 0.1196 | 是 | 否 |
| 13 | 研究 | 2017-04-05–2017-06-30 | 0.1958 | 0.0840 | 2.33 | 8.17 | 98% | 61 | 0.1958 | 0.1908 | 是 | 否 |
| 14 | 研究 | 2017-07-03–2017-09-30 | 0.0733 | 0.1097 | 0.67 | 2.34 | 80% | 66 | 0.0733 | 0.0475 | 是 | 否 |
| 15 | 研究 | 2017-10-02–2017-12-29 | 0.1477 | 0.0804 | 1.84 | 4.63 | 92% | 62 | 0.1477 | 0.1451 | 是 | 否 |
| 16 | 研究 | 2018-01-02–2018-03-31 | 0.1539 | 0.1246 | 1.23 | 3.61 | 90% | 58 | 0.1539 | 0.1767 | 否 | 否 |
| 17 | 研究 | 2018-04-02–2018-06-29 | 0.1587 | 0.1514 | 1.05 | 2.52 | 78% | 60 | 0.1587 | 0.1472 | 是 | 否 |
| 18 | 研究 | 2018-07-02–2018-09-28 | 0.2590 | 0.0806 | 3.21 | 8.35 | 100% | 64 | 0.2590 | 0.2562 | 是 | 否 |
| 19 | 研究 | 2018-10-01–2018-12-28 | 0.0336 | 0.1999 | 0.17 | 0.52 | 58% | 65 | 0.0336 | 0.0253 | 是 | 否 |
| 20 | 研究 | 2019-01-02–2019-03-29 | -0.0214 | 0.1447 | -0.15 | -0.42 | 37% | 54 | -0.0214 | -0.0172 | 否 | 否 |
| 21 | 研究 | 2019-04-01–2019-06-28 | 0.1320 | 0.2799 | 0.47 | 0.97 | 56% | 61 | 0.1320 | 0.1545 | 否 | 否 |
| 22 | 研究 | 2019-07-01–2019-09-27 | 0.0416 | 0.1026 | 0.41 | 1.71 | 71% | 63 | 0.0416 | 0.0598 | 否 | 否 |
| 23 | 研究 | 2019-10-01–2019-12-31 | 0.0907 | 0.0545 | 1.66 | 5.13 | 98% | 64 | 0.0907 | 0.1648 | 否 | 否 |
| 24 | 研究 | 2020-01-02–2020-03-31 | 0.0529 | 0.1672 | 0.32 | 0.83 | 55% | 56 | 0.0529 | 0.1231 | 否 | 否 |
| 25 | 研究 | 2020-04-01–2020-06-30 | 0.0209 | 0.1120 | 0.19 | 0.67 | 58% | 60 | 0.0209 | -0.0182 | 是 | 否 |
| 26 | 研究 | 2020-07-01–2020-09-30 | 0.0551 | 0.1285 | 0.43 | 1.19 | 64% | 66 | 0.0551 | 0.0994 | 否 | 否 |
| 27 | 研究 | 2020-10-05–2020-12-31 | 0.0636 | 0.1015 | 0.63 | 1.82 | 68% | 63 | 0.0636 | 0.1366 | 否 | 否 |
| 28 | 研究 | 2021-01-04–2021-03-31 | 0.0187 | 0.1404 | 0.13 | 0.43 | 56% | 55 | 0.0187 | 0.0266 | 否 | 否 |
| 29 | 研究 | 2021-04-01–2021-06-30 | 0.1911 | 0.1155 | 1.65 | 4.23 | 98% | 61 | 0.1911 | 0.1456 | 是 | 否 |
| 30 | 研究 | 2021-07-01–2021-09-30 | 0.2417 | 0.1114 | 2.17 | 9.54 | 100% | 64 | 0.2417 | 0.2104 | 是 | 否 |
| 31 | 研究 | 2021-10-01–2021-12-30 | 0.0681 | 0.1855 | 0.37 | 0.84 | 52% | 64 | 0.0681 | 0.0965 | 否 | 否 |
| 32 | 研究 | 2022-01-03–2022-03-31 | 0.1472 | 0.0878 | 1.68 | 6.08 | 96% | 56 | 0.1472 | 0.0840 | 是 | 是 |
| 33 | 研究 | 2022-04-01–2022-06-30 | 0.1635 | 0.1352 | 1.21 | 4.32 | 87% | 61 | 0.1635 | 0.1486 | 是 | 否 |
| 34 | 研究 | 2022-07-01–2022-09-30 | 0.1940 | 0.1795 | 1.08 | 2.54 | 86% | 65 | 0.1940 | 0.1286 | 是 | 是 |
| 35 | 研究 | 2022-10-03–2022-12-30 | 0.0052 | 0.2187 | 0.02 | 0.06 | 45% | 64 | 0.0052 | 0.0215 | 否 | 否 |
| 36 | 研究 | 2023-01-03–2023-03-31 | 0.0477 | 0.1199 | 0.40 | 1.09 | 74% | 54 | 0.0477 | 0.0527 | 否 | 否 |
| 37 | 研究 | 2023-04-06–2023-06-30 | 0.1057 | 0.0543 | 1.94 | 7.91 | 100% | 59 | 0.1057 | 0.1019 | 是 | 否 |
| 38 | 研究 | 2023-07-03–2023-09-28 | 0.1886 | 0.1092 | 1.73 | 4.42 | 100% | 63 | 0.1886 | 0.1823 | 是 | 否 |
| 39 | 研究 | 2023-10-02–2023-12-29 | 0.0898 | 0.1233 | 0.73 | 2.22 | 73% | 63 | 0.0898 | 0.0976 | 否 | 否 |
| 40 | 研究 | 2024-01-02–2024-03-29 | 0.1568 | 0.1045 | 1.50 | 3.41 | 91% | 56 | 0.1568 | 0.1633 | 否 | 否 |
| 41 | 研究 | 2024-04-01–2024-06-28 | 0.0615 | 0.1078 | 0.57 | 1.56 | 66% | 61 | 0.0615 | 0.0594 | 是 | 否 |
| 42 | 研究 | 2024-07-01–2024-09-30 | 0.1542 | 0.1250 | 1.23 | 3.04 | 94% | 63 | 0.1542 | 0.1449 | 是 | 否 |
| 43 | 研究 | 2024-10-01–2024-12-03 | 0.1844 | 0.0588 | 3.14 | 11.85 | 100% | 42 | 0.1844 | 0.1797 | 是 | 否 |

### 每年

| 年 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0.1161 | 0.1120 | 1.04 | 4.41 | 86% | 244 |
| 2017 | 0.1334 | 0.1070 | 1.25 | 6.13 | 88% | 246 |
| 2018 | 0.1506 | 0.1669 | 0.90 | 4.12 | 81% | 247 |
| 2019 | 0.0633 | 0.1753 | 0.36 | 1.56 | 67% | 242 |
| 2020 | 0.0484 | 0.1289 | 0.38 | 1.93 | 62% | 245 |
| 2021 | 0.1333 | 0.1669 | 0.80 | 3.54 | 77% | 244 |
| 2022 | 0.1267 | 0.1801 | 0.70 | 3.33 | 78% | 246 |
| 2023 | 0.1102 | 0.1167 | 0.94 | 4.69 | 87% | 239 |
| 2024 | 0.1351 | 0.1143 | 1.18 | 5.18 | 86% | 222 |

### 成交金額三分位

| 組 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 低 | 0.1226 | 0.1537 | 0.80 | 10.97 | 78% | 2175 |
| 中 | 0.1233 | 0.1489 | 0.83 | 11.35 | 80% | 2175 |
| 高 | 0.0931 | 0.1741 | 0.53 | 7.01 | 71% | 2175 |

### 十分位（扣成本）

| 樣本 | 組合 | 月報酬 | 月化超額 | 超額最大回撤 | 年化換手率（倍） | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | D1 | -0.28% | -1.15% | -72.14% | 7.85 | 2154 |
| 全部 | D2 | 0.03% | -0.84% | -59.94% | 9.90 | 2154 |
| 全部 | D3 | 0.26% | -0.60% | -48.90% | 10.37 | 2154 |
| 全部 | D4 | 0.34% | -0.53% | -43.95% | 10.62 | 2154 |
| 全部 | D5 | 0.58% | -0.28% | -26.92% | 10.66 | 2154 |
| 全部 | D6 | 0.54% | -0.32% | -28.96% | 10.58 | 2154 |
| 全部 | D7 | 0.52% | -0.34% | -31.21% | 10.40 | 2154 |
| 全部 | D8 | 0.81% | -0.05% | -14.53% | 10.12 | 2154 |
| 全部 | D9 | 0.96% | 0.10% | -22.33% | 9.43 | 2154 |
| 全部 | D10 | 1.20% | 0.34% | -25.01% | 6.95 | 2154 |
| 全部 | universe | 0.87% | 0.00% | 0.00% | 1.87 | 2154 |
| 全部 | D10-D1 | 0.77% | 0.77% | -50.80% | 14.80 | 2154 |
| 成交金額前三分之一 | D1 | -0.55% | -1.37% | -79.48% | 8.58 | 2154 |
| 成交金額前三分之一 | D2 | 0.16% | -0.65% | -52.18% | 10.26 | 2154 |
| 成交金額前三分之一 | D3 | 0.42% | -0.40% | -37.59% | 10.72 | 2154 |
| 成交金額前三分之一 | D4 | 0.63% | -0.19% | -20.94% | 10.85 | 2154 |
| 成交金額前三分之一 | D5 | 0.64% | -0.18% | -20.21% | 10.85 | 2154 |
| 成交金額前三分之一 | D6 | 0.53% | -0.28% | -28.84% | 10.74 | 2154 |
| 成交金額前三分之一 | D7 | 0.54% | -0.28% | -28.88% | 10.49 | 2154 |
| 成交金額前三分之一 | D8 | 0.69% | -0.13% | -26.64% | 10.21 | 2154 |
| 成交金額前三分之一 | D9 | 0.72% | -0.09% | -33.31% | 9.52 | 2154 |
| 成交金額前三分之一 | D10 | 0.97% | 0.15% | -36.54% | 7.05 | 2154 |
| 成交金額前三分之一 | universe | 0.82% | 0.00% | 0.00% | 2.59 | 2154 |
| 成交金額前三分之一 | D10-D1 | 0.74% | 0.74% | -61.14% | 15.63 | 2154 |

## 特徵重要度前 10 名

| 摺 | 前 10 名（gain） | 和前一摺重疊 |
| --- | --- | --- |
| 8 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, fund_gross_margin, fund_debt_ratio, fund_op_margin, fund_ep, fund_roe_ttm, tech_mom_250_20 | — |
| 9 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, fund_debt_ratio, fund_op_margin, fund_roe_ttm, fund_ep, fund_gross_margin, tech_high_52w_gap | 9 |
| 10 | tech_limit_up_cnt_20d, chip_margin_usage, tech_limit_down_cnt_20d, fund_roe_ttm, fund_dividend_yield, fund_op_margin, tech_high_52w_gap, risk_idio_vol_60d, fund_rev_yoy, tech_ma_gap_60d | 7 |
| 11 | chip_margin_usage, tech_limit_up_cnt_20d, fund_roe_ttm, tech_limit_down_cnt_20d, fund_dividend_yield, fund_ep, tech_ma_gap_60d, risk_idio_vol_60d, tech_high_52w_gap, fund_rev_yoy | 9 |
| 12 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, risk_idio_vol_60d, fund_ep, fund_dividend_yield, tech_ma_gap_60d, fund_cfo_to_ni, fund_rev_yoy | 9 |
| 13 | chip_margin_usage, tech_limit_down_cnt_20d, risk_idio_vol_60d, fund_roe_ttm, fund_ep, fund_gross_margin, fund_dividend_yield, fund_debt_ratio, fund_rev_yoy, tech_limit_up_cnt_20d | 8 |
| 14 | chip_margin_usage, tech_limit_down_cnt_20d, risk_idio_vol_60d, fund_dividend_yield, fund_ep, fund_roe_ttm, fund_gross_margin, tech_limit_up_cnt_20d, fund_rev_yoy, tech_ma_gap_60d | 9 |
| 15 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, fund_rev_yoy, fund_roe_ttm, fund_ep, fund_rev_3m_yoy, fund_gross_margin, fund_debt_ratio | 8 |
| 16 | chip_margin_usage, tech_limit_down_cnt_20d, fund_rev_3m_yoy, fund_roe_ttm, fund_rev_yoy, fund_gross_margin, fund_ep, fund_dividend_yield, fund_opm_yoy_chg, fund_debt_ratio | 9 |
| 17 | chip_margin_usage, tech_limit_down_cnt_20d, fund_rev_3m_yoy, fund_roe_ttm, risk_idio_vol_60d, fund_dividend_yield, fund_ep, fund_rev_yoy, fund_gross_margin, fund_gm_yoy_chg | 8 |
| 18 | chip_margin_usage, risk_idio_vol_60d, tech_limit_down_cnt_20d, fund_roe_ttm, fund_dividend_yield, fund_rev_3m_yoy, fund_gm_yoy_chg, fund_debt_ratio, fund_gross_margin, tech_limit_up_cnt_20d | 8 |
| 19 | risk_idio_vol_60d, chip_margin_usage, fund_dividend_yield, fund_roe_ttm, fund_debt_ratio, tech_limit_down_cnt_20d, tech_turnover_rate_5d, fund_gm_yoy_chg, fund_ep, tech_high_52w_gap | 7 |
| 20 | risk_idio_vol_60d, chip_margin_usage, tech_limit_down_cnt_20d, fund_debt_ratio, fund_dividend_yield, fund_roe_ttm, fund_gm_yoy_chg, tech_turnover_rate_5d, tech_high_52w_gap, tech_limit_up_cnt_20d | 9 |
| 21 | chip_margin_usage, risk_idio_vol_60d, tech_limit_down_cnt_20d, fund_debt_ratio, fund_dividend_yield, fund_gm_yoy_chg, fund_roe_ttm, tech_turnover_rate_5d, tech_high_52w_gap, fund_rev_3m_yoy | 9 |
| 22 | chip_margin_usage, risk_idio_vol_60d, fund_debt_ratio, tech_limit_down_cnt_20d, fund_dividend_yield, fund_gm_yoy_chg, fund_roe_ttm, fund_rev_3m_yoy, risk_beta_250d, tech_high_52w_gap | 9 |
| 23 | chip_margin_usage, tech_limit_down_cnt_20d, fund_debt_ratio, fund_roe_ttm, risk_idio_vol_60d, fund_dividend_yield, tech_limit_up_cnt_20d, fund_gm_yoy_chg, risk_beta_250d, fund_rev_3m_yoy | 9 |
| 24 | risk_idio_vol_60d, tech_limit_up_cnt_20d, fund_dividend_yield, chip_margin_usage, fund_opm_yoy_chg, tech_high_52w_gap, fund_rev_3m_yoy, chip_fi_holding_pct, risk_beta_250d, chip_short_ratio | 6 |
| 25 | fund_dividend_yield, risk_idio_vol_60d, chip_margin_usage, tech_limit_up_cnt_20d, fund_gm_yoy_chg, tech_limit_down_cnt_20d, tech_high_52w_gap, chip_fi_holding_pct, fund_roe_ttm, fund_rev_3m_yoy | 7 |
| 26 | tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, fund_debt_ratio, risk_beta_250d, fund_roe_ttm, fund_gm_yoy_chg, tech_limit_up_cnt_20d, tech_high_52w_gap, tech_mom_250_20 | 7 |
| 27 | fund_debt_ratio, tech_limit_down_cnt_20d, chip_margin_usage, risk_beta_250d, fund_roe_ttm, fund_dividend_yield, fund_gm_yoy_chg, fund_eps_yoy, fund_gross_margin, fund_cfo_to_ni | 7 |
| 28 | fund_dividend_yield, chip_margin_usage, tech_limit_down_cnt_20d, risk_idio_vol_60d, fund_gm_yoy_chg, tech_limit_up_cnt_20d, fund_roe_ttm, chip_fi_holding_pct, size_log_mcap, tech_high_52w_gap | 5 |
| 29 | tech_limit_down_cnt_20d, fund_dividend_yield, chip_margin_usage, tech_limit_up_cnt_20d, risk_beta_250d, fund_debt_ratio, tech_high_52w_gap, fund_roe_ttm, risk_idio_vol_60d, fund_gm_yoy_chg | 8 |
| 30 | tech_limit_down_cnt_20d, risk_beta_250d, chip_margin_usage, fund_dividend_yield, tech_limit_up_cnt_20d, fund_debt_ratio, fund_roe_ttm, fund_gross_margin, fund_gm_yoy_chg, chip_fi_holding_pct | 8 |
| 31 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_beta_250d, tech_limit_up_cnt_20d, chip_margin_usage, tech_high_52w_gap, fund_debt_ratio, fund_roe_ttm, fund_rev_yoy_q, fund_rev_yoy | 7 |
| 32 | tech_limit_down_cnt_20d, risk_beta_250d, tech_limit_up_cnt_20d, fund_dividend_yield, tech_high_52w_gap, chip_margin_usage, fund_debt_ratio, tech_attention_t, fund_rev_yoy_q, fund_gm_yoy_chg | 8 |
| 33 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_limit_up_cnt_20d, fund_dividend_yield, fund_debt_ratio, risk_beta_250d, tech_high_52w_gap, fund_roe_ttm, fund_eps_yoy | 7 |
| 34 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, fund_debt_ratio, risk_beta_250d, fund_roe_ttm, fund_eps_yoy, tech_high_52w_gap, tech_vol_20d | 9 |
| 35 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_roe_ttm, fund_eps_yoy, risk_beta_250d, fund_debt_ratio, fund_rev_yoy_q | 9 |
| 36 | tech_limit_down_cnt_20d, risk_idio_vol_60d, risk_beta_250d, chip_margin_usage, fund_debt_ratio, fund_roe_ttm, tech_high_52w_gap, tech_limit_up_cnt_20d, fund_eps_yoy, fund_dividend_yield | 9 |
| 37 | tech_limit_down_cnt_20d, risk_beta_250d, risk_idio_vol_60d, fund_debt_ratio, fund_roe_ttm, chip_margin_usage, fund_dividend_yield, tech_high_52w_gap, fund_gm_yoy_chg, fund_eps_yoy | 9 |
| 38 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_high_52w_gap, risk_beta_250d, chip_short_ratio, fund_debt_ratio, fund_eps_yoy, fund_roe_ttm, tech_vol_20d | 8 |
| 39 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, risk_beta_250d, tech_high_52w_gap, fund_debt_ratio, fund_eps_yoy, fund_dividend_yield, fund_roe_ttm, chip_short_ratio | 9 |
| 40 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_high_52w_gap, risk_beta_250d, fund_eps_yoy, fund_debt_ratio, chip_short_ratio, fund_roe_ttm | 10 |
| 41 | tech_limit_down_cnt_20d, risk_idio_vol_60d, risk_beta_250d, fund_dividend_yield, fund_debt_ratio, fund_eps_yoy, fund_roe_ttm, chip_margin_usage, tech_high_52w_gap, tech_limit_up_cnt_20d | 9 |
| 42 | risk_idio_vol_60d, fund_dividend_yield, tech_limit_down_cnt_20d, fund_debt_ratio, risk_beta_250d, fund_eps_yoy, tech_high_52w_gap, fund_roe_ttm, chip_margin_usage, fund_gm_yoy_chg | 9 |
| 43 | fund_dividend_yield, tech_limit_down_cnt_20d, risk_beta_250d, risk_idio_vol_60d, fund_roe_ttm, fund_debt_ratio, fund_eps_yoy, tech_high_52w_gap, chip_margin_usage, fund_gm_yoy_chg | 10 |

- 進前 10 名的摺數：chip_margin_usage 36、tech_limit_down_cnt_20d 35、fund_roe_ttm 34、fund_dividend_yield 33、fund_debt_ratio 28、risk_idio_vol_60d 27、tech_high_52w_gap 25、tech_limit_up_cnt_20d 24、risk_beta_250d 20、fund_gm_yoy_chg 17、fund_eps_yoy 12、fund_ep 10、fund_gross_margin 10、fund_rev_3m_yoy 9、fund_rev_yoy 9、chip_fi_holding_pct 4、chip_short_ratio 4、tech_ma_gap_60d 4、fund_op_margin 3、fund_rev_yoy_q 3、tech_turnover_rate_5d 3、fund_cfo_to_ni 2、fund_opm_yoy_chg 2、tech_mom_250_20 2、tech_vol_20d 2、size_log_mcap 1、tech_attention_t 1

## 關卡：排名 20 日

- 研究期 IC 均值 0.1129 大於 0 且 NW t 10.17 大於 2：通過
- 贏 Ridge 的研究摺 21／36 摺 達 60%：未過
- Ridge 研究期 IC 均值 0.1138，NW t 10.65，2175 天
- 重要度變動只列研究摺，調參摺不計
