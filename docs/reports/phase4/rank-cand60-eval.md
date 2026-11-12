# 評估報表：rank-cand60（5 日）

> 手動附註：候選設計，未凍結。本機分支 exp/p4-rank-cand60（ad7ac40，develop 93302e0 加一個 commit，未 push）把第二批後的 60 個特徵（第一批 55 個 + chip 4 個 + fund_days_to_ex_date）寫進 production.DESIGNS 的排名設計；其餘同 rank-std1109（60 個月窗、重訓輪數 max(最佳輪數, 200)、每日一列、regression，標準摺 5–43，seed 0，Ridge 用第一批篩選後的 55 個特徵）。篩選 p3-final-20261109

## 排名 5 日

- 研究期 IC 均值 0.1407，標準差 0.1341，NW t 28.08，ICIR 1.05，2175 天
- IC 為正的研究摺比例：100%
- 中性化 IC：0.0905，NW t 42.46，2175 天
- 贏 baseline 的研究摺比例：92%（36／36 摺可比）
- 🔴 洩漏警訊：第 24、25 摺比同摺 baseline 的 IC 高 0.05 以上，先當成洩漏追查
- 打亂 label：這次沒跑，未判定

### 每摺

| 摺 | 類別 | 測試期 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 | 共同列 IC | baseline IC | 贏 baseline | 洩漏 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 調參 | 2015-04-01–2015-06-30 | 0.1490 | 0.1267 | 1.18 | 4.96 | 87% | 61 | 0.1490 | 0.1186 | 是 | — |
| 6 | 調參 | 2015-07-01–2015-09-30 | 0.1274 | 0.1170 | 1.09 | 4.28 | 78% | 63 | 0.1274 | 0.0985 | 是 | — |
| 7 | 調參 | 2015-10-01–2015-12-03 | 0.1039 | 0.1253 | 0.83 | 4.02 | 80% | 45 | 0.1039 | 0.0809 | 是 | — |
| 8 | 研究 | 2016-01-04–2016-03-31 | 0.1349 | 0.0923 | 1.46 | 6.32 | 93% | 57 | 0.1349 | 0.1155 | 是 | 否 |
| 9 | 研究 | 2016-04-01–2016-06-30 | 0.1503 | 0.1043 | 1.44 | 5.92 | 89% | 61 | 0.1503 | 0.1295 | 是 | 否 |
| 10 | 研究 | 2016-07-01–2016-09-30 | 0.1682 | 0.0918 | 1.83 | 9.69 | 97% | 62 | 0.1682 | 0.1313 | 是 | 否 |
| 11 | 研究 | 2016-10-03–2016-12-30 | 0.1505 | 0.1184 | 1.27 | 5.86 | 91% | 64 | 0.1505 | 0.1309 | 是 | 否 |
| 12 | 研究 | 2017-01-03–2017-03-31 | 0.1394 | 0.1122 | 1.24 | 5.91 | 86% | 57 | 0.1394 | 0.1206 | 是 | 否 |
| 13 | 研究 | 2017-04-05–2017-06-30 | 0.2120 | 0.1054 | 2.01 | 9.73 | 97% | 61 | 0.2120 | 0.1910 | 是 | 否 |
| 14 | 研究 | 2017-07-03–2017-09-30 | 0.1290 | 0.1150 | 1.12 | 4.76 | 89% | 66 | 0.1290 | 0.0964 | 是 | 否 |
| 15 | 研究 | 2017-10-02–2017-12-29 | 0.1684 | 0.0875 | 1.92 | 9.35 | 95% | 62 | 0.1684 | 0.1378 | 是 | 否 |
| 16 | 研究 | 2018-01-02–2018-03-31 | 0.1728 | 0.1207 | 1.43 | 6.80 | 93% | 58 | 0.1728 | 0.1522 | 是 | 否 |
| 17 | 研究 | 2018-04-02–2018-06-29 | 0.1601 | 0.1151 | 1.39 | 5.79 | 92% | 60 | 0.1601 | 0.1455 | 是 | 否 |
| 18 | 研究 | 2018-07-02–2018-09-28 | 0.1963 | 0.1123 | 1.75 | 9.85 | 98% | 64 | 0.1963 | 0.1813 | 是 | 否 |
| 19 | 研究 | 2018-10-01–2018-12-28 | 0.1203 | 0.2131 | 0.56 | 2.62 | 74% | 65 | 0.1203 | 0.0982 | 是 | 否 |
| 20 | 研究 | 2019-01-02–2019-03-29 | 0.0995 | 0.1276 | 0.78 | 3.28 | 76% | 54 | 0.0995 | 0.0855 | 是 | 否 |
| 21 | 研究 | 2019-04-01–2019-06-28 | 0.1632 | 0.1665 | 0.98 | 4.20 | 82% | 61 | 0.1632 | 0.1492 | 是 | 否 |
| 22 | 研究 | 2019-07-01–2019-09-27 | 0.1084 | 0.1210 | 0.90 | 4.30 | 83% | 63 | 0.1084 | 0.0793 | 是 | 否 |
| 23 | 研究 | 2019-10-01–2019-12-31 | 0.1610 | 0.1017 | 1.58 | 7.55 | 95% | 64 | 0.1610 | 0.1296 | 是 | 否 |
| 24 | 研究 | 2020-01-02–2020-03-31 | 0.1518 | 0.1448 | 1.05 | 6.36 | 88% | 56 | 0.1518 | 0.0948 | 是 | 是 |
| 25 | 研究 | 2020-04-01–2020-06-30 | 0.1143 | 0.1234 | 0.93 | 4.63 | 82% | 60 | 0.1143 | 0.0425 | 是 | 是 |
| 26 | 研究 | 2020-07-01–2020-09-30 | 0.1472 | 0.1427 | 1.03 | 5.03 | 79% | 66 | 0.1472 | 0.1116 | 是 | 否 |
| 27 | 研究 | 2020-10-05–2020-12-31 | 0.1219 | 0.1150 | 1.06 | 4.77 | 83% | 63 | 0.1219 | 0.1100 | 是 | 否 |
| 28 | 研究 | 2021-01-04–2021-03-31 | 0.1229 | 0.1469 | 0.84 | 3.47 | 75% | 55 | 0.1229 | 0.0917 | 是 | 否 |
| 29 | 研究 | 2021-04-01–2021-06-30 | 0.1473 | 0.1588 | 0.93 | 4.72 | 85% | 61 | 0.1473 | 0.1302 | 是 | 否 |
| 30 | 研究 | 2021-07-01–2021-09-30 | 0.1911 | 0.1250 | 1.53 | 8.67 | 98% | 64 | 0.1911 | 0.1746 | 是 | 否 |
| 31 | 研究 | 2021-10-01–2021-12-30 | 0.1136 | 0.1097 | 1.04 | 4.72 | 86% | 64 | 0.1136 | 0.0925 | 是 | 否 |
| 32 | 研究 | 2022-01-03–2022-03-31 | 0.1591 | 0.1260 | 1.26 | 5.47 | 93% | 56 | 0.1591 | 0.1558 | 是 | 否 |
| 33 | 研究 | 2022-04-01–2022-06-30 | 0.1290 | 0.1118 | 1.15 | 5.84 | 92% | 61 | 0.1290 | 0.1299 | 否 | 否 |
| 34 | 研究 | 2022-07-01–2022-09-30 | 0.0932 | 0.1771 | 0.53 | 2.42 | 66% | 65 | 0.0932 | 0.0715 | 是 | 否 |
| 35 | 研究 | 2022-10-03–2022-12-30 | 0.1029 | 0.1940 | 0.53 | 2.40 | 75% | 64 | 0.1029 | 0.1118 | 否 | 否 |
| 36 | 研究 | 2023-01-03–2023-03-31 | 0.0980 | 0.1557 | 0.63 | 2.87 | 74% | 54 | 0.0980 | 0.0824 | 是 | 否 |
| 37 | 研究 | 2023-04-06–2023-06-30 | 0.1125 | 0.1503 | 0.75 | 3.76 | 76% | 59 | 0.1125 | 0.0937 | 是 | 否 |
| 38 | 研究 | 2023-07-03–2023-09-28 | 0.1416 | 0.1266 | 1.12 | 4.89 | 86% | 63 | 0.1416 | 0.1278 | 是 | 否 |
| 39 | 研究 | 2023-10-02–2023-12-29 | 0.1385 | 0.1228 | 1.13 | 5.17 | 83% | 63 | 0.1385 | 0.1266 | 是 | 否 |
| 40 | 研究 | 2024-01-02–2024-03-29 | 0.1132 | 0.1208 | 0.94 | 3.77 | 88% | 56 | 0.1132 | 0.1159 | 否 | 否 |
| 41 | 研究 | 2024-04-01–2024-06-28 | 0.1278 | 0.1215 | 1.05 | 4.34 | 82% | 61 | 0.1278 | 0.1061 | 是 | 否 |
| 42 | 研究 | 2024-07-01–2024-09-30 | 0.1550 | 0.1196 | 1.30 | 6.24 | 95% | 63 | 0.1550 | 0.1151 | 是 | 否 |
| 43 | 研究 | 2024-10-01–2024-12-03 | 0.1418 | 0.1559 | 0.91 | 4.08 | 81% | 42 | 0.1418 | 0.1218 | 是 | 否 |

### 每年

| 年 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0.1513 | 0.1027 | 1.47 | 13.02 | 92% | 244 |
| 2017 | 0.1619 | 0.1098 | 1.47 | 12.94 | 92% | 246 |
| 2018 | 0.1620 | 0.1496 | 1.08 | 9.68 | 89% | 247 |
| 2019 | 0.1341 | 0.1334 | 1.01 | 8.70 | 84% | 242 |
| 2020 | 0.1337 | 0.1321 | 1.01 | 9.88 | 82% | 245 |
| 2021 | 0.1445 | 0.1383 | 1.04 | 9.67 | 86% | 244 |
| 2022 | 0.1196 | 0.1585 | 0.75 | 6.85 | 81% | 246 |
| 2023 | 0.1238 | 0.1390 | 0.89 | 8.19 | 80% | 239 |
| 2024 | 0.1345 | 0.1282 | 1.05 | 8.96 | 87% | 222 |

### 成交金額三分位

| 組 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 低 | 0.1467 | 0.1369 | 1.07 | 29.20 | 86% | 2175 |
| 中 | 0.1453 | 0.1380 | 1.05 | 28.17 | 85% | 2175 |
| 高 | 0.1284 | 0.1639 | 0.78 | 21.24 | 78% | 2175 |

### 十分位（扣成本）

| 樣本 | 組合 | 月報酬 | 月化超額 | 超額最大回撤 | 年化換手率（倍） | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | D1 | -4.65% | -4.61% | -99.42% | 34.50 | 2169 |
| 全部 | D2 | -3.33% | -3.29% | -97.36% | 41.23 | 2169 |
| 全部 | D3 | -2.64% | -2.60% | -94.32% | 42.71 | 2169 |
| 全部 | D4 | -2.07% | -2.02% | -89.21% | 43.41 | 2169 |
| 全部 | D5 | -1.77% | -1.73% | -85.07% | 43.58 | 2169 |
| 全部 | D6 | -1.52% | -1.47% | -80.05% | 43.27 | 2169 |
| 全部 | D7 | -1.16% | -1.12% | -70.37% | 42.64 | 2169 |
| 全部 | D8 | -0.81% | -0.77% | -57.17% | 41.48 | 2169 |
| 全部 | D9 | -0.31% | -0.26% | -31.61% | 38.42 | 2169 |
| 全部 | D10 | 0.87% | 0.91% | -9.47% | 27.79 | 2169 |
| 全部 | universe | -0.04% | 0.00% | 0.00% | 2.92 | 2169 |
| 全部 | D10-D1 | 2.44% | 2.44% | -12.14% | 62.28 | 2169 |
| 成交金額前三分之一 | D1 | -4.89% | -4.72% | -99.50% | 35.32 | 2169 |
| 成交金額前三分之一 | D2 | -3.49% | -3.33% | -97.49% | 41.76 | 2169 |
| 成交金額前三分之一 | D3 | -2.73% | -2.56% | -94.15% | 43.09 | 2169 |
| 成交金額前三分之一 | D4 | -2.00% | -1.83% | -86.89% | 43.84 | 2169 |
| 成交金額前三分之一 | D5 | -1.84% | -1.68% | -84.41% | 43.89 | 2169 |
| 成交金額前三分之一 | D6 | -1.62% | -1.46% | -79.97% | 43.59 | 2169 |
| 成交金額前三分之一 | D7 | -1.32% | -1.15% | -71.89% | 42.76 | 2169 |
| 成交金額前三分之一 | D8 | -0.85% | -0.68% | -53.88% | 41.43 | 2169 |
| 成交金額前三分之一 | D9 | -0.33% | -0.16% | -30.52% | 37.76 | 2169 |
| 成交金額前三分之一 | D10 | 1.01% | 1.18% | -11.88% | 27.84 | 2169 |
| 成交金額前三分之一 | universe | -0.17% | 0.00% | 0.00% | 4.16 | 2169 |
| 成交金額前三分之一 | D10-D1 | 2.74% | 2.74% | -16.14% | 63.16 | 2169 |

## 特徵重要度前 10 名

| 摺 | 前 10 名（gain） | 和前一摺重疊 |
| --- | --- | --- |
| 8 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_ret_1d, tech_kd_d, tech_turnover_rate_5d, tech_mom_250_20, tech_limit_up_t | — |
| 9 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, tech_mom_250_20, tech_kd_d, tech_turnover_rate_5d, tech_limit_up_cnt_20d, tech_ret_1d | 9 |
| 10 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_ret_1d, tech_limit_up_cnt_20d, tech_turnover_rate_5d, tech_kd_d, tech_mom_250_20 | 10 |
| 11 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_limit_down_cnt_20d, fund_dividend_yield, tech_turnover_rate_5d, tech_kd_d, chip_it_net_5d, tech_ret_5d | 8 |
| 12 | risk_idio_vol_60d, tech_ma_gap_5d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ret_1d, tech_turnover_rate_5d, fund_dividend_yield, tech_high_52w_gap, chip_it_net_5d, tech_kd_d | 9 |
| 13 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_turnover_rate_5d, fund_dividend_yield, tech_limit_up_cnt_20d, chip_it_net_5d, tech_kd_d | 9 |
| 14 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_turnover_rate_5d, fund_dividend_yield, tech_limit_up_cnt_20d, chip_it_net_5d, tech_kd_d | 10 |
| 15 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_1d, tech_ret_5d, chip_it_net_5d, tech_kd_d, tech_turnover_rate_5d, tech_ma_gap_60d | 8 |
| 16 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_1d, tech_ret_5d, tech_kd_d, chip_it_net_5d, tech_limit_up_cnt_20d, tech_high_52w_gap | 8 |
| 17 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_kd_d, tech_limit_down_cnt_20d, tech_ret_5d, chip_it_net_5d, tech_turnover_rate_5d, tech_limit_up_cnt_20d | 9 |
| 18 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_1d, tech_kd_d, tech_ret_5d, chip_it_net_5d, tech_limit_up_cnt_20d, tech_vol_20d | 9 |
| 19 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_kd_d, tech_ret_5d, tech_ret_1d, tech_limit_down_cnt_20d, tech_vol_20d, tech_limit_up_cnt_20d, tech_high_52w_gap | 9 |
| 20 | risk_idio_vol_60d, tech_ma_gap_5d, tech_ret_5d, chip_margin_usage, tech_kd_d, tech_ret_1d, tech_vol_20d, tech_limit_down_cnt_20d, chip_it_net_5d, tech_limit_up_cnt_20d | 9 |
| 21 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_ret_5d, tech_vol_20d, tech_kd_d, tech_ret_1d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_it_net_5d | 10 |
| 22 | risk_idio_vol_60d, tech_ma_gap_5d, chip_margin_usage, tech_limit_down_cnt_20d, tech_ret_5d, tech_limit_up_cnt_20d, tech_vol_20d, tech_kd_d, tech_ret_1d, tech_high_52w_gap | 9 |
| 23 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, fund_dividend_yield, fund_roe_ttm, tech_high_52w_gap, fund_days_since_fs, tech_vol_20d | 7 |
| 24 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_limit_up_cnt_20d, fund_dividend_yield, fund_ep, fund_rev_yoy, fund_roe_ttm, fund_gm_yoy_chg, fund_rev_3m_yoy | 6 |
| 25 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_limit_up_cnt_20d, fund_dividend_yield, tech_kd_d, tech_ret_5d, tech_vol_20d, tech_high_52w_gap | 5 |
| 26 | risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, tech_ret_5d, tech_high_52w_gap, tech_vol_20d, tech_ma_gap_60d | 9 |
| 27 | risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, tech_ma_gap_5d, chip_margin_usage, fund_dividend_yield, tech_high_52w_gap, tech_vol_20d, tech_ret_5d, chip_day_trade_ratio_5d | 9 |
| 28 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, fund_dividend_yield, tech_high_52w_gap, tech_ret_5d, tech_vol_20d, risk_beta_250d | 9 |
| 29 | risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, chip_margin_usage, tech_ma_gap_5d, fund_dividend_yield, tech_high_52w_gap, tech_ret_5d, tech_vol_20d, chip_day_trade_ratio_5d | 9 |
| 30 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_vol_20d, tech_ret_1d, fund_dividend_yield, tech_ret_5d, tech_high_52w_gap | 9 |
| 31 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_high_52w_gap, tech_vol_20d, tech_attention_t, fund_dividend_yield | 9 |
| 32 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_high_52w_gap, tech_vol_20d, tech_attention_t, tech_ret_5d | 9 |
| 33 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_ma_gap_5d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ret_1d, tech_vol_20d, tech_high_52w_gap, fund_dividend_yield, tech_ret_5d | 9 |
| 34 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_ret_1d, tech_vol_20d, tech_high_52w_gap, fund_dividend_yield, tech_attention_t | 9 |
| 35 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_high_52w_gap, tech_vol_20d, tech_ret_1d, tech_attention_t, fund_dividend_yield | 10 |
| 36 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, tech_high_52w_gap, risk_beta_250d, tech_attention_t, tech_vol_20d, fund_dividend_yield | 9 |
| 37 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, chip_margin_usage, tech_high_52w_gap, tech_attention_t, fund_dividend_yield, tech_vol_20d, tech_ret_1d | 9 |
| 38 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_attention_t, tech_ma_gap_5d, fund_dividend_yield, risk_beta_250d, tech_high_52w_gap, tech_limit_up_t | 8 |
| 39 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, tech_ma_gap_5d, fund_dividend_yield, tech_high_52w_gap, tech_attention_t, risk_beta_250d, tech_limit_up_t | 10 |
| 40 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_ma_gap_5d, fund_dividend_yield, chip_margin_usage, tech_high_52w_gap, tech_limit_up_t, risk_beta_250d, tech_attention_t | 10 |
| 41 | tech_limit_down_cnt_20d, risk_idio_vol_60d, tech_limit_up_cnt_20d, chip_margin_usage, fund_dividend_yield, risk_beta_250d, tech_high_52w_gap, tech_attention_t, tech_ma_gap_5d, tech_limit_up_t | 10 |
| 42 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, chip_margin_usage, risk_beta_250d, tech_high_52w_gap, tech_ma_gap_5d, tech_limit_up_t, tech_attention_t | 10 |
| 43 | risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, tech_high_52w_gap, chip_margin_usage, tech_ma_gap_5d, tech_turnover_rate_5d, tech_limit_up_t, tech_ma_gap_20d | 8 |

- 進前 10 名的摺數：chip_margin_usage 36、risk_idio_vol_60d 36、tech_limit_down_cnt_20d 36、tech_ma_gap_5d 35、tech_limit_up_cnt_20d 32、fund_dividend_yield 27、tech_high_52w_gap 24、tech_ret_1d 22、tech_vol_20d 19、tech_ret_5d 17、tech_kd_d 16、tech_attention_t 11、chip_it_net_5d 10、tech_turnover_rate_5d 10、risk_beta_250d 7、tech_limit_up_t 7、tech_mom_250_20 3、chip_day_trade_ratio_5d 2、fund_roe_ttm 2、tech_ma_gap_60d 2、fund_days_since_fs 1、fund_ep 1、fund_gm_yoy_chg 1、fund_rev_3m_yoy 1、fund_rev_yoy 1、tech_ma_gap_20d 1

## 關卡：排名 5 日

- 研究期 IC 均值 0.1407 大於 0 且 NW t 28.08 大於 2：通過
- 贏 Ridge 的研究摺 33／36 摺 達 60%：通過
- Ridge 研究期 IC 均值 0.1191，NW t 21.64，2175 天
- 重要度變動只列研究摺，調參摺不計

# 評估報表：rank-cand60（20 日）

## 排名 20 日

- 研究期 IC 均值 0.1343，標準差 0.1376，NW t 13.66，ICIR 0.98，2175 天
- IC 為正的研究摺比例：97%
- 中性化 IC：0.0839，NW t 22.83，2175 天
- 贏 baseline 的研究摺比例：75%（36／36 摺可比）
- 🔴 洩漏警訊：第 19、25、29 摺比同摺 baseline 的 IC 高 0.05 以上，先當成洩漏追查
- 打亂 label：這次沒跑，未判定

### 每摺

| 摺 | 類別 | 測試期 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 | 共同列 IC | baseline IC | 贏 baseline | 洩漏 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 調參 | 2015-04-01–2015-06-30 | 0.1604 | 0.1024 | 1.57 | 3.58 | 100% | 61 | 0.1604 | 0.1966 | 否 | — |
| 6 | 調參 | 2015-07-01–2015-09-30 | 0.0663 | 0.1916 | 0.35 | 0.77 | 54% | 63 | 0.0663 | 0.0790 | 否 | — |
| 7 | 調參 | 2015-10-01–2015-12-03 | 0.0675 | 0.0769 | 0.88 | 3.40 | 78% | 45 | 0.0675 | 0.0321 | 是 | — |
| 8 | 研究 | 2016-01-04–2016-03-31 | 0.1270 | 0.0885 | 1.43 | 3.24 | 98% | 57 | 0.1270 | 0.1161 | 是 | 否 |
| 9 | 研究 | 2016-04-01–2016-06-30 | 0.1346 | 0.1187 | 1.13 | 2.70 | 82% | 61 | 0.1346 | 0.1271 | 是 | 否 |
| 10 | 研究 | 2016-07-01–2016-09-30 | 0.1887 | 0.0841 | 2.24 | 7.95 | 98% | 62 | 0.1887 | 0.1547 | 是 | 否 |
| 11 | 研究 | 2016-10-03–2016-12-30 | 0.0709 | 0.0884 | 0.80 | 2.18 | 73% | 64 | 0.0709 | 0.0465 | 是 | 否 |
| 12 | 研究 | 2017-01-03–2017-03-31 | 0.1211 | 0.1214 | 1.00 | 2.23 | 81% | 57 | 0.1211 | 0.1225 | 否 | 否 |
| 13 | 研究 | 2017-04-05–2017-06-30 | 0.2040 | 0.0868 | 2.35 | 8.51 | 98% | 61 | 0.2040 | 0.1848 | 是 | 否 |
| 14 | 研究 | 2017-07-03–2017-09-30 | 0.0662 | 0.1173 | 0.56 | 1.96 | 77% | 66 | 0.0662 | 0.0520 | 是 | 否 |
| 15 | 研究 | 2017-10-02–2017-12-29 | 0.1533 | 0.0786 | 1.95 | 5.09 | 95% | 62 | 0.1533 | 0.1436 | 是 | 否 |
| 16 | 研究 | 2018-01-02–2018-03-31 | 0.1836 | 0.1191 | 1.54 | 4.49 | 90% | 58 | 0.1836 | 0.1784 | 是 | 否 |
| 17 | 研究 | 2018-04-02–2018-06-29 | 0.1538 | 0.1566 | 0.98 | 2.34 | 77% | 60 | 0.1538 | 0.1393 | 是 | 否 |
| 18 | 研究 | 2018-07-02–2018-09-28 | 0.2890 | 0.0886 | 3.26 | 9.57 | 100% | 64 | 0.2890 | 0.2703 | 是 | 否 |
| 19 | 研究 | 2018-10-01–2018-12-28 | 0.1234 | 0.1896 | 0.65 | 2.83 | 82% | 65 | 0.1234 | 0.0275 | 是 | 是 |
| 20 | 研究 | 2019-01-02–2019-03-29 | -0.0002 | 0.1424 | -0.00 | -0.00 | 41% | 54 | -0.0002 | 0.0011 | 否 | 否 |
| 21 | 研究 | 2019-04-01–2019-06-28 | 0.1741 | 0.2390 | 0.73 | 1.51 | 62% | 61 | 0.1741 | 0.1657 | 是 | 否 |
| 22 | 研究 | 2019-07-01–2019-09-27 | 0.0690 | 0.0906 | 0.76 | 3.10 | 83% | 63 | 0.0690 | 0.0527 | 是 | 否 |
| 23 | 研究 | 2019-10-01–2019-12-31 | 0.1472 | 0.1028 | 1.43 | 4.18 | 91% | 64 | 0.1472 | 0.1640 | 否 | 否 |
| 24 | 研究 | 2020-01-02–2020-03-31 | 0.1384 | 0.1993 | 0.69 | 1.98 | 77% | 56 | 0.1384 | 0.1218 | 是 | 否 |
| 25 | 研究 | 2020-04-01–2020-06-30 | 0.0608 | 0.1005 | 0.61 | 2.69 | 70% | 60 | 0.0608 | -0.0165 | 是 | 是 |
| 26 | 研究 | 2020-07-01–2020-09-30 | 0.0971 | 0.1099 | 0.88 | 2.53 | 76% | 66 | 0.0971 | 0.1028 | 否 | 否 |
| 27 | 研究 | 2020-10-05–2020-12-31 | 0.1102 | 0.0870 | 1.27 | 4.04 | 87% | 63 | 0.1102 | 0.1372 | 否 | 否 |
| 28 | 研究 | 2021-01-04–2021-03-31 | 0.0858 | 0.1174 | 0.73 | 2.22 | 75% | 55 | 0.0858 | 0.0501 | 是 | 否 |
| 29 | 研究 | 2021-04-01–2021-06-30 | 0.2094 | 0.1003 | 2.09 | 7.17 | 98% | 61 | 0.2094 | 0.1379 | 是 | 是 |
| 30 | 研究 | 2021-07-01–2021-09-30 | 0.2065 | 0.0835 | 2.47 | 10.66 | 100% | 64 | 0.2065 | 0.2248 | 否 | 否 |
| 31 | 研究 | 2021-10-01–2021-12-30 | 0.0941 | 0.1836 | 0.51 | 1.19 | 56% | 64 | 0.0941 | 0.0979 | 否 | 否 |
| 32 | 研究 | 2022-01-03–2022-03-31 | 0.1715 | 0.0895 | 1.91 | 6.15 | 100% | 56 | 0.1715 | 0.1559 | 是 | 否 |
| 33 | 研究 | 2022-04-01–2022-06-30 | 0.1690 | 0.1052 | 1.61 | 5.79 | 92% | 61 | 0.1690 | 0.1614 | 是 | 否 |
| 34 | 研究 | 2022-07-01–2022-09-30 | 0.1891 | 0.1758 | 1.08 | 2.60 | 86% | 65 | 0.1891 | 0.1497 | 是 | 否 |
| 35 | 研究 | 2022-10-03–2022-12-30 | 0.0280 | 0.1994 | 0.14 | 0.34 | 52% | 64 | 0.0280 | 0.0440 | 否 | 否 |
| 36 | 研究 | 2023-01-03–2023-03-31 | 0.0752 | 0.1219 | 0.62 | 1.88 | 74% | 54 | 0.0752 | 0.0520 | 是 | 否 |
| 37 | 研究 | 2023-04-06–2023-06-30 | 0.1398 | 0.0717 | 1.95 | 6.25 | 98% | 59 | 0.1398 | 0.1097 | 是 | 否 |
| 38 | 研究 | 2023-07-03–2023-09-28 | 0.1882 | 0.1013 | 1.86 | 4.99 | 98% | 63 | 0.1882 | 0.1992 | 否 | 否 |
| 39 | 研究 | 2023-10-02–2023-12-29 | 0.1090 | 0.1069 | 1.02 | 3.07 | 83% | 63 | 0.1090 | 0.1065 | 是 | 否 |
| 40 | 研究 | 2024-01-02–2024-03-29 | 0.1535 | 0.0924 | 1.66 | 3.86 | 98% | 56 | 0.1535 | 0.1387 | 是 | 否 |
| 41 | 研究 | 2024-04-01–2024-06-28 | 0.0851 | 0.1199 | 0.71 | 1.83 | 70% | 61 | 0.0851 | 0.0650 | 是 | 否 |
| 42 | 研究 | 2024-07-01–2024-09-30 | 0.1451 | 0.1257 | 1.15 | 2.88 | 98% | 63 | 0.1451 | 0.1396 | 是 | 否 |
| 43 | 研究 | 2024-10-01–2024-12-03 | 0.1680 | 0.0689 | 2.44 | 7.36 | 100% | 42 | 0.1680 | 0.1651 | 是 | 否 |

### 每年

| 年 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 0.1299 | 0.1044 | 1.24 | 5.47 | 88% | 244 |
| 2017 | 0.1350 | 0.1140 | 1.18 | 5.78 | 88% | 246 |
| 2018 | 0.1878 | 0.1567 | 1.20 | 6.14 | 87% | 247 |
| 2019 | 0.1008 | 0.1674 | 0.60 | 2.69 | 70% | 242 |
| 2020 | 0.1010 | 0.1314 | 0.77 | 4.76 | 78% | 245 |
| 2021 | 0.1505 | 0.1399 | 1.08 | 5.03 | 82% | 244 |
| 2022 | 0.1382 | 0.1648 | 0.84 | 3.89 | 82% | 246 |
| 2023 | 0.1298 | 0.1092 | 1.19 | 6.20 | 89% | 239 |
| 2024 | 0.1351 | 0.1111 | 1.22 | 5.47 | 91% | 222 |

### 成交金額三分位

| 組 | IC 均值 | 標準差 | ICIR | NW t | IC 為正的天數比例 | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 低 | 0.1415 | 0.1447 | 0.98 | 14.02 | 83% | 2175 |
| 中 | 0.1438 | 0.1367 | 1.05 | 15.25 | 85% | 2175 |
| 高 | 0.1162 | 0.1678 | 0.69 | 9.54 | 76% | 2175 |

### 十分位（扣成本）

| 樣本 | 組合 | 月報酬 | 月化超額 | 超額最大回撤 | 年化換手率（倍） | 天數 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部 | D1 | -0.77% | -1.64% | -83.13% | 8.20 | 2154 |
| 全部 | D2 | -0.12% | -0.98% | -65.49% | 10.21 | 2154 |
| 全部 | D3 | 0.07% | -0.79% | -57.65% | 10.65 | 2154 |
| 全部 | D4 | 0.34% | -0.53% | -43.51% | 10.84 | 2154 |
| 全部 | D5 | 0.46% | -0.41% | -35.64% | 10.85 | 2154 |
| 全部 | D6 | 0.59% | -0.27% | -25.66% | 10.80 | 2154 |
| 全部 | D7 | 0.73% | -0.14% | -16.82% | 10.71 | 2154 |
| 全部 | D8 | 0.86% | -0.01% | -10.29% | 10.42 | 2154 |
| 全部 | D9 | 1.17% | 0.30% | -8.76% | 9.72 | 2154 |
| 全部 | D10 | 1.52% | 0.66% | -11.51% | 7.07 | 2154 |
| 全部 | universe | 0.87% | 0.00% | 0.00% | 1.87 | 2154 |
| 全部 | D10-D1 | 1.51% | 1.51% | -23.33% | 15.27 | 2154 |
| 成交金額前三分之一 | D1 | -1.01% | -1.83% | -87.13% | 8.90 | 2154 |
| 成交金額前三分之一 | D2 | -0.10% | -0.92% | -63.53% | 10.54 | 2154 |
| 成交金額前三分之一 | D3 | 0.26% | -0.56% | -46.36% | 10.89 | 2154 |
| 成交金額前三分之一 | D4 | 0.47% | -0.35% | -33.69% | 11.02 | 2154 |
| 成交金額前三分之一 | D5 | 0.54% | -0.28% | -27.39% | 11.02 | 2154 |
| 成交金額前三分之一 | D6 | 0.69% | -0.12% | -16.28% | 10.92 | 2154 |
| 成交金額前三分之一 | D7 | 0.72% | -0.10% | -15.41% | 10.78 | 2154 |
| 成交金額前三分之一 | D8 | 0.83% | 0.02% | -15.69% | 10.49 | 2154 |
| 成交金額前三分之一 | D9 | 0.92% | 0.10% | -17.69% | 9.75 | 2154 |
| 成交金額前三分之一 | D10 | 1.34% | 0.52% | -16.86% | 7.09 | 2154 |
| 成交金額前三分之一 | universe | 0.82% | 0.00% | 0.00% | 2.59 | 2154 |
| 成交金額前三分之一 | D10-D1 | 1.51% | 1.51% | -30.27% | 15.99 | 2154 |

## 特徵重要度前 10 名

| 摺 | 前 10 名（gain） | 和前一摺重疊 |
| --- | --- | --- |
| 8 | chip_margin_usage, fund_dividend_yield, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, tech_mom_250_20, risk_idio_vol_60d, tech_high_52w_gap, risk_beta_250d, fund_roe_ttm, chip_day_trade_ratio_5d | — |
| 9 | chip_margin_usage, fund_dividend_yield, tech_limit_down_cnt_20d, tech_mom_250_20, tech_limit_up_cnt_20d, fund_debt_ratio, risk_beta_250d, chip_fi_holding_pct, fund_gross_margin, risk_idio_vol_60d | 7 |
| 10 | tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, tech_ma_gap_60d, tech_mom_250_20, risk_idio_vol_60d, fund_roe_ttm, tech_turnover_rate_5d | 6 |
| 11 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_roe_ttm, tech_ma_gap_60d, tech_turnover_rate_5d, risk_idio_vol_60d, tech_mom_250_20 | 10 |
| 12 | tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_high_52w_gap, tech_limit_up_cnt_20d, fund_roe_ttm, tech_turnover_rate_5d, tech_ma_gap_60d, risk_idio_vol_60d, tech_mom_250_20 | 10 |
| 13 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, fund_roe_ttm, tech_high_52w_gap, risk_idio_vol_60d, tech_mom_250_20, fund_gross_margin, fund_debt_ratio | 8 |
| 14 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, fund_dividend_yield, tech_high_52w_gap, tech_turnover_rate_5d, tech_ma_gap_60d, tech_mom_250_20, fund_ep | 7 |
| 15 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, tech_high_52w_gap, fund_dividend_yield, tech_ma_gap_60d, fund_gross_margin, fund_ep, fund_op_margin | 8 |
| 16 | chip_margin_usage, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, fund_roe_ttm, tech_high_52w_gap, risk_idio_vol_60d, fund_dividend_yield, fund_gross_margin, fund_ep, tech_ma_gap_60d | 9 |
| 17 | chip_margin_usage, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, fund_dividend_yield, fund_roe_ttm, tech_high_52w_gap, fund_ep, risk_idio_vol_60d, fund_gross_margin, tech_ma_gap_60d | 10 |
| 18 | chip_margin_usage, tech_limit_up_cnt_20d, tech_limit_down_cnt_20d, fund_roe_ttm, fund_dividend_yield, tech_high_52w_gap, risk_idio_vol_60d, fund_ep, fund_gross_margin, fund_op_margin | 9 |
| 19 | chip_margin_usage, risk_idio_vol_60d, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, fund_roe_ttm, tech_high_52w_gap, fund_gm_yoy_chg, fund_rev_yoy, tech_turnover_rate_5d | 7 |
| 20 | risk_idio_vol_60d, chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_roe_ttm, fund_dividend_yield, tech_high_52w_gap, fund_ep, fund_rev_yoy, fund_gm_yoy_chg | 9 |
| 21 | chip_margin_usage, risk_idio_vol_60d, tech_limit_down_cnt_20d, fund_dividend_yield, fund_roe_ttm, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_gm_yoy_chg, fund_rev_yoy, fund_gross_margin | 9 |
| 22 | chip_margin_usage, risk_idio_vol_60d, tech_limit_down_cnt_20d, fund_dividend_yield, fund_roe_ttm, fund_gm_yoy_chg, tech_high_52w_gap, tech_limit_up_cnt_20d, fund_rev_yoy, fund_rev_3m_yoy | 9 |
| 23 | chip_margin_usage, tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, fund_gm_yoy_chg, fund_roe_ttm, fund_rev_yoy, tech_high_52w_gap, fund_rev_3m_yoy | 10 |
| 24 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, fund_roe_ttm, fund_gm_yoy_chg, risk_idio_vol_60d, fund_rev_3m_yoy, fund_ep, fund_rev_yoy, tech_limit_up_cnt_20d | 9 |
| 25 | chip_margin_usage, risk_idio_vol_60d, fund_dividend_yield, tech_limit_down_cnt_20d, fund_rev_3m_yoy, tech_limit_up_cnt_20d, fund_roe_ttm, tech_high_52w_gap, fund_gm_yoy_chg, fund_rev_yoy | 9 |
| 26 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, fund_gm_yoy_chg, tech_limit_up_cnt_20d, fund_roe_ttm, risk_idio_vol_60d, risk_beta_250d, fund_rev_3m_yoy, fund_gross_margin | 8 |
| 27 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, fund_roe_ttm, fund_gm_yoy_chg, fund_rev_3m_yoy, risk_beta_250d, fund_ep, fund_gross_margin, fund_debt_ratio | 8 |
| 28 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, fund_gm_yoy_chg, tech_high_52w_gap, fund_roe_ttm, risk_beta_250d, tech_limit_up_cnt_20d, fund_rev_yoy | 6 |
| 29 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_gm_yoy_chg, risk_beta_250d, fund_roe_ttm, fund_rev_3m_yoy | 9 |
| 30 | chip_margin_usage, tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, risk_beta_250d, fund_gm_yoy_chg, risk_idio_vol_60d, tech_high_52w_gap, fund_gross_margin, fund_roe_ttm | 9 |
| 31 | tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, risk_idio_vol_60d, fund_dividend_yield, tech_high_52w_gap, risk_beta_250d, fund_gm_yoy_chg, fund_roe_ttm, fund_rev_3m_yoy | 9 |
| 32 | tech_limit_down_cnt_20d, tech_limit_up_cnt_20d, chip_margin_usage, fund_dividend_yield, risk_idio_vol_60d, risk_beta_250d, tech_high_52w_gap, fund_gm_yoy_chg, fund_rev_3m_yoy, fund_rev_yoy | 9 |
| 33 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, risk_beta_250d, fund_gm_yoy_chg, fund_rev_3m_yoy, fund_rev_yoy | 10 |
| 34 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, tech_high_52w_gap, risk_beta_250d, fund_dividend_yield, fund_gm_yoy_chg, fund_roe_ttm, fund_rev_yoy | 9 |
| 35 | risk_idio_vol_60d, tech_limit_down_cnt_20d, chip_margin_usage, tech_limit_up_cnt_20d, tech_high_52w_gap, risk_beta_250d, fund_gm_yoy_chg, fund_roe_ttm, fund_dividend_yield, fund_eps_yoy | 9 |
| 36 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_high_52w_gap, tech_limit_up_cnt_20d, risk_beta_250d, fund_dividend_yield, fund_gm_yoy_chg, fund_roe_ttm, fund_eps_yoy | 10 |
| 37 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, tech_limit_up_cnt_20d, tech_high_52w_gap, risk_beta_250d, fund_dividend_yield, fund_gm_yoy_chg, fund_roe_ttm, fund_eps_yoy | 10 |
| 38 | tech_limit_down_cnt_20d, risk_idio_vol_60d, chip_margin_usage, risk_beta_250d, fund_dividend_yield, tech_limit_up_cnt_20d, tech_high_52w_gap, fund_gm_yoy_chg, fund_eps_yoy, fund_gross_margin | 9 |
| 39 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, tech_high_52w_gap, chip_margin_usage, tech_limit_up_cnt_20d, risk_beta_250d, fund_eps_yoy, fund_gm_yoy_chg, chip_short_ratio | 9 |
| 40 | tech_limit_down_cnt_20d, risk_idio_vol_60d, fund_dividend_yield, tech_high_52w_gap, tech_limit_up_cnt_20d, risk_beta_250d, chip_margin_usage, fund_eps_yoy, chip_short_ratio, fund_gm_yoy_chg | 10 |
| 41 | tech_limit_down_cnt_20d, fund_dividend_yield, risk_idio_vol_60d, tech_limit_up_cnt_20d, tech_high_52w_gap, risk_beta_250d, chip_margin_usage, fund_eps_yoy, fund_gm_yoy_chg, chip_short_ratio | 10 |
| 42 | tech_limit_down_cnt_20d, fund_dividend_yield, tech_limit_up_cnt_20d, risk_beta_250d, risk_idio_vol_60d, tech_high_52w_gap, chip_margin_usage, fund_eps_yoy, fund_gm_yoy_chg, fund_roe_ttm | 9 |
| 43 | fund_dividend_yield, tech_limit_down_cnt_20d, risk_idio_vol_60d, risk_beta_250d, tech_high_52w_gap, chip_margin_usage, tech_limit_up_cnt_20d, fund_eps_yoy, fund_gm_yoy_chg, fund_debt_ratio | 9 |

- 進前 10 名的摺數：chip_margin_usage 36、fund_dividend_yield 36、tech_limit_down_cnt_20d 36、tech_limit_up_cnt_20d 35、risk_idio_vol_60d 33、tech_high_52w_gap 32、fund_roe_ttm 28、fund_gm_yoy_chg 25、risk_beta_250d 20、fund_gross_margin 11、fund_rev_yoy 11、fund_rev_3m_yoy 10、fund_eps_yoy 9、fund_ep 8、tech_ma_gap_60d 7、tech_mom_250_20 7、tech_turnover_rate_5d 5、fund_debt_ratio 4、chip_short_ratio 3、fund_op_margin 2、chip_day_trade_ratio_5d 1、chip_fi_holding_pct 1

## 關卡：排名 20 日

- 研究期 IC 均值 0.1343 大於 0 且 NW t 13.66 大於 2：通過
- 贏 Ridge 的研究摺 27／36 摺 達 60%：通過
- Ridge 研究期 IC 均值 0.1193，NW t 11.35，2175 天
- 重要度變動只列研究摺，調參摺不計
