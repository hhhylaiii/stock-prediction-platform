# Phase 2 驗收報告

狀態：觀察中。驗收條件是連續 5 個交易日每日排程無人工介入（規格 §9 Phase 2、plan PR 2.6）；Phase 3 第一支 PR 開分支前填完。

## 驗收條件

| 條件 | 結果 | 證據 |
| --- | --- | --- |
| 連續 5 個交易日無人工介入 | 待填 | 下表 |
| 品質檢查能攔下人為注入的錯誤資料 | 通過 | PR #34 的注入錯誤測試（`tests/etl/test_quality.py`） |

## 每日紀錄

每個交易日隔天早上填：`ready_for_research` 的結果、當天 `data_quality_log` 最高的 severity、有沒有手動操作。

```bash
docker compose exec worker python manage.py shell -c \
  "from datetime import date; from apps.etl.tasks import ready_for_research; print(ready_for_research(date(2026, 11, 2)))"
```

| 交易日 | `ready_for_research` | 最高 severity | 失敗或重試用完的 task | 人工介入 |
| --- | --- | --- | --- | --- |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

## 備註

- 中斷（Mac 睡著、Docker 停掉）後隔天由 15:10 的 catchup 補齊，算無人工介入；需要手動跑指令才算介入。
