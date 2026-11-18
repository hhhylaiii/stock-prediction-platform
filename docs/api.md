# Pages and API fields

Which notes the three views of the UI (see the view table in the [README](../README.md)) show and when, and the API fields behind each note.

## When a model is not live

- When no model is live yet, the prediction blocks show 「預測資料尚未產出」 ("predictions have not been produced yet"), while the candlestick chart, ETL records and quality checks are shown as usual.
- A model goes live only after its holdout verdict passes, or after the user releases a verdict that fell short: the empty ranking says so, and when the ranking is live but another model has no predictions for the day, a note above the table names the columns left blank, and the stock page says the same under its forecast table for each horizon the stock has a prediction for, there also covering the volatility model.

## Released verdicts and forced swaps

- A model built on a released verdict shows the verdict, what fell short and the user's reason under the rows it covers on the models page (`holdout_release` in `/api/models`, without the report path), and the model-output checks keep each scope's latest `model_holdout_release` alert pinned.
- The today and stock pages name every model shown there that went live on a released verdict, or that was swapped in by force past promotion gates it failed (`released_verdict` and `forced_gates` on each entry of `models` in `/api/predictions` and `/api/stocks/<code>/predictions`), and point to the models page, where the forced swap's day, reason and failed gates sit under that model (`forced_promotion` in `/api/models`).

## The 20-day interval and the dividend basis

- The note beside the 20-day interval says how often the 80% interval held: first on the holdout, which ran once after the design was frozen, then over the research period.
- When the interval model is not live and the band is instead estimated from the stock's last 20 days of volatility around the latest close, the note says so and gives how often that estimate held over the same two periods, naming the quarters where it fell short.
- The today and stock pages also say that predicted prices and price bands are on a dividend-adjusted basis: for a stock that goes ex-dividend within the horizon, the market price will be lower by about the cash dividend even when the predicted return is exactly right (ex-rights lowers it by the allotment ratio), and the pages do not deduct it.

## Horizons

- Each horizon's ranking is dated on its own, so a horizon can be without a live ranking model: `horizons` in `/api/status` dates each horizon's ranking (`prediction_date` is null when no ranking model of that horizon is live, `ready` says whether it has caught up with the latest prices), `prediction_date` is the latest of them and `predictions_ready` is true when any horizon is ready.
- The today and stock pages open on 20 days and fall back to the horizon that has a ranking when 20 days has none, unless the reader has already picked a horizon; the empty horizon says that no ranking model is live for it, which horizon has one, and points to the models page; once any horizon has caught up, the status bar names each other horizon that is not live or whose predictions are behind (when none has, it only says the predictions are not out yet); and the stock page names each horizon's ranking model when the two are not one build.

## Sorting the today table

- The close, predicted return, predicted price, probability and previous-rank headers on the today page sort the table: the first click puts the highest first (for the previous rank, rank 1), the next click reverses it, blanks stay last either way and equal values keep the ranking model's order. While sorted, the first column shows the position in that order, a 「模型名次」 column beside it shows the ranking model's rank, and clicking that header returns to the ranking.
- `/api/predictions` sorts the whole market after the market filter and the search, before paging, by `ordering`: `rank` by default, or `close`, `ret_pred`, `price_pred`, `prob_up_5pct` or `prev_rank`, each with a leading `-` for descending. Each value is worked out as the row shows it; `meta.orderable` lists the columns that have something to sort by that day (a header not listed is disabled), and `meta.ordering` is the order applied, which is `rank` when the asked column is not orderable. Watch-only sorts the watched stocks on the page by the same rules.
- While the return model serves, a note under the controls says that the rank and the predicted return come from two models estimated apart, so a stock ranked near the top can still have a negative predicted return; that the return models are judged on their prediction error and were never checked for ordering, so an order by predicted return is not validated the way the rank is; and names the headers that can sort that day.

## The holding rule's list

- On 5 days the today page leads with the holding rule's list (H70/N30, a research record and not investment advice) from `/api/holding`. The endpoint serves only the main book `H70N30-C0`; the candidate books and the books paired with them never go through the API or the pages (`docs/reports/phase4/holding-shadow-prereg.md` §7, item 3). On 20 days the list and the rule column are not shown and a note says the rule uses only the 5-day ranking.
- `list` is the latest list: its signal and trade days, `status` (`listed`, or `no_prediction` with no actions), `backfilled`, the counts, and `actions` with each stock's code, name, `action` (`sell`, `keep`, `buy`, `backup`), `priority` and `pct`, in sell, keep, buy, backup order. `fill` is the latest fill: the book's `value` after that day's close (costs paid, starting from cash 1.0), `cash`, `held`, `paid`, `traded`, `backfilled`, and `stuck`, the sells that could not be filled. The rule column of the table reads both, so a stock can read 「賣出、賣不掉」. `meta.predictions_date` is the 5-day ranking's day; when the list is older, the page says this morning's book step has not written yet.
- `since.points` compares the book with the equal-weight universe and TAIEX_TR from the close of the first fill, where all three are 1.0: the book is its value over the first fill's value, the universe chains each later fill's `universe_return`, and the index is that day's TAIEX_TR close over the first fill's, read from `index_daily` at request time. The stored `index_return` of the fill rows is not used, so a close that arrives late fills its day in later, and a correction to `index_daily` changes the line; the fill rows stay the record. A missing close blanks only its own day (the whole line when it is the first fill's); a missing universe return blanks the universe from that day on.
- When the list a fill follows has an empty universe (no predictions that day), the fill records a universe return of 0 by the preregistration (§5, item 1: no universe row on d−1), so the universe line stays flat that day; `since.no_prediction_days` counts exactly those fills after the first one, by the list's empty `universe` rather than its `no_prediction` status (a day whose universe stocks all lack a score is `no_prediction` but still has a real universe return), and the page discloses them when there is at least one.
- Each fill must sign on the day the fill before it traded; when one is missing the comparison is refused (`since.error` is `ledger_invalid`, `points` is empty, the reason is logged) and the page says the book's data is off while the list and the latest fill still show.

## The models page

- On the models page the monthly rank IC chart and the rank score column name the holdout when their figures come from it instead of a research run (`meta.source` in `/api/models/ic-monthly`, `metric_source` in `/api/models`); the chart falls back to the latest rank holdout when no research run of a live ranking model has monthly figures, which is the case before any ranking model is live.
- When `horizon` is not given, `/api/models/ic-monthly` charts 20 days unless only the 5-day ranking is live, and the chart's caption says which horizon it draws.
- A horizon with no live model that the latest promotion of its scope judged and held back is explained under the model table with the gates it failed (`meta.not_live` in `/api/models`; a held horizon whose older model still serves is not listed), and the promotion record names what came in and what was held when only part of a build was swapped in (`promoted_slots` and `held_slots` in the `model_promotion` row).
- A scope whose latest holdout claim was never released and none of whose models serve is listed in `meta.unreleased` of `/api/models`, `/api/predictions` and `/api/stocks/{id}/predictions` (its `scope`, `run_key` and `slots`; only the interval scope, the 20-day quantile, volatility and classification models, is checked). For the claim `interval-holdout-20261112` the models page says under the table that those three models are not live, with the verdict and the user's decision of 2026-11-12 (`docs/SPEC.md:313`); the today page's 20-day view (watch-only included) and the stock page add a shorter form, which on the stock page ends with a link to the models page. On the stock page, for either horizon, the hint says 「N 日排名目前沒有上線中的模型」 when no ranking model serves that horizon instead of 「N 日預測尚未產出」, and the rank figure says 「排名未上線」 when another horizon has a ranking model.

## The status bar

- The status bar at the top reads `/api/status`: when the day's quality checks report an error, no predictions are produced for that day; `quality.warnings` is the number of checks with warnings and `quality.warning_rows` is the number of rows with warnings.
- The count of degraded models in the status bar reads every model on its own latest served day.
