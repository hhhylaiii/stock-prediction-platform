# Pages and API fields

What the three views of the UI (see the view table in the [README](../README.md)) show when a model is not live, went live on a released verdict or was swapped in by force, and the API fields behind each note.

## When no model is live

- When no model is live yet, the prediction blocks show 「預測資料尚未產出」 ("predictions have not been produced yet"), while the candlestick chart, ETL records and quality checks are shown as usual.
- A model goes live only after its holdout verdict passes, or after the user releases a verdict that fell short: the empty ranking says so, and when the ranking is live but another model has no predictions for the day, a note above the table names the columns left blank, and the stock page says the same under its forecast table for each horizon the stock has a prediction for, there also covering the volatility model.

## Released verdicts and forced swaps

- A model built on a released verdict shows the verdict, what fell short and the user's reason under the rows it covers on the models page (`holdout_release` in `/api/models`, without the report path), and the model-output checks keep each scope's latest `model_holdout_release` alert pinned.
- The today and stock pages name every model shown there that went live on a released verdict, or that was swapped in by force past promotion gates it failed (`released_verdict` and `forced_gates` on each entry of `models` in `/api/predictions` and `/api/stocks/<code>/predictions`), and point to the models page, where the forced swap's day, reason and failed gates sit under that model (`forced_promotion` in `/api/models`).

## The 20-day interval and the dividend basis

- The note beside the 20-day interval says how often the 80% interval held: first on the holdout, which ran once after the design was frozen, then over the research period.
- When the interval model is not live and the band is instead estimated from the stock's last 20 days of volatility around the latest close, the note says so and gives how often that estimate held over the same two periods, naming the quarters where it fell short.
- Both pages also say that predicted prices and price bands are on a dividend-adjusted basis: for a stock that goes ex-dividend within the horizon, the market price will be lower by about the cash dividend even when the predicted return is exactly right (ex-rights lowers it by the allotment ratio), and the pages do not deduct it.

## Horizons

- Each horizon's ranking is dated on its own, so a horizon can be without a live ranking model: `horizons` in `/api/status` dates each horizon's ranking (`prediction_date` is null when no ranking model of that horizon is live, `ready` says whether it has caught up with the latest prices), `prediction_date` is the latest of them and `predictions_ready` is true when any horizon is ready.
- The today and stock pages open on 20 days and fall back to the horizon that has a ranking when 20 days has none, unless the reader has already picked a horizon; the empty horizon says that no ranking model is live for it, which horizon has one, and points to the models page; once any horizon has caught up, the status bar names each other horizon that is not live or whose predictions are behind (when none has, it only says the predictions are not out yet); and the stock page names each horizon's ranking model when the two are not one build.

## The models page

- On the models page the monthly rank IC chart and the rank score column name the holdout when their figures come from it instead of a research run (`meta.source` in `/api/models/ic-monthly`, `metric_source` in `/api/models`); the chart falls back to the latest rank holdout when no research run of a live ranking model has monthly figures, which is the case before any ranking model is live.
- When `horizon` is not given, `/api/models/ic-monthly` charts 20 days unless only the 5-day ranking is live, and the chart's caption says which horizon it draws.
- A horizon with no live model that the latest promotion of its scope judged and held back is explained under the model table with the gates it failed (`meta.not_live` in `/api/models`; a held horizon whose older model still serves is not listed), and the promotion record names what came in and what was held when only part of a build was swapped in (`promoted_slots` and `held_slots` in the `model_promotion` row).

## The status bar

- The status bar at the top reads `/api/status`: when the day's quality checks report an error, no predictions are produced for that day; `quality.warnings` is the number of checks with warnings and `quality.warning_rows` is the number of rows with warnings.
- The count of degraded models in the status bar reads every model on its own latest served day.
