(function () {
  'use strict';

  const api = window.TwpApi;
  const STOCK_CODE = /^[0-9A-Za-z]{1,10}$/;
  const PRICE_DAYS = 60;
  const HISTORY_HORIZON = 20;
  const HISTORY_ROWS = 8;
  const QUALITY_DAYS = 30;
  const SOURCES = { twse: '證交所', tpex: '櫃買中心', finmind: 'FinMind', tdcc: '集保', yahoo: 'Yahoo', datagov: 'data.gov.tw', mops: 'MOPS', stock_master: '股票清單', trading_calendar: '交易日曆', twse_tpex: '證交所／櫃買中心' };
  const RUN_STATUS = { success: ['完成', 'in'], failed: ['失敗', 'warn'], running: ['執行中', 'idle'] };
  const CHECKS = {
    row_count: '各表當日列數',
    big_move: '普通股漲跌幅超過 10% 且無除權息紀錄',
    missing_values: 'standard 表普通股當日缺值',
    trading_calendar: '所有日期都在交易日曆內',
    index_freshness: '市場指數的最新收盤',
    etl_runs: '資料集與排程工作都已更新',
    stock_master: '股票清單異動',
    us_close: '美股收盤',
    adj_factor_changed: '還原因子更新',
    statement_filed_late: '財報逾期申報',
    statement_filed_later: '更正財報逾期申報',
    statement_restated: '財報重編',
    statement_overwritten: '財報數字以 MOPS 覆寫',
    statement_margin_cleared: '清除累計期間的利益率',
    statement_viewer_mismatch: '下市公司財報由 XBRL 檢視頁補',
    revenue_company_mismatch: '月營收由 MOPS 單一公司查詢補缺',
    fi_holding_zero_replaced: '外資持股 0 值覆寫',
    market_history_rebuilt: '市場歷史表重建',
    statement_source_mismatch: '財報兩來源數字不同',
    statement_type_mismatch: '財報種類不同',
    revenue_history_mismatch: '月營收由 MOPS 歷史彙總頁補缺',
    revenue_source_mismatch: '月營收兩來源數字不同',
    statement_history_mismatch: '財報由 MOPS 歷史彙總頁補缺',
    dividend_source_mismatch: '股利兩來源不同',
    dividend_notice: '除權除息預告表',
    ex_date_unannounced: '除息日沒有公告日',
    exchange_fill: '交易所逐日表補缺',
    industry_fill: '交易所產業補缺',
    output_width_ratio: '區間寬度比異常',
    output_odds_vs_q50: '機率與中位數矛盾',
    monitor_unscored: '到期預測缺 label',
    monitor_feature_coverage: '線上特徵覆蓋率',
    monitor_ex_date_calendar: '除息日超出交易日曆',
    model_alert: '模型告警',
    model_promotion: '換模',
    model_holdout_release: 'holdout 放行後建模',
    research_evening: '研究晚間排程',
    research_morning: '研究早盤排程',
    research_retrain: '每月重訓',
    research_promote: '自動換模',
    research_rebuild: '研究資料重建'
  };
  const ALERT_REASONS = { below_threshold: '低於門檻', outside_range: '超出範圍', above_baseline: '輸給 baseline', missing_reference: '缺參考值' };
  const ALERT_METRICS = { ic: 'IC', coverage: '覆蓋率', brier: 'Brier' };
  const GATES = { agreement: '與現役模型的預測相關', gain_overlap: '與現役模型的重要特徵重疊', shadow_oos: '影子模型最近 6 個月 OOS', data_quality: '訓練窗內的資料品質' };
  const SAMPLE_UNITS = { stocks: ' 檔', features: ' 個特徵', days: ' 天' };
  const OBJECTIVES = { rank: '排名', regression: '絕對報酬', quantile: '報酬區間', volatility: '波動率', classification: '跑贏大盤機率' };
  const SCOPES = { rank: '排名', return: '絕對報酬', interval: '區間與機率' };
  const MODEL_REQUIRES = '模型要在 holdout 判定達標之後才會上線，判定未達標或明顯退化時要經使用者放行';
  const SEVERITY = { info: ['通過', 'in'], warning: ['警告', 'warn'], error: ['錯誤', 'warn'] };

  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtPrice = n => n >= 1000 ? Math.round(n).toLocaleString('en-US') : n >= 100 ? n.toFixed(1) : n.toFixed(2);
  const fmtInt = n => Number(n).toLocaleString('en-US');
  const sgn = (x, d) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d === undefined ? 1 : d) + '%';
  const signed = (x, digits) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(digits);
  const colorOf = x => x > 0 ? 'var(--up)' : x < 0 ? 'var(--down)' : 'var(--ink-2)';
  const f1 = v => v.toFixed(1);
  const md = iso => Number(iso.slice(5, 7)) + '/' + Number(iso.slice(8, 10));
  const hhmm = iso => iso ? iso.slice(11, 16) : '—';
  const errorBox = err => '<div class="error" role="alert">' + esc(err && err.message ? err.message : '發生錯誤。') + '</div>';

  function ma(values, n) {
    return values.map((v, i) => {
      if (i < n - 1) return null;
      const win = values.slice(i - n + 1, i + 1);
      return win.some(x => x === null) ? null : win.reduce((a, b) => a + b, 0) / n;
    });
  }

  function niceTicks(lo, hi, n) {
    const span = hi - lo, raw = span / (n - 1), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const steps = [1, 2, 2.5, 5, 10];
    let step = mag * 10;
    for (const k of steps) { if (raw <= k * mag) { step = k * mag; break; } }
    const start = Math.ceil(lo / step) * step, out = [];
    for (let v = start; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
    return out;
  }

  const state = { h: 20, mkt: 'all', q: '', watchOnly: false, watch: new Set(), stock: '2330', chart: 'candle', stockToken: 0, featureToken: 0, todayToken: 0, page: 1, watchCache: null };
  try { const c = localStorage.getItem('twp-chart'); if (c === 'line' || c === 'candle') state.chart = c; } catch (e) { state.chart = 'candle'; }
  try { const w = JSON.parse(localStorage.getItem('twp-watch') || '[]'); if (Array.isArray(w)) w.filter(c => STOCK_CODE.test(c)).forEach(c => state.watch.add(c)); } catch (e) { state.watch.clear(); }
  function saveWatch() { try { localStorage.setItem('twp-watch', JSON.stringify([...state.watch])); } catch (e) { return; } }
  try { const s = localStorage.getItem('twp-stock'); if (s && STOCK_CODE.test(s)) state.stock = s; } catch (e) { state.stock = '2330'; }
  function saveStock() { try { localStorage.setItem('twp-stock', state.stock); } catch (e) { return; } }

  const statusEl = document.getElementById('status');
  const summaryOf = day => ({ passed: day.passed, worst: day.worst, missing_checks: day.missing_checks, warnings: new Set(day.checks.filter(c => c.severity === 'warning').map(c => c.check_name)).size });
  const sizeOf = v => Array.isArray(v) ? v.length : Number(v || 0);
  const keyCount = v => v && typeof v === 'object' ? Object.keys(v).length : 0;

  function verdictOf(day) {
    const warnings = day.warnings;
    if (day.worst === 'error') return { label: '未通過', cls: 'warn', text: '有檢查未通過，當日不產出預測。' };
    if (!day.worst) return { label: '尚未執行', cls: 'idle', text: '檢查尚未執行。' };
    if (day.missing_checks.length) return { label: '尚未完成', cls: 'idle', text: '已執行的資料檢查沒有錯誤，還有 ' + day.missing_checks.length + ' 項尚未執行。' };
    if (day.passed === false) return { label: '未通過', cls: 'warn', text: '模型輸出檢查有錯誤。' };
    if (warnings) return { label: '通過（' + warnings + ' 項警告）', cls: 'warn', text: '檢查通過，但有 ' + warnings + ' 項警告。' };
    return { label: '通過', cls: 'in', text: '全部檢查通過。' };
  }

  async function renderStatus() {
    try {
      const { data } = await api.get('status');
      const verdict = verdictOf(data.quality);
      const cls = verdict.cls === 'in' ? '' : ' ' + verdict.cls;
      const prices = data.price_date ? esc(data.price_date) + ' 收盤資料' : '尚無收盤資料';
      const parts = ['<span>' + prices + '</span>', '<span><i class="dot' + cls + '"></i>品質檢查' + esc(verdict.label) + '</span>'];
      if (data.updated_at) parts.push('<span>' + esc(hhmm(data.updated_at)) + ' 更新</span>');
      if (!data.predictions_ready) parts.push('<span><i class="dot idle"></i>預測尚未產出</span>');
      if (data.degraded.length) parts.push('<span><i class="dot warn"></i>' + data.degraded.length + ' 個模型降級中</span>');
      statusEl.innerHTML = parts.join('');
    } catch (err) {
      statusEl.innerHTML = '<span><i class="dot warn"></i>' + esc(err.message) + '</span>';
    }
  }

  const watchOnlyBtn = document.getElementById('watch-only');
  const qEl = document.getElementById('q');
  const gotoEl = document.getElementById('today-goto');

  const rowsEl = document.getElementById('rows');
  const countEl = document.getElementById('count');
  const todayMetaEl = document.getElementById('today-meta');
  const bandNoteEl = document.getElementById('band-note');
  const fallbackNoteEl = document.getElementById('fallback-note');
  const modelNoteEl = document.getElementById('model-note');
  const prevBtn = document.getElementById('prev-page');
  const nextBtn = document.getElementById('next-page');
  const RANGE_SPAN = { 5: [-8, 12], 20: [-16, 24] };
  const SOURCE_NOTE = { volatility: '波動率區間', tech_vol_20d: '近 20 日波動估計' };
  const PROB_HEAD = '跑贏大盤 5 個百分點機率';
  const PROB_TIP = '20 日報酬跑贏大盤（加權股價報酬指數，含息）超過 5 個百分點的機率';
  const BAND_NOTE = '20 日價格區間是 80% 區間。設計定案後只跑一次的 holdout（2025 年起，到 2026-11-11 已到期的預測）裡，實際報酬落在區間內的比例是 75.0%；7 個完整的季度在 59% 到 84% 之間，偏低的是 2025 年第 1 季 59.4%、2026 年第 2 季 69.6%，最後一段只有 2026 年 10 月 1 日至 12 日的 7 個交易日（樣本很少）、40.4%。' +
    '定案前的研究期 2016–2024 年是 77.7%，大盤急跌的季度曾明顯偏低：2020 年第 1 季 49%、2022 年第 2 季 66%、第 3 季 68%。';
  const FALLBACK_NOTE = '20 日價格區間目前不是模型的預測：報酬區間模型沒有上線，區間是用近 20 日的波動估的，以最新收盤價為中心、上下各 1.28 個標準差（換算成 20 日）。' +
    '這個估法的 80% 區間在 holdout（2025 年起，到 2026-11-11 已到期的預測）裡，實際報酬落在區間內的比例是 78.7%；7 個完整的季度在 62.8% 到 89.4% 之間，偏低的是 2025 年第 1 季 62.8%，最後一段只有 2026 年 10 月 1 日至 12 日的 7 個交易日（樣本很少）、43.4%。' +
    '更早的研究期 2016–2024 年是 79.2%，大盤急跌的季度曾明顯偏低：2020 年第 1 季 55%。';
  const MODEL_NAMES = { lightgbm: 'LightGBM', ridge: 'Ridge', xgboost: 'XGBoost' };
  const SEARCH_MAX = 32;
  const pctOf = (x, digits) => x === null || x === undefined ? '—' : sgn(x * 100, digits);
  const priceOf = x => x === null || x === undefined ? '—' : fmtPrice(x);
  const onFallbackBand = row => Boolean(row) && row.interval_source === 'tech_vol_20d';

  function controls() {
    document.querySelectorAll('[data-h]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.h) === state.h)));
    document.querySelectorAll('[data-mkt]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mkt === state.mkt)));
    document.getElementById('ret-head').textContent = state.h + ' 日預測報酬';
    document.getElementById('prob-head').textContent = PROB_HEAD + (state.h === 20 ? '' : '（僅 20 日）');
    document.getElementById('band-head').textContent = '80% 預測區間' + (state.h === 20 ? '（價格）' : '（僅 20 日）');
    watchOnlyBtn.hidden = state.watch.size === 0;
    watchOnlyBtn.textContent = '只看追蹤（' + state.watch.size + '）';
    watchOnlyBtn.setAttribute('aria-pressed', String(watchingOnly()));
  }
  const watchingOnly = () => state.watchOnly && state.watch.size > 0;

  function bandHtml(row) {
    const iv = state.h === 20 ? row.interval : null;
    if (!iv) return '<div class="range"></div><div class="faint small">—</div>';
    const [lo, hi] = RANGE_SPAN[state.h];
    const at = x => Math.max(0, Math.min(100, (x * 100 - lo) / (hi - lo) * 100));
    const note = SOURCE_NOTE[row.interval_source];
    const median = row.ret_pred === null ? '' : '<div class="med" style="left:' + at(row.ret_pred) + '%"></div>';
    return '<div class="range"' + (note ? ' title="' + esc(note) + '"' : '') + '><div class="zero" style="left:' + at(0) + '%"></div>' +
      '<div class="band" style="left:' + at(iv.ret_low) + '%;width:' + (at(iv.ret_high) - at(iv.ret_low)) + '%"></div>' + median + '</div>' +
      '<div class="muted small">' + priceOf(iv.low) + ' – ' + priceOf(iv.high) + (note ? '<span class="faint">（' + esc(note) + '）</span>' : '') + '</div>';
  }

  function todayRowHtml(row) {
    return '<a class="trow rank-cols" href="#stock-' + encodeURIComponent(row.stock_id) + '">' +
      '<div class="num rank">' + (row.rank === null ? '—' : row.rank) + '</div>' +
      '<div class="stockcell"><span class="muted">' + esc(row.stock_id) + '</span><span class="name">' + esc(row.name || '—') + '</span>' +
      (row.in_universe ? '' : '<span class="faint small">訓練範圍外</span>') + '</div>' +
      '<div class="faint small market">' + esc(row.market_label || '—') + '</div>' +
      '<div class="right">' + priceOf(row.close) + '</div>' +
      '<div class="right num" style="color:' + colorOf(row.ret_pred || 0) + '">' + pctOf(row.ret_pred, 1) + '</div>' +
      '<div class="right">' + priceOf(row.price_pred) + '</div>' +
      bandHtml(row) +
      '<div class="right prob">' + (row.prob_up_5pct === null ? '—' : Math.round(row.prob_up_5pct * 100) + '%') + '</div>' +
      '<div class="right faint small">' + (row.prev_rank === null ? '—' : '昨 ' + row.prev_rank) + '</div></a>';
  }

  const shownModels = (models, horizon) => (models || []).filter(model => horizon === 20 || ['rank', 'return'].includes(model.role));

  function caveatText(models) {
    const order = Object.keys(OBJECTIVES);
    const sorted = [...models].sort((a, b) => order.indexOf(a.objective) - order.indexOf(b.objective) || b.horizon - a.horizon);
    const grouped = (keyOf, clause) => {
      const groups = new Map();
      sorted.forEach(model => {
        const key = keyOf(model), name = (OBJECTIVES[model.objective] || model.objective) + ' ' + model.horizon + ' 日';
        const names = groups.get(key) || [];
        if (key && !names.includes(name)) groups.set(key, [...names, name]);
      });
      return [...groups].map(([key, names]) => names.join('、') + clause(key));
    };
    return [
      ...grouped(model => model.released_verdict, verdict => '模型的 holdout 判定' + verdict + '，經使用者放行上線'),
      ...grouped(model => (model.forced_gates || []).map(gate => GATES[gate] || gate).join('、'), gates => '模型的換模門檻沒過（' + gates + '），經使用者強制換上')
    ].join('；');
  }

  function todayMetaText(meta) {
    if (!meta.date) return '';
    const [y, m, d] = meta.date.split('-').map(Number);
    const parts = [y + ' 年 ' + m + ' 月 ' + d + ' 日收盤後產出，涵蓋 ' + fmtInt(meta.universe) + ' 檔訓練範圍內的普通股'];
    const rank = (meta.models || []).find(model => model.role === 'rank');
    if (rank) parts.push('排名模型為 ' + (MODEL_NAMES[rank.model_type] || rank.model_type) + ' ' + rank.version);
    const monitor = meta.monitor || {};
    if (monitor.rank_ic !== null && monitor.rank_ic !== undefined) parts.push('近 ' + monitor.rank_ic_days + ' 個評估日 rank IC ' + monitor.rank_ic.toFixed(3));
    if (monitor.coverage !== null && monitor.coverage !== undefined) parts.push('80% 區間的實際覆蓋率 ' + Math.round(monitor.coverage * 100) + '%');
    const degraded = meta.degraded || [];
    const degradedRuns = new Set((meta.models || []).filter(model => degraded.includes(model.role)).map(model => model.run_id)).size;
    if (degraded.length) parts.push(degradedRuns + ' 個模型降級中，結果僅供參考' + (meta.interval_alert ? '，區間改用波動率區間' : ''));
    const caveats = caveatText(shownModels(meta.models, state.h));
    return parts.join('，') + '。' + (caveats ? caveats + '。原因見模型頁。' : '');
  }

  function watchedStocks() {
    const key = [...state.watch].join(',');
    if (state.watchCache && state.watchCache.key === key) return state.watchCache.promise;
    const promise = Promise.allSettled([...state.watch].map(code => api.get('stocks/' + code + '/predictions'))).then(settled => {
      const failed = settled.find(result => result.status === 'rejected' && result.reason.status !== 404);
      if (failed) throw failed.reason;
      return settled.filter(result => result.status === 'fulfilled').map(result => result.value.data);
    });
    state.watchCache = { key: key, promise: promise };
    promise.catch(() => { if (state.watchCache && state.watchCache.promise === promise) state.watchCache = null; });
    return promise;
  }

  async function watchedToday() {
    const entries = (await watchedStocks()).map(stock => stock.horizons.find(entry => entry.horizon === state.h)).filter(Boolean);
    const q = state.q.trim().toLowerCase();
    const rows = entries.map(entry => entry.prediction)
      .filter(row => row && (state.mkt === 'all' || row.market === state.mkt) && (!q || row.stock_id.toLowerCase().startsWith(q) || (row.name || '').toLowerCase().includes(q)))
      .sort((a, b) => (a.rank === null) - (b.rank === null) || a.rank - b.rank);
    const latest = entries.filter(entry => entry.date).sort((a, b) => (a.date < b.date) - (a.date > b.date))[0];
    return { rows: rows, meta: latest ? { date: latest.date, universe: latest.universe, models: latest.models, degraded: latest.degraded, interval_alert: latest.interval_alert } : {} };
  }

  function emptyHtml(meta) {
    if (state.q.trim()) return '<div class="empty">今日預測裡沒有符合「' + esc(state.q.trim()) + '」的股票。試試代號或公司名稱的一部分。</div>';
    if (watchingOnly()) return '<div class="empty">追蹤的股票沒有這一期的預測。</div>';
    if (!meta.date) return '<div class="empty" id="today-empty">預測資料尚未產出。排名' + MODEL_REQUIRES + '；上線後，這裡會列出全市場的排名。</div>';
    return '<div class="empty">沒有符合條件的股票。</div>';
  }

  function lackingText(models, horizon, own) {
    const twenty = horizon === 20;
    const serving = new Set((models || []).map(model => model.role));
    const estimated = twenty && !serving.has('interval') && (!own || Boolean(own.interval));
    const lacking = [], blank = [];
    if (!serving.has('return')) { lacking.push(OBJECTIVES[twenty ? 'quantile' : 'regression']); blank.push('預測報酬', '預測價'); }
    if (twenty && own && !serving.has('interval') && !own.interval) blank.push('80% 區間');
    if (twenty && own && !serving.has('volatility')) { lacking.push(OBJECTIVES.volatility); blank.push('預測 20 日波動'); }
    if (twenty && !serving.has('probability')) { lacking.push(OBJECTIVES.classification); blank.push('跑贏大盤機率'); }
    if (!lacking.length) return '';
    return '沒有' + lacking.join('、') + '模型的預測：' + blank.join('、') + '留空' + (estimated ? '，價格區間改用波動估計' : '') + '。';
  }

  function modelNoteText(meta, shown) {
    if (!meta.date || !shown) return '';
    const lacking = lackingText(meta.models, state.h);
    return lacking ? '這一期' + lacking + MODEL_REQUIRES + '；上線中的模型列在模型頁。' : '';
  }

  function showNotes(meta, rows) {
    const serving = new Set((meta.models || []).map(model => model.role));
    bandNoteEl.hidden = !serving.has('interval') || Boolean(meta.interval_alert);
    fallbackNoteEl.hidden = state.h !== 20 || serving.has('interval') || serving.has('volatility') || !rows.some(onFallbackBand);
    modelNoteEl.textContent = modelNoteText(meta, rows.length);
    modelNoteEl.hidden = !modelNoteEl.textContent;
  }

  function countText(meta, shown) {
    if (!meta.total) return '';
    if (state.mkt !== 'all' || state.q.trim()) return '符合條件 ' + fmtInt(meta.total) + ' 檔';
    const first = (meta.page - 1) * meta.page_size + 1;
    const span = fmtInt(first) + '–' + fmtInt(first + shown - 1);
    if (meta.total === meta.universe) return '第 ' + span + ' 名，共 ' + fmtInt(meta.total) + ' 檔';
    return '第 ' + span + ' 列，共 ' + fmtInt(meta.total) + ' 檔，其中 ' + fmtInt(meta.universe) + ' 檔在訓練範圍內';
  }

  async function fetchToday() {
    if (watchingOnly()) return watchedToday();
    const params = { horizon: state.h, page: state.page };
    if (state.mkt !== 'all') params.market = state.mkt;
    if (state.q.trim()) params.q = state.q.trim().slice(0, SEARCH_MAX);
    const { data, meta } = await api.get('predictions', params);
    return { rows: data, meta: meta };
  }

  const findStocks = async text => (await api.get('stocks', { q: text.slice(0, SEARCH_MAX) })).data;

  const goto = { text: '', ticket: 0 };

  async function loadGoto() {
    const text = state.q.trim();
    if (text === goto.text) return;
    goto.text = text;
    const mine = ++goto.ticket;
    gotoEl.hidden = true;
    gotoEl.innerHTML = '';
    if (!STOCK_CODE.test(text)) return;
    let hits;
    try { hits = await findStocks(text); } catch (err) { hits = []; if (mine === goto.ticket) goto.text = null; }
    const hit = hits.find(row => row.stock_id.toLowerCase() === text.toLowerCase());
    if (mine !== goto.ticket || !hit) return;
    gotoEl.innerHTML = '<a href="#stock-' + encodeURIComponent(hit.stock_id) + '">查看 ' + esc(hit.stock_id) + ' ' + esc(hit.name) + ' 個股頁</a>';
    gotoEl.hidden = false;
  }

  async function loadToday() {
    const token = ++state.todayToken;
    controls();
    loadGoto();
    prevBtn.disabled = nextBtn.disabled = true;
    let result;
    try {
      result = await fetchToday();
    } catch (err) {
      if (token !== state.todayToken) return;
      rowsEl.innerHTML = errorBox(err);
      countEl.textContent = todayMetaEl.textContent = '';
      showNotes({}, []);
      return;
    }
    if (token !== state.todayToken) return;
    const { rows, meta } = result;
    rowsEl.innerHTML = rows.map(todayRowHtml).join('') || emptyHtml(meta);
    todayMetaEl.textContent = todayMetaText(meta);
    showNotes(meta, rows);
    if (watchingOnly()) {
      countEl.textContent = '追蹤中的 ' + rows.length + ' 檔';
      return;
    }
    state.page = meta.page || 1;
    countEl.textContent = countText(meta, rows.length);
    prevBtn.disabled = state.page <= 1;
    nextBtn.disabled = state.page >= (meta.pages || 1);
  }

  function reloadToday() { state.page = 1; loadToday(); }
  let searchTimer = null;
  document.querySelectorAll('[data-h]').forEach(b => b.addEventListener('click', () => { state.h = Number(b.dataset.h); reloadToday(); }));
  document.querySelectorAll('[data-mkt]').forEach(b => b.addEventListener('click', () => { state.mkt = b.dataset.mkt; reloadToday(); }));
  qEl.addEventListener('input', () => { state.q = qEl.value; clearTimeout(searchTimer); searchTimer = setTimeout(reloadToday, 250); });
  watchOnlyBtn.addEventListener('click', () => { state.watchOnly = !state.watchOnly; state.watchCache = null; reloadToday(); });
  prevBtn.addEventListener('click', () => { state.page -= 1; loadToday(); });
  nextBtn.addEventListener('click', () => { state.page += 1; loadToday(); });

  const stockEl = document.getElementById('view-stock');
  const backLink = '<a class="back" href="#today"><svg aria-hidden="true" width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2.5L4.5 7L9 11.5"></path></svg>今日預測</a>';
  const waiting = '<p class="placeholder">預測資料尚未產出。</p>';
  const stockTop = () => '<div class="stock-top">' + backLink + '<form class="search stock-go" id="stock-go" role="search">' +
    '<input type="search" id="stock-q" placeholder="前往個股：代號或名稱" aria-label="前往個股：代號或名稱" autocomplete="off" maxlength="' + SEARCH_MAX + '">' +
    '<div class="stock-hits" id="stock-hits" hidden></div></form></div>';
  const hitHtml = hit => '<a href="#stock-' + encodeURIComponent(hit.stock_id) + '"><span class="code">' + esc(hit.stock_id) + '</span> ' + esc(hit.name) +
    ' <span class="faint small">' + esc(hit.market_label) + (hit.delisted ? ' 已下市' : '') + '</span></a>';

  function bindStockSearch() {
    const form = document.getElementById('stock-go'), input = document.getElementById('stock-q'), list = document.getElementById('stock-hits');
    let timer = null, ticket = 0, shown = { text: null, hits: [] };
    const show = html => { list.innerHTML = html; list.hidden = !html; };
    const close = () => { clearTimeout(timer); ticket += 1; show(''); };
    const lookup = async text => {
      const mine = ++ticket;
      try {
        const hits = await findStocks(text);
        if (mine !== ticket) return null;
        shown = { text: text, hits: hits };
        show(hits.length ? hits.map(hitHtml).join('') : '<div class="empty small">找不到這檔股票。</div>');
        return hits;
      } catch (err) {
        if (mine === ticket) show(errorBox(err));
        return null;
      }
    };
    input.addEventListener('input', () => {
      clearTimeout(timer);
      const text = input.value.trim();
      if (!text) { ticket += 1; show(''); return; }
      timer = setTimeout(() => lookup(text), 250);
    });
    form.addEventListener('submit', async e => {
      e.preventDefault();
      clearTimeout(timer);
      const text = input.value.trim();
      if (!text) return;
      const hits = shown.text === text ? shown.hits : await lookup(text);
      if (!hits || !hits.length) return;
      close();
      location.hash = '#stock-' + encodeURIComponent(hits[0].stock_id);
    });
    form.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
    form.addEventListener('focusout', e => { if (!form.contains(e.relatedTarget)) close(); });
    list.addEventListener('mousedown', e => e.preventDefault());
    list.addEventListener('click', e => { if (e.target.closest('a')) close(); });
  }

  function showStockProblem(html) {
    stockEl.innerHTML = stockTop() + html;
    bindStockSearch();
  }
  const missingHtml = code => '<div class="empty" id="stock-missing">找不到這檔股票（' + esc(code) + '）。</div>';

  function quoteHtml(quote) {
    if (!quote) return '<div class="price"><span class="faint small">尚無成交資料</span></div>';
    const change = quote.change === null ? '' :
      '<span class="chg" style="color:' + colorOf(quote.change) + '">' + signed(quote.change, quote.close >= 100 ? 1 : 2) + '（' + sgn(quote.change_ratio * 100, 2) + '）</span>';
    return '<div class="price"><span class="big" id="stock-close">' + fmtPrice(quote.close) + '</span>' + change + '<span class="faint small">' + esc(quote.date) + ' 收盤</span></div>';
  }

  const SUPERSCRIPT = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  const minus = (v, text) => (v < 0 ? '−' : '') + text;
  function percentText(v, signedOut) {
    const x = Math.abs(v * 100), text = x.toFixed(x >= 10 ? 1 : x >= 0.01 ? 2 : 3);
    if (Number(text) === 0) return text + '%';
    return (v < 0 ? '−' : signedOut ? '+' : '') + text + '%';
  }
  function numberText(v) {
    const x = Math.abs(v);
    if (x === 0) return '0';
    if (x >= 1000) return minus(v, fmtInt(Math.round(x)));
    if (x >= 0.001) return minus(v, String(Number(x.toPrecision(4))));
    const [mantissa, exponent] = x.toExponential(2).split('e');
    return minus(v, mantissa + '×10' + String(Number(exponent)).split('').map(c => SUPERSCRIPT[c]).join(''));
  }
  const UNIT_TEXT = {
    ratio: v => percentText(v, true),
    percent: v => percentText(v, false),
    points: v => signed(v * 100, 2) + ' 個百分點',
    number: numberText,
    count: v => fmtInt(Math.round(v)),
    flag: v => v ? '是' : '否',
    category: v => String(v)
  };
  const valueText = (v, unit) => v === null || v === undefined ? '—' : (UNIT_TEXT[unit] || UNIT_TEXT.number)(v);
  const loading = '<p class="placeholder">載入中…</p>';
  const settle = promise => promise.then(value => ({ status: 'fulfilled', value: value }), reason => ({ status: 'rejected', reason: reason }));

  function stockHtml(s) {
    const watching = state.watch.has(s.stock_id);
    const tags = [s.market_label, s.industry, s.delist_date ? s.delist_date + ' 下市' : ''].filter(Boolean);
    return stockTop() +
      '<div class="stockhead"><div><div class="id"><h1><span class="code">' + esc(s.stock_id) + '</span><span id="stock-name">' + esc(s.name) + '</span></h1>' +
      '<span class="tags">' + tags.map(t => '<span>' + esc(t) + '</span>').join('') + '</span></div>' + quoteHtml(s.quote) + '</div>' +
      '<div class="aside"><div class="kpi" id="stock-rank"><div class="l"></div><div class="v">—</div></div>' +
      '<button type="button" class="ghost" id="watch-btn" aria-pressed="' + watching + '" style="height:36px">' + (watching ? '已追蹤' : '加入追蹤') + '</button></div></div>' +
      '<div class="section"><div class="sechead"><div class="control"><h2 id="chart-title"></h2>' +
      '<div class="seg" role="group" aria-label="預測期間" id="stock-seg"><button type="button" data-sh="5">5 日</button><button type="button" data-sh="20">20 日</button></div><span class="faint small" id="stock-seg-hint"></span>' +
      '<div class="seg" role="group" aria-label="圖表類型" id="chart-seg"><button type="button" data-ct="candle" aria-pressed="' + (state.chart === 'candle') + '">K 線</button><button type="button" data-ct="line" aria-pressed="' + (state.chart === 'line') + '">收盤線</button></div></div>' +
      '<div class="legend" id="chart-legend"></div></div><div class="chart" id="price-chart"></div></div>' +
      '<div class="two"><section id="features"><h2>這次預測怎麼來的</h2><div class="body">' + loading + '</div></section>' +
      '<section id="forecasts"><h2>三類預測</h2><div class="body">' + loading + '</div></section></div>' +
      '<div class="section" id="history"><div class="sechead"><h2>過去對本股的 ' + HISTORY_HORIZON + ' 日預測，與實際結果</h2><span class="muted small"></span></div><div class="body">' + loading + '</div></div>';
  }

  function rankHtml(entry, status) {
    if (status === 'pending') return '—<small> 載入中…</small>';
    if (status === 'rejected') return '—<small> 讀取失敗</small>';
    const p = entry && entry.prediction;
    if (!p) return '—<small> 尚未產出</small>';
    if (p.rank === null) return '—<small> 訓練範圍外</small>';
    return '第 ' + p.rank + ' 名<small> / ' + fmtInt(entry.universe) + '</small>';
  }

  function forecastRowHtml(entry) {
    const p = entry.prediction || {};
    const iv = entry.horizon === 20 ? p.interval : null;
    const note = iv ? SOURCE_NOTE[p.interval_source] : null;
    const ret = p.ret_pred === null || p.ret_pred === undefined ? null : p.ret_pred;
    return '<div class="trow h-cols"><div class="muted">' + entry.horizon + ' 日</div>' +
      '<div class="right num" style="color:' + colorOf(ret || 0) + '">' + pctOf(ret, 1) + '</div>' +
      '<div class="right">' + priceOf(p.price_pred) + '</div>' +
      '<div class="right muted band-text">' + (iv ? priceOf(iv.low) + ' – ' + priceOf(iv.high) + (note ? '（' + esc(note) + '）' : '') : '—') + '</div>' +
      '<div class="right prob">' + (p.prob_up_5pct === null || p.prob_up_5pct === undefined ? '—' : Math.round(p.prob_up_5pct * 100) + '%') + '</div></div>';
  }

  function forecastsHtml(data) {
    const entries = [...data.horizons].sort((a, b) => a.horizon - b.horizon);
    const based = [...entries].reverse().filter(entry => entry.prediction && entry.prediction.close !== null);
    if (!based.length) return waiting;
    const twenty = entries.find(entry => entry.horizon === 20) || {};
    const sameClose = based.every(entry => entry.prediction.close_date === based[0].prediction.close_date);
    const closeText = p => esc(p.close_date) + ' 收盤 ' + fmtPrice(p.close);
    const basis = sameClose ? closeText(based[0].prediction) : based.map(entry => entry.horizon + ' 日以 ' + closeText(entry.prediction)).join('、');
    const vol = twenty.prediction ? twenty.prediction.vol_pred : null;
    const quantileBand = Boolean(twenty.prediction && twenty.prediction.interval && twenty.prediction.interval_source === 'quantile');
    const fallbackBand = onFallbackBand(twenty.prediction) && !(twenty.models || []).some(model => model.role === 'interval');
    const realised = data.realized_vol && data.realized_vol.tech_vol_60d !== null ? data.realized_vol.tech_vol_60d : null;
    const rank = (twenty.models || []).find(model => model.role === 'rank') || (entries.flatMap(entry => entry.models || []).find(model => model.role === 'rank'));
    const degraded = entries.some(entry => (entry.degraded || []).length);
    const caveats = caveatText(entries.flatMap(entry => shownModels(entry.models, entry.horizon)));
    const lacking = entries.filter(entry => entry.prediction).map(entry => [entry.horizon, lackingText(entry.models, entry.horizon, entry.prediction)])
      .filter(([, text]) => text).map(([horizon, text]) => horizon + ' 日' + text).join('');
    return '<p class="sub">排名、報酬與區間機率各由自己的模型產出；排名模型多用了幾個籌碼與除息日特徵，其餘特徵相同。價格以 ' + basis + ' 換算。</p><div class="list">' +
      '<div class="thead h-cols" style="padding-top:10px"><div>期間</div><div class="right">預測報酬</div><div class="right">預測價</div><div class="right">80% 區間</div><div class="right prob-head" title="' + esc(PROB_TIP) + '">' + PROB_HEAD + '</div></div>' +
      entries.map(forecastRowHtml).join('') + '</div>' + (quantileBand ? '<p class="note band-note">' + BAND_NOTE + '</p>' : '') +
      (fallbackBand ? '<p class="note fallback-note">' + FALLBACK_NOTE + '</p>' : '') +
      (lacking ? '<p class="note model-note">' + lacking + MODEL_REQUIRES + '；<a href="#models">上線中的模型列在模型頁</a>。</p>' : '') +
      '<div class="pair"><div><div class="faint tiny">預測 20 日波動（年化）</div><div class="v">' + (vol === null || vol === undefined ? '—' : (vol * 100).toFixed(1) + '%') + '</div>' +
      (realised === null ? '' : '<div class="muted tiny">過去 60 日實際為 ' + (realised * 100).toFixed(1) + '%</div>') + '</div>' +
      '<div><div class="faint tiny">排名模型版本</div><div style="font-size:15px;font-weight:500;line-height:1.6">' + (rank ? esc((MODEL_NAMES[rank.model_type] || rank.model_type) + ' ' + rank.version) : '—') + '</div>' +
      (rank ? '<div class="muted tiny">訓練資料到 ' + esc(rank.train_end) + '，<a href="#models">成績在模型頁</a></div>' : '') + '</div></div>' +
      (degraded ? '<p class="note">有模型降級中，結果僅供參考。</p>' : '') +
      (caveats ? '<p class="note caveat-note">' + esc(caveats) + '。<a href="#models">原因在模型頁</a></p>' : '');
  }

  function featuresHtml(data, meta) {
    if (!meta.run_id) return waiting;
    if (!data.items.length) return '<p class="placeholder">這一期沒有這檔股票的特徵貢獻。</p>';
    const scale = Math.max(...data.items.map(item => Math.abs(item.contrib)), 1e-9);
    return '<p class="sub">對 ' + meta.horizon + ' 日排名分數影響最大的 ' + data.items.length + ' 個特徵；右側是對模型分數的貢獻，正值推高排名。</p><div class="list">' +
      data.items.map(item => {
        const w = Math.abs(item.contrib) / scale * 50, pos = item.contrib >= 0;
        return '<div class="feat"><span>' + esc(item.label) + '<span class="faint small">　</span><span class="faint small value">' + esc(valueText(item.value, item.unit)) + '</span></span>' +
          '<span class="right num contrib"' + (pos ? '' : ' style="color:var(--ink-2)"') + '>' + signed(item.contrib, 3) + '</span>' +
          '<div class="bar"><div class="zero"></div><div class="fill" style="background:' + (pos ? 'var(--model)' : 'var(--ink-4)') + ';left:' + (pos ? 50 : 50 - w) + '%;width:' + w + '%"></div></div></div>';
      }).join('') + '</div>' +
      '<p class="note">共 ' + fmtInt(data.n_features) + ' 個特徵' + (data.rest ? '，其餘 ' + fmtInt(data.rest.n_features) + ' 個合計 ' + signed(data.rest.contrib, 3) : '') + '。</p>' +
      '<p class="faint small recompute">特徵值取自目前的特徵表，每晚重算近期資料後可能更新，和預測當時不一定相同；貢獻值是預測當時算的。</p>';
  }

  function historyRowHtml(row) {
    const result = row.inside === null ? '<div class="res faint">—</div>' :
      '<div class="res"><i class="' + (row.inside ? 'in' : 'out') + '"></i><span' + (row.inside ? '' : ' class="muted"') + '>' + (row.inside ? '落在區間內' : '超出區間') + '</span></div>';
    return '<div class="trow short hist-cols"><div class="muted">' + esc(row.date) + (row.in_universe ? '' : '<span class="faint small">　訓練範圍外</span>') + '</div>' +
      '<div class="right" style="color:' + colorOf(row.ret_pred || 0) + '">' + pctOf(row.ret_pred, 1) + '</div>' +
      '<div class="right num" style="color:' + colorOf(row.actual) + '">' + pctOf(row.actual, 1) + '</div>' +
      '<div class="muted">' + (row.interval ? pctOf(row.interval.ret_low, 1) + ' 到 ' + pctOf(row.interval.ret_high, 1) : '—') + '</div>' + result + '</div>';
  }

  function historyHtml(rows, meta) {
    if (!rows.length) return '<p class="placeholder">還沒有到期的預測。</p>';
    return '<div class="scroll"><div class="table auto" style="min-width:760px;margin-top:16px">' +
      '<div class="thead hist-cols" style="padding-top:10px"><div>預測日</div><div class="right">預測報酬</div><div class="right">實際報酬</div><div>當時的 80% 區間</div><div>結果</div></div>' +
      rows.map(historyRowHtml).join('') + '</div></div>' +
      (rows.length < meta.total ? '<p class="note"><button type="button" class="ghost" id="history-all">顯示全部 ' + fmtInt(meta.total) + ' 次預測</button></p>' : '');
  }

  function showHistory(host, result) {
    const head = host.querySelector('.sechead span'), body = host.querySelector('.body');
    if (result.status === 'rejected') { body.innerHTML = errorBox(result.reason); return; }
    const { data, meta } = result.value;
    head.textContent = meta.with_interval ? '最近 ' + meta.with_interval + ' 次有區間的預測中，實際報酬有 ' + meta.inside + ' 次落在 80% 區間內' : '';
    body.innerHTML = historyHtml(data, meta);
    const more = body.querySelector('#history-all');
    if (more) more.addEventListener('click', () => {
      more.disabled = true;
      const note = more.parentNode;
      note.querySelectorAll('.error').forEach(box => box.remove());
      api.getAll('stocks/' + state.stock + '/history', { horizon: HISTORY_HORIZON }).then(
        all => showHistory(host, { status: 'fulfilled', value: all }),
        err => { note.insertAdjacentHTML('beforeend', errorBox(err)); more.disabled = false; });
    });
  }

  async function loadFeatures(code, token) {
    const mine = ++state.featureToken;
    const body = document.querySelector('#features .body');
    body.innerHTML = loading;
    try {
      const { data, meta } = await api.get('stocks/' + code + '/features', { horizon: state.h });
      if (token === state.stockToken && mine === state.featureToken) body.innerHTML = featuresHtml(data, meta);
    } catch (err) {
      if (token === state.stockToken && mine === state.featureToken) body.innerHTML = errorBox(err);
    }
  }

  function fanOf(entry) {
    const p = entry && entry.prediction;
    if (!p || p.ret_pred === null || p.close === null) return null;
    const band = entry.horizon === 20 && p.interval ? { low: p.interval.ret_low, high: p.interval.ret_high } : null;
    return { h: entry.horizon, day: entry.date, date: p.close_date, ret: p.ret_pred, band: band };
  }

  async function renderStock(code) {
    const token = ++state.stockToken;
    if (!STOCK_CODE.test(code)) {
      showStockProblem(missingHtml(code));
      return;
    }
    stockEl.innerHTML = backLink + loading;
    const later = {
      forecasts: settle(api.get('stocks/' + code + '/predictions')),
      history: settle(api.get('stocks/' + code + '/history', { horizon: HISTORY_HORIZON, page_size: HISTORY_ROWS }))
    };
    const [stock, prices] = await Promise.all([settle(api.get('stocks/' + code)), settle(api.get('stocks/' + code + '/prices', { days: PRICE_DAYS }))]);
    if (token !== state.stockToken) return;
    if (stock.status === 'rejected') {
      showStockProblem(stock.reason && stock.reason.status === 404 ? missingHtml(code) : errorBox(stock.reason));
      return;
    }
    const s = stock.value.data;
    state.stock = s.stock_id;
    saveStock();
    document.querySelector('[data-nav="stock"]').setAttribute('href', '#stock-' + encodeURIComponent(state.stock));
    stockEl.innerHTML = stockHtml(s);
    bindStockSearch();
    document.getElementById('watch-btn').addEventListener('click', e => {
      if (state.watch.has(s.stock_id)) state.watch.delete(s.stock_id); else state.watch.add(s.stock_id);
      saveWatch();
      const on = state.watch.has(s.stock_id);
      e.currentTarget.setAttribute('aria-pressed', String(on));
      e.currentTarget.textContent = on ? '已追蹤' : '加入追蹤';
      controls();
    });
    let forecasts = { status: 'pending' };
    const entryOf = () => forecasts.status === 'fulfilled' ? forecasts.value.data.horizons.find(entry => entry.horizon === state.h) : null;
    const drawChart = () => {
      const bars = prices.status === 'fulfilled' ? prices.value.data : [];
      const candidate = fanOf(entryOf());
      const fan = candidate && bars.some(b => b.date === candidate.day) && bars.some(b => b.date === candidate.date && b.close !== null) ? candidate : null;
      document.getElementById('chart-title').textContent = '過去 ' + PRICE_DAYS + ' 個交易日' + (fan ? '，與未來 ' + fan.h + ' 日的預測' : '');
      document.querySelectorAll('#chart-seg [data-ct]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.ct === state.chart)));
      const forecastLegend = fan ? '<span><i class="sw-dash"></i>預測中位數（內插）</span>' + (fan.band ? '<span><i class="sw-band"></i>80% 預測區間（內插）</span>' : '') : '';
      document.getElementById('chart-legend').innerHTML = (state.chart === 'candle'
        ? '<span><i class="sw-up"></i>收高於開</span><span><i class="sw-down"></i>收低於開</span><span><i class="sw-ma5"></i>5 日均線</span><span><i class="sw-ma20"></i>20 日均線</span>'
        : '<span><i class="sw-line"></i>收盤價（還原）</span>') + forecastLegend;
      const host = document.getElementById('price-chart');
      if (prices.status === 'rejected') host.innerHTML = errorBox(prices.reason);
      else drawPriceChart(host, s, bars, state.chart, fan);
    };
    const showRank = () => { document.querySelector('#stock-rank .v').innerHTML = rankHtml(entryOf(), forecasts.status); };
    const showSeg = () => {
      if (forecasts.status !== 'fulfilled') return;
      const ready = new Set(forecasts.value.data.horizons.filter(entry => entry.prediction).map(entry => entry.horizon));
      const buttons = [...document.querySelectorAll('#stock-seg [data-sh]')];
      buttons.forEach(b => { b.disabled = !ready.has(Number(b.dataset.sh)); });
      const missing = buttons.filter(b => b.disabled).map(b => b.dataset.sh + ' 日');
      const none = s.delist_date ? '已下市，沒有預測' : '預測產出後可切換 5 日／20 日';
      document.getElementById('stock-seg-hint').textContent = !ready.size ? none : missing.length ? missing.join('、') + '預測尚未產出' : '';
    };
    const drawHorizon = () => {
      controls();
      document.querySelectorAll('#stock-seg [data-sh]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.sh) === state.h)));
      document.querySelector('#stock-rank .l').textContent = state.h + ' 日預測排名';
      showRank();
      drawChart();
      loadFeatures(s.stock_id, token);
    };
    document.querySelectorAll('#stock-seg [data-sh]').forEach(b => b.addEventListener('click', () => { state.h = Number(b.dataset.sh); state.page = 1; drawHorizon(); }));
    document.querySelectorAll('#chart-seg [data-ct]').forEach(b => b.addEventListener('click', () => {
      state.chart = b.dataset.ct;
      try { localStorage.setItem('twp-chart', state.chart); } catch (e) { state.chart = b.dataset.ct; }
      drawChart();
    }));
    drawHorizon();
    later.forecasts.then(result => {
      if (token !== state.stockToken) return;
      forecasts = result;
      document.querySelector('#forecasts .body').innerHTML = result.status === 'rejected' ? errorBox(result.reason) : forecastsHtml(result.value.data);
      showRank();
      showSeg();
      if (fanOf(entryOf())) drawChart();
    });
    later.history.then(result => { if (token === state.stockToken) showHistory(document.getElementById('history'), result); });
  }

  function fanPoints(fan, anchor, base) {
    const steps = [...Array(fan.h + 1).keys()];
    const at = (t, q) => {
      const share = t / fan.h;
      return { i: anchor + t, v: base * (1 + fan.ret * share + (q - fan.ret) * Math.sqrt(share)) };
    };
    return {
      median: steps.map(t => at(t, fan.ret)),
      top: fan.band ? steps.map(t => at(t, fan.band.high)) : [],
      bottom: fan.band ? steps.map(t => at(t, fan.band.low)) : []
    };
  }

  function drawPriceChart(host, s, bars, kind, fan) {
    const traded = bars.filter(b => b.close !== null);
    if (!traded.length) { host.innerHTML = '<p class="placeholder">這段期間沒有成交資料。</p>'; return; }
    const candle = kind === 'candle';
    const n = bars.length;
    const anchor = fan ? bars.findIndex(b => b.date === fan.day) : -1;
    const base = fan ? bars.find(b => b.date === fan.date && b.close !== null) : null;
    const fanAt = anchor < 0 || !base ? null : fanPoints(fan, anchor, base.close);
    const slots = fanAt ? Math.max(n, anchor + fan.h + 1) : n;
    const fanValues = fanAt ? [...fanAt.median, ...fanAt.top, ...fanAt.bottom].map(pt => pt.v) : [];
    const closes = bars.map(b => b.close);
    const ma5 = ma(closes, 5), ma20 = ma(closes, 20);
    const prevClose = [];
    bars.reduce((last, b, i) => { prevClose[i] = last; return b.close !== null ? b.close : last; }, null);
    const W = 1280, padL = 56, padR = 24, padT = 20, plotH = 300, gapV = 18, volH = 72, padB = 36;
    const volTop = padT + plotH + gapV, volBase = volTop + volH, H = volBase + padB;
    const innerW = W - padL - padR, step = slots > 1 ? innerW / (slots - 1) : 0;
    const lows = [...traded.map(b => candle ? b.low : b.close), ...fanValues], highs = [...traded.map(b => candle ? b.high : b.close), ...fanValues];
    const vmin = Math.min(...lows), vmax = Math.max(...highs), pad = (vmax - vmin) * 0.08 || vmax * 0.02 || 1;
    const ticks = niceTicks(vmin - pad, vmax + pad, 6);
    const yMin = Math.min(vmin - pad, ticks[0]), yMax = Math.max(vmax + pad, ticks[ticks.length - 1]);
    const x = i => slots > 1 ? padL + i * step : padL + innerW / 2;
    const y = v => padT + (yMax - v) / (yMax - yMin) * plotH;
    const lots = bars.map(b => b.volume === null ? null : b.volume / 1000);
    const maxVol = Math.max(1, ...lots.filter(v => v !== null));
    const yv = v => volBase - v / maxVol * volH;
    const labelPrice = v => fmtPrice(v);
    const bw = Math.max(3, Math.round(Math.max(step, 6) * 0.62 * 10) / 10);
    const path = arr => arr.map((v, i) => v === null ? '' : ((i === 0 || arr[i - 1] === null) ? 'M' : 'L') + f1(x(i)) + ' ' + f1(y(v))).join(' ');
    const xl = bars.map((b, i) => ({ i, label: md(b.date) })).filter(t => t.i % 10 === 0 || t.i === n - 1);
    if (fanAt) xl.push({ i: anchor + fan.h, label: '+' + fan.h + ' 日' });
    const line = pts => pts.map((pt, k) => (k ? 'L' : 'M') + f1(x(pt.i)) + ' ' + f1(y(pt.v))).join(' ');
    const fanLayer = !fanAt ? '' :
      (fanAt.top.length ? '<path class="fan-band" d="' + line(fanAt.top) + ' L' + line([...fanAt.bottom].reverse()).slice(1) + ' Z" fill="var(--model-soft)" stroke="none"></path>' : '') +
      '<path class="fan-median" d="' + line(fanAt.median) + '" fill="none" stroke="var(--model)" stroke-width="2" stroke-dasharray="6 4"></path>';

    let priceLayer;
    if (candle) {
      priceLayer = bars.map((d, i) => {
        if (d.close === null) return '';
        const up = d.close >= d.open, c = up ? 'var(--up)' : 'var(--down)';
        const yo = y(d.open), yc = y(d.close), bodyTop = Math.min(yo, yc), bodyH = Math.max(1, Math.abs(yo - yc));
        return '<line x1="' + f1(x(i)) + '" x2="' + f1(x(i)) + '" y1="' + f1(y(d.high)) + '" y2="' + f1(y(d.low)) + '" stroke="' + c + '" stroke-width="1"></line>' +
          '<rect class="candle" x="' + f1(x(i) - bw / 2) + '" y="' + f1(bodyTop) + '" width="' + bw + '" height="' + f1(bodyH) + '" fill="' + c + '"></rect>';
      }).join('') +
        '<path d="' + path(ma20) + '" fill="none" stroke="var(--ink-4)" stroke-width="1.5" stroke-linejoin="round"></path>' +
        '<path d="' + path(ma5) + '" fill="none" stroke="var(--ink-2)" stroke-width="1.5" stroke-linejoin="round"></path>';
    } else {
      priceLayer = '<path class="close-line" d="' + path(closes) + '" fill="none" stroke="var(--ink)" stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round"></path>';
    }
    const volLayer = bars.map((d, i) => {
      if (lots[i] === null) return '';
      const up = d.close !== null && prevClose[i] !== null && d.close >= prevClose[i];
      return '<rect x="' + f1(x(i) - bw / 2) + '" y="' + f1(yv(lots[i])) + '" width="' + bw + '" height="' + f1(volBase - yv(lots[i])) + '" fill="' + (up ? 'var(--up)' : 'var(--down)') + '" fill-opacity="0.55"></rect>';
    }).join('');

    host.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(s.name) + '過去 ' + n + ' 個交易日' + (candle ? 'K 線' : '收盤價') + '與成交量' + (fanAt ? '，與未來 ' + fan.h + ' 日的預測' : '') + '">' +
      ticks.filter(v => v >= yMin && v <= yMax).map(v => '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + f1(y(v)) + '" y2="' + f1(y(v)) + '" stroke="#2A2A28"></line><text x="' + (padL - 10) + '" y="' + f1(y(v) + 4) + '" text-anchor="end" fill="#8A887F" font-size="12">' + labelPrice(v) + '</text>').join('') +
      '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + f1(volTop) + '" y2="' + f1(volTop) + '" stroke="#2A2A28"></line>' +
      '<text x="' + (padL - 10) + '" y="' + f1(volTop + 4) + '" text-anchor="end" fill="#8A887F" font-size="12">' + fmtInt(Math.round(maxVol)) + '</text>' +
      '<text x="' + (padL - 10) + '" y="' + f1(volBase) + '" text-anchor="end" fill="#8A887F" font-size="12">張</text>' +
      '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + f1(volBase) + '" y2="' + f1(volBase) + '" stroke="#2A2A28"></line>' +
      xl.map(t => '<text x="' + f1(x(t.i)) + '" y="' + (H - 12) + '" text-anchor="middle" fill="#8A887F" font-size="12">' + t.label + '</text>').join('') +
      fanLayer + priceLayer + volLayer +
      '<g class="hover" style="display:none"><line class="hx" y1="' + padT + '" y2="' + volBase + '" stroke="#6B6A64"></line><circle class="hc" r="4.5" stroke="#1B1B1A" stroke-width="2"></circle></g>' +
      '<rect class="overlay" x="' + padL + '" y="' + padT + '" width="' + innerW + '" height="' + (volBase - padT) + '" fill="transparent"></rect></svg>' +
      '<div class="tip" style="display:none"><div class="t"></div><div class="b"></div></div>';

    const svg = host.querySelector('svg'), overlay = host.querySelector('.overlay'), hover = host.querySelector('.hover');
    const hx = host.querySelector('.hx'), hc = host.querySelector('.hc'), tip = host.querySelector('.tip'), tt = tip.querySelector('.t'), tb = tip.querySelector('.b');
    overlay.addEventListener('pointermove', ev => {
      const rect = svg.getBoundingClientRect();
      const ux = (ev.clientX - rect.left) * (W / rect.width);
      const idx = slots > 1 ? Math.max(0, Math.min(slots - 1, Math.round((ux - padL) / step))) : 0;
      const d = bars[idx], px = x(idx);
      if (!d) {
        const t = idx - anchor, median = fanAt.median[t].v;
        tt.textContent = '+' + t + ' 個交易日';
        tb.innerHTML = '<div>預測中位數 ' + fmtPrice(median) + '</div>' +
          (fanAt.top.length ? '<div class="muted">80% 區間 ' + fmtPrice(fanAt.bottom[t].v) + ' – ' + fmtPrice(fanAt.top[t].v) + '</div>' : '');
        hc.setAttribute('cy', y(median));
        hc.setAttribute('fill', '#ECEAE4');
        hc.style.display = '';
      } else if (d.close === null) {
        tt.textContent = d.date.replace(/-/g, '/');
        tb.textContent = '無成交資料';
        hc.style.display = 'none';
      } else {
        tt.textContent = d.date.replace(/-/g, '/');
        const prior = prevClose[idx];
        const chg = prior ? (d.close / prior - 1) * 100 : 0;
        const volume = lots[idx] === null ? '' : '成交量 ' + fmtInt(Math.round(lots[idx])) + ' 張';
        tb.innerHTML = candle
          ? '<div>開 ' + fmtPrice(d.open) + '　高 ' + fmtPrice(d.high) + '　低 ' + fmtPrice(d.low) + '　收 ' + fmtPrice(d.close) + '　<span style="color:' + colorOf(chg) + '">' + sgn(chg, 2) + '</span></div><div class="muted">' + volume + (ma5[idx] !== null ? '　5 日均 ' + fmtPrice(ma5[idx]) : '') + (ma20[idx] !== null ? '　20 日均 ' + fmtPrice(ma20[idx]) : '') + '</div>'
          : '<div>收盤 ' + fmtPrice(d.close) + '　<span style="color:' + colorOf(chg) + '">' + sgn(chg, 2) + '</span></div><div class="muted">' + volume + '</div>';
        hc.setAttribute('cy', y(d.close));
        hc.setAttribute('fill', candle ? (d.close >= d.open ? '#E08E8A' : '#4FAE97') : '#ECEAE4');
        hc.style.display = '';
      }
      hx.setAttribute('x1', px); hx.setAttribute('x2', px); hc.setAttribute('cx', px);
      hover.style.display = ''; tip.style.display = '';
      tip.style.left = (px / W * 100) + '%';
      tip.style.transform = idx > slots - 12 ? 'translateX(-100%)' : idx < 6 ? 'translateX(0)' : 'translateX(-50%)';
    });
    overlay.addEventListener('pointerleave', () => { hover.style.display = 'none'; tip.style.display = 'none'; });
  }

  const pct1 = v => (v * 100).toFixed(1) + '%';
  const num3 = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3);
  const has = v => v !== null && v !== undefined;
  const modelName = type => MODEL_NAMES[type] || type;
  const monthOf = iso => iso ? iso.slice(0, 7) : '';

  function recentText(row) {
    const r = row.recent || {};
    if (!r.days) return '尚無到期的評估日';
    const parts = {
      rank: [has(r.ic) ? 'rank IC ' + num3(r.ic) : '', has(r.icir) ? 'ICIR ' + r.icir.toFixed(2) : ''],
      regression: [has(r.mae) ? 'MAE ' + pct1(r.mae) : '', has(r.direction_acc) ? '方向準確率 ' + pct1(r.direction_acc) : ''],
      quantile: [has(r.coverage) ? '覆蓋率 ' + pct1(r.coverage) : '', has(r.width) ? '平均寬度 ' + (r.width * 100).toFixed(1) + ' 個百分點' : ''],
      volatility: [has(r.mae) ? 'MAE ' + (r.mae * 100).toFixed(1) + ' 個百分點（年化）' : ''],
      classification: [has(r.auc) ? 'AUC ' + num3(r.auc) : '', has(r.brier) ? 'Brier ' + num3(r.brier) : '']
    }[row.objective] || [];
    return parts.filter(Boolean).join('，') + '（近 ' + r.days + ' 個評估日）';
  }

  function researchText(row) {
    if (row.objective !== 'rank') return '—';
    if (!row.research) return '回測成績尚未產出';
    const m = row.research.metrics || {};
    const own = (row.metric_source === 'holdout' ? 'holdout' : '研究期') + ' rank IC ' + (has(m.rank_ic) ? num3(m.rank_ic) : '—') + (has(m.rank_icir) ? '，ICIR ' + m.rank_icir.toFixed(2) : '');
    const b = row.baseline ? row.baseline.metrics || {} : null;
    return own + (b ? '；Ridge ' + (has(b.rank_ic) ? num3(b.rank_ic) : '—') + (has(b.rank_icir) ? '，ICIR ' + b.rank_icir.toFixed(2) : '') : '');
  }

  function degradedText(row) {
    const w = row.degraded_window;
    return [row.degraded_on ? row.degraded_on + ' 起' : '', w ? windowText(w.own_rows, w.days, w.mostly_predecessors) : ''].filter(Boolean).join('；');
  }

  const releaseKey = row => row && row.holdout_release ? JSON.stringify(row.holdout_release) : '';

  function releaseSpans(rows) {
    return rows.map((row, i) => {
      const key = releaseKey(row);
      if (!key || releaseKey(rows[i + 1]) === key) return 0;
      let span = 1;
      while (releaseKey(rows[i - span]) === key) span += 1;
      return span;
    });
  }

  function releasedHtml(row, span) {
    const r = row.holdout_release;
    if (!span) return '';
    const short = [r.failed.length ? '未達的門檻：' + r.failed.join('、') : '', r.flagged.length ? '退化旗標：' + r.flagged.join('、') : ''].filter(Boolean).join('；');
    const lines = [(span > 1 ? '以上 ' + span + ' 個模型：' : '') + 'holdout 判定「' + r.status + '」' + (short ? '，' + short : ''), r.released_at.slice(0, 10) + ' 經使用者放行：' + r.reason];
    return '<div class="released">' + lines.map(line => '<div>' + esc(line) + '</div>').join('') + '</div>';
  }

  function forcedHtml(row) {
    const f = row.forced_promotion;
    if (!f) return '';
    const lines = ['換模門檻沒過：' + f.gates.map(gateText).join('、'), [f.promoted_on, '經使用者強制換上：' + (f.reason || '沒有寫原因')].filter(Boolean).join(' ')];
    return '<div class="forced">' + lines.map(line => '<div>' + esc(line) + '</div>').join('') + '</div>';
  }

  function modelRowHtml(row, span) {
    const degraded = row.status === 'degraded' ? '<span class="faint small">　降級中</span>' : '';
    const detail = row.status === 'degraded' ? degradedText(row) : '';
    const notes = forcedHtml(row) + releasedHtml(row, span);
    return '<div class="trow model-cols' + (notes ? ' has-release' : '') + '"><div class="num">' + esc(OBJECTIVES[row.objective] || row.objective) + '<span class="faint small">　' + row.horizon + ' 日</span></div>' +
      '<div>' + esc(modelName(row.model_type)) + degraded + (detail ? '<div class="faint small degraded">' + esc(detail) + '</div>' : '') + '</div>' +
      '<div class="muted version">' + esc(row.version) + '</div>' +
      '<div class="muted">' + esc(monthOf(row.train_start)) + ' 到 ' + esc(monthOf(row.train_end)) + '</div>' +
      '<div>' + esc(recentText(row)) + '</div>' +
      '<div class="muted">' + esc(researchText(row)) + '</div></div>' + notes;
  }

  async function renderModelTable() {
    const host = document.getElementById('model-table');
    try {
      const { data } = await api.get('models');
      const spans = releaseSpans(data);
      host.innerHTML = !data.length ? '<p class="placeholder">模型尚未上線。</p>' :
        '<div class="scroll" style="margin-top:14px"><div class="table" style="min-width:1100px">' +
        '<div class="thead model-cols"><div>目標</div><div>模型</div><div>版本</div><div>訓練區間</div><div>近 6 個月成績</div><div>回測成績與 baseline</div></div>' +
        data.map((row, i) => modelRowHtml(row, spans[i])).join('') + '</div></div>';
    } catch (err) {
      host.innerHTML = errorBox(err);
    }
  }

  function icTicks(lo, hi) {
    const step = [0.01, 0.02, 0.05, 0.1, 0.2].find(s => (hi - lo) / s <= 5) || 0.5;
    const first = Math.floor(lo / step + 1e-9), last = Math.max(first + 1, Math.ceil(hi / step - 1e-9));
    return Array.from({ length: last - first + 1 }, (_, k) => Number(((first + k) * step).toFixed(2)));
  }

  function drawIcChart(host, rows, meta) {
    const model = modelName(meta.model_type), baseline = modelName(meta.baseline);
    const values = rows.flatMap(row => [row.model, row.baseline]).filter(has);
    const W = 1280, H = 300, padL = 48, padR = 24, padT = 16, padB = 32, innerW = W - padL - padR, innerH = H - padT - padB;
    const ticks = icTicks(Math.min(0, ...values), Math.max(0, ...values));
    const vMin = ticks[0], vMax = ticks[ticks.length - 1];
    const y = v => padT + (vMax - v) / (vMax - vMin) * innerH, y0 = y(0), gw = innerW / rows.length;
    const bar = (cls, i, v, dx) => !has(v) ? '' : '<rect class="' + cls + (rows[i].complete ? '' : ' partial') + '" x="' + f1(padL + i * gw + gw / 2 + dx) + '" y="' + f1(Math.min(y(v), y0)) + '" width="' + f1(Math.max(2, gw / 2 - 3)) + '" height="' + f1(Math.abs(y(v) - y0)) + '" rx="2" fill="' + (cls === 'bar-model' ? 'var(--model)' : 'var(--ink-4)') + '"' + (rows[i].complete ? '' : ' fill-opacity="0.35"') + '></rect>';
    host.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="近 ' + rows.length + ' 個月每月 rank IC，' + esc(model) + ' 與 ' + esc(baseline) + ' 並列長條圖">' +
      ticks.map(v => '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + f1(y(v)) + '" y2="' + f1(y(v)) + '" stroke="' + (v === 0 ? '#4A4A46' : '#2A2A28') + '"></line><text x="' + (padL - 8) + '" y="' + f1(y(v) + 4) + '" text-anchor="end" fill="#8A887F" font-size="12">' + v.toFixed(2) + '</text>').join('') +
      '<g class="hl" style="display:none"><rect y="' + padT + '" width="' + f1(gw) + '" height="' + innerH + '" fill="rgba(255,255,255,0.04)"></rect></g>' +
      rows.map((row, i) => bar('bar-model', i, row.model, -gw / 2 + 2) + bar('bar-baseline', i, row.baseline, 1)).join('') +
      rows.map((row, i) => (i % 3 === 0 || i === rows.length - 1) ? '<text x="' + f1(padL + i * gw + gw / 2) + '" y="' + (H - 8) + '" text-anchor="middle" fill="#8A887F" font-size="12">' + esc(row.month.slice(2, 4) + '/' + row.month.slice(5)) + '</text>' : '').join('') +
      '<rect class="overlay" x="' + padL + '" y="' + padT + '" width="' + innerW + '" height="' + innerH + '" fill="transparent"></rect></svg>' +
      '<div class="tip" style="display:none;top:6px"><div class="t"></div><div class="b"></div></div>';
    const svg = host.querySelector('svg'), overlay = host.querySelector('.overlay'), hl = host.querySelector('.hl'), hlRect = hl.querySelector('rect');
    const tip = host.querySelector('.tip'), tt = tip.querySelector('.t'), tb = tip.querySelector('.b');
    overlay.addEventListener('pointermove', ev => {
      const rect = svg.getBoundingClientRect();
      const i = Math.max(0, Math.min(rows.length - 1, Math.floor(((ev.clientX - rect.left) * (W / rect.width) - padL) / gw)));
      const row = rows[i], gx = padL + i * gw;
      hlRect.setAttribute('x', gx); hl.style.display = '';
      tt.textContent = row.month.replace('-', ' 年 ').replace(/ 0(\d)$/, ' $1') + ' 月';
      tb.textContent = model + ' ' + (has(row.model) ? num3(row.model) : '—') + '，' + baseline + ' ' + (has(row.baseline) ? num3(row.baseline) : '—') +
        (row.complete ? '' : '（只有 ' + (row.days || 0) + ' 天，不計入比較）');
      tip.style.display = ''; tip.style.left = ((gx + gw / 2) / W * 100) + '%';
      tip.style.transform = i > rows.length - 4 ? 'translateX(-100%)' : i < 2 ? 'translateX(0)' : 'translateX(-50%)';
    });
    overlay.addEventListener('pointerleave', () => { hl.style.display = 'none'; tip.style.display = 'none'; });
  }

  const IC_SOURCES = { params: '上線模型指定的研究', self: '上線模型的研究' };

  async function renderIcChart() {
    const host = document.getElementById('ic-chart'), sub = document.getElementById('ic-sub'), legend = document.getElementById('ic-legend');
    try {
      const { data, meta } = await api.get('models/ic-monthly', { horizon: 20, months: 24 });
      if (!data.length) {
        host.innerHTML = '<p class="placeholder">回測結果尚未產出。</p>';
        sub.textContent = 'walk-forward 研究期的月平均。';
        legend.innerHTML = '';
        return;
      }
      const model = modelName(meta.model_type), baseline = modelName(meta.baseline);
      legend.innerHTML = '<span><i class="sw-sq" style="background:var(--model)"></i>' + esc(model) + '</span><span><i class="sw-sq" style="background:var(--ink-4)"></i>' + esc(baseline) + '</span>';
      sub.textContent = (meta.source === 'holdout' ? 'holdout（設計定案後才跑的驗證期）的月平均，近 ' + data.length + ' 個月（' : 'walk-forward 研究期的月平均，近 ' + data.length + ' 個月（' + (IC_SOURCES[meta.source] || meta.source) + '，') + meta.run_key + '）。' +
        model + ' 有 ' + meta.above_baseline + ' 個月高於 ' + baseline + '，負值月份 ' + meta.negative + ' 個；天數少於 ' + meta.min_days + ' 天的月份不計，以淡色表示。';
      drawIcChart(host, data, meta);
    } catch (err) {
      host.innerHTML = errorBox(err);
      sub.textContent = '';
      legend.innerHTML = '';
    }
  }

  function etlRowHtml(run) {
    const status = RUN_STATUS[run.status] || [run.status, 'idle'];
    return '<div class="trow short etl-cols"><div class="muted">' + esc(hhmm(run.started_at)) + '</div><div>' + esc(run.dataset) + '</div>' +
      '<div class="muted">' + esc(SOURCES[run.source] || run.source) + '</div><div class="right">' + fmtInt(run.rows) + '</div>' +
      '<div class="res"><i class="' + status[1] + '"></i><span' + (run.status === 'success' ? '' : ' class="muted"') + ' title="' + esc(run.error) + '">' + esc(status[0]) + '</span></div></div>';
  }

  async function renderEtl() {
    const sub = document.getElementById('etl-sub'), host = document.getElementById('etl');
    try {
      const { data, meta } = await api.getAll('etl/runs', {});
      sub.textContent = meta.date + ' 的每日增量，依開始時間排序。';
      host.innerHTML = data.map(etlRowHtml).join('') || '<div class="empty">這個交易日還沒有資料更新。</div>';
    } catch (err) {
      sub.textContent = '';
      host.innerHTML = errorBox(err);
    }
  }

  function checkSummary(check) {
    const d = check.detail || {};
    switch (check.check_name) {
      case 'row_count': return (d.table || '') + ' ' + fmtInt(d.count || 0) + ' 列' + (d.previous ? '，前一交易日 ' + fmtInt(d.previous) + ' 列' : '');
      case 'big_move': return d.count ? fmtInt(d.count) + ' 檔，已標記待查' : '沒有';
      case 'missing_values': {
        const cols = Object.entries(d.columns || {}).filter(([, gap]) => gap.missing > 0);
        return (d.table || '') + (cols.length ? '：' + cols.map(([col, gap]) => (col === 'row' ? '整列' : col) + ' 缺 ' + fmtInt(gap.missing) + ' 檔').join('、') : ' 沒有缺值');
      }
      case 'trading_calendar': {
        const tables = Object.entries(d.rows || {});
        return tables.length ? '有資料落在休市日：' + tables.map(([table, days]) => table + '（' + Object.keys(days).join('、') + '）').join('、') : '沒有資料落在休市日';
      }
      case 'index_freshness': return Object.keys(d.stale || {}).length ? '過期：' + Object.keys(d.stale).join('、') : '全部在期限內';
      case 'adj_factor_changed': {
        const days = Object.values(d.stocks || {});
        const earliest = d.earliest || days.slice().sort()[0];
        return fmtInt(has(d.count) ? d.count : days.length) + ' 檔' + (earliest ? '，最早 ' + earliest : '');
      }
      case 'us_close': return Object.keys(d.stale || {}).length ? '未更新：' + Object.keys(d.stale).join('、') : '全部已更新';
      case 'etl_runs': {
        const blocking = d.blocking || [], rest = (d.missing || []).filter(key => !blocking.includes(key));
        if (!blocking.length && !rest.length) return '全部資料集與排程工作都已更新';
        return [blocking.length ? '缺閘門資料集：' + blocking.join('、') : '', rest.length ? '尚未更新：' + rest.join('、') : ''].filter(Boolean).join('；');
      }
      case 'stock_master': {
        const parts = [['新增', d.added], ['換市場', d.moved], ['恢復上市', d.relisted], ['下市', d.delisted], ['消失未下市', d.vanished]]
          .filter(([, v]) => sizeOf(v) > 0).map(([label, v]) => label + ' ' + fmtInt(sizeOf(v)) + ' 檔');
        return parts.length ? parts.join('、') : '沒有異動';
      }
      default: return '';
    }
  }

  const SEVERITY_RANK = ['info', 'warning', 'error'];
  const worstOf = severities => severities.reduce((a, b) => SEVERITY_RANK.indexOf(b) > SEVERITY_RANK.indexOf(a) ? b : a, 'info');
  const perStock = rows => rows.every(row => row.detail && row.detail.stock_id !== undefined && row.detail.stock_id !== null);

  function lineHtml(name, severity, summary) {
    const sev = SEVERITY[severity] || [severity, 'idle'];
    return '<div class="check"><i class="' + (severity === 'info' ? '' : sev[1]) + '"></i><div><div>' + esc(CHECKS[name] || name) + '（' + esc(sev[0]) + '）</div>' +
      '<div class="muted small">' + esc(summary) + '</div></div></div>';
  }

  function dayChecksHtml(checks) {
    const names = [...new Set(checks.map(check => check.check_name))];
    return names.map(name => {
      const rows = checks.filter(check => check.check_name === name);
      if (rows.length < 2 || !perStock(rows)) return rows.map(checkHtml).join('');
      const stocks = new Set(rows.map(row => row.detail.stock_id)).size;
      return lineHtml(name, worstOf(rows.map(row => row.severity)), fmtInt(stocks) + ' 檔');
    }).join('');
  }

  const spanText = (first, last) => first === last ? first : first + ' 至 ' + last;
  const correctionHtml = row => lineHtml(row.check_name, row.severity, (row.stocks ? fmtInt(row.stocks) + ' 檔、' : '') + fmtInt(row.rows) + ' 列，' + spanText(row.first_date, row.last_date));

  async function renderCorrections() {
    const host = document.getElementById('corrections');
    try {
      const { data } = await api.get('data-quality/corrections');
      host.innerHTML = data.map(correctionHtml).join('') || '<p class="placeholder">沒有資料更正紀錄。</p>';
    } catch (err) {
      host.innerHTML = errorBox(err);
    }
  }

  const checkHtml = check => lineHtml(check.check_name, check.severity, checkSummary(check));

  const metricText = (metric, v) => metric === 'coverage' ? (v * 100).toFixed(1) + '%' : Number(v).toFixed(3);

  function windowText(own, days, mostly) {
    const share = Number.isInteger(own) && Number.isInteger(days) ? '窗口 ' + own + '／' + days + ' 列為本 run' + (days > 0 ? '（' + Math.round(own / days * 100) + '%）' : '') : '';
    if (mostly !== true) return share;
    return share ? share + '，多為前任 run 的列' : '窗口多為前任 run 的列';
  }

  function modeText(run, date) {
    if (run.mode === 'reset' && run.reset_rows === 0 && run.restored_on && date && run.restored_on > date) return '恢復前的告警（' + run.restored_on + ' restore）';
    if (run.mode === 'grace') return (run.grace_start && run.grace_end ? '寬限期 ' + run.grace_start + '–' + run.grace_end : '寬限期內') + '，只告警不降級';
    if (run.mode !== 'reset') return '';
    const count = v => has(v) ? v : '—';
    const parts = [run.restored_on ? run.restored_on + ' restore' : '', '已累積 ' + count(run.reset_rows) + '／' + count(run.reset_needed) + ' 列'];
    return '此 slot 在 restore 重設期（' + parts.filter(Boolean).join('，') + '），只告警不降級';
  }

  function slotText(slot) {
    const [objective, ...rest] = String(slot || '').split(' ');
    return [OBJECTIVES[objective] || objective, ...rest].join(' ');
  }

  function gateText(gate) {
    const value = typeof gate.value === 'number' ? '實際 ' + (Number.isInteger(gate.value) ? fmtInt(gate.value) : gate.value.toFixed(3)) : '';
    const sample = Number.isInteger(gate.sample) ? '樣本 ' + fmtInt(gate.sample) + (SAMPLE_UNITS[gate.sample_unit] || '') : '';
    const threshold = has(gate.threshold) && gate.threshold !== '' ? '門檻 ' + gate.threshold : '';
    const parts = [gate.reason, value, threshold, sample].filter(Boolean);
    return (GATES[gate.name] || gate.name) + (parts.length ? '（' + parts.join('，') + '）' : '');
  }

  function promotionLines(d) {
    if (d.promoted === true && !d.forced) return ['已換上 ' + (d.run_key || '')];
    const head = (d.forced ? '強制換模：' + (d.reason || '沒有寫原因') : '換模門檻未過，續用現役模型') + (d.run_key ? '（' + d.run_key + '）' : '');
    return [head, ...(d.failed || []).map(item => slotText(item.slot) + (item.run_id ? '（' + item.run_id + '）' : '') + '：' + (item.gates || []).map(gateText).join('、'))];
  }

  function alertLine(run, date) {
    const metric = ALERT_METRICS[run.metric] || run.metric || '';
    const threshold = Array.isArray(run.threshold) ? run.threshold.map(v => metricText(run.metric, v)).join('–') :
      run.threshold === null || run.threshold === undefined ? null : metricText(run.metric, run.threshold);
    return run.run_id + '（' + (OBJECTIVES[run.objective] || run.objective) + ' ' + run.horizon + ' 日）：' + metric + (/[A-Za-z]$/.test(metric) ? ' ' : '') + (ALERT_REASONS[run.reason] || run.reason) +
      (run.value === null || run.value === undefined ? '' : '，近 ' + run.days + ' 日 ' + metricText(run.metric, run.value)) +
      (threshold === null ? '' : '，門檻 ' + threshold) +
      [windowText(run.own_rows, run.days, run.mostly_predecessors), modeText(run, date)].filter(Boolean).map(part => '；' + part).join('');
  }

  const namesOf = v => Array.isArray(v) ? v : [];

  const share2 = v => typeof v === 'number' ? (v * 100).toFixed(2) + '%' : '—';

  function coverageLine(entry) {
    const stocks = Number.isInteger(entry.count) ? '（' + fmtInt(entry.count) + ' 檔）' : '';
    return (entry.collapsed ? '崩落：' : '偏低：') + entry.feature + ' 線上有值 ' + share2(entry.share) + stocks +
      '，參考中位數 ' + share2(entry.median) + '、下緣 ' + share2(entry.low) + '、門檻 ' + share2(entry.bound);
  }

  function coverageLines(d) {
    const flagged = namesOf(d.flagged), appeared = namesOf(d.appeared);
    if (d.error) return ['檢查沒有跑成：' + d.error, flagged.length ? '當天先前標記：' + flagged.map(entry => entry.feature).join('、') : ''].filter(Boolean);
    const collapsed = flagged.filter(entry => entry.collapsed), low = flagged.filter(entry => !entry.collapsed);
    const head = collapsed.length ? collapsed.length + ' 個特徵崩落，當天不產出預測' + (low.length ? '；另有 ' + low.length + ' 個偏低' : '') :
      low.length ? low.length + ' 個特徵偏低' : '都在參考範圍內';
    return [
      head + '（判 ' + fmtInt(d.judged || 0) + '／' + fmtInt(d.features || 0) + ' 個特徵）',
      ...collapsed.map(coverageLine),
      ...low.map(coverageLine),
      appeared.length ? '參考期全空、線上有值：' + appeared.join('、') : ''
    ].filter(Boolean);
  }

  function releaseLines(d) {
    const failed = namesOf(d.failed), flagged = namesOf(d.flagged);
    const at = d.released_at ? d.released_at.slice(0, 10) + ' ' + hhmm(d.released_at) + ' 放行' : '';
    return [
      (SCOPES[d.scope] || d.scope || '—') + (d.run_key ? '（' + d.run_key + '）' : '') + '：holdout 判定「' + (d.status || '—') + '」，經使用者放行後建模',
      failed.length ? '未達的門檻：' + failed.join('、') : '',
      flagged.length ? '退化旗標：' + flagged.join('、') : '',
      [at, d.holdout ? 'holdout ' + d.holdout : ''].filter(Boolean).join('，')
    ].filter(Boolean);
  }

  const scopeReasons = (reasons, word) => Object.entries(reasons || {}).map(([scope, reason]) => (SCOPES[scope] || scope) + word + '：' + reason);

  const leftDetail = d => d.handled === 'by_hand' || has(d.limit);
  const handNote = '人工重建後，這列會在 ' + QUALITY_DAYS + ' 個交易日內或出現新的人工重建紀錄時消失';

  function rebuildText(d) {
    if (d.error) return '重建失敗：' + d.error;
    const span = fmtInt(d.stocks || 0) + ' 檔，自 ' + (d.earliest || '—') + ' 起';
    const days = has(d.limit_days) ? d.limit_days : '—';
    switch (d.handled) {
      case 'by_hand': return '最近 ' + days + ' 個交易日以外，未自動重建：' + span;
      case 'covered': return '已由每晚 ' + days + ' 日重算涵蓋';
      case 'rebuilt': return '已重建 ' + span;
      default: return has(d.limit) ? '超過上限，未自動重建：' + span : '重建 ' + span;
    }
  }

  function modelCheckState(check, alertShare, date) {
    const d = check.detail || {};
    const stocks = fmtInt(d.flagged || 0) + '／' + fmtInt(d.n_stocks || 0) + ' 檔';
    switch (check.check_name) {
      case 'output_width_ratio':
        if ((d.share || 0) > alertShare) return ['warn', ['警告，' + stocks]];
        return ['', [d.flagged ? '少量標記，' + stocks : '沒有標記']];
      case 'output_odds_vs_q50': return ['', [d.flagged ? '參考，' + stocks : '沒有標記']];
      case 'monitor_unscored': return ['warn', [fmtInt(sizeOf(d.runs)) + ' 筆']];
      case 'monitor_feature_coverage': return [check.severity === 'info' ? '' : 'warn', coverageLines(d)];
      case 'monitor_ex_date_calendar': return ['warn', [fmtInt(d.count || 0) + ' 筆，最晚 ' + (d.latest_ex_date || '—') + '；交易日曆只到 ' + (d.calendar_end || '—')]];
      case 'model_alert': return ['warn', (d.runs || []).map(run => alertLine(run, date))];
      case 'model_promotion': return [check.severity === 'info' ? '' : 'warn', promotionLines(d)];
      case 'model_holdout_release': return ['warn', releaseLines(d)];
      case 'research_retrain': return [check.severity === 'info' ? '' : 'warn', ['建 ' + keyCount(d.built) + '、拒 ' + keyCount(d.refused) + '、失敗 ' + keyCount(d.failed), ...scopeReasons(d.refused, '拒建'), ...scopeReasons(d.failed, '失敗')]];
      case 'research_rebuild': return [check.severity === 'info' ? '' : 'warn', [rebuildText(d), ...(leftDetail(d) ? [handNote] : [])]];
      case 'research_promote': return ['warn', ['失敗 ' + keyCount(d.failed) + '、完成 ' + sizeOf(d.done), ...Object.entries(d.failed || {}).map(([key, error]) => key + '：' + error)]];
      default: return [check.severity === 'info' ? '' : 'warn', [(SEVERITY[check.severity] || [check.severity])[0]]];
    }
  }

  function modelCheckHtml(check, alertShare, date, note) {
    const [state, lines] = modelCheckState(check, alertShare, date);
    const failed = check.severity === 'error';
    const cls = failed ? 'warn' : state;
    const shown = failed ? ['錯誤，' + (lines[0] || ''), ...lines.slice(1)] : lines;
    const runClass = ['model_alert', 'model_promotion', 'model_holdout_release', 'monitor_feature_coverage'].includes(check.check_name) ? ' run' : '';
    return '<div class="check"><i class="' + cls + '"></i><div><div>' + esc(CHECKS[check.check_name] || check.check_name) + (note ? '　<span class="faint small pinned">' + esc(note) + '</span>' : '') + '</div>' +
      shown.map(line => '<div class="muted small' + runClass + '">' + esc(line) + '</div>').join('') + '</div></div>';
  }

  const isPromotion = check => check.check_name === 'model_promotion';
  const isRelease = check => check.check_name === 'model_holdout_release';
  const isResearch = check => check.check_name.startsWith('research_');
  const isDaily = check => !isPromotion(check) && !isRelease(check) && !isResearch(check);
  const scopeOf = check => (check.detail || {}).scope || (check.detail || {}).run_key;
  const RESEARCH_ORDER = ['research_retrain', 'research_promote', 'research_rebuild', 'research_evening', 'research_morning'];
  const researchRank = check => {
    const rank = RESEARCH_ORDER.indexOf(check.check_name);
    return rank < 0 ? RESEARCH_ORDER.length : rank;
  };

  function latestBy(data, wanted, keyOf) {
    const all = data.flatMap(entry => (entry.model_checks || []).filter(wanted).map(check => ({ date: entry.date, check: check })))
      .sort((a, b) => (b.check.id || 0) - (a.check.id || 0));
    return { total: all.length, shown: all.filter((item, i) => all.findIndex(other => keyOf(other.check) === keyOf(item.check)) === i) };
  }

  const leftByHand = check => check.check_name === 'research_rebuild' && leftDetail(check.detail || {});

  function researchPins(data) {
    const dayOf = top => ((data.find(entry => entry.date === top.date) || {}).model_checks || [])
      .filter(check => check.check_name === top.check.check_name)
      .sort((a, b) => (a.id || 0) - (b.id || 0))
      .map(check => ({ date: top.date, check: check }));
    const pins = latestBy(data, isResearch, check => check.check_name).shown
      .slice().sort((a, b) => researchRank(a.check) - researchRank(b.check))
      .flatMap(dayOf);
    const hand = latestBy(data, leftByHand, () => 'by_hand').shown[0];
    if (!hand || pins.some(item => item.check.id === hand.check.id)) return pins;
    const at = pins.findIndex(item => item.check.check_name === hand.check.check_name);
    return [...pins.slice(0, at), hand, ...pins.slice(at)];
  }

  function renderModelChecks(data, meta) {
    const sub = document.getElementById('model-checks-sub'), host = document.getElementById('model-checks');
    const monitored = check => isDaily(check) && check.check_name !== 'monitor_ex_date_calendar';
    const day = data.find(entry => (entry.model_checks || []).some(monitored)) || data.find(entry => (entry.model_checks || []).some(isDaily));
    const share = meta.output_alert_share;
    sub.textContent = day ? '最近一次寫入的交易日：' + day.date + '；區間寬度比的標記超過 ' + Math.round(share * 100) + '% 才算警告，機率矛盾只做參考。' : '';
    const promotions = latestBy(data, isPromotion, scopeOf);
    const count = promotions.total > promotions.shown.length ? '，近 ' + data.length + ' 個交易日共 ' + promotions.total + ' 筆' : '';
    const research = researchPins(data);
    const pinned = promotions.shown.map(item => modelCheckHtml(item.check, share, item.date, item.date + count)).join('') +
      [...latestBy(data, isRelease, scopeOf).shown, ...research].map(item => modelCheckHtml(item.check, share, item.date, item.date)).join('');
    const rest = day ? day.model_checks.filter(isDaily).map(check => modelCheckHtml(check, share, day.date)).join('') : '';
    host.innerHTML = pinned + rest || '<p class="placeholder">還沒有模型輸出檢查。</p>';
  }

  async function renderQuality() {
    const sub = document.getElementById('quality-sub'), host = document.getElementById('quality'), days = document.getElementById('quality-days');
    try {
      const { data, meta } = await api.get('data-quality', { days: QUALITY_DAYS });
      const latest = data[0];
      sub.textContent = latest.date + '：' + verdictOf(summaryOf(latest)).text;
      host.innerHTML = dayChecksHtml(latest.checks) +
        latest.missing_checks.map(name => '<div class="check"><i class="idle"></i><div><div>' + esc(CHECKS[name] || name) + '</div><div class="muted small">尚未執行</div></div></div>').join('');
      days.innerHTML = '<div class="list">' + data.map(day => {
        const verdict = verdictOf(summaryOf(day));
        return '<div class="trow short qh-cols"><div class="muted">' + esc(day.date) + '</div><div class="res"><i class="' + verdict.cls + '"></i><span>' + esc(verdict.label) + '</span></div>' +
          '<div class="muted small">' + (day.passed === false && day.worst !== 'error' && !day.missing_checks.length ? '模型輸出檢查有錯誤' :
            (day.worst ? '最嚴重：' + esc((SEVERITY[day.worst] || [day.worst])[0]) : '') + (day.missing_checks.length && day.checks.length ? '，缺 ' + day.missing_checks.length + ' 項' : '')) + '</div></div>';
      }).join('') + '</div>';
      renderModelChecks(data, meta);
    } catch (err) {
      sub.textContent = '';
      host.innerHTML = errorBox(err);
      days.innerHTML = '';
      document.getElementById('model-checks-sub').textContent = '';
      document.getElementById('model-checks').innerHTML = errorBox(err);
    }
  }

  const views = { today: document.getElementById('view-today'), stock: stockEl, models: document.getElementById('view-models') };
  function route() {
    let hash;
    try { hash = decodeURIComponent((location.hash || '#today').slice(1)); } catch (e) { hash = 'today'; }
    let name = 'today';
    if (hash === 'models') name = 'models';
    let code = state.stock;
    if (hash === 'stock' || hash.startsWith('stock-')) { name = 'stock'; code = hash.slice('stock-'.length) || state.stock; }
    Object.keys(views).forEach(k => { views[k].hidden = k !== name; });
    document.querySelectorAll('[data-nav]').forEach(a => { if (a.dataset.nav === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    document.querySelector('[data-nav="stock"]').setAttribute('href', '#stock-' + encodeURIComponent(state.stock));
    if (name === 'today') { state.watchCache = null; loadToday(); }
    if (name === 'stock') renderStock(code);
    if (name === 'models') { renderModelTable(); renderIcChart(); renderEtl(); renderQuality(); renderCorrections(); }
    renderStatus();
    window.scrollTo(0, 0);
  }
  document.getElementById('prob-head').title = PROB_TIP;
  bandNoteEl.textContent = BAND_NOTE;
  fallbackNoteEl.textContent = FALLBACK_NOTE;
  window.addEventListener('hashchange', route);
  route();
})();
