(function () {
  'use strict';

  const api = window.TwpApi;
  const STOCK_CODE = /^[0-9A-Za-z]{1,10}$/;
  const PRICE_DAYS = 60;
  const QUALITY_DAYS = 30;
  const SOURCES = { twse: '證交所', tpex: '櫃買中心', finmind: 'FinMind', tdcc: '集保', yahoo: 'Yahoo', datagov: 'data.gov.tw', mops: 'MOPS', stock_master: '股票清單', trading_calendar: '交易日曆', twse_tpex: '證交所／櫃買中心' };
  const RUN_STATUS = { success: ['完成', 'in'], failed: ['失敗', 'warn'], running: ['執行中', 'idle'] };
  const CHECKS = {
    row_count: '各表當日列數',
    big_move: '漲跌幅超過 10% 且無除權息紀錄',
    missing_values: 'standard 表當日缺值',
    trading_calendar: '所有日期都在交易日曆內',
    index_freshness: '市場指數的最新收盤',
    etl_runs: '每日資料集都已更新',
    stock_master: '股票清單異動',
    us_close: '美股收盤',
    output_width_ratio: '區間寬度比異常',
    output_odds_vs_q50: '機率與中位數矛盾',
    monitor_unscored: '到期預測缺 label',
    model_alert: '模型告警'
  };
  const ALERT_REASONS = { below_threshold: '滾動 IC 低於門檻', outside_range: '超出範圍', above_baseline: '輸給 baseline', missing_reference: '缺參考值' };
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

  const state = { h: 20, mkt: 'all', q: '', watchOnly: false, watch: new Set(), stock: '2330', chart: 'candle', stockToken: 0, todayToken: 0, page: 1, watchCache: null };
  try { const c = localStorage.getItem('twp-chart'); if (c === 'line' || c === 'candle') state.chart = c; } catch (e) { state.chart = 'candle'; }
  try { const w = JSON.parse(localStorage.getItem('twp-watch') || '[]'); if (Array.isArray(w)) w.filter(c => STOCK_CODE.test(c)).forEach(c => state.watch.add(c)); } catch (e) { state.watch.clear(); }
  function saveWatch() { try { localStorage.setItem('twp-watch', JSON.stringify([...state.watch])); } catch (e) { return; } }

  const statusEl = document.getElementById('status');
  const summaryOf = day => ({ passed: day.passed, worst: day.worst, missing_checks: day.missing_checks, warnings: day.checks.filter(c => c.severity === 'warning').length });
  const sizeOf = v => Array.isArray(v) ? v.length : Number(v || 0);

  function verdictOf(day) {
    const warnings = day.warnings;
    if (day.worst === 'error') return { label: '未通過', cls: 'warn', text: '有檢查未通過，當日不產出預測。' };
    if (!day.worst) return { label: '尚未執行', cls: 'idle', text: '檢查尚未執行。' };
    if (day.missing_checks.length) return { label: '尚未完成', cls: 'idle', text: '已執行的檢查沒有錯誤，還有 ' + day.missing_checks.length + ' 項尚未執行。' };
    if (day.passed === false) return { label: '未通過', cls: 'warn', text: '模型輸出檢查有錯誤，當日不產出預測。' };
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

  const rowsEl = document.getElementById('rows');
  const countEl = document.getElementById('count');
  const todayMetaEl = document.getElementById('today-meta');
  const prevBtn = document.getElementById('prev-page');
  const nextBtn = document.getElementById('next-page');
  const RANGE_SPAN = { 5: [-8, 12], 20: [-16, 24] };
  const SOURCE_NOTE = { volatility: '波動率區間', tech_vol_20d: '近 20 日波動估計' };
  const MODEL_NAMES = { lightgbm: 'LightGBM', ridge: 'Ridge', xgboost: 'XGBoost' };
  const SEARCH_MAX = 32;
  const pctOf = (x, digits) => x === null || x === undefined ? '—' : sgn(x * 100, digits);
  const priceOf = x => x === null || x === undefined ? '—' : fmtPrice(x);

  function controls() {
    document.querySelectorAll('[data-h]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.h) === state.h)));
    document.querySelectorAll('[data-mkt]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mkt === state.mkt)));
    document.getElementById('ret-head').textContent = state.h + ' 日預測報酬';
    document.getElementById('prob-head').textContent = '漲幅超過 5% 機率' + (state.h === 20 ? '' : '（僅 20 日）');
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
    if (degraded.length) parts.push(degradedRuns + ' 個模型降級中，結果僅供參考' + (degraded.includes('interval') ? '，區間改用波動率區間' : ''));
    return parts.join('，') + '。';
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
    return { rows: rows, meta: latest ? { date: latest.date, universe: latest.universe, models: latest.models, degraded: latest.degraded } : {} };
  }

  function emptyHtml(meta) {
    if (state.q.trim()) return '<div class="empty">沒有符合「' + esc(state.q.trim()) + '」的股票。試試代號或公司名稱的一部分。</div>';
    if (watchingOnly()) return '<div class="empty">追蹤的股票沒有這一期的預測。</div>';
    if (!meta.date) return '<div class="empty" id="today-empty">預測資料尚未產出。模型上線後，這裡會列出全市場的排名。</div>';
    return '<div class="empty">沒有符合條件的股票。</div>';
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

  async function loadToday() {
    const token = ++state.todayToken;
    controls();
    prevBtn.disabled = nextBtn.disabled = true;
    let result;
    try {
      result = await fetchToday();
    } catch (err) {
      if (token !== state.todayToken) return;
      rowsEl.innerHTML = errorBox(err);
      countEl.textContent = todayMetaEl.textContent = '';
      return;
    }
    if (token !== state.todayToken) return;
    const { rows, meta } = result;
    rowsEl.innerHTML = rows.map(todayRowHtml).join('') || emptyHtml(meta);
    todayMetaEl.textContent = todayMetaText(meta);
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

  function quoteHtml(quote) {
    if (!quote) return '<div class="price"><span class="faint small">尚無成交資料</span></div>';
    const change = quote.change === null ? '' :
      '<span class="chg" style="color:' + colorOf(quote.change) + '">' + signed(quote.change, quote.close >= 100 ? 1 : 2) + '（' + sgn(quote.change_ratio * 100, 2) + '）</span>';
    return '<div class="price"><span class="big" id="stock-close">' + fmtPrice(quote.close) + '</span>' + change + '<span class="faint small">' + esc(quote.date) + ' 收盤</span></div>';
  }

  function stockHtml(s) {
    const watching = state.watch.has(s.stock_id);
    const tags = [s.market_label, s.industry, s.delist_date ? s.delist_date + ' 下市' : ''].filter(Boolean);
    return backLink +
      '<div class="stockhead"><div><div class="id"><h1><span class="code">' + esc(s.stock_id) + '</span><span id="stock-name">' + esc(s.name) + '</span></h1>' +
      '<span class="tags">' + tags.map(t => '<span>' + esc(t) + '</span>').join('') + '</span></div>' + quoteHtml(s.quote) + '</div>' +
      '<div class="aside"><div class="kpi"><div class="l">' + state.h + ' 日預測排名</div><div class="v">—<small> 尚未產出</small></div></div>' +
      '<button type="button" class="ghost" id="watch-btn" aria-pressed="' + watching + '" style="height:36px">' + (watching ? '已追蹤' : '加入追蹤') + '</button></div></div>' +
      '<div class="section"><div class="sechead"><div class="control"><h2>過去 ' + PRICE_DAYS + ' 個交易日</h2>' +
      '<div class="seg" role="group" aria-label="圖表類型" id="chart-seg"><button type="button" data-ct="candle" aria-pressed="' + (state.chart === 'candle') + '">K 線</button><button type="button" data-ct="line" aria-pressed="' + (state.chart === 'line') + '">收盤線</button></div></div>' +
      '<div class="legend" id="chart-legend"></div></div><div class="chart" id="price-chart"></div></div>' +
      '<div class="two"><section><h2>這次預測怎麼來的</h2>' + waiting + '</section>' +
      '<section><h2>三類預測</h2>' + waiting + '</section></div>' +
      '<div class="section"><h2>過去對本股的 20 日預測，與實際結果</h2>' + waiting + '</div>';
  }

  async function renderStock(code) {
    const token = ++state.stockToken;
    if (!STOCK_CODE.test(code)) {
      stockEl.innerHTML = backLink + errorBox({ message: '找不到股票 ' + code + '。' });
      return;
    }
    stockEl.innerHTML = backLink + '<p class="placeholder">載入中…</p>';
    const [stock, prices] = await Promise.allSettled([api.get('stocks/' + code), api.get('stocks/' + code + '/prices', { days: PRICE_DAYS })]);
    if (token !== state.stockToken) return;
    if (stock.status === 'rejected') {
      stockEl.innerHTML = backLink + errorBox(stock.reason);
      return;
    }
    const s = stock.value.data;
    state.stock = s.stock_id;
    document.querySelector('[data-nav="stock"]').setAttribute('href', '#stock-' + encodeURIComponent(state.stock));
    stockEl.innerHTML = stockHtml(s);
    document.getElementById('watch-btn').addEventListener('click', e => {
      if (state.watch.has(s.stock_id)) state.watch.delete(s.stock_id); else state.watch.add(s.stock_id);
      saveWatch();
      const on = state.watch.has(s.stock_id);
      e.currentTarget.setAttribute('aria-pressed', String(on));
      e.currentTarget.textContent = on ? '已追蹤' : '加入追蹤';
      controls();
    });
    const drawChart = () => {
      document.querySelectorAll('#chart-seg [data-ct]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.ct === state.chart)));
      document.getElementById('chart-legend').innerHTML = state.chart === 'candle'
        ? '<span><i class="sw-up"></i>收高於開</span><span><i class="sw-down"></i>收低於開</span><span><i class="sw-ma5"></i>5 日均線</span><span><i class="sw-ma20"></i>20 日均線</span>'
        : '<span><i class="sw-line"></i>收盤價（還原）</span>';
      const host = document.getElementById('price-chart');
      if (prices.status === 'rejected') host.innerHTML = errorBox(prices.reason);
      else drawPriceChart(host, s, prices.value.data, state.chart);
    };
    document.querySelectorAll('#chart-seg [data-ct]').forEach(b => b.addEventListener('click', () => {
      state.chart = b.dataset.ct;
      try { localStorage.setItem('twp-chart', state.chart); } catch (e) { state.chart = b.dataset.ct; }
      drawChart();
    }));
    drawChart();
  }

  function drawPriceChart(host, s, bars, kind) {
    const traded = bars.filter(b => b.close !== null);
    if (!traded.length) { host.innerHTML = '<p class="placeholder">這段期間沒有成交資料。</p>'; return; }
    const candle = kind === 'candle';
    const n = bars.length;
    const closes = bars.map(b => b.close);
    const ma5 = ma(closes, 5), ma20 = ma(closes, 20);
    const prevClose = [];
    bars.reduce((last, b, i) => { prevClose[i] = last; return b.close !== null ? b.close : last; }, null);
    const W = 1280, padL = 56, padR = 24, padT = 20, plotH = 300, gapV = 18, volH = 72, padB = 36;
    const volTop = padT + plotH + gapV, volBase = volTop + volH, H = volBase + padB;
    const innerW = W - padL - padR, step = n > 1 ? innerW / (n - 1) : 0;
    const lows = traded.map(b => candle ? b.low : b.close), highs = traded.map(b => candle ? b.high : b.close);
    const vmin = Math.min(...lows), vmax = Math.max(...highs), pad = (vmax - vmin) * 0.08 || vmax * 0.02 || 1;
    const ticks = niceTicks(vmin - pad, vmax + pad, 6);
    const yMin = Math.min(vmin - pad, ticks[0]), yMax = Math.max(vmax + pad, ticks[ticks.length - 1]);
    const x = i => n > 1 ? padL + i * step : padL + innerW / 2;
    const y = v => padT + (yMax - v) / (yMax - yMin) * plotH;
    const lots = bars.map(b => b.volume === null ? null : b.volume / 1000);
    const maxVol = Math.max(1, ...lots.filter(v => v !== null));
    const yv = v => volBase - v / maxVol * volH;
    const labelPrice = v => fmtPrice(v);
    const bw = Math.max(3, Math.round(Math.max(step, 6) * 0.62 * 10) / 10);
    const path = arr => arr.map((v, i) => v === null ? '' : ((i === 0 || arr[i - 1] === null) ? 'M' : 'L') + f1(x(i)) + ' ' + f1(y(v))).join(' ');
    const xl = bars.map((b, i) => ({ i, label: md(b.date) })).filter(t => t.i % 10 === 0 || t.i === n - 1);

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
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(s.name) + '過去 ' + n + ' 個交易日' + (candle ? 'K 線' : '收盤價') + '與成交量">' +
      ticks.filter(v => v >= yMin && v <= yMax).map(v => '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + f1(y(v)) + '" y2="' + f1(y(v)) + '" stroke="#2A2A28"></line><text x="' + (padL - 10) + '" y="' + f1(y(v) + 4) + '" text-anchor="end" fill="#8A887F" font-size="12">' + labelPrice(v) + '</text>').join('') +
      '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + f1(volTop) + '" y2="' + f1(volTop) + '" stroke="#2A2A28"></line>' +
      '<text x="' + (padL - 10) + '" y="' + f1(volTop + 4) + '" text-anchor="end" fill="#8A887F" font-size="12">' + fmtInt(Math.round(maxVol)) + '</text>' +
      '<text x="' + (padL - 10) + '" y="' + f1(volBase) + '" text-anchor="end" fill="#8A887F" font-size="12">張</text>' +
      '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + f1(volBase) + '" y2="' + f1(volBase) + '" stroke="#2A2A28"></line>' +
      xl.map(t => '<text x="' + f1(x(t.i)) + '" y="' + (H - 12) + '" text-anchor="middle" fill="#8A887F" font-size="12">' + t.label + '</text>').join('') +
      priceLayer + volLayer +
      '<g class="hover" style="display:none"><line class="hx" y1="' + padT + '" y2="' + volBase + '" stroke="#6B6A64"></line><circle class="hc" r="4.5" stroke="#1B1B1A" stroke-width="2"></circle></g>' +
      '<rect class="overlay" x="' + padL + '" y="' + padT + '" width="' + innerW + '" height="' + (volBase - padT) + '" fill="transparent"></rect></svg>' +
      '<div class="tip" style="display:none"><div class="t"></div><div class="b"></div></div>';

    const svg = host.querySelector('svg'), overlay = host.querySelector('.overlay'), hover = host.querySelector('.hover');
    const hx = host.querySelector('.hx'), hc = host.querySelector('.hc'), tip = host.querySelector('.tip'), tt = tip.querySelector('.t'), tb = tip.querySelector('.b');
    overlay.addEventListener('pointermove', ev => {
      const rect = svg.getBoundingClientRect();
      const ux = (ev.clientX - rect.left) * (W / rect.width);
      const idx = n > 1 ? Math.max(0, Math.min(n - 1, Math.round((ux - padL) / step))) : 0;
      const d = bars[idx], px = x(idx);
      tt.textContent = d.date.replace(/-/g, '/');
      if (d.close === null) {
        tb.textContent = '無成交資料';
        hc.style.display = 'none';
      } else {
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
      tip.style.transform = idx > n - 12 ? 'translateX(-100%)' : idx < 6 ? 'translateX(0)' : 'translateX(-50%)';
    });
    overlay.addEventListener('pointerleave', () => { hover.style.display = 'none'; tip.style.display = 'none'; });
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
      case 'us_close': return Object.keys(d.stale || {}).length ? '未更新：' + Object.keys(d.stale).join('、') : '全部已更新';
      case 'etl_runs': {
        const blocking = d.blocking || [], rest = (d.missing || []).filter(key => !blocking.includes(key));
        if (!blocking.length && !rest.length) return '全部資料集都已更新';
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

  function checkHtml(check) {
    const sev = SEVERITY[check.severity] || [check.severity, 'idle'];
    return '<div class="check"><i class="' + (check.severity === 'info' ? '' : sev[1]) + '"></i><div><div>' + esc(CHECKS[check.check_name] || check.check_name) + '（' + esc(sev[0]) + '）</div>' +
      '<div class="muted small">' + esc(checkSummary(check)) + '</div></div></div>';
  }

  function modelCheckState(check, alertShare) {
    const d = check.detail || {};
    const stocks = fmtInt(d.flagged || 0) + '／' + fmtInt(d.n_stocks || 0) + ' 檔';
    switch (check.check_name) {
      case 'output_width_ratio':
      case 'output_odds_vs_q50':
        if ((d.share || 0) > alertShare) return ['warn', '警告，' + stocks];
        return ['', d.flagged ? '少量標記，' + stocks : '沒有標記'];
      case 'monitor_unscored': return ['warn', fmtInt(sizeOf(d.runs)) + ' 筆'];
      case 'model_alert': return ['warn', (ALERT_REASONS[d.reason] || String(d.reason || '')) + (d.run_id ? '（' + d.run_id + '）' : '')];
      default: return [check.severity === 'info' ? '' : 'warn', (SEVERITY[check.severity] || [check.severity])[0]];
    }
  }

  function modelCheckHtml(check, alertShare) {
    const [cls, text] = modelCheckState(check, alertShare);
    return '<div class="check"><i class="' + cls + '"></i><div><div>' + esc(CHECKS[check.check_name] || check.check_name) + '</div><div class="muted small">' + esc(text) + '</div></div></div>';
  }

  function renderModelChecks(data, meta) {
    const sub = document.getElementById('model-checks-sub'), host = document.getElementById('model-checks');
    const day = data.find(entry => (entry.model_checks || []).length);
    const share = meta.output_alert_share;
    sub.textContent = day ? '最近一次寫入的交易日：' + day.date + '；區間寬度比與機率矛盾的標記超過 ' + Math.round(share * 100) + '% 才算警告。' : '';
    host.innerHTML = day ? day.model_checks.map(check => modelCheckHtml(check, share)).join('') : '<p class="placeholder">還沒有模型輸出檢查。</p>';
  }

  async function renderQuality() {
    const sub = document.getElementById('quality-sub'), host = document.getElementById('quality'), days = document.getElementById('quality-days');
    try {
      const { data, meta } = await api.get('data-quality', { days: QUALITY_DAYS });
      const latest = data[0];
      sub.textContent = latest.date + '：' + verdictOf(summaryOf(latest)).text;
      host.innerHTML = latest.checks.map(checkHtml).join('') +
        latest.missing_checks.map(name => '<div class="check"><i class="idle"></i><div><div>' + esc(CHECKS[name] || name) + '</div><div class="muted small">尚未執行</div></div></div>').join('');
      days.innerHTML = '<div class="list">' + data.map(day => {
        const verdict = verdictOf(summaryOf(day));
        return '<div class="trow short qh-cols"><div class="muted">' + esc(day.date) + '</div><div class="res"><i class="' + verdict.cls + '"></i><span>' + esc(verdict.label) + '</span></div>' +
          '<div class="muted small">' + (day.worst ? '最嚴重：' + esc((SEVERITY[day.worst] || [day.worst])[0]) : '') + (day.missing_checks.length && day.checks.length ? '，缺 ' + day.missing_checks.length + ' 項' : '') + '</div></div>';
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
    if (name === 'models') { renderEtl(); renderQuality(); }
    renderStatus();
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', route);
  route();
})();
