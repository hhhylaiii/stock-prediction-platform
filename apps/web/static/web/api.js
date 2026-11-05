(function (global) {
  'use strict';

  class ApiError extends Error {
    constructor(status, code, message) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.code = code;
    }
  }

  function urlFor(path, params) {
    const url = new URL('/api/' + path, global.location.origin);
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') url.searchParams.set(key, String(value));
    });
    return url;
  }

  async function get(path, params) {
    let response;
    try {
      response = await fetch(urlFor(path, params), { headers: { Accept: 'application/json' } });
    } catch (err) {
      throw new ApiError(0, 'network', '無法連線到伺服器。');
    }
    let body = null;
    try {
      body = await response.json();
    } catch (err) {
      body = null;
    }
    if (!body || typeof body !== 'object' || typeof body.success !== 'boolean') {
      throw new ApiError(response.status, 'bad_response', '伺服器回應不是預期的格式（HTTP ' + response.status + '）。');
    }
    if (!body.success) {
      const error = body.error || {};
      throw new ApiError(response.status, error.code || 'error', error.message || '發生錯誤。');
    }
    return { data: body.data, meta: body.meta || {} };
  }

  async function getAll(path, params, pageSize) {
    const rows = [];
    let page = 1;
    let meta = {};
    for (;;) {
      const result = await get(path, Object.assign({}, params, { page: page, page_size: pageSize || 100 }));
      rows.push(...result.data);
      meta = result.meta;
      if (!meta.pages || page >= meta.pages) break;
      page += 1;
    }
    return { data: rows, meta: meta };
  }

  global.TwpApi = { get: get, getAll: getAll, ApiError: ApiError };
})(window);
