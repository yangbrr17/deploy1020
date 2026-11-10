/* Charts for password-protected pages.  After a page is unlocked, renderViz(box) looks for
     <a class="viz-data" data-b64="...">        one JSON document with filters, series and chart specs
     <div class="viz-filters"></div>            where the filter row goes
     <div class="viz" data-chart="ID"></div>    one chart each
   and draws SVG charts with a hover read-out, a clickable legend and a table view.  No data lives in this file. */
(function () {
  var NS = 'http://www.w3.org/2000/svg';
  var CSS = [
    '.viz-root{--s0:#52514e;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;--vgrid:#e1e0d9;--vaxis:#c3c2b7;--vmuted:#898781}',
    '@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .viz-root{--s0:#c3c2b7;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--vgrid:#2c2c2a;--vaxis:#383835}}',
    ':root[data-theme="dark"] .viz-root{--s0:#c3c2b7;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--vgrid:#2c2c2a;--vaxis:#383835}',
    '.viz-filters{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:1rem 0;padding:10px 14px;border:1px solid var(--line);border-radius:8px;background:var(--panel)}',
    '.viz-filters label{font-size:.9rem;color:var(--muted);display:flex;gap:6px;align-items:center}',
    '.viz-filters select{font:inherit;font-size:.9rem;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);max-width:100%}',
    '.viz{position:relative;margin:1.2rem 0 1.6rem;padding:14px 16px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg)}',
    '.viz-title{font-weight:600;line-height:1.4}',
    '.viz-sub{font-size:.88rem;color:var(--muted);line-height:1.55;margin:.15rem 0 .5rem}',
    '.viz-legend{display:flex;flex-wrap:wrap;gap:4px 6px;margin:.3rem 0 .2rem}',
    '.viz-chip{font:inherit;font-size:.82rem;line-height:1.3;display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border:1px solid var(--line);border-radius:999px;background:var(--bg);color:var(--ink);cursor:pointer}',
    '.viz-chip[aria-pressed="false"]{color:var(--vmuted);background:transparent}',
    '.viz-chip[aria-pressed="false"] .viz-key{background:transparent!important;box-shadow:inset 0 0 0 1.5px var(--vmuted)}',
    '.viz-chip.static{cursor:default}',
    '.viz-key{display:inline-block;width:14px;height:3px;border-radius:2px;flex:none}',
    '.viz svg{display:block;width:100%;overflow:visible;touch-action:pan-y}',
    '.viz svg text{font:12px -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;fill:var(--vmuted);font-variant-numeric:tabular-nums}',
    '.viz svg text.lab{fill:var(--muted)}',
    '.viz svg text.val{fill:var(--ink)}',
    '.viz-tip{position:absolute;pointer-events:none;z-index:6;min-width:120px;max-width:300px;padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);box-shadow:0 2px 10px rgba(0,0,0,.12);font-size:.82rem;line-height:1.5;display:none}',
    '.viz-tip .h{color:var(--muted);margin-bottom:2px}',
    '.viz-tip .r{display:flex;align-items:center;gap:7px;white-space:nowrap}',
    '.viz-tip .r b{font-variant-numeric:tabular-nums;min-width:3.6em}',
    '.viz-tip .r span.n{color:var(--muted)}',
    '.viz-note{color:var(--muted);font-size:.9rem;padding:1.2rem 0}',
    '.viz details{margin-top:.3rem;font-size:.85rem}',
    '.viz details summary{cursor:pointer;color:var(--muted)}',
    '.viz details table{font-size:.82rem;margin:.4rem 0}',
    '.viz details td,.viz details th{padding:3px 8px;text-align:right}',
    '.viz details td:first-child,.viz details th:first-child{text-align:left}'
  ].join('\n');

  function el(name, attrs, parent, text) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(name, cls, parent, text) {
    var e = document.createElement(name);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(v, d) {
    if (v == null || isNaN(v)) return '—';
    var s = Math.abs(v).toFixed(d == null ? 3 : d);
    return (v < 0 && +s !== 0 ? '−' : '') + s;
  }
  function col(slot) { return 'var(--s' + (slot || 0) + ')'; }
  function ticks(lo, hi, n) {
    if (lo === hi) { lo -= 1; hi += 1; }
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var out = [], t = Math.ceil(lo / step - 1e-9) * step;
    for (; t <= hi + step * 1e-9; t += step) out.push(+t.toFixed(10));
    return { v: out, d: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
  }
  function decodeData(a) {
    var b = atob(a.getAttribute('data-b64')), u = new Uint8Array(b.length);
    for (var i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
    return JSON.parse(new TextDecoder().decode(u));
  }

  function App(box, spec) {
    this.box = box; this.spec = spec; this.state = {}; this.vis = {}; this.charts = [];
    var self = this;
    (spec.filters || []).forEach(function (f) { self.state[f.id] = f.def != null ? f.def : f.options[0].v; });
    for (var k in spec.series || {}) this.vis[k] = spec.series[k].on !== false;
    for (var id in spec.charts) for (var q in spec.charts[id].series || {}) this.vis[q] = spec.charts[id].series[q].on !== false;
    var fr = box.querySelector('.viz-filters');
    if (fr) {
      fr.classList.add('viz-root');
      (spec.filters || []).forEach(function (f) {
        var lab = h('label', '', fr, f.label), sel = h('select', '', lab);
        f.options.forEach(function (o) { var op = h('option', '', sel, o.t); op.value = o.v; });
        sel.value = self.state[f.id];
        sel.addEventListener('change', function () { self.state[f.id] = sel.value; self.draw(); });
      });
    }
    box.querySelectorAll('.viz[data-chart]').forEach(function (node) {
      var c = spec.charts[node.getAttribute('data-chart')];
      if (c) self.charts.push({ node: node, c: c });
    });
    var timer;
    window.addEventListener('resize', function () { clearTimeout(timer); timer = setTimeout(function () { self.draw(); }, 150); });
    this.draw();
  }
  App.prototype.key = function (c) {
    var s = this.state;
    return (c.by && c.by.length) ? c.by.map(function (id) { return s[id]; }).join('|') : '_';
  };
  App.prototype.seriesOf = function (c) { return c.series || this.spec.series || {}; };
  App.prototype.draw = function () {
    var self = this;
    this.charts.forEach(function (ch) { self.card(ch.node, ch.c); });
  };
  App.prototype.card = function (node, c) {
    var self = this, data = (c.data || {})[this.key(c)], S = this.seriesOf(c);
    node.textContent = ''; node.classList.add('viz-root');
    h('div', 'viz-title', node, c.title);
    if (c.sub) h('div', 'viz-sub', node, c.sub);
    if (!data) { h('div', 'viz-note', node, c.missing || '这个组合没有对应的数据。'); return; }
    var keys = c.type === 'rows' || c.type === 'bars' ? [] : Object.keys(S).filter(function (k) { return data[k]; });
    if (keys.length > 1) {
      var lg = h('div', 'viz-legend', node);
      keys.forEach(function (k) {
        var b = h('button', 'viz-chip' + (c.fixed ? ' static' : ''), lg);
        b.type = 'button'; b.setAttribute('aria-pressed', self.vis[k] !== false || c.fixed ? 'true' : 'false');
        var key = h('span', 'viz-key', b); key.style.background = col(S[k].slot);
        h('span', '', b, S[k].name);
        if (!c.fixed) b.addEventListener('click', function () { self.vis[k] = self.vis[k] === false; self.draw(); });
      });
    }
    var shown = keys.filter(function (k) { return c.fixed || self.vis[k] !== false; });
    var tip = h('div', 'viz-tip', node);
    var W = Math.max(280, node.clientWidth - 34), svg;
    if (c.type === 'line') svg = this.line(node, c, data, S, shown, W, tip);
    else if (c.type === 'xy') svg = this.xy(node, c, data, S, shown, W, tip);
    else if (c.type === 'rows') svg = this.rows(node, c, data, W, tip);
    else if (c.type === 'bars') svg = this.bars(node, c, data, W, tip);
    if (svg) { svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', c.title); }
    this.table(node, c, data, S, keys);
  };

  function showTip(tip, node, x, y, build) {
    tip.textContent = ''; build(tip); tip.style.display = 'block';
    var w = tip.offsetWidth, left = x + 14;
    if (left + w > node.clientWidth - 4) left = Math.max(4, x - w - 14);
    tip.style.left = left + 'px'; tip.style.top = Math.max(4, y) + 'px';
  }
  function tipRow(tip, slot, value, name) {
    var r = h('div', 'r', tip), k = h('span', 'viz-key', r);
    k.style.background = col(slot); h('b', '', r, value); h('span', 'n', r, name);
  }
  function frame(node, W, H, m, yT, yOf, c) {
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, height: H }, null);
    node.insertBefore(svg, node.querySelector('.viz-tip'));
    yT.v.forEach(function (t) {
      var y = yOf(t), zero = Math.abs(t) < 1e-12;
      el('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: zero ? 'var(--vaxis)' : 'var(--vgrid)', 'stroke-width': zero ? 1.5 : 1 }, svg);
      el('text', { x: m.l - 8, y: y + 4, 'text-anchor': 'end' }, svg, fmt(t, yT.d));
    });
    if (c.ylab) el('text', { x: 4, y: 12, class: 'lab' }, svg, c.ylab);
    if (c.xlab) el('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'lab' }, svg, c.xlab);
    return svg;
  }

  App.prototype.line = function (node, c, data, S, shown, W, tip) {
    var n = c.x.length, H = c.height || 290, m = { l: 52, r: 16, t: c.ylab ? 26 : 12, b: c.xlab ? 48 : 30 };
    var lo = Infinity, hi = -Infinity;
    shown.forEach(function (k) { data[k].forEach(function (v) { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });
    if (!shown.length) { lo = 0; hi = 1; }
    if (c.zero !== false) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    var pad = (hi - lo) * 0.06 || 0.01, yT = ticks(lo - pad, hi + pad, 5);
    var y0 = Math.min(lo - pad, yT.v[0]), y1 = Math.max(hi + pad, yT.v[yT.v.length - 1]);
    var pw = W - m.l - m.r, ph = H - m.t - m.b, inset = n > 30 ? 0 : 14;
    function xOf(i) { return m.l + inset + (n > 1 ? i / (n - 1) : 0.5) * (pw - 2 * inset); }
    function yOf(v) { return m.t + (1 - (v - y0) / (y1 - y0)) * ph; }
    var svg = frame(node, W, H, m, yT, yOf, c);
    var xt = c.xticks || c.x.map(function (t, i) { return [i, t]; }), lastX = -1e9;
    xt.forEach(function (p) {
      var x = xOf(p[0]);
      if (x - lastX < 44) return;
      lastX = x; el('text', { x: x, y: H - m.b + 18, 'text-anchor': 'middle' }, svg, p[1]);
    });
    var cross = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--vaxis)', 'stroke-width': 1, visibility: 'hidden' }, svg);
    var dots = [];
    shown.forEach(function (k) {
      var d = '', pen = false;
      data[k].forEach(function (v, i) {
        if (v == null) { pen = false; return; }
        d += (pen ? 'L' : 'M') + xOf(i).toFixed(1) + ' ' + yOf(v).toFixed(1); pen = true;
      });
      el('path', { d: d, fill: 'none', stroke: col(S[k].slot), 'stroke-width': S[k].slot ? 2 : 1.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
      if (n <= 20) data[k].forEach(function (v, i) {
        if (v != null) el('circle', { cx: xOf(i), cy: yOf(v), r: 3.5, fill: col(S[k].slot), stroke: 'var(--bg)', 'stroke-width': 1.5 }, svg);
      });
      dots.push(el('circle', { r: 5, fill: col(S[k].slot), stroke: 'var(--bg)', 'stroke-width': 2, visibility: 'hidden' }, svg));
    });
    var hit = el('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent' }, svg);
    function move(ev) {
      var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
      var i = Math.round((px - m.l - inset) / Math.max(1, pw - 2 * inset) * (n - 1));
      i = Math.max(0, Math.min(n - 1, i));
      cross.setAttribute('x1', xOf(i)); cross.setAttribute('x2', xOf(i)); cross.setAttribute('visibility', 'visible');
      shown.forEach(function (k, j) {
        var v = data[k][i];
        dots[j].setAttribute('visibility', v == null ? 'hidden' : 'visible');
        if (v != null) { dots[j].setAttribute('cx', xOf(i)); dots[j].setAttribute('cy', yOf(v)); }
      });
      var nr = node.getBoundingClientRect();
      showTip(tip, node, xOf(i) * r.width / W + r.left - nr.left, ev.clientY - nr.top - 20, function (t) {
        h('div', 'h', t, (c.tipx || c.x)[i]);
        shown.forEach(function (k) {
          var se = c.err && c.err[key(c)] && c.err[key(c)][k];
          tipRow(t, S[k].slot, fmt(data[k][i], c.d) + (se && se[i] != null ? ' ± ' + fmt(se[i], c.d) : ''), S[k].name);
        });
      });
    }
    var self = this;
    function key(cc) { return self.key(cc); }
    function leave() { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dots.forEach(function (d) { d.setAttribute('visibility', 'hidden'); }); }
    hit.addEventListener('pointermove', move); hit.addEventListener('pointerdown', move); hit.addEventListener('pointerleave', leave);
    return svg;
  };

  App.prototype.xy = function (node, c, data, S, shown, W, tip) {
    var H = c.height || 320, m = { l: 52, r: 16, t: c.ylab ? 26 : 12, b: 48 };
    var xl = Infinity, xh = -Infinity, yl = Infinity, yh = -Infinity;
    shown.forEach(function (k) { data[k].forEach(function (p) { xl = Math.min(xl, p[0]); xh = Math.max(xh, p[0]); yl = Math.min(yl, p[1]); yh = Math.max(yh, p[1]); }); });
    if (!shown.length) { xl = yl = -1; xh = yh = 1; }
    if (c.diag) { yl = Math.min(yl, xl); yh = Math.max(yh, xh); }
    var px_ = (xh - xl) * 0.06 || 0.01, py_ = (yh - yl) * 0.06 || 0.01;
    var xT = ticks(xl - px_, xh + px_, 6), yT = ticks(yl - py_, yh + py_, 5);
    var x0 = Math.min(xl - px_, xT.v[0]), x1 = Math.max(xh + px_, xT.v[xT.v.length - 1]);
    var y0 = Math.min(yl - py_, yT.v[0]), y1 = Math.max(yh + py_, yT.v[yT.v.length - 1]);
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    function xOf(v) { return m.l + (v - x0) / (x1 - x0) * pw; }
    function yOf(v) { return m.t + (1 - (v - y0) / (y1 - y0)) * ph; }
    var svg = frame(node, W, H, m, yT, yOf, c);
    xT.v.forEach(function (t) {
      if (Math.abs(t) < 1e-12) el('line', { x1: xOf(t), x2: xOf(t), y1: m.t, y2: H - m.b, stroke: 'var(--vaxis)', 'stroke-width': 1.5 }, svg);
      el('text', { x: xOf(t), y: H - m.b + 18, 'text-anchor': 'middle' }, svg, fmt(t, xT.d));
    });
    if (c.diag) {
      var a = Math.max(x0, y0), b = Math.min(x1, y1);
      el('line', { x1: xOf(a), y1: yOf(a), x2: xOf(b), y2: yOf(b), stroke: 'var(--vmuted)', 'stroke-width': 1 }, svg);
      el('text', { x: xOf(b) - 4, y: yOf(b) + 14, 'text-anchor': 'end', class: 'lab' }, svg, c.diag);
    }
    var pts = [];
    shown.forEach(function (k) {
      var d = data[k].map(function (p, i) { return (i ? 'L' : 'M') + xOf(p[0]).toFixed(1) + ' ' + yOf(p[1]).toFixed(1); }).join('');
      el('path', { d: d, fill: 'none', stroke: col(S[k].slot), 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
      data[k].forEach(function (p, i) {
        var e = el('circle', { cx: xOf(p[0]), cy: yOf(p[1]), r: 4, fill: col(S[k].slot), stroke: 'var(--bg)', 'stroke-width': 2 }, svg);
        pts.push({ k: k, p: p, i: i, x: xOf(p[0]), y: yOf(p[1]), e: e });
      });
    });
    var hit = el('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent' }, svg), cur;
    function move(ev) {
      var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width, py = (ev.clientY - r.top) * W / r.width, best, bd = 1e18;
      pts.forEach(function (q) { var dd = (q.x - px) * (q.x - px) + (q.y - py) * (q.y - py); if (dd < bd) { bd = dd; best = q; } });
      if (!best) return;
      if (cur) cur.e.setAttribute('r', 4);
      cur = best; best.e.setAttribute('r', 6);
      var nr = node.getBoundingClientRect();
      showTip(tip, node, best.x * r.width / W + r.left - nr.left, best.y * r.width / W + r.top - nr.top - 30, function (t) {
        h('div', 'h', t, S[best.k].name + ' · 第 ' + (best.i + 1) + ' 组');
        tipRow(t, S[best.k].slot, fmt(best.p[1], c.d), c.yname || 'y');
        tipRow(t, S[best.k].slot, fmt(best.p[0], c.d), c.xname || 'x');
      });
    }
    hit.addEventListener('pointermove', move); hit.addEventListener('pointerdown', move);
    hit.addEventListener('pointerleave', function () { tip.style.display = 'none'; if (cur) cur.e.setAttribute('r', 4); });
    return svg;
  };

  App.prototype.rows = function (node, c, data, W, tip) {
    var rows = data, rh = 30, labW = Math.min(230, Math.max(120, W * 0.36)), m = { l: labW, r: 52, t: 26, b: 26 };
    var H = m.t + rows.length * rh + m.b, lo = c.ref, hi = c.ref;
    rows.forEach(function (r) { lo = Math.min(lo, r.v); hi = Math.max(hi, r.v); });
    var pad = (hi - lo) * 0.1 || 0.05, xT = ticks(lo - pad, hi + pad, 5), x0 = Math.min(lo - pad, xT.v[0]), x1 = Math.max(hi + pad, xT.v[xT.v.length - 1]);
    function xOf(v) { return m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r); }
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, height: H }, null);
    node.insertBefore(svg, tip);
    xT.v.forEach(function (t) {
      el('line', { x1: xOf(t), x2: xOf(t), y1: m.t, y2: H - m.b, stroke: 'var(--vgrid)', 'stroke-width': 1 }, svg);
      el('text', { x: xOf(t), y: H - m.b + 16, 'text-anchor': 'middle' }, svg, fmt(t, xT.d));
    });
    el('line', { x1: xOf(c.ref), x2: xOf(c.ref), y1: m.t - 6, y2: H - m.b, stroke: 'var(--vmuted)', 'stroke-width': 1.5 }, svg);
    el('text', { x: xOf(c.ref), y: m.t - 10, 'text-anchor': 'middle', class: 'lab' }, svg, c.reflab || '');
    rows.forEach(function (r, i) {
      var y = m.t + (i + 0.5) * rh, g = el('g', { tabindex: 0 }, svg);
      el('text', { x: m.l - 12, y: y + 4, 'text-anchor': 'end', class: 'val' }, g, r.name);
      el('line', { x1: xOf(c.ref), x2: xOf(r.v), y1: y, y2: y, stroke: col(r.slot), 'stroke-width': 2, 'stroke-linecap': 'round', opacity: 0.45 }, g);
      var dot = el('circle', { cx: xOf(r.v), cy: y, r: 5, fill: col(r.slot), stroke: 'var(--bg)', 'stroke-width': 2 }, g);
      el('text', { x: Math.max(xOf(r.v), xOf(c.ref)) + 10, y: y + 4, class: 'val' }, g, fmt(r.v, c.d));
      var hit = el('rect', { x: 0, y: y - rh / 2, width: W, height: rh, fill: 'transparent' }, g);
      function on() { dot.setAttribute('r', 7); showTip(tip, node, xOf(r.v) + 17, y - 6, function (t) { h('div', 'h', t, r.name); tipRow(t, r.slot, fmt(r.v, c.d), c.vname || ''); }); }
      function off() { dot.setAttribute('r', 5); tip.style.display = 'none'; }
      hit.addEventListener('pointerenter', on); hit.addEventListener('pointerleave', off); g.addEventListener('focus', on); g.addEventListener('blur', off);
    });
    return svg;
  };

  App.prototype.bars = function (node, c, data, W, tip) {
    var H = c.height || 240, m = { l: 52, r: 16, t: c.ylab ? 30 : 18, b: c.xlab ? 50 : 32 }, n = data.length;
    var lo = 0, hi = 0;
    data.forEach(function (r) { lo = Math.min(lo, r.v); hi = Math.max(hi, r.v); });
    var pad = (hi - lo) * 0.12, yT = ticks(lo - (lo < 0 ? pad : 0), hi + pad, 4), y0 = Math.min(lo, yT.v[0]), y1 = Math.max(hi + pad, yT.v[yT.v.length - 1]);
    var pw = Math.min(W - m.l - m.r, n * 110), band = pw / n, bw = Math.min(24, band * 0.5);
    function yOf(v) { return m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b); }
    var svg = frame(node, W, H, m, yT, yOf, c);
    if (c.ref != null) {
      el('line', { x1: m.l, x2: m.l + pw, y1: yOf(c.ref), y2: yOf(c.ref), stroke: 'var(--vmuted)', 'stroke-width': 1.5 }, svg);
      el('text', { x: m.l + pw + 4, y: yOf(c.ref) + 4, class: 'lab' }, svg, c.reflab || '');
    }
    data.forEach(function (r, i) {
      var cx = m.l + (i + 0.5) * band, yv = yOf(r.v), yb = yOf(0), top = Math.min(yv, yb), hh = Math.abs(yv - yb), rr = Math.min(4, hh);
      var g = el('g', { tabindex: 0 }, svg), x = cx - bw / 2, up = r.v >= 0;
      var d = up ? 'M' + x + ' ' + yb + 'V' + (top + rr) + 'Q' + x + ' ' + top + ' ' + (x + rr) + ' ' + top + 'H' + (x + bw - rr) + 'Q' + (x + bw) + ' ' + top + ' ' + (x + bw) + ' ' + (top + rr) + 'V' + yb + 'Z'
                 : 'M' + x + ' ' + yb + 'V' + (yv - rr) + 'Q' + x + ' ' + yv + ' ' + (x + rr) + ' ' + yv + 'H' + (x + bw - rr) + 'Q' + (x + bw) + ' ' + yv + ' ' + (x + bw) + ' ' + (yv - rr) + 'V' + yb + 'Z';
      var bar = el('path', { d: d, fill: col(c.slot || 1) }, g);
      if (n <= 6) el('text', { x: cx, y: up ? top - 6 : yv + 15, 'text-anchor': 'middle', class: 'val' }, g, fmt(r.v, c.d) + (c.unit || ''));
      el('text', { x: cx, y: H - m.b + 18, 'text-anchor': 'middle' }, g, r.name);
      var hit = el('rect', { x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b, fill: 'transparent' }, g);
      function on() { bar.setAttribute('opacity', 0.8); showTip(tip, node, cx + 17, top - 10, function (t) { h('div', 'h', t, r.name); tipRow(t, c.slot || 1, fmt(r.v, c.d) + (c.unit || ''), c.vname || ''); if (r.note) h('div', 'h', t, r.note); }); }
      function off() { bar.removeAttribute('opacity'); tip.style.display = 'none'; }
      hit.addEventListener('pointerenter', on); hit.addEventListener('pointerleave', off); g.addEventListener('focus', on); g.addEventListener('blur', off);
    });
    return svg;
  };

  App.prototype.table = function (node, c, data, S, keys) {
    var det = h('details', '', node); h('summary', '', det, '用表格看这张图的数据');
    var tb = h('table', '', det), tr = h('tr', '', h('thead', '', tb)), body = h('tbody', '', tb);
    if (c.type === 'rows' || c.type === 'bars') {
      h('th', '', tr, c.rowname || ''); h('th', '', tr, c.vname || '数值');
      data.forEach(function (r) { var x = h('tr', '', body); h('td', '', x, r.name); h('td', '', x, fmt(r.v, c.d) + (c.unit || '')); });
    } else if (c.type === 'xy') {
      h('th', '', tr, '组');
      keys.forEach(function (k) { h('th', '', tr, S[k].name + '：' + (c.xname || 'x')); h('th', '', tr, c.yname || 'y'); });
      for (var i = 0; i < data[keys[0]].length; i++) {
        var x = h('tr', '', body); h('td', '', x, String(i + 1));
        keys.forEach(function (k) { h('td', '', x, fmt(data[k][i][0], c.d)); h('td', '', x, fmt(data[k][i][1], c.d)); });
      }
    } else {
      h('th', '', tr, c.xname || ''); keys.forEach(function (k) { h('th', '', tr, S[k].name); });
      var step = c.x.length > 40 ? Math.ceil(c.x.length / 24) : 1;
      for (var j = 0; j < c.x.length; j += step) {
        var y = h('tr', '', body); h('td', '', y, (c.tipx || c.x)[j]);
        keys.forEach(function (k) { h('td', '', y, fmt(data[k][j], c.d)); });
      }
    }
  };

  window.renderViz = function (box) {
    var a = box.querySelector('a.viz-data[data-b64]');
    if (!a) return;
    if (!document.getElementById('viz-css')) { var st = document.createElement('style'); st.id = 'viz-css'; st.textContent = CSS; document.head.appendChild(st); }
    try { new App(box, decodeData(a)); } catch (e) { console.error('viz', e); }
  };
})();
