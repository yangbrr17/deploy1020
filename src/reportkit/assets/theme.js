/* reportkit theme v2 — runtime for report pages. No dependencies, no network.
   On DOMContentLoaded it
     1. builds the sticky section index (.rk-nav) from the section heads;
     2. draws every <div class="rk-chart"> from the JSON spec in its child <script type="application/json">;
        types: line, bar, waterfall, dumbbell, hist, heatmap, scatter, box, multiples (+ sparklines .rk-spark);
     3. adds tooltips (hover, arrow keys) and the "数据" table view to every chart panel;
     4. adds copy buttons to code and math, renders math with KaTeX when it is on the page;
     5. wires the theme / print buttons; before printing, lays the sheet out at A4 width and redraws.
   House style (docs/report-theme/README.md): value axis on the right, solid hairline grid, ink zero line,
   last values in coloured tags at the right edge, legend in the panel head, units in the subtitle.
   Series colours follow the series name (or "key") across the report. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var doc = document, root = doc.documentElement;

  /* ================================================================ utilities */
  function svg(tag, attrs, parent) {
    var e = doc.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function h(tag, cls, parent, text) {
    var e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, attrs) {
    var t = svg('text', Object.assign({ x: r1(x), y: r1(y) }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function r1(v) { return Math.round(v * 10) / 10; }
  function px(v) { return Math.round(v) + 0.5; }           // crisp 1px lines
  function nums(a) { return (a || []).filter(isNum); }
  function minOf(a) { return Math.min.apply(null, a); }
  function maxOf(a) { return Math.max.apply(null, a); }

  var canvas, fontFamily;
  function textWidth(s, size, weight) {
    canvas = canvas || doc.createElement('canvas');
    fontFamily = fontFamily || getComputedStyle(doc.body).fontFamily;
    var c = canvas.getContext('2d');
    c.font = (weight || 400) + ' ' + (size || 11) + 'px ' + fontFamily;
    return c.measureText(String(s)).width;
  }
  function ellipsize(s, maxW, size, weight) {
    s = String(s);
    if (textWidth(s, size, weight) <= maxW) return s;
    while (s.length > 1 && textWidth(s + '…', size, weight) > maxW) s = s.slice(0, -1);
    return s + '…';
  }

  /* ================================================================ numbers */
  // mini-language: "auto", ".2f", ",.0f", "+.1f", ".1%", "+.1%", ".2s" (K/M/B), ".2e", "int"
  function compact(v, p) {
    var a = Math.abs(v), u = '', d = 1;
    if (a >= 1e9) { u = 'B'; d = 1e9; } else if (a >= 1e6) { u = 'M'; d = 1e6; } else if (a >= 1e3) { u = 'K'; d = 1e3; }
    var s = (v / d).toFixed(p == null ? (a / d >= 100 ? 0 : 1) : p);
    if (p == null) s = s.replace(/\.0$/, '');
    return s + u;
  }
  function group(s) { var p = s.split('.'); p[0] = p[0].replace(/\B(?=(\d{3})+(?!\d))/g, ','); return p.join('.'); }
  function fmt(v, f) {
    if (v == null || (typeof v === 'number' && !isFinite(v))) return '–';
    if (typeof v !== 'number') return String(v);
    f = f || 'auto';
    var s;
    if (f === 'auto') {
      var a = Math.abs(v);
      if (a === 0) s = '0';
      else if (a >= 1e4) s = compact(v);
      else if (a >= 100) s = group(v.toFixed(0));
      else if (a >= 1) s = String(+v.toFixed(2));
      else s = String(+v.toPrecision(3));
    } else if (f === 'int') {
      s = group(Math.round(v).toFixed(0));
    } else {
      var m = /^([+])?(,)?(?:\.(\d+))?([f%se])?$/.exec(f);
      if (!m) return String(v);
      var p = m[3] == null ? null : +m[3], t = m[4] || 'f';
      if (t === '%') s = (v * 100).toFixed(p == null ? 0 : p) + '%';
      else if (t === 's') s = compact(v, p);
      else if (t === 'e') s = v.toExponential(p == null ? 2 : p);
      else { s = p == null ? String(v) : v.toFixed(p); if (m[2]) s = group(s); }
      if (m[1] && v > 0) s = '+' + s;
    }
    return s.replace(/^-/, '−').replace('e-', 'e−');
  }
  function signed(v, f) { return fmt(v, '+' + String(f || '.3f').replace(/^\+/, '')); }

  /* ================================================================ scales */
  function tickStep(lo, hi, n) {
    var raw = (hi - lo) / Math.max(1, n), p = Math.pow(10, Math.floor(Math.log10(raw))), e = raw / p;
    return (e >= 7.07 ? 10 : e >= 3.16 ? 5 : e >= 1.41 ? 2 : 1) * p;
  }
  function clean(v) { return +v.toPrecision(12); }
  function linear(d0, d1, r0, rr, opts) {
    opts = opts || {};
    var lo = d0, hi = d1;
    if (!(hi > lo)) { var pad = lo === 0 ? 1 : Math.abs(lo) * 0.1; lo -= pad; hi += pad; }
    var step = tickStep(lo, hi, opts.ticks || 5);
    if (opts.nice !== false) { lo = Math.floor(clean(lo / step)) * step; hi = Math.ceil(clean(hi / step)) * step; }
    var f = function (v) { return r0 + (v - lo) / (hi - lo) * (rr - r0); };
    f.domain = [lo, hi]; f.step = step;
    f.ticks = function () { var out = []; for (var t = Math.ceil(clean(lo / step)) * step; t <= hi + step * 1e-9; t += step) out.push(clean(t)); return out; };
    f.invert = function (p) { return lo + (p - r0) / (rr - r0) * (hi - lo); };
    return f;
  }
  function logScale(d0, d1, r0, rr) {
    var a = Math.log10(d0), b = Math.log10(d1);
    if (!(b > a)) b = a + 1;
    var f = function (v) { return r0 + (Math.log10(v) - a) / (b - a) * (rr - r0); };
    f.domain = [d0, d1]; f.log = true;
    f.ticks = function () {
      var out = [], dec = b - a;
      for (var k = Math.floor(a); k <= Math.ceil(b); k++) [1, 2, 5].forEach(function (m) {
        var v = clean(m * Math.pow(10, k));
        if ((m === 1 || dec <= 2) && v >= d0 * (1 - 1e-9) && v <= d1 * (1 + 1e-9)) out.push(v);
      });
      return out;
    };
    f.invert = function (p) { return Math.pow(10, a + (p - r0) / (rr - r0) * (b - a)); };
    return f;
  }
  function decimalsFor(step) { return Math.max(0, -Math.floor(Math.log10(step) + 1e-9)); }
  // axis labels: decimals from the tick step; keep %, compact and sign semantics of the value format
  function tickFmt(scale, f) {
    if (scale.log) return function (v) { return Math.abs(v) >= 1e4 ? fmt(v, 's') : group(String(v)); };
    f = f || '';
    var sign = f.charAt(0) === '+' ? '+' : '';
    if (/%$/.test(f)) { var dp = decimalsFor(scale.step * 100); return function (v) { return fmt(v, sign + '.' + dp + '%'); }; }
    if (/s$/.test(f)) return function (v) { return fmt(v, 's'); };
    var d = decimalsFor(scale.step);
    return function (v) { return Math.abs(v) >= 1e4 ? fmt(v, 's') : fmt(v, sign + ',.' + d + 'f'); };
  }

  /* ================================================================ colours */
  var registry = {}, nextSlot = 1;
  function slotVar(s) { return s === 'gray' ? 'var(--rk-gray)' : s === 'ink' ? 'var(--rk-ink)' : 'var(--rk-' + s + ')'; }
  // colour follows the series name (or key) across the report; one new series alone takes slot 1 without
  // claiming it; series.color ("s1".."s8", "gray", "ink") always wins; emphasis greys out the others
  function assignColors(series, emphasis) {
    var multi = series.length > 1;
    series.forEach(function (s) {
      if (s.color) { s._c = slotVar(s.color); return; }
      if (emphasis && emphasis.indexOf(s.name) < 0 && emphasis.indexOf(s.key) < 0) { s._c = slotVar('gray'); s._muted = true; return; }
      var key = s.key || s.name;
      if (registry[key]) { s._c = slotVar(registry[key]); return; }
      if (!multi && !emphasis) { s._c = slotVar('s1'); return; }
      if (nextSlot <= 8) { registry[key] = 's' + nextSlot++; s._c = slotVar(registry[key]); } else s._c = slotVar('gray');
    });
  }
  function specSeries(sp) { return sp ? (sp.series || (sp.a ? [sp.a, sp.b] : sp.groups) || []) : []; }
  function isDark() {
    if (root.getAttribute('data-print') === '1') return false;
    var t = root.getAttribute('data-theme');
    if (t) return t === 'dark';
    return !!(window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches);
  }
  function resolve(c) {                       // "var(--rk-s1)" -> "#2154d8"
    var m = /var\((--[\w-]+)\)/.exec(c);
    return m ? getComputedStyle(root).getPropertyValue(m[1]).trim() || '#888888' : c;
  }
  function hex2rgb(x) { x = x.replace('#', ''); if (x.length === 3) x = x.replace(/./g, '$&$&'); return [0, 2, 4].map(function (i) { return parseInt(x.substr(i, 2), 16) / 255; }); }
  function lin(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function delin(c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; }
  function toOk(hex) {
    var c = hex2rgb(hex).map(lin);
    var l = Math.cbrt(0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2]);
    var m = Math.cbrt(0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2]);
    var s = Math.cbrt(0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2]);
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
  }
  function fromOk(o) {
    var l = Math.pow(o[0] + 0.3963377774 * o[1] + 0.2158037573 * o[2], 3), m = Math.pow(o[0] - 0.1055613458 * o[1] - 0.0638541728 * o[2], 3), s = Math.pow(o[0] - 0.0894841775 * o[1] - 1.2914855480 * o[2], 3);
    var rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
    return '#' + rgb.map(function (c) { var v = Math.round(clamp(delin(clamp(c, 0, 1)), 0, 1) * 255); return (v < 16 ? '0' : '') + v.toString(16); }).join('');
  }
  function ramp(stops) {
    var ok = stops.map(toOk);
    return function (t) {
      t = clamp(t, 0, 1) * (ok.length - 1);
      var i = Math.min(Math.floor(t), ok.length - 2), u = t - i, a = ok[i], b = ok[i + 1];
      return fromOk([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]);
    };
  }
  function luminance(hex) { var c = hex2rgb(hex).map(lin); return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; }
  function inkOn(hex) { var L = luminance(hex); return 1.05 / (L + 0.05) >= (L + 0.05) / 0.0545 ? '#ffffff' : '#111111'; }
  function heatScale(kind) {
    var dark = isDark();
    if (kind === 'sequential') return ramp(dark ? ['#101a33', '#1f3f8f', '#3e71e9', '#9db8f5'] : ['#f2f5fc', '#b9cbf3', '#5f86e6', '#2154d8', '#13348c']);
    return ramp(dark ? ['#3e71e9', '#1c1c1c', '#e34480'] : ['#2154d8', '#f1f1ef', '#e34480']);
  }

  /* ================================================================ chrome: tooltip, legend, axes */
  function tooltip(box) {
    var tip = h('div', 'rk-tip', box);
    tip.setAttribute('role', 'status');
    return {
      show: function (x, y, head, rows) {
        tip.textContent = '';
        if (head != null && head !== '') h('div', 'rk-tip-h', tip, head);
        rows.forEach(function (r) {
          var row = h('div', 'rk-tip-r', tip);
          if (r.color) { var k = h('i', '', row); k.style.background = r.color; if (r.rect) { k.style.height = '9px'; k.style.width = '9px'; } }
          h('b', r.tone || '', row, r.value);
          if (r.name) h('span', '', row, r.name);
        });
        tip.classList.add('on');
        var W = box.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
        tip.style.left = clamp(x + 14 + tw > W ? x - tw - 14 : x + 14, 0, Math.max(0, W - tw)) + 'px';
        tip.style.top = clamp(y - th / 2, -6, box.clientHeight - th + 6) + 'px';
      },
      hide: function () { tip.classList.remove('on'); }
    };
  }
  // the legend lives in the panel head (so all heads of a row stay one height); outside a panel, above the chart
  function legendSlot(box) {
    var panel = box.closest('.rk-panel'), host = panel ? panel.querySelector('.rk-ph') : box.parentNode;
    var old = host && host.querySelector('.rk-legend');
    if (old) old.parentNode.removeChild(old);
    return panel ? host : null;
  }
  function legend(box, items) {
    var host = legendSlot(box);
    if (!items.length) return;
    var lg = h('div', 'rk-legend');
    items.forEach(function (it) {
      var s = h('span', '', lg), k = h('i', 'rk-k-' + it.kind, s);
      if (it.kind !== 'ref') k.style.background = it.color;
      s.appendChild(doc.createTextNode(it.name));
    });
    if (host) host.appendChild(lg); else box.parentNode.insertBefore(lg, box);
  }
  function frame(box, W, H) {
    var el = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', tabindex: 0 }, box);
    return { el: el, grid: svg('g', {}, el), marks: svg('g', {}, el), over: svg('g', {}, el), hov: svg('g', { 'pointer-events': 'none' }, el) };
  }
  // horizontal gridlines across [x0, x1] and labels right of x1; labels near an `avoid` y (value tags) are skipped
  function rightAxis(g, y, x0, x1, f, avoid) {
    y.ticks().forEach(function (t) {
      var py = y(t);
      svg('line', { x1: x0, x2: x1, y1: px(py), y2: px(py), 'class': 'rk-grid' }, g);
      if ((avoid || []).some(function (a) { return Math.abs(a - py) < 15; })) return;
      txt(g, x1 + 7, py + 3.5, f(t), { 'text-anchor': 'start' });
    });
  }
  function bottomAxis(g, pos, ticks, y0, label, x0, x1) {
    var last = -1e9;
    ticks.forEach(function (t, i) {
      var p = pos(t), s = label(t, i), w = textWidth(s);
      if (p < x0 - 1 || p > x1 + 1 || p - w / 2 < last + 10) return;
      var anchor = 'middle', tx = p;
      if (p - w / 2 < x0 - 2) { anchor = 'start'; tx = Math.max(p, x0); }
      svg('line', { x1: px(p), x2: px(p), y1: y0, y2: y0 + 4, 'class': 'rk-tick' }, g);
      txt(g, tx, y0 + 16, s, { 'text-anchor': anchor });
      last = anchor === 'start' ? tx + w : p + w / 2;
    });
  }
  function baseline(g, x0, x1, y) { svg('line', { x1: x0, x2: x1, y1: px(y), y2: px(y), 'class': 'rk-base' }, g); }
  function xTitle(g, spec, x1, H) { if (spec.xLabel) txt(g, x1, H - 1, spec.xLabel, { 'class': 'rk-ax', 'text-anchor': 'end' }); }
  // value tags at the right edge, dodged so they never overlap
  function dodge(items, lo, hi, gap) {
    items.sort(function (a, b) { return a.y - b.y; });
    for (var i = 1; i < items.length; i++) if (items[i].y < items[i - 1].y + gap) items[i].y = items[i - 1].y + gap;
    var over = items.length ? items[items.length - 1].y - hi : 0;
    if (over > 0) items.forEach(function (t) { t.y -= over; });
    items.forEach(function (t) { t.y = Math.max(lo, t.y); });
    return items;
  }
  function drawTag(g, x, y, w, color, text) {
    var c = resolve(color);
    svg('rect', { x: r1(x), y: r1(y - 8), width: r1(w), height: 16, fill: c }, g);
    txt(g, x + 5, y + 4, text, { 'class': 'rk-tag', style: 'fill:' + inkOn(c) });
  }
  function pointer(e, el, W) { var r = el.getBoundingClientRect(), k = W / r.width; return [(e.clientX - r.left) * k, (e.clientY - r.top) * k, k]; }
  function scaleK(el, W) { return el.getBoundingClientRect().width / W; }
  function keyNav(el, n, show, clear, step2) {
    var cur = -1;
    el.addEventListener('keydown', function (e) {
      var d = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: step2 || 1, ArrowUp: -(step2 || 1) }[e.key];
      if (!d || !n) return;
      e.preventDefault(); cur = clamp(cur + d, 0, n - 1); show(cur);
    });
    el.addEventListener('blur', clear);
    el.addEventListener('pointerleave', clear);
  }
  function yTicks(H, top, bottom) { return Math.max(3, Math.round((H - bottom - top) / 50)); }
  function tickWidth(y, f) { return maxOf(y.ticks().map(function (t) { return textWidth(f(t)); })); }

  /* ================================================================ line (time series) */
  function renderLine(box, spec, W, H) {
    var series = (spec.series || []).map(function (s) { return Object.assign({}, s); });
    assignColors(series, spec.emphasis == null ? null : [].concat(spec.emphasis));
    var xs = spec.x || [], cat = xs.some(function (v) { return typeof v === 'string'; });
    var yv = [];
    series.forEach(function (s) { yv = yv.concat(nums(s.values)); });
    (spec.refs || []).forEach(function (r) { if (isNum(r.y)) yv.push(r.y); });
    var ymin = spec.yMin != null ? spec.yMin : minOf(yv), ymax = spec.yMax != null ? spec.yMax : maxOf(yv);
    if (spec.zero) { ymin = Math.min(0, ymin); ymax = Math.max(0, ymax); }
    var top = 8, bottom = 24 + (spec.xLabel ? 14 : 0);
    var y = linear(ymin, ymax, H - bottom, top, { nice: !(spec.yMin != null && spec.yMax != null), ticks: yTicks(H, top, bottom) });
    var yf = tickFmt(y, spec.yFormat), vf = spec.yFormat || 'auto';
    var tags = [];
    if (spec.tags !== false && series.length <= 6) series.forEach(function (s) {
      if (s._muted) return;
      for (var i = s.values.length - 1; i >= 0; i--) if (isNum(s.values[i])) { tags.push({ s: s, i: i, v: s.values[i], y: y(s.values[i]), text: fmt(s.values[i], vf) }); break; }
    });
    var tagW = tags.length ? maxOf(tags.map(function (t) { return textWidth(t.text, 11, 700); })) + 10 : 0;
    var x0 = 2, x1 = W - Math.max(tickWidth(y, yf) + 12, tagW + 6) - 2;
    var x;
    if (cat) { var band = (x1 - x0) / xs.length; x = function (i) { return x0 + band * (i + 0.5); }; }
    else {
      var xv = nums(xs), xmin = spec.xMin != null ? spec.xMin : minOf(xv), xmax = spec.xMax != null ? spec.xMax : maxOf(xv);
      x = spec.xLog ? logScale(xmin, xmax, x0, x1) : linear(xmin, xmax, x0, x1, { nice: false, ticks: Math.max(2, Math.floor((x1 - x0) / 80)) });
    }
    var X = function (i) { return cat ? x(i) : x(xs[i]); };
    var lg = series.length > 1 ? series.map(function (s) { return { name: s.name, color: s._c, kind: 'line' }; }) : [];
    if (typeof spec.diagonal === 'string') lg.push({ name: spec.diagonal, kind: 'ref' });
    (spec.refs || []).forEach(function (r) { if (r.legend && r.label) lg.push({ name: r.label, kind: 'ref' }); });
    if (lg.length && series.length === 1) lg.unshift({ name: series[0].name, color: series[0]._c, kind: 'line' });
    legend(box, lg);
    var F = frame(box, W, H), y0 = H - bottom;
    dodge(tags, top + 8, y0 - 8, 17);
    rightAxis(F.grid, y, x0, x1, yf, tags.map(function (t) { return t.y; }));
    if (y.domain[0] < 0 && y.domain[1] > 0) baseline(F.grid, x0, x1, y(0));
    baseline(F.grid, x0, x1, y0);
    if (cat) {
      var maxw = maxOf(xs.map(function (s) { return textWidth(s); })), every = Math.max(1, Math.ceil((maxw + 12) / ((x1 - x0) / xs.length)));
      bottomAxis(F.grid, x, xs.map(function (_, i) { return i; }).filter(function (i) { return i % every === 0; }), y0, function (i) { return String(xs[i]); }, x0, x1);
    } else bottomAxis(F.grid, x, x.ticks(), y0, tickFmt(x, spec.xFormat), x0, x1);
    xTitle(F.grid, spec, x1, H);
    if (spec.diagonal) {
      var lo = Math.max(y.domain[0], x.domain ? x.domain[0] : 0), hi = Math.min(y.domain[1], x.domain ? x.domain[1] : 1);
      svg('line', { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), 'class': 'rk-refl' }, F.marks);
    }
    (spec.refs || []).forEach(function (r) {
      if (isNum(r.y)) { svg('line', { x1: x0, x2: x1, y1: y(r.y), y2: y(r.y), 'class': 'rk-refl' }, F.marks); if (r.label && !r.legend) txt(F.marks, x0 + 4, y(r.y) - 5, r.label, { 'class': 'rk-dl2' }); }
      else if (isNum(r.x)) { var p = x(r.x); svg('line', { x1: p, x2: p, y1: top, y2: y0, 'class': 'rk-refl' }, F.marks); if (r.label && !r.legend) txt(F.marks, p + 4, top + 10, r.label, { 'class': 'rk-dl2' }); }
    });
    series.slice().sort(function (a, b) { return (b._muted ? 1 : 0) - (a._muted ? 1 : 0); }).forEach(function (s) {
      var d = '', pen = false;
      s.values.forEach(function (v, i) { if (!isNum(v)) { pen = false; return; } d += (pen ? 'L' : 'M') + r1(X(i)) + ',' + r1(y(v)); pen = true; });
      if (s.area || (spec.area && series.length === 1)) {
        var b0 = y(clamp(0, y.domain[0], y.domain[1]));
        d.split('M').filter(Boolean).forEach(function (sg) {
          var ps = sg.split('L'), a = ps[0].split(','), b = ps[ps.length - 1].split(',');
          svg('path', { d: 'M' + a[0] + ',' + b0 + 'L' + sg + 'L' + b[0] + ',' + b0 + 'Z', style: 'fill:' + s._c, 'fill-opacity': 0.12, 'class': 'rk-mark' }, F.marks);
        });
      }
      svg('path', { d: d, fill: 'none', style: 'stroke:' + s._c, 'stroke-width': s._muted ? 1.5 : (s.width || 2), 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dash ? '5 4' : null, 'class': 'rk-mark' }, F.marks);
      if (spec.markers || s.markers) s.values.forEach(function (v, i) {
        if (isNum(v)) svg('circle', { cx: r1(X(i)), cy: r1(y(v)), r: 3.2, style: 'fill:' + s._c + ';stroke:var(--rk-bg);stroke-width:1.5' }, F.marks);
      });
    });
    (spec.labels || []).forEach(function (l) {
      var lx = x(l.x), ly = y(l.y);
      svg('circle', { cx: lx, cy: ly, r: 4, style: 'fill:var(--rk-ink);stroke:var(--rk-bg);stroke-width:2' }, F.over);
      txt(F.over, lx + (l.dx != null ? l.dx : 8), ly + (l.dy != null ? l.dy : -8), l.text, { 'class': 'rk-dl', 'text-anchor': l.anchor || 'start' });
    });
    tags.forEach(function (t) { drawTag(F.over, x1 + 4, t.y, tagW, t.s._c, t.text); });
    var tip = tooltip(box), n = xs.length;
    function show(i) {
      F.hov.textContent = '';
      var p = X(i), rows = [];
      svg('line', { x1: px(p), x2: px(p), y1: top, y2: y0, 'class': 'rk-cross' }, F.hov);
      series.forEach(function (s) {
        var v = s.values[i];
        if (isNum(v)) svg('circle', { cx: p, cy: y(v), r: 4, style: 'fill:' + s._c + ';stroke:var(--rk-bg);stroke-width:2' }, F.hov);
        rows.push({ color: s._c, value: fmt(v, vf), name: series.length > 1 ? s.name : null, y: isNum(v) ? v : -Infinity });
      });
      rows.sort(function (a, b) { return b.y - a.y; });
      var k = scaleK(F.el, W);
      tip.show(p * k, (top + (y0 - top) * 0.3) * k, (spec.xName ? spec.xName + ' ' : '') + (cat ? xs[i] : fmt(xs[i], spec.xFormat)), rows);
    }
    function clear() { tip.hide(); F.hov.textContent = ''; }
    F.el.addEventListener('pointermove', function (e) {
      var q = pointer(e, F.el, W);
      if (q[0] < x0 - 8 || q[0] > x1 + 8) { clear(); return; }
      var best = 0, bd = Infinity;
      for (var i = 0; i < n; i++) { var d = Math.abs(X(i) - q[0]); if (d < bd) { bd = d; best = i; } }
      show(best);
    });
    keyNav(F.el, n, show, clear);
  }

  /* ================================================================ bar: columns, horizontal ranking, stacked */
  function barPath(x, y, w, hh, r, end) {
    r = Math.max(0, Math.min(r, w / 2, hh / 2));
    if (end === 'top') return 'M' + x + ',' + (y + hh) + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + (y + hh) + 'Z';
    if (end === 'bottom') return 'M' + x + ',' + y + 'V' + (y + hh - r) + 'Q' + x + ',' + (y + hh) + ' ' + (x + r) + ',' + (y + hh) + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + (y + hh) + ' ' + (x + w) + ',' + (y + hh - r) + 'V' + y + 'Z';
    if (end === 'right') return 'M' + x + ',' + y + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + (y + hh - r) + 'Q' + (x + w) + ',' + (y + hh) + ' ' + (x + w - r) + ',' + (y + hh) + 'H' + x + 'Z';
    if (end === 'left') return 'M' + (x + w) + ',' + y + 'H' + (x + r) + 'Q' + x + ',' + y + ' ' + x + ',' + (y + r) + 'V' + (y + hh - r) + 'Q' + x + ',' + (y + hh) + ' ' + (x + r) + ',' + (y + hh) + 'H' + (x + w) + 'Z';
    return 'M' + x + ',' + y + 'h' + w + 'v' + hh + 'h' + (-w) + 'Z';
  }
  function renderBar(box, spec, W, H) {
    var cats = spec.categories || [], series = (spec.series || []).map(function (s) { return Object.assign({}, s); });
    var single = series.length === 1, stacked = !!spec.stacked && !single, tone = spec.tone === 'sign' && single;
    assignColors(series, null);
    var hl = spec.highlight ? [].concat(spec.highlight) : null;
    function colorOf(s, ci) {
      var v = s.values[ci];
      if (tone) return v >= 0 ? 'var(--rk-up)' : 'var(--rk-down)';
      if (hl && single) return hl.indexOf(cats[ci]) >= 0 ? s._c : 'var(--rk-gray)';
      return s._c;
    }
    var vals = [];
    cats.forEach(function (_, ci) {
      if (stacked) { var p = 0, q = 0; series.forEach(function (s) { var v = s.values[ci]; if (isNum(v)) { if (v >= 0) p += v; else q += v; } }); vals.push(p, q); }
      else series.forEach(function (s) { if (isNum(s.values[ci])) vals.push(s.values[ci]); });
    });
    var vmin = spec.min != null ? spec.min : Math.min(0, minOf(vals)), vmax = spec.max != null ? spec.max : Math.max(0, maxOf(vals));
    var vf = spec.valueFormat || 'auto', nS = stacked ? 1 : series.length, gap = 2;
    var labelsOn = spec.labels != null ? spec.labels : (single || stacked) && cats.length <= 14;
    var catW = cats.map(function (c) { return textWidth(c, 11.5); });
    var horiz = spec.horizontal;
    // horizontal when names are long and do not fit under the columns; short period labels (D1, 2024Q1) thin out instead
    if (horiz == null) horiz = maxOf(catW.concat([0])) > Math.max(48, (W - 50) / Math.max(1, cats.length) - 8);
    var lg = single ? [] : series.map(function (s) { return { name: s.name, color: s._c, kind: 'rect' }; });
    if (tone) lg = [{ name: '为正', color: 'var(--rk-up)', kind: 'rect' }, { name: '为负', color: 'var(--rk-down)', kind: 'rect' }];
    legend(box, lg);
    var F, tip = tooltip(box), marks = [];
    if (!horiz) {
      var top = labelsOn ? 18 : 8, bottom = 24;
      var y = linear(vmin, vmax, H - bottom, top, { ticks: yTicks(H, top, bottom) }), yf = tickFmt(y, spec.valueFormat);
      var x0 = 2, x1 = W - tickWidth(y, yf) - 14, band = (x1 - x0) / cats.length;
      var bw = Math.min(single || stacked ? 30 : 18, (band * 0.7 - gap * (nS - 1)) / nS);
      F = frame(box, W, H);
      rightAxis(F.grid, y, x0, x1, yf);
      var every = Math.max(1, Math.ceil((maxOf(catW) + 10) / band));
      cats.forEach(function (c, ci) {
        var cx = x0 + band * (ci + 0.5), gw = bw * nS + gap * (nS - 1), bx = cx - gw / 2;
        if (ci % every === 0) txt(F.grid, cx, H - bottom + 16, c, { 'text-anchor': 'middle', 'class': 'rk-cat' });
        var acc = { p: 0, n: 0 }, grp = [];
        series.forEach(function (s, si) {
          var v = s.values[ci]; if (!isNum(v)) return;
          var xx = stacked ? bx : bx + si * (bw + gap), a, b;
          if (stacked) { if (v >= 0) { a = acc.p; acc.p += v; b = acc.p; } else { a = acc.n; acc.n += v; b = acc.n; } } else { a = 0; b = v; }
          var ya = y(a), yb = y(b), t0 = Math.min(ya, yb), hh = Math.abs(ya - yb);
          if (stacked && si > 0) { if (v >= 0) hh -= 1; else { t0 += 1; hh -= 1; } }
          grp.push(svg('path', { d: barPath(r1(xx), r1(t0), r1(bw), Math.max(0, r1(hh)), 1.5, v >= 0 ? 'top' : 'bottom'), style: 'fill:' + colorOf(s, ci), 'class': 'rk-mark' }, F.marks));
          if (labelsOn && !stacked) txt(F.over, xx + bw / 2, v >= 0 ? yb - 5 : yb + 13, fmt(v, vf), { 'text-anchor': 'middle', 'class': 'rk-dl' });
        });
        if (labelsOn && stacked) txt(F.over, cx, y(acc.p) - 5, fmt(acc.p + acc.n, vf), { 'text-anchor': 'middle', 'class': 'rk-dl' });
        marks.push({ hit: svg('rect', { x: x0 + band * ci, y: top, width: band, height: H - bottom - top, 'class': 'rk-hit' }, F.over), grp: grp, ci: ci, cx: cx, cy: top + 20 });
      });
      baseline(F.grid, x0, x1, y(0));
    } else {
      var topH = 2, bottomH = spec.axis ? 22 : 4;
      var rowH = clamp((H - topH - bottomH) / Math.max(1, cats.length), 18, 40);
      var need = topH + bottomH + rowH * cats.length;
      if (need > H + 1) { H = Math.ceil(need); box.style.minHeight = H + 'px'; }
      var bwH = Math.min(single || stacked ? 15 : 11, (rowH * 0.72 - gap * (nS - 1)) / nS);
      var labW = Math.min(maxOf(catW.concat([0])) + 12, W * 0.38);
      var tipW = labelsOn ? maxOf(vals.map(function (v) { return textWidth(fmt(v, vf), 11.5, 700); })) + 10 : 8;
      var xa = labW, xb = W - tipW;
      var xs = linear(vmin, vmax, xa, xb, { ticks: Math.max(2, Math.floor((xb - xa) / 90)), nice: !!spec.axis });
      var y0 = topH + rowH * cats.length;
      F = frame(box, W, H);
      if (spec.axis) xs.ticks().forEach(function (t) { var p = px(xs(t)); svg('line', { x1: p, x2: p, y1: topH, y2: y0, 'class': 'rk-grid' }, F.grid); txt(F.grid, p, y0 + 15, tickFmt(xs, spec.valueFormat)(t), { 'text-anchor': 'middle' }); });
      cats.forEach(function (c, ci) {
        var cy = topH + rowH * (ci + 0.5), gh = bwH * nS + gap * (nS - 1), by = cy - gh / 2;
        txt(F.grid, labW - 10, cy + 4, ellipsize(c, labW - 12, 11.5), { 'text-anchor': 'end', 'class': 'rk-cat' });
        if (ci > 0) svg('line', { x1: 0, x2: W, y1: px(topH + rowH * ci), y2: px(topH + rowH * ci), 'class': 'rk-grid' }, F.grid);
        var acc = { p: 0, n: 0 }, grp = [];
        series.forEach(function (s, si) {
          var v = s.values[ci]; if (!isNum(v)) return;
          var yy = stacked ? by : by + si * (bwH + gap), a, b;
          if (stacked) { if (v >= 0) { a = acc.p; acc.p += v; b = acc.p; } else { a = acc.n; acc.n += v; b = acc.n; } } else { a = 0; b = v; }
          var p0 = xs(a), p1 = xs(b), l = Math.min(p0, p1), w = Math.abs(p1 - p0);
          if (stacked && si > 0) { if (v >= 0) { l += 1; w -= 1; } else w -= 1; }
          grp.push(svg('path', { d: barPath(r1(l), r1(yy), Math.max(0, r1(w)), r1(bwH), 1.5, v >= 0 ? 'right' : 'left'), style: 'fill:' + colorOf(s, ci), 'class': 'rk-mark' }, F.marks));
          if (labelsOn && !stacked) txt(F.over, v >= 0 ? p1 + 5 : p1 - 5, yy + bwH / 2 + 4, fmt(v, vf), { 'class': 'rk-dl', 'text-anchor': v >= 0 ? 'start' : 'end' });
        });
        if (labelsOn && stacked) txt(F.over, xs(acc.p) + 5, cy + 4, fmt(acc.p + acc.n, vf), { 'class': 'rk-dl' });
        marks.push({ hit: svg('rect', { x: 0, y: topH + rowH * ci, width: W, height: rowH, 'class': 'rk-hit' }, F.over), grp: grp, ci: ci, cy: cy });
      });
      svg('line', { x1: px(xs(0)), x2: px(xs(0)), y1: topH, y2: y0, 'class': 'rk-base' }, F.grid);
      (spec.refs || []).forEach(function (r) { if (isNum(r.x)) { svg('line', { x1: xs(r.x), x2: xs(r.x), y1: topH, y2: y0, 'class': 'rk-refl' }, F.marks); if (r.label) txt(F.marks, xs(r.x) + 4, y0 - 4, r.label, { 'class': 'rk-dl2' }); } });
    }
    function show(m) {
      F.el.classList.add('rk-hovering');
      marks.forEach(function (o) { o.grp.forEach(function (p) { p.classList.toggle('rk-dimmed', o !== m); }); });
      var rows = series.map(function (s) { return { color: colorOf(s, m.ci), rect: true, value: fmt(s.values[m.ci], vf), name: single ? null : s.name }; });
      if (stacked) rows.push({ value: fmt(series.reduce(function (a, s) { return a + (isNum(s.values[m.ci]) ? s.values[m.ci] : 0); }, 0), vf), name: '合计' });
      var k = scaleK(F.el, W);
      tip.show((m.cx != null ? m.cx : W * 0.55) * k, m.cy * k, String(cats[m.ci]), rows);
    }
    function clear() { tip.hide(); F.el.classList.remove('rk-hovering'); marks.forEach(function (o) { o.grp.forEach(function (p) { p.classList.remove('rk-dimmed'); }); }); }
    marks.forEach(function (m) { m.hit.addEventListener('pointerenter', function () { show(m); }); });
    keyNav(F.el, marks.length, function (i) { show(marks[i]); }, clear);
  }

  /* ================================================================ waterfall (bridge) */
  function renderWaterfall(box, spec, W, H) {
    var steps = spec.steps || [], vf = spec.valueFormat || 'auto', better = spec.better || 'up';
    var run = 0, bars = [], floor = spec.yMin != null ? spec.yMin : 0;
    steps.forEach(function (s) {
      if (s.kind === 'total') { run = s.value; bars.push({ s: s, a: floor, b: s.value, total: true }); }
      else { bars.push({ s: s, a: run, b: run + s.value, total: false }); run += s.value; }
    });
    var all = []; bars.forEach(function (b) { all.push(b.a, b.b); });
    var lo = spec.yMin != null ? spec.yMin : Math.min(0, minOf(all)), hi = spec.yMax != null ? spec.yMax : maxOf(all);
    var top = 18, bottom = 24;
    var y = linear(lo, hi, H - bottom, top, { nice: spec.yMin == null, ticks: yTicks(H, top, bottom) }), yf = tickFmt(y, vf);
    var x0 = 2, x1 = W - tickWidth(y, yf) - 14, band = (x1 - x0) / steps.length, bw = Math.min(46, band * 0.62);
    var upC = better === 'up' ? 'var(--rk-up)' : 'var(--rk-down)', dnC = better === 'up' ? 'var(--rk-down)' : 'var(--rk-up)';
    legend(box, [{ name: '增加', color: upC, kind: 'rect' }, { name: '减少', color: dnC, kind: 'rect' }, { name: '合计', color: 'var(--rk-ink-2)', kind: 'rect' }]);
    var F = frame(box, W, H), tip = tooltip(box), marks = [], y0 = H - bottom;
    rightAxis(F.grid, y, x0, x1, yf);
    baseline(F.grid, x0, x1, y0);
    bars.forEach(function (b, i) {
      var cx = x0 + band * (i + 0.5), l = cx - bw / 2, ya = y(b.a), yb = y(b.b), t0 = Math.min(ya, yb), hh = Math.max(1.5, Math.abs(ya - yb));
      var color = b.total ? (b.s.color ? slotVar(b.s.color) : 'var(--rk-ink-2)') : (b.b >= b.a ? upC : dnC);
      svg('rect', { x: r1(l), y: r1(t0), width: r1(bw), height: r1(hh), style: 'fill:' + color, 'class': 'rk-mark' }, F.marks);
      if (b.total && floor > 0) {             // truncated axis: a break mark at the foot of each total bar
        var yb0 = y0 - 8, zz = 'M' + r1(l - 1) + ',' + r1(yb0 + 2);
        for (var z = 1; z <= 4; z++) zz += 'L' + r1(l - 1 + (bw + 2) * z / 4) + ',' + r1(yb0 + (z % 2 ? -2 : 2));
        svg('path', { d: zz, fill: 'none', style: 'stroke:var(--rk-bg)', 'stroke-width': 3 }, F.marks);
      }
      if (i < bars.length - 1) svg('line', { x1: r1(l + bw), x2: r1(x0 + band * (i + 1.5) - bw / 2), y1: px(yb), y2: px(yb), 'class': 'rk-conn' }, F.marks);
      txt(F.over, cx, t0 - 5, b.total ? fmt(b.b, vf) : signed(b.s.value, vf === 'auto' ? '.3f' : vf), { 'text-anchor': 'middle', 'class': 'rk-dl', style: b.total ? null : 'fill:' + color });
      txt(F.grid, cx, y0 + 16, ellipsize(b.s.label, band - 4, 11.5), { 'text-anchor': 'middle', 'class': 'rk-cat' });
      marks.push({ i: i, cx: cx, cy: t0, hit: svg('rect', { x: x0 + band * i, y: top, width: band, height: y0 - top, 'class': 'rk-hit' }, F.over) });
    });
    function show(m) {
      var b = bars[m.i], k = scaleK(F.el, W);
      tip.show(m.cx * k, (m.cy + 24) * k, b.s.label, b.total ? [{ value: fmt(b.b, vf), name: '合计' }] :
        [{ value: signed(b.s.value, vf === 'auto' ? '.3f' : vf), name: '变化' }, { value: fmt(b.b, vf), name: '累计' }]);
    }
    marks.forEach(function (m) { m.hit.addEventListener('pointerenter', function () { show(m); }); });
    keyNav(F.el, marks.length, function (i) { show(marks[i]); }, function () { tip.hide(); });
  }

  /* ================================================================ dumbbell (two values per row) */
  function renderDumbbell(box, spec, W, H) {
    var cats = spec.categories || [], A = Object.assign({}, spec.a), B = Object.assign({}, spec.b), vf = spec.valueFormat || 'auto';
    if (spec.a.color) assignColors([A, B], null); else { A._c = 'var(--rk-gray)'; assignColors([B], null); }
    var all = nums(A.values).concat(nums(B.values));
    var labW = Math.min(maxOf(cats.map(function (c) { return textWidth(c, 11.5); })) + 12, W * 0.36);
    var valW = maxOf(all.map(function (v) { return textWidth(fmt(v, vf), 11.5, 700); })) + 12;
    var topH = 2, bottomH = 22, rowH = clamp((H - topH - bottomH) / Math.max(1, cats.length), 18, 40);
    var need = topH + bottomH + rowH * cats.length; if (need > H + 1) { H = Math.ceil(need); box.style.minHeight = H + 'px'; }
    var xs = linear(spec.min != null ? spec.min : minOf(all), spec.max != null ? spec.max : maxOf(all), labW + 6, W - valW, { ticks: Math.max(2, Math.floor((W - labW - valW) / 90)) });
    legend(box, [{ name: A.name, color: A._c, kind: 'dot' }, { name: B.name, color: B._c, kind: 'dot' }]);
    var F = frame(box, W, H), tip = tooltip(box), y0 = topH + rowH * cats.length, marks = [], xf = tickFmt(xs, vf);
    xs.ticks().forEach(function (t) { var p = px(xs(t)); svg('line', { x1: p, x2: p, y1: topH, y2: y0, 'class': 'rk-grid' }, F.grid); txt(F.grid, p, y0 + 15, xf(t), { 'text-anchor': 'middle' }); });
    baseline(F.grid, labW, W - valW + 4, y0);
    cats.forEach(function (c, i) {
      var cy = topH + rowH * (i + 0.5), a = A.values[i], b = B.values[i];
      txt(F.grid, labW - 8, cy + 4, ellipsize(c, labW - 10, 11.5), { 'text-anchor': 'end', 'class': 'rk-cat' });
      if (isNum(a) && isNum(b)) svg('line', { x1: xs(a), x2: xs(b), y1: cy, y2: cy, style: 'stroke:var(--rk-line-2)', 'stroke-width': 3 }, F.marks);
      if (isNum(a)) svg('circle', { cx: r1(xs(a)), cy: r1(cy), r: 5, style: 'fill:' + A._c + ';stroke:var(--rk-bg);stroke-width:1.5' }, F.marks);
      if (isNum(b)) svg('circle', { cx: r1(xs(b)), cy: r1(cy), r: 5, style: 'fill:' + B._c + ';stroke:var(--rk-bg);stroke-width:1.5' }, F.marks);
      var far = Math.max(isNum(a) ? a : -Infinity, isNum(b) ? b : -Infinity);
      if (isNum(b)) txt(F.over, xs(far) + 10, cy + 4, fmt(b, vf), { 'class': 'rk-dl' });
      marks.push({ i: i, cy: cy, hit: svg('rect', { x: 0, y: topH + rowH * i, width: W, height: rowH, 'class': 'rk-hit' }, F.over) });
    });
    function show(m) {
      var a = A.values[m.i], b = B.values[m.i];
      tip.show(W * 0.5 * scaleK(F.el, W), m.cy * scaleK(F.el, W), cats[m.i], [{ color: A._c, value: fmt(a, vf), name: A.name }, { color: B._c, value: fmt(b, vf), name: B.name },
        { value: isNum(a) && isNum(b) ? signed(b - a, vf === 'auto' ? '.3f' : vf) : '–', name: '差' }]);
    }
    marks.forEach(function (m) { m.hit.addEventListener('pointerenter', function () { show(m); }); });
    keyNav(F.el, marks.length, function (i) { show(marks[i]); }, function () { tip.hide(); });
  }

  /* ================================================================ histogram */
  function renderHist(box, spec, W, H) {
    var edges = spec.edges, nb = edges.length - 1, series = (spec.series || []).map(function (s) { return Object.assign({}, s); });
    assignColors(series, null);
    var vals = []; series.forEach(function (s) { vals = vals.concat(nums(s.counts)); });
    var top = 8, bottom = 24 + (spec.xLabel ? 14 : 0);
    var y = linear(0, maxOf(vals), H - bottom, top, { ticks: yTicks(H, top, bottom) }), yf = tickFmt(y, spec.yFormat);
    var x0 = 2, x1 = W - tickWidth(y, yf) - 14;
    var x = spec.xLog ? logScale(edges[0], edges[nb], x0, x1) : linear(edges[0], edges[nb], x0, x1, { nice: false, ticks: Math.max(2, Math.floor((x1 - x0) / 80)) });
    var lg = series.length > 1 ? series.map(function (s) { return { name: s.name, color: s._c, kind: 'line' }; }) : [];
    (spec.refs || []).forEach(function (r) { if (r.legend && r.label) lg.push({ name: r.label, kind: 'ref' }); });
    legend(box, lg);
    var F = frame(box, W, H), y0 = H - bottom;
    rightAxis(F.grid, y, x0, x1, yf);
    bottomAxis(F.grid, x, x.ticks(), y0, tickFmt(x, spec.xFormat), x0, x1);
    xTitle(F.grid, spec, x1, H);
    series.forEach(function (s) {
      if (series.length === 1) {
        s.counts.forEach(function (v, i) {
          if (!isNum(v) || v <= 0) return;
          var a = x(edges[i]) + 0.5, b = x(edges[i + 1]) - 0.5;
          svg('rect', { x: r1(a), y: r1(y(v)), width: r1(Math.max(0.6, b - a)), height: r1(y0 - y(v)), style: 'fill:' + s._c, 'class': 'rk-mark' }, F.marks);
        });
      } else {
        var d = 'M' + r1(x(edges[0])) + ',' + r1(y0);
        s.counts.forEach(function (v, i) { d += 'V' + r1(y(isNum(v) ? v : 0)) + 'H' + r1(x(edges[i + 1])); });
        d += 'V' + r1(y0);
        svg('path', { d: d + 'Z', style: 'fill:' + s._c, 'fill-opacity': 0.12, 'class': 'rk-mark' }, F.marks);
        svg('path', { d: d, fill: 'none', style: 'stroke:' + s._c, 'stroke-width': 1.75, 'class': 'rk-mark' }, F.marks);
      }
    });
    baseline(F.grid, x0, x1, y0);
    (spec.refs || []).forEach(function (r) {
      if (!isNum(r.x)) return;
      var p = x(r.x);
      svg('line', { x1: p, x2: p, y1: top, y2: y0, 'class': 'rk-refl' }, F.over);
      if (r.label && !r.legend) txt(F.over, p + 4, top + 9, r.label, { 'class': 'rk-dl2' });
    });
    var tip = tooltip(box), xf = spec.xFormat || 'auto';
    function show(i) {
      F.hov.textContent = '';
      var a = x(edges[i]), b = x(edges[i + 1]);
      svg('rect', { x: a, y: top, width: Math.max(1, b - a), height: y0 - top, style: 'fill:var(--rk-ink)', 'fill-opacity': 0.06 }, F.hov);
      var k = scaleK(F.el, W);
      tip.show((a + b) / 2 * k, (top + (y0 - top) * 0.3) * k, '[' + fmt(edges[i], xf) + ', ' + fmt(edges[i + 1], xf) + ')',
        series.map(function (s) { return { color: s._c, rect: series.length === 1, value: fmt(s.counts[i], spec.yFormat), name: series.length > 1 ? s.name : null }; }));
    }
    function clear() { tip.hide(); F.hov.textContent = ''; }
    F.el.addEventListener('pointermove', function (e) {
      var q = pointer(e, F.el, W), v = x.invert(q[0]);
      if (v < edges[0] || v >= edges[nb]) { clear(); return; }
      for (var i = 0; i < nb; i++) if (v < edges[i + 1]) { show(i); return; }
    });
    keyNav(F.el, nb, show, clear);
  }

  /* ================================================================ heatmap */
  function renderHeat(box, spec, W, H) {
    var rows = spec.rows, cols = spec.cols, M = spec.values, kind = spec.scale || 'diverging', vf = spec.valueFormat || '.2f';
    var all = []; M.forEach(function (r) { all = all.concat(nums(r)); });
    var dom = spec.domain || (kind === 'diverging' ? (function () { var a = maxOf(all.map(Math.abs)); return [-a, a]; })() : [minOf(all), maxOf(all)]);
    var col = heatScale(kind), t = function (v) { return (v - dom[0]) / (dom[1] - dom[0]); };
    legendSlot(box);
    var labW = Math.min(maxOf(rows.map(function (r) { return textWidth(r, 11.5); })) + 12, W * 0.3);
    var cw = (W - labW) / cols.length, colW = maxOf(cols.map(function (c) { return textWidth(c, 11.5); }));
    var many = cols.length > 12, rot = !many && colW > cw - 6;
    var colLab = cols.map(function (c, j) { return rot ? ellipsize(c, Math.min(110, (W - labW - cw * (j + 0.5)) / 0.7071 - 4), 11.5) : c; });
    var topH = many ? 4 : rot ? maxOf(colLab.map(function (c) { return textWidth(c, 11.5); })) * 0.7071 + 14 : 20;
    var scaleH = 26, xLabH = many ? 18 : 0;
    var ch = (H - topH - scaleH - xLabH) / rows.length;
    if (ch < 15) { ch = 15; H = Math.ceil(topH + scaleH + xLabH + ch * rows.length); box.style.minHeight = H + 'px'; }
    ch = Math.min(ch, spec.cellHeight || 44);
    var gridH = ch * rows.length;
    var F = frame(box, W, H), tip = tooltip(box), cells = [];
    if (many) {
      var every = Math.ceil((colW + 8) / cw);
      cols.forEach(function (c, j) { if (j % every === 0) txt(F.grid, labW + cw * (j + 0.5), topH + gridH + 14, c, { 'text-anchor': 'middle' }); });
    } else cols.forEach(function (c, j) {
      var cx = labW + cw * (j + 0.5);
      if (rot) txt(F.grid, cx - 2, topH - 6, colLab[j], { 'class': 'rk-cat', transform: 'rotate(-45 ' + r1(cx - 2) + ' ' + r1(topH - 6) + ')' });
      else txt(F.grid, cx, topH - 7, c, { 'text-anchor': 'middle', 'class': 'rk-cat' });
    });
    var labMax = maxOf(all.map(function (v) { return textWidth(fmt(v, vf), 10.5, 600); }));
    var showLab = spec.labels !== false && labMax < cw - 6 && ch >= 15;
    rows.forEach(function (r, i) {
      var cy = topH + ch * i;
      txt(F.grid, labW - 8, cy + ch / 2 + 4, ellipsize(r, labW - 10, 11.5), { 'text-anchor': 'end', 'class': 'rk-cat' });
      cols.forEach(function (c, j) {
        var v = M[i][j], fill = isNum(v) ? col(t(v)) : 'none';
        var cell = svg('rect', { x: r1(labW + cw * j), y: r1(cy), width: r1(cw - 1), height: r1(ch - 1), fill: fill, 'class': 'rk-mark' }, F.marks);
        if (!isNum(v)) cell.setAttribute('style', 'stroke:var(--rk-line)');
        if (showLab && isNum(v)) txt(F.marks, labW + cw * (j + 0.5) - 0.5, cy + ch / 2 + 3.5, fmt(v, vf), { 'text-anchor': 'middle', 'class': 'rk-in', style: 'fill:' + inkOn(fill) });
        cells.push({ el: cell, i: i, j: j, x: labW + cw * (j + 0.5), y: cy + ch / 2 });
      });
    });
    var sy = topH + gridH + xLabH + 11, sw = Math.min(160, Math.max(60, W - labW - 120)), sx = labW + 34;
    var gid = 'rkg' + Math.random().toString(36).slice(2, 8), defs = svg('defs', {}, F.el), lgr = svg('linearGradient', { id: gid }, defs);
    for (var k = 0; k <= 10; k++) svg('stop', { offset: (k * 10) + '%', 'stop-color': col(k / 10) }, lgr);
    svg('rect', { x: sx, y: sy, width: sw, height: 7, fill: 'url(#' + gid + ')' }, F.grid);
    txt(F.grid, sx - 6, sy + 7, fmt(dom[0], vf), { 'text-anchor': 'end' });
    txt(F.grid, sx + sw + 6, sy + 7, fmt(dom[1], vf) + (spec.scaleLabel ? '   ' + spec.scaleLabel : ''), {});
    function show(c) {
      cells.forEach(function (o) { o.el.setAttribute('stroke', o === c ? 'var(--rk-ink)' : 'none'); o.el.setAttribute('stroke-width', o === c ? 1.5 : 0); });
      var kk = scaleK(F.el, W);
      tip.show(c.x * kk, c.y * kk, rows[c.i] + ' × ' + cols[c.j], [{ value: fmt(M[c.i][c.j], vf), name: spec.valueName || null }]);
    }
    function clear() { tip.hide(); cells.forEach(function (o) { o.el.setAttribute('stroke', 'none'); }); }
    cells.forEach(function (c) { c.el.addEventListener('pointerenter', function () { show(c); }); });
    keyNav(F.el, cells.length, function (i) { show(cells[i]); }, clear, cols.length);
  }

  /* ================================================================ scatter (+ regression) */
  function renderScatter(box, spec, W, H) {
    var series = (spec.series || []).map(function (s) { return Object.assign({}, s); });
    assignColors(series, null);
    var xv = [], yv = [];
    series.forEach(function (s) { s.x.forEach(function (v, i) { if (isNum(v) && isNum(s.y[i])) { xv.push(v); yv.push(s.y[i]); } }); });
    var top = spec.yLabel ? 18 : 8, bottom = 24 + (spec.xLabel ? 14 : 0);
    var lo = spec.yMin != null ? spec.yMin : minOf(yv), hi = spec.yMax != null ? spec.yMax : maxOf(yv);
    var xlo = spec.xMin != null ? spec.xMin : minOf(xv), xhi = spec.xMax != null ? spec.xMax : maxOf(xv);
    if (spec.diagonal && spec.square !== false) { lo = xlo = Math.min(lo, xlo); hi = xhi = Math.max(hi, xhi); }
    var y = linear(lo, hi, H - bottom, top, { ticks: yTicks(H, top, bottom) }), yf = tickFmt(y, spec.yFormat);
    var x0 = 6, x1 = W - tickWidth(y, yf) - 14;
    var x = linear(xlo, xhi, x0, x1, { ticks: Math.max(2, Math.floor((x1 - x0) / 80)) });
    var lg = series.length > 1 ? series.map(function (s) { return { name: s.name, color: s._c, kind: 'dot' }; }) : [];
    if (typeof spec.diagonal === 'string') lg.push({ name: spec.diagonal, kind: 'ref' });
    if (spec.fit) lg.push({ name: '最小二乘拟合', color: series.length === 1 ? series[0]._c : 'var(--rk-ink-2)', kind: 'line' });
    legend(box, lg);
    var F = frame(box, W, H), y0 = H - bottom;
    rightAxis(F.grid, y, x0, x1, yf);
    if (spec.yLabel) txt(F.grid, x0, 10, '↑ ' + spec.yLabel, { 'class': 'rk-ax' });
    bottomAxis(F.grid, x, x.ticks(), y0, tickFmt(x, spec.xFormat), x0, x1);
    baseline(F.grid, x0, x1, y0);
    xTitle(F.grid, spec, x1, H);
    if (spec.diagonal) { var a = Math.max(x.domain[0], y.domain[0]), b = Math.min(x.domain[1], y.domain[1]); svg('line', { x1: x(a), y1: y(a), x2: x(b), y2: y(b), 'class': 'rk-refl' }, F.marks); }
    var pts = [], notes = [];
    series.forEach(function (s) {
      s.x.forEach(function (v, i) {
        if (!isNum(v) || !isNum(s.y[i])) return;
        var cx = r1(x(v)), cy = r1(y(s.y[i]));
        svg('circle', { cx: cx, cy: cy, r: spec.r || 4, style: 'fill:' + s._c + ';stroke:var(--rk-bg);stroke-width:1.25', 'fill-opacity': spec.opacity || 0.9, 'class': 'rk-mark' }, F.marks);
        pts.push({ x: cx, y: cy, s: s, i: i });
      });
      if (spec.fit) {
        var X = [], Y = []; s.x.forEach(function (v, i) { if (isNum(v) && isNum(s.y[i])) { X.push(v); Y.push(s.y[i]); } });
        var n = X.length; if (n < 3) return;
        var mx = X.reduce(function (p, q) { return p + q; }, 0) / n, my = Y.reduce(function (p, q) { return p + q; }, 0) / n, sxy = 0, sxx = 0, syy = 0;
        for (var k = 0; k < n; k++) { sxy += (X[k] - mx) * (Y[k] - my); sxx += (X[k] - mx) * (X[k] - mx); syy += (Y[k] - my) * (Y[k] - my); }
        var b1 = sxy / sxx, b0 = my - b1 * mx, xa = minOf(X), xb = maxOf(X);
        svg('line', { x1: x(xa), y1: y(b0 + b1 * xa), x2: x(xb), y2: y(b0 + b1 * xb), style: 'stroke:' + (series.length === 1 ? s._c : 'var(--rk-ink-2)'), 'stroke-width': 1.5 }, F.over);
        notes.push((series.length > 1 ? s.name + '：' : '') + 'R² ' + (sxy * sxy / (sxx * syy)).toFixed(2) + '，斜率 ' + fmt(b1, '.2f'));
      }
    });
    if (notes.length) txt(F.over, x1 - 4, y0 - 8, notes.join('   '), { 'class': 'rk-dl', 'text-anchor': 'end' });
    (spec.labels || []).forEach(function (l) { txt(F.over, x(l.x) + (l.dx || 7), y(l.y) + (l.dy || -7), l.text, { 'class': 'rk-dl2', 'text-anchor': l.anchor || 'start' }); });
    var tip = tooltip(box);
    function showPt(p) {
      F.hov.textContent = '';
      svg('circle', { cx: p.x, cy: p.y, r: 6.5, style: 'fill:none;stroke:var(--rk-ink);stroke-width:1.5' }, F.hov);
      var k = scaleK(F.el, W), name = p.s.labels ? p.s.labels[p.i] : (series.length > 1 ? p.s.name : null);
      tip.show(p.x * k, p.y * k, name, [{ value: fmt(p.s.x[p.i], spec.xFormat), name: spec.xLabel || 'x' }, { value: fmt(p.s.y[p.i], spec.yFormat), name: spec.yLabel || 'y' }]);
    }
    function clear() { tip.hide(); F.hov.textContent = ''; }
    F.el.addEventListener('pointermove', function (e) {
      var q = pointer(e, F.el, W), best = null, bd = 24 * q[2] * 24 * q[2];
      pts.forEach(function (p) { var d2 = (p.x - q[0]) * (p.x - q[0]) + (p.y - q[1]) * (p.y - q[1]); if (d2 < bd) { bd = d2; best = p; } });
      if (best) showPt(best); else clear();
    });
    keyNav(F.el, pts.length, function (i) { showPt(pts[i]); }, clear);
  }

  /* ================================================================ box plot (distribution per group) */
  function renderBox(box, spec, W, H) {
    var groups = (spec.groups || []).map(function (g) { return Object.assign({}, g); }), vf = spec.valueFormat || 'auto';
    assignColors(groups, null);
    var all = []; groups.forEach(function (g) { all.push(g.min, g.max); all = all.concat(nums(g.points)); });
    var top = 10, bottom = 24;
    top = 22;                                   // room for the median label above each whisker
    var y = linear(spec.yMin != null ? spec.yMin : minOf(all), spec.yMax != null ? spec.yMax : maxOf(all), H - bottom, top, { ticks: yTicks(H, top, bottom) });
    var yf = tickFmt(y, vf), x0 = 2, x1 = W - tickWidth(y, yf) - 14, band = (x1 - x0) / groups.length, bw = Math.min(40, band * 0.36);
    legendSlot(box);
    var F = frame(box, W, H), tip = tooltip(box), marks = [], y0 = H - bottom;
    rightAxis(F.grid, y, x0, x1, yf);
    baseline(F.grid, x0, x1, y0);
    groups.forEach(function (g, i) {
      var cx = x0 + band * (i + 0.5) - bw * 0.35, l = cx - bw / 2;
      svg('line', { x1: px(cx), x2: px(cx), y1: y(g.max), y2: y(g.q3), style: 'stroke:var(--rk-ink-2)' }, F.marks);
      svg('line', { x1: px(cx), x2: px(cx), y1: y(g.q1), y2: y(g.min), style: 'stroke:var(--rk-ink-2)' }, F.marks);
      [g.min, g.max].forEach(function (v) { svg('line', { x1: cx - bw / 4, x2: cx + bw / 4, y1: px(y(v)), y2: px(y(v)), style: 'stroke:var(--rk-ink-2)' }, F.marks); });
      svg('rect', { x: r1(l), y: r1(y(g.q3)), width: r1(bw), height: r1(Math.max(1, y(g.q1) - y(g.q3))), style: 'fill:' + g._c + ';stroke:' + g._c, 'fill-opacity': 0.16, 'stroke-width': 1.5 }, F.marks);
      svg('line', { x1: l, x2: l + bw, y1: y(g.median), y2: y(g.median), style: 'stroke:' + g._c, 'stroke-width': 2.5 }, F.marks);
      (g.points || []).forEach(function (v, k) {         // the raw values, jittered to the right of the box
        if (!isNum(v)) return;
        var jx = cx + bw * 0.75 + ((k * 0.6180339887) % 1) * bw * 0.6;
        svg('circle', { cx: r1(jx), cy: r1(y(v)), r: 2.6, style: 'fill:' + g._c, 'fill-opacity': 0.75 }, F.marks);
      });
      txt(F.grid, x0 + band * (i + 0.5), y0 + 16, ellipsize(g.name, band - 4, 11.5), { 'text-anchor': 'middle', 'class': 'rk-cat' });
      txt(F.over, cx, y(g.max) - 6, fmt(g.median, vf), { 'text-anchor': 'middle', 'class': 'rk-dl' });
      marks.push({ i: i, cx: cx, hit: svg('rect', { x: x0 + band * i, y: top, width: band, height: y0 - top, 'class': 'rk-hit' }, F.over) });
    });
    function show(m) {
      var g = groups[m.i], k = scaleK(F.el, W);
      tip.show(m.cx * k, (top + 30) * k, g.name, [{ value: fmt(g.max, vf), name: '最大' }, { value: fmt(g.q3, vf), name: '上四分位' }, { color: g._c, value: fmt(g.median, vf), name: '中位数' },
        { value: fmt(g.q1, vf), name: '下四分位' }, { value: fmt(g.min, vf), name: '最小' }, { value: String(g.n), name: '样本' }]);
    }
    marks.forEach(function (m) { m.hit.addEventListener('pointerenter', function () { show(m); }); });
    keyNav(F.el, marks.length, function (i) { show(marks[i]); }, function () { tip.hide(); });
  }

  /* ================================================================ small multiples */
  function renderMultiples(box, spec, W, H) {
    var panels = spec.panels || [], xs = spec.x || [], n = panels.length, vf = spec.yFormat || 'auto';
    var cols = Math.min(spec.cols || 3, W < 520 ? 2 : 6, n), rowsN = Math.ceil(n / cols);
    var cellH = Math.max(76, (H - 2) / rowsN);
    if (cellH * rowsN > H + 1) { H = Math.ceil(cellH * rowsN + 2); box.style.minHeight = H + 'px'; }
    var gx = 18, cw = (W - gx * (cols - 1)) / cols;
    var all = []; panels.forEach(function (p) { all = all.concat(nums(p.values)); });
    (spec.refs || []).forEach(function (r) { if (isNum(r.y)) all.push(r.y); });
    var ylo = spec.yMin != null ? spec.yMin : minOf(all), yhi = spec.yMax != null ? spec.yMax : maxOf(all);
    legendSlot(box);
    var F = frame(box, W, H), tip = tooltip(box), cells = [];
    panels.forEach(function (p, k) {
      var c = k % cols, r = Math.floor(k / cols), ox = c * (cw + gx), oy = r * cellH;
      var t0 = oy + 32, b0 = oy + cellH - (r === rowsN - 1 ? 18 : 12);
      var lo = ylo, hi = yhi;
      if (spec.shareY === false) { var pv = nums(p.values); lo = minOf(pv); hi = maxOf(pv); }
      var y = linear(lo, hi, b0, t0, { ticks: 2 });
      var x = function (i) { return ox + (xs.length < 2 ? cw / 2 : i / (xs.length - 1) * (cw - 6)); };
      var color = p.color ? slotVar(p.color) : (spec.color ? slotVar(spec.color) : 'var(--rk-s1)');
      svg('line', { x1: ox, x2: ox + cw, y1: px(oy), y2: px(oy), style: 'stroke:var(--rk-ink)', 'stroke-width': 1 }, F.grid);
      txt(F.grid, ox, oy + 15, ellipsize(p.name, cw * 0.62, 11.5, 700), { 'class': 'rk-mt' });
      var first = p.values.findIndex(isNum), last = null;
      for (var i = p.values.length - 1; i >= 0; i--) if (isNum(p.values[i])) { last = i; break; }
      if (last != null) txt(F.grid, ox + cw, oy + 15, fmt(p.values[last], vf), { 'text-anchor': 'end', 'class': 'rk-dl' });
      var tk = y.ticks();
      tk.forEach(function (t) { svg('line', { x1: ox, x2: ox + cw, y1: px(y(t)), y2: px(y(t)), 'class': 'rk-grid' }, F.grid); });
      if (c === 0) [tk[0], tk[tk.length - 1]].forEach(function (t) { txt(F.grid, ox + 1, y(t) - 3, tickFmt(y, vf)(t), {}); });
      (spec.refs || []).forEach(function (rf) { if (isNum(rf.y) && rf.y >= y.domain[0] && rf.y <= y.domain[1]) svg('line', { x1: ox, x2: ox + cw, y1: y(rf.y), y2: y(rf.y), 'class': 'rk-refl' }, F.marks); });
      var d = '', pen = false;
      p.values.forEach(function (v, i) { if (!isNum(v)) { pen = false; return; } d += (pen ? 'L' : 'M') + r1(x(i)) + ',' + r1(y(v)); pen = true; });
      var ab = y(clamp(spec.areaBase != null ? spec.areaBase : y.domain[0], y.domain[0], y.domain[1]));
      if (d && last != null) svg('path', { d: d + 'L' + r1(x(last)) + ',' + r1(ab) + 'L' + r1(x(first)) + ',' + r1(ab) + 'Z', style: 'fill:' + color, 'fill-opacity': 0.1 }, F.marks);
      svg('path', { d: d, fill: 'none', style: 'stroke:' + color, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }, F.marks);
      if (last != null) svg('circle', { cx: r1(x(last)), cy: r1(y(p.values[last])), r: 2.8, style: 'fill:' + color }, F.marks);
      if (r === rowsN - 1 && xs.length) { txt(F.grid, ox, oy + cellH - 3, String(xs[0]), {}); txt(F.grid, ox + cw, oy + cellH - 3, String(xs[xs.length - 1]), { 'text-anchor': 'end' }); }
      cells.push({ p: p, ox: ox, oy: oy, x: x, y: y, color: color, t0: t0, b0: b0 });
    });
    function show(cell, i) {
      F.hov.textContent = '';
      var v = cell.p.values[i], X = cell.x(i);
      svg('line', { x1: px(X), x2: px(X), y1: cell.t0, y2: cell.b0, 'class': 'rk-cross' }, F.hov);
      if (isNum(v)) svg('circle', { cx: X, cy: cell.y(v), r: 3.5, style: 'fill:' + cell.color + ';stroke:var(--rk-bg);stroke-width:1.5' }, F.hov);
      var k = scaleK(F.el, W);
      tip.show(X * k, (cell.t0 + 10) * k, cell.p.name + ' · ' + xs[i], [{ color: cell.color, value: fmt(v, vf) }]);
    }
    function clear() { tip.hide(); F.hov.textContent = ''; }
    F.el.addEventListener('pointermove', function (e) {
      var q = pointer(e, F.el, W), hit = null;
      cells.forEach(function (c) { if (q[0] >= c.ox && q[0] <= c.ox + cw && q[1] >= c.oy && q[1] <= c.oy + cellH) hit = c; });
      if (!hit || !xs.length) { clear(); return; }
      show(hit, clamp(Math.round((q[0] - hit.ox) / (cw - 6) * (xs.length - 1)), 0, xs.length - 1));
    });
    F.el.addEventListener('pointerleave', clear);
  }

  /* ================================================================ table view (the accessible twin of every chart) */
  function dataTable(spec) {
    var head = [], body = [], T = spec.type, f = spec.valueFormat;
    if (T === 'line') {
      head = [spec.xName || spec.xLabel || 'x'].concat(spec.series.map(function (s) { return s.name; }));
      spec.x.forEach(function (xv, i) { body.push([typeof xv === 'string' ? xv : fmt(xv, spec.xFormat)].concat(spec.series.map(function (s) { return fmt(s.values[i], spec.yFormat); }))); });
    } else if (T === 'bar') {
      head = [spec.categoryName || '类别'].concat(spec.series.map(function (s) { return s.name; }));
      spec.categories.forEach(function (c, i) { body.push([c].concat(spec.series.map(function (s) { return fmt(s.values[i], f); }))); });
    } else if (T === 'waterfall') {
      head = ['步骤', '变化', '累计']; var run = 0;
      spec.steps.forEach(function (s) { if (s.kind === 'total') { run = s.value; body.push([s.label, '', fmt(run, f)]); } else { run += s.value; body.push([s.label, signed(s.value, f || '.3f'), fmt(run, f)]); } });
    } else if (T === 'dumbbell') {
      head = ['', spec.a.name, spec.b.name, '差'];
      spec.categories.forEach(function (c, i) { var a = spec.a.values[i], b = spec.b.values[i]; body.push([c, fmt(a, f), fmt(b, f), isNum(a) && isNum(b) ? signed(b - a, f || '.3f') : '–']); });
    } else if (T === 'hist') {
      head = ['区间'].concat(spec.series.map(function (s) { return s.name; }));
      for (var i = 0; i < spec.edges.length - 1; i++) body.push(['[' + fmt(spec.edges[i], spec.xFormat) + ', ' + fmt(spec.edges[i + 1], spec.xFormat) + ')'].concat(spec.series.map(function (s) { return fmt(s.counts[i], spec.yFormat); })));
    } else if (T === 'heatmap') {
      head = [''].concat(spec.cols);
      spec.rows.forEach(function (r, i) { body.push([r].concat(spec.values[i].map(function (v) { return fmt(v, f || '.2f'); }))); });
    } else if (T === 'scatter') {
      head = ['点', spec.xLabel || 'x', spec.yLabel || 'y'];
      spec.series.forEach(function (s) { s.x.forEach(function (xv, i) { if (body.length < 500) body.push([s.labels ? s.labels[i] : s.name, fmt(xv, spec.xFormat), fmt(s.y[i], spec.yFormat)]); }); });
    } else if (T === 'box') {
      head = ['组', '样本', '最小', '下四分位', '中位数', '上四分位', '最大'];
      spec.groups.forEach(function (g) { body.push([g.name, String(g.n)].concat([g.min, g.q1, g.median, g.q3, g.max].map(function (v) { return fmt(v, f); }))); });
    } else if (T === 'multiples') {
      head = [''].concat(spec.panels.map(function (p) { return p.name; }));
      spec.x.forEach(function (xv, i) { body.push([String(xv)].concat(spec.panels.map(function (p) { return fmt(p.values[i], spec.yFormat); }))); });
    }
    var wrap = h('div', 'rk-tablewrap'), tb = h('table', 'rk-table', wrap), tr = h('tr', '', h('thead', '', tb));
    head.forEach(function (c, j) { h('th', j ? 'rk-r' : '', tr, c); });
    var tbody = h('tbody', '', tb);
    body.forEach(function (r) { var row = h('tr', '', tbody); r.forEach(function (c, j) { h('td', j ? 'rk-r' : '', row, c); }); });
    return wrap;
  }

  /* ================================================================ driver */
  var RENDER = { line: renderLine, bar: renderBar, waterfall: renderWaterfall, dumbbell: renderDumbbell, hist: renderHist,
    heatmap: renderHeat, scatter: renderScatter, box: renderBox, multiples: renderMultiples };
  function readSpec(box) {
    if (box._spec !== undefined) return box._spec;
    var s = box.querySelector('script[type="application/json"]');
    try { box._spec = JSON.parse(s ? s.textContent : box.getAttribute('data-spec')); } catch (e) { box._spec = null; }
    return box._spec;
  }
  function draw(box) {
    var spec = readSpec(box), keep = box.querySelector('script[type="application/json"]');
    Array.prototype.slice.call(box.childNodes).forEach(function (n) { if (n !== keep && !(n.classList && n.classList.contains('rk-dataview'))) box.removeChild(n); });
    if (!spec || !RENDER[spec.type]) { h('p', 'rk-muted', box, '图表数据无法读取。'); return; }
    box.style.minHeight = spec.height ? spec.height + 'px' : '';
    var W = Math.max(200, Math.floor(box.clientWidth)), H = Math.max(80, Math.floor(box.clientHeight));
    try { RENDER[spec.type](box, spec, W, H); }
    catch (e) { h('p', 'rk-muted', box, '图表渲染失败：' + e.message); if (window.console) console.error(e); }
    box._w = Math.floor(box.clientWidth); box._h = Math.floor(box.clientHeight);
    var s2 = box.querySelector('svg'), panel = box.closest('.rk-panel'), cap = panel && panel.querySelector('.rk-pt');
    if (s2 && cap) s2.setAttribute('aria-label', cap.textContent);
  }
  function setupPanel(box) {
    var panel = box.closest('.rk-panel');
    if (!panel || panel._rk) return;
    panel._rk = true;
    var foot = panel.querySelector('.rk-pf');
    if (!foot) return;
    var btn = h('button', 'rk-btn', foot, '数据');
    btn.type = 'button'; btn.setAttribute('aria-pressed', 'false'); btn.title = '显示图背后的数字';
    var view = h('div', 'rk-dataview', box); view.hidden = true;
    btn.onclick = function () {
      var on = view.hidden;
      if (on && !view.firstChild) view.appendChild(dataTable(readSpec(box)));
      view.hidden = !on; btn.setAttribute('aria-pressed', String(on));
    };
  }
  function spark(el) {
    var v; try { v = JSON.parse(el.getAttribute('data-spark')); } catch (e) { return; }
    el.textContent = '';
    var W = Math.max(40, el.clientWidth), H = el.clientHeight || 26, vals = nums(v);
    if (vals.length < 2) return;
    var lo = minOf(vals), hi = maxOf(vals);
    var y = function (t) { return hi === lo ? H / 2 : 2 + (1 - (t - lo) / (hi - lo)) * (H - 4); };
    var x = function (i) { return 1 + i / (v.length - 1) * (W - 5); };
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, 'aria-hidden': 'true' }, el), d = '', pen = false, last = null, first = null;
    v.forEach(function (t, i) { if (!isNum(t)) { pen = false; return; } d += (pen ? 'L' : 'M') + r1(x(i)) + ',' + r1(y(t)); pen = true; last = [x(i), y(t)]; if (!first) first = [x(i), y(t)]; });
    var c = el.getAttribute('data-color') ? slotVar(el.getAttribute('data-color')) : 'var(--rk-accent)';
    svg('path', { d: d + 'L' + r1(last[0]) + ',' + H + 'L' + r1(first[0]) + ',' + H + 'Z', style: 'fill:' + c, 'fill-opacity': 0.1 }, s);
    svg('path', { d: d, fill: 'none', style: 'stroke:' + c, 'stroke-width': 1.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, s);
    svg('circle', { cx: r1(last[0]), cy: r1(last[1]), r: 2.6, style: 'fill:' + c }, s);
  }
  function renderAll(force) {
    registry = {}; nextSlot = 1;
    doc.querySelectorAll('.rk-chart').forEach(function (box) {
      if (force || box._w !== Math.floor(box.clientWidth) || box._h !== Math.floor(box.clientHeight)) draw(box);
      else { var sp = readSpec(box); assignColors(specSeries(sp).map(function (s) { return Object.assign({}, s); }), sp && sp.emphasis != null ? [].concat(sp.emphasis) : null); }
    });
    doc.querySelectorAll('.rk-spark[data-spark]').forEach(spark);
  }

  /* ================================================================ section index, copy, math, tools */
  function buildNav() {
    var nav = doc.querySelector('.rk-nav');
    if (!nav || nav.getAttribute('data-static') != null) return;
    var secs = doc.querySelectorAll('.rk-sec[id]');
    if (!secs.length) { nav.parentNode.removeChild(nav); return; }
    nav.textContent = '';
    var links = [];
    secs.forEach(function (sec) {
      var hd = sec.querySelector('.rk-sec-head h2'); if (!hd) return;
      var a = h('a', '', nav); a.href = '#' + sec.id;
      var n = sec.querySelector('.rk-sec-head .rk-n');
      if (n) h('span', 'rk-n', a, n.textContent);
      h('span', '', a, hd.textContent);
      links.push([sec, a]);
    });
    var on = function () {
      var cur = null;
      links.forEach(function (l) { if (l[0].getBoundingClientRect().top < 90) cur = l; });
      links.forEach(function (l) { l[1].classList.toggle('on', l === cur); });
      if (cur && nav.scrollWidth > nav.clientWidth) { var r = cur[1].getBoundingClientRect(), nr = nav.getBoundingClientRect(); if (r.left < nr.left || r.right > nr.right) nav.scrollLeft += r.left - nr.left - 20; }
    };
    window.addEventListener('scroll', function () { window.requestAnimationFrame(on); }, { passive: true });
    on();
  }
  function copyText(t, btn) {
    function done() { var o = btn.textContent; btn.textContent = '已复制'; setTimeout(function () { btn.textContent = o; }, 1200); }
    function fallback() { var a = doc.createElement('textarea'); a.value = t; a.style.position = 'fixed'; a.style.opacity = '0'; doc.body.appendChild(a); a.select(); try { doc.execCommand('copy'); done(); } catch (e) { } doc.body.removeChild(a); }
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(t).then(done, fallback); else fallback();
  }
  function copyButtons() {
    doc.querySelectorAll('.rk-code').forEach(function (w) {
      if (w.querySelector('.rk-copy')) return;
      var code = w.querySelector('code') || w.querySelector('pre'), b = h('button', 'rk-btn rk-copy', w, '复制');
      b.type = 'button'; b.onclick = function () { copyText(code.textContent, b); };
    });
    doc.querySelectorAll('.rk-math[data-tex]').forEach(function (w) {
      if (w.querySelector('.rk-copy')) return;
      var b = h('button', 'rk-btn rk-copy', w, '复制 LaTeX');
      b.type = 'button'; b.onclick = function () { copyText(w.getAttribute('data-tex'), b); };
    });
  }
  function math() {
    if (!window.katex) return;
    doc.querySelectorAll('.rk-math[data-tex] .rk-tex').forEach(function (t) {
      try { window.katex.render(t.parentNode.getAttribute('data-tex'), t, { displayMode: true, throwOnError: false }); t.classList.remove('rk-tex'); } catch (e) { }
    });
    doc.querySelectorAll('.rk-im[data-tex]').forEach(function (t) { try { window.katex.render(t.getAttribute('data-tex'), t, { throwOnError: false }); } catch (e) { } });
  }
  function beforePrint() { root.classList.add('rk-printing'); root.setAttribute('data-print', '1'); renderAll(true); }
  function afterPrint() { root.classList.remove('rk-printing'); root.removeAttribute('data-print'); renderAll(true); }
  function tools() {
    var tb = doc.querySelector('[data-rk="theme"]'), pb = doc.querySelector('[data-rk="print"]');
    if (tb) tb.onclick = function () {
      var n = isDark() ? 'light' : 'dark';
      root.setAttribute('data-theme', n);
      try { localStorage.setItem('rk-theme', n); } catch (e) { }
      renderAll(true);
    };
    if (pb) pb.onclick = function () { window.print(); };
    if (window.matchMedia) {
      var mq = matchMedia('(prefers-color-scheme: dark)'), on = function () { if (!root.getAttribute('data-theme')) renderAll(true); };
      if (mq.addEventListener) mq.addEventListener('change', on); else if (mq.addListener) mq.addListener(on);
    }
    window.addEventListener('beforeprint', beforePrint);
    window.addEventListener('afterprint', afterPrint);
  }

  function init() {
    buildNav(); copyButtons(); math(); tools();
    doc.querySelectorAll('.rk-chart').forEach(setupPanel);
    renderAll(true);
    if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(function () { canvas = null; fontFamily = null; renderAll(true); });
    var t = null, again = function () { clearTimeout(t); t = setTimeout(function () { renderAll(false); }, 100); };
    if ('ResizeObserver' in window) { var ro = new ResizeObserver(again); doc.querySelectorAll('.rk-chart,.rk-spark').forEach(function (b) { ro.observe(b); }); }
    else window.addEventListener('resize', again);
  }
  window.reportkit = { render: renderAll, format: fmt, print: { before: beforePrint, after: afterPrint } };
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();
})();
