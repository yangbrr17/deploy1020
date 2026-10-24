/* reportkit theme v1 — runtime for report pages. No dependencies, no network.
   What it does, all on DOMContentLoaded:
     1. builds the table of contents from h2/h3 inside .rk-paper (needs ids);
     2. renders every <div class="rk-chart"> from the JSON spec in its child <script type="application/json">
        (types: line, bar, hist, heatmap, scatter) and every .rk-spark[data-spark];
     3. adds hover / keyboard tooltips and the "表格" (table view) toggle to each figure;
     4. adds copy buttons to code and math blocks, renders math with KaTeX when it is on the page;
     5. wires the theme and print buttons in .rk-tools.
   Charts re-render on resize, on a theme change and before printing. Spec format: see the docs page
   (docs/report-theme/README.md, section "图表 spec"). Series colours follow the series name across the
   whole report (first appearance takes the next slot), so one model keeps one colour in every chart. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var doc = document, root = doc.documentElement;

  /* ---------------- small utilities ---------------- */
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
    var t = svg('text', Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function r1(v) { return Math.round(v * 10) / 10; }

  var canvas;
  function textWidth(s, size, weight) {
    canvas = canvas || doc.createElement('canvas');
    var c = canvas.getContext('2d');
    c.font = (weight || 400) + ' ' + (size || 11.5) + 'px ' + getComputedStyle(doc.body).fontFamily;
    return c.measureText(String(s)).width;
  }

  /* number formatting: "auto", ".2f", ",.0f", "+.1f", ".1%", ".2s" (compact K/M/B), ".2e", or "int" */
  function compact(v, p) {
    var a = Math.abs(v), u = '', d = 1;
    if (a >= 1e9) { u = 'B'; d = 1e9; } else if (a >= 1e6) { u = 'M'; d = 1e6; } else if (a >= 1e3) { u = 'K'; d = 1e3; }
    var s = (v / d).toFixed(p == null ? (a / d >= 100 ? 0 : 1) : p);
    if (p == null) s = s.replace(/\.0$/, '');
    return s + u;
  }
  function group(s) {
    var parts = s.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return parts.join('.');
  }
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

  /* ---------------- scales ---------------- */
  function tickStep(lo, hi, n) {
    var raw = (hi - lo) / Math.max(1, n), p = Math.pow(10, Math.floor(Math.log10(raw))), e = raw / p;
    return (e >= 7.07 ? 10 : e >= 3.16 ? 5 : e >= 1.41 ? 2 : 1) * p;
  }
  function clean(v) { return +v.toPrecision(12); }
  function linear(d0, d1, r0, r1_, opts) {
    opts = opts || {};
    var lo = d0, hi = d1;
    if (lo === hi) { var pad = lo === 0 ? 1 : Math.abs(lo) * 0.1; lo -= pad; hi += pad; }
    var n = opts.ticks || 5, step = tickStep(lo, hi, n);
    if (opts.nice !== false) { lo = Math.floor(clean(lo / step)) * step; hi = Math.ceil(clean(hi / step)) * step; }
    var f = function (v) { return r0 + (v - lo) / (hi - lo) * (r1_ - r0); };
    f.domain = [lo, hi];
    f.ticks = function () {
      var out = [], t = Math.ceil(clean(lo / step)) * step;
      for (; t <= hi + step * 1e-9; t += step) out.push(clean(t));
      return out;
    };
    f.step = step;
    f.invert = function (px) { return lo + (px - r0) / (r1_ - r0) * (hi - lo); };
    return f;
  }
  function logScale(d0, d1, r0, r1_) {
    var a = Math.log10(d0), b = Math.log10(d1);
    if (a === b) b = a + 1;
    var f = function (v) { return r0 + (Math.log10(v) - a) / (b - a) * (r1_ - r0); };
    f.domain = [d0, d1];
    f.log = true;
    f.ticks = function () {
      var out = [], dec = b - a;
      for (var k = Math.floor(a); k <= Math.ceil(b); k++) {
        [1, 2, 5].forEach(function (m) {
          var v = clean(m * Math.pow(10, k));
          if ((m === 1 || dec <= 2) && v >= d0 * (1 - 1e-9) && v <= d1 * (1 + 1e-9)) out.push(v);
        });
      }
      return out;
    };
    f.invert = function (px) { return Math.pow(10, a + (px - r0) / (r1_ - r0) * (b - a)); };
    return f;
  }
  function decimalsFor(step) { return Math.max(0, -Math.floor(Math.log10(step) + 1e-9)); }
  function tickFmt(scale, f) {
    if (scale.log) return function (v) { return Math.abs(v) >= 1e4 ? fmt(v, 's') : group(String(v)); };
    f = f || '';
    if (/%$/.test(f)) { var dp = decimalsFor(scale.step * 100); return function (v) { return fmt(v, '.' + dp + '%'); }; }
    if (/s$/.test(f)) return function (v) { return fmt(v, 's'); };
    var d = decimalsFor(scale.step);
    return function (v) { return Math.abs(v) >= 1e4 ? fmt(v, 's') : fmt(v, ',.' + d + 'f'); };
  }

  /* ---------------- colours ---------------- */
  var registry = {}, nextSlot = 1;
  function slotVar(s) { return s === 'gray' ? 'var(--rk-gray)' : 'var(--rk-' + s + ')'; }
  // Series colour follows the series name across the report. A chart with one new series uses slot 1
  // without claiming it; explicit series.color ("s3", "gray") always wins.
  function assignColors(series, emphasis) {
    var multi = series.length > 1;
    series.forEach(function (s) {
      if (s.color) { s._c = slotVar(s.color); return; }
      if (emphasis != null && emphasis.indexOf(s.name) < 0) { s._c = slotVar('gray'); s._muted = true; return; }
      var key = s.key || s.name;
      if (registry[key]) { s._c = slotVar(registry[key]); return; }
      if (!multi && emphasis == null) { s._c = slotVar('s1'); return; }
      if (nextSlot <= 8) { registry[key] = 's' + nextSlot++; s._c = slotVar(registry[key]); }
      else { s._c = slotVar('gray'); }
    });
  }
  function isDark() {
    if (root.getAttribute('data-print') === '1') return false;
    var t = root.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches;
  }
  // OKLab interpolation for heatmap scales
  function hex2rgb(x) { x = x.replace('#', ''); return [0, 2, 4].map(function (i) { return parseInt(x.substr(i, 2), 16) / 255; }); }
  function lin(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  function delin(c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; }
  function toOk(hex) {
    var c = hex2rgb(hex).map(lin);
    var l = Math.cbrt(0.4122214708 * c[0] + 0.5363325363 * c[1] + 0.0514459929 * c[2]);
    var m = Math.cbrt(0.2119034982 * c[0] + 0.6806995451 * c[1] + 0.1073969566 * c[2]);
    var s = Math.cbrt(0.0883024619 * c[0] + 0.2817188376 * c[1] + 0.6299787005 * c[2]);
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
  }
  function fromOk(o) {
    var l = Math.pow(o[0] + 0.3963377774 * o[1] + 0.2158037573 * o[2], 3);
    var m = Math.pow(o[0] - 0.1055613458 * o[1] - 0.0638541728 * o[2], 3);
    var s = Math.pow(o[0] - 0.0894841775 * o[1] - 1.2914855480 * o[2], 3);
    var rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
      -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
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
  function inkOn(hex) { var L = luminance(hex); return (1.05 / (L + 0.05)) >= ((L + 0.05) / 0.0545) ? '#ffffff' : '#0b0b0b'; }
  function heatScale(kind) {
    var dark = isDark();
    if (kind === 'sequential') return ramp(dark ? ['#184f95', '#2a78d6', '#86b6ef', '#cde2fb'] : ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b']);
    return ramp(dark ? ['#3987e5', '#383835', '#e66767'] : ['#2a78d6', '#f0efec', '#e34948']);
  }

  /* ---------------- tooltip ---------------- */
  function tooltip(box) {
    var tip = h('div', 'rk-tip', box);
    tip.setAttribute('role', 'status');
    return {
      show: function (x, y, head, rows) {
        tip.textContent = '';
        if (head != null) h('div', 'rk-tip-h', tip, head);
        rows.forEach(function (r) {
          var row = h('div', 'rk-tip-r', tip);
          if (r.color) { var k = h('i', '', row); k.style.background = r.color; if (r.rect) { k.style.height = '8px'; k.style.width = '8px'; k.style.borderRadius = '2px'; } }
          h('b', '', row, r.value);
          if (r.name) h('span', '', row, r.name);
        });
        tip.classList.add('on');
        var W = box.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
        var left = x + 14 + tw > W ? x - tw - 14 : x + 14;
        tip.style.left = clamp(left, 0, Math.max(0, W - tw)) + 'px';
        tip.style.top = clamp(y - th / 2, -8, box.clientHeight - th + 8) + 'px';
      },
      hide: function () { tip.classList.remove('on'); }
    };
  }
  function pointer(e, svgEl, W) {
    var r = svgEl.getBoundingClientRect(), k = W / r.width;
    return [(e.clientX - r.left) * k, (e.clientY - r.top) * k, k];
  }

  /* ---------------- shared chart frame ---------------- */
  function legend(box, items, kind, force) {
    if (items.length < 2 && !force) return;
    var lg = h('div', 'rk-legend', box);
    items.forEach(function (it) {
      var s = h('span', '', lg), k = h('i', 'rk-k-' + (it.kind || kind), s);
      if ((it.kind || kind) !== 'ref') k.style.background = it.color;
      s.appendChild(doc.createTextNode(it.name));
    });
  }
  function axisY(g, y, x0, x1, f, label, H) {
    y.ticks().forEach(function (t) {
      var py = Math.round(y(t)) + 0.5;
      svg('line', { x1: x0, x2: x1, y1: py, y2: py, 'class': 'rk-grid' }, g);
      txt(g, x0 - 8, py + 4, f(t), { 'text-anchor': 'end' });
    });
    if (label) txt(g, x0, 12, label, { 'class': 'rk-ax-title', 'text-anchor': 'start' });
  }
  function axisX(g, x, ticks, y0, f, x0, x1) {
    var last = -1e9;
    ticks.forEach(function (t) {
      var px = x(t), s = f(t), w = textWidth(s);
      if (px < x0 - 1 || px > x1 + 1) return;
      if (px - w / 2 < last + 8) return;
      var anchor = 'middle';
      if (px - w / 2 < x0 - 30) anchor = 'start';
      txt(g, r1(px), y0 + 17, s, { 'text-anchor': anchor });
      last = px + w / 2;
    });
  }
  function hasXLabel(spec) { return !!spec.xLabel; }

  /* ---------------- line & scatter ---------------- */
  function renderXY(box, spec, W) {
    var scatter = spec.type === 'scatter';
    var series = (spec.series || []).map(function (s) { return Object.assign({}, s); });
    assignColors(series, spec.emphasis == null ? null : [].concat(spec.emphasis));
    var cat = !scatter && spec.x && spec.x.some(function (v) { return typeof v === 'string'; });
    var xs = scatter ? null : spec.x;
    var yv = [], xv = [];
    series.forEach(function (s) {
      var ys = scatter ? s.y : s.values;
      ys.forEach(function (v, i) { if (isNum(v)) { yv.push(v); if (scatter) xv.push(s.x[i]); } });
    });
    if (!scatter && !cat) xs.forEach(function (v) { if (isNum(v)) xv.push(v); });
    (spec.refs || []).forEach(function (r) { if (isNum(r.y)) yv.push(r.y); if (isNum(r.x)) xv.push(r.x); });
    var ymin = spec.yMin != null ? spec.yMin : Math.min.apply(null, yv), ymax = spec.yMax != null ? spec.yMax : Math.max.apply(null, yv);
    if (spec.yMin == null && spec.zero) ymin = Math.min(0, ymin);
    var H = spec.height || 300, legendItems = [];
    series.forEach(function (s) { legendItems.push({ name: s.name, color: s._c, kind: scatter ? 'dot' : 'line' }); });
    // margins
    var top = spec.yLabel ? 26 : 10, bottom = 26 + (hasXLabel(spec) ? 18 : 0);
    var yScale0 = linear(ymin, ymax, H - bottom, top, { nice: spec.yMin == null || spec.yMax == null });
    if (spec.yMin != null && spec.yMax != null) yScale0 = linear(spec.yMin, spec.yMax, H - bottom, top, { nice: false });
    var yf = tickFmt(yScale0, spec.yFormat);
    var left = Math.max.apply(null, yScale0.ticks().map(function (t) { return textWidth(yf(t)); })) + 14;
    // direct end labels (lines, <= 4 series, only when they do not collide)
    var right = 14, ends = [];
    if (!scatter && series.length > 1 && series.length <= 4 && spec.endLabels !== false) {
      series.forEach(function (s) {
        for (var i = s.values.length - 1; i >= 0; i--) if (isNum(s.values[i])) { ends.push({ s: s, y: yScale0(s.values[i]), i: i }); break; }
      });
      ends.sort(function (a, b) { return a.y - b.y; });
      var ok = ends.every(function (e, k) { return k === 0 || e.y - ends[k - 1].y >= 15; });
      if (ok) right = 14 + Math.max.apply(null, ends.map(function (e) { return textWidth(e.s.name, 12, 600); })) + 8; else ends = [];
    }
    var x0 = left, x1 = W - right, y0 = H - bottom, y1 = top;
    var x;
    if (cat) {
      var n = xs.length, band = (x1 - x0) / n;
      x = function (i) { return x0 + band * (i + 0.5); };
    } else {
      var xmin = spec.xMin != null ? spec.xMin : Math.min.apply(null, xv), xmax = spec.xMax != null ? spec.xMax : Math.max.apply(null, xv);
      x = spec.xLog ? logScale(xmin, xmax, x0, x1) : linear(xmin, xmax, x0, x1, { nice: !!scatter && spec.xMin == null, ticks: Math.max(2, Math.floor((x1 - x0) / 80)) });
    }
    var y = yScale0;
    var refItems = (spec.refs || []).filter(function (r) { return r.label && r.legend; }).map(function (r) { return { name: r.label, kind: 'ref' }; });
    if (typeof spec.diagonal === 'string') refItems.unshift({ name: spec.diagonal, kind: 'ref' });
    legend(box, legendItems.concat(refItems), 'line', !!refItems.length);
    var el = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', role: 'img', tabindex: 0 }, box);
    el.style.height = 'auto';
    var g = svg('g', {}, el);
    axisY(g, y, x0, x1, yf, spec.yLabel, H);
    var xf = cat ? function (i) { return String(xs[i]); } : tickFmt(x, spec.xFormat);
    if (cat) {
      var idx = xs.map(function (_, i) { return i; }), every = Math.max(1, Math.ceil(Math.max.apply(null, xs.map(function (s) { return textWidth(s); })) / ((x1 - x0) / xs.length) * 1.15));
      axisX(g, x, idx.filter(function (i) { return i % every === 0; }), y0, xf, x0, x1);
    } else {
      axisX(g, x, x.ticks(), y0, xf, x0, x1);
    }
    svg('line', { x1: x0, x2: x1, y1: Math.round(y0) + 0.5, y2: Math.round(y0) + 0.5, 'class': 'rk-base' }, g);
    if (spec.xLabel) txt(g, x1, H - 2, spec.xLabel, { 'class': 'rk-ax-title', 'text-anchor': 'end' });
    // reference lines
    var rg = svg('g', {}, el);
    if (spec.diagonal) {
      var lo = Math.max(y.domain[0], x.domain ? x.domain[0] : 0), hi = Math.min(y.domain[1], x.domain ? x.domain[1] : 1);
      svg('line', { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), 'class': 'rk-refl' }, rg);
    }
    (spec.refs || []).forEach(function (r) {
      if (isNum(r.y)) {
        svg('line', { x1: x0, x2: x1, y1: y(r.y), y2: y(r.y), 'class': 'rk-refl' }, rg);
        if (r.label && !r.legend) txt(rg, x0 + 6, y(r.y) - 5, r.label, {});
      } else if (isNum(r.x)) {
        var px = cat ? x(r.x) : x(r.x);
        svg('line', { x1: px, x2: px, y1: y1, y2: y0, 'class': 'rk-refl' }, rg);
        if (r.label && !r.legend) txt(rg, px + 5, y1 + 10, r.label, {});
      }
    });
    var mg = svg('g', {}, el);
    var px_ = function (i, s) { return scatter ? x(s.x[i]) : (cat ? x(i) : x(xs[i])); };
    var order = series.slice().sort(function (a, b) { return (b._muted ? 1 : 0) - (a._muted ? 1 : 0); });
    var pts = [];
    order.forEach(function (s) {
      var ys = scatter ? s.y : s.values;
      if (scatter) {
        var gg = svg('g', { 'class': 'rk-mark' }, mg);
        ys.forEach(function (v, i) {
          if (!isNum(v) || !isNum(s.x[i])) return;
          var cx = r1(x(s.x[i])), cy = r1(y(v));
          svg('circle', { cx: cx, cy: cy, r: spec.r || 3.5, style: 'fill:' + s._c + ';stroke:var(--rk-paper);stroke-width:1', 'fill-opacity': spec.opacity || 0.8 }, gg);
          pts.push({ x: cx, y: cy, s: s, i: i });
        });
        return;
      }
      var d = '', pen = false, first = null, lastPt = null;
      ys.forEach(function (v, i) {
        if (!isNum(v)) { pen = false; return; }
        var p = [r1(px_(i, s)), r1(y(v))];
        d += (pen ? 'L' : 'M') + p[0] + ',' + p[1];
        pen = true; if (!first) first = p; lastPt = p;
      });
      if (s.area || (spec.area && series.length === 1)) {
        var base = y(clamp(0, y.domain[0], y.domain[1]));
        var segs = d.split('M').filter(Boolean);
        segs.forEach(function (sg) {
          var ps = sg.split('L'), a = ps[0].split(','), b = ps[ps.length - 1].split(',');
          svg('path', { d: 'M' + a[0] + ',' + base + 'L' + sg + 'L' + b[0] + ',' + base + 'Z', style: 'fill:' + s._c, 'fill-opacity': 0.1, 'class': 'rk-mark' }, mg);
        });
      }
      svg('path', { d: d, fill: 'none', style: 'stroke:' + s._c, 'stroke-width': s._muted ? 1.5 : 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dash ? '5 4' : null, 'class': 'rk-mark' }, mg);
      if (spec.markers || s.markers) ys.forEach(function (v, i) {
        if (isNum(v)) svg('circle', { cx: r1(px_(i, s)), cy: r1(y(v)), r: 4, style: 'fill:' + s._c + ';stroke:var(--rk-paper);stroke-width:2' }, mg);
      });
    });
    // direct labels at line ends
    ends.forEach(function (e) {
      txt(mg, x1 + 8, e.y + 4, e.s.name, { 'class': 'rk-dl' });
    });
    (spec.labels || []).forEach(function (l) {   // optional point annotations {x, y, text}
      var lx = cat ? x(l.x) : x(l.x), ly = y(l.y);
      svg('circle', { cx: lx, cy: ly, r: 4, style: 'fill:var(--rk-ink);stroke:var(--rk-paper);stroke-width:2' }, mg);
      txt(mg, lx + (l.dx != null ? l.dx : 8), ly + (l.dy != null ? l.dy : -8), l.text, { 'class': 'rk-dl', 'text-anchor': l.anchor || 'start' });
    });
    // hover
    var tip = tooltip(box);
    var hov = svg('g', { 'pointer-events': 'none' }, el);
    var cross = null, dots = [];
    function clear() { tip.hide(); hov.textContent = ''; el.classList.remove('rk-hovering'); }
    if (scatter) {
      function showPt(p) {
        hov.textContent = '';
        svg('circle', { cx: p.x, cy: p.y, r: 5, style: 'fill:' + p.s._c + ';stroke:var(--rk-paper);stroke-width:2' }, hov);
        var k = el.getBoundingClientRect().width / W;
        tip.show(p.x * k, p.y * k, series.length > 1 ? p.s.name : (p.s.labels ? p.s.labels[p.i] : null), [
          { value: fmt(p.s.x[p.i], spec.xFormat), name: spec.xLabel || 'x' },
          { value: fmt(p.s.y[p.i], spec.yFormat), name: spec.yLabel || 'y' }]);
      }
      el.addEventListener('pointermove', function (e) {
        var q = pointer(e, el, W), best = null, bd = 24 * q[2] * 24 * q[2];
        pts.forEach(function (p) { var d2 = (p.x - q[0]) * (p.x - q[0]) + (p.y - q[1]) * (p.y - q[1]); if (d2 < bd) { bd = d2; best = p; } });
        if (best) showPt(best); else clear();
      });
      el.addEventListener('pointerleave', clear);
      var ki = -1;
      el.addEventListener('keydown', function (e) {
        if (!pts.length || (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft')) return;
        e.preventDefault(); ki = (ki + (e.key === 'ArrowRight' ? 1 : pts.length - 1)) % pts.length; showPt(pts[ki]);
      });
      el.addEventListener('blur', clear);
      return;
    }
    var nX = cat ? xs.length : xs.length;
    function showAt(i) {
      hov.textContent = '';
      var px = cat ? x(i) : x(xs[i]);
      svg('line', { x1: Math.round(px) + 0.5, x2: Math.round(px) + 0.5, y1: y1, y2: y0, 'class': 'rk-cross' }, hov);
      var rows = [];
      series.forEach(function (s) {
        var v = s.values[i];
        if (isNum(v)) svg('circle', { cx: px, cy: y(v), r: 4.5, style: 'fill:' + s._c + ';stroke:var(--rk-paper);stroke-width:2' }, hov);
        rows.push({ color: s._c, value: fmt(v, spec.yFormat), name: series.length > 1 ? s.name : null, y: isNum(v) ? v : -Infinity });
      });
      rows.sort(function (a, b) { return b.y - a.y; });
      var k = el.getBoundingClientRect().width / W;
      tip.show(px * k, (y1 + (y0 - y1) * 0.35) * k, (spec.xName ? spec.xName + ' ' : '') + (cat ? xs[i] : fmt(xs[i], spec.xFormat)), rows);
    }
    var cur = -1;
    el.addEventListener('pointermove', function (e) {
      var q = pointer(e, el, W);
      if (q[0] < x0 - 10 || q[0] > x1 + 10) { clear(); return; }
      var best = 0, bd = Infinity;
      for (var i = 0; i < nX; i++) { var d = Math.abs((cat ? x(i) : x(xs[i])) - q[0]); if (d < bd) { bd = d; best = i; } }
      cur = best; showAt(best);
    });
    el.addEventListener('pointerleave', clear);
    el.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault(); cur = clamp(cur + (e.key === 'ArrowRight' ? 1 : -1), 0, nX - 1); showAt(cur);
    });
    el.addEventListener('blur', clear);
  }

  /* ---------------- bar ---------------- */
  // rounded data end of radius r on side `end` (top|bottom|right|left), square at the baseline
  function barPath(x, y, w, hh, r, end) {
    r = Math.max(0, Math.min(r, w / 2, hh / 2));
    if (end === 'top') return 'M' + x + ',' + (y + hh) + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + (y + hh) + 'Z';
    if (end === 'bottom') return 'M' + x + ',' + y + 'V' + (y + hh - r) + 'Q' + x + ',' + (y + hh) + ' ' + (x + r) + ',' + (y + hh) + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + (y + hh) + ' ' + (x + w) + ',' + (y + hh - r) + 'V' + y + 'Z';
    if (end === 'right') return 'M' + x + ',' + y + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + 'V' + (y + hh - r) + 'Q' + (x + w) + ',' + (y + hh) + ' ' + (x + w - r) + ',' + (y + hh) + 'H' + x + 'Z';
    if (end === 'left') return 'M' + (x + w) + ',' + y + 'H' + (x + r) + 'Q' + x + ',' + y + ' ' + x + ',' + (y + r) + 'V' + (y + hh - r) + 'Q' + x + ',' + (y + hh) + ' ' + (x + r) + ',' + (y + hh) + 'H' + (x + w) + 'Z';
    return 'M' + x + ',' + y + 'h' + w + 'v' + hh + 'h' + (-w) + 'Z';
  }
  function ellipsize(s, maxW, size) {
    s = String(s);
    if (textWidth(s, size) <= maxW) return s;
    while (s.length > 1 && textWidth(s + '…', size) > maxW) s = s.slice(0, -1);
    return s + '…';
  }
  function renderBar(box, spec, W) {
    var cats = spec.categories || [], series = (spec.series || []).map(function (s) { return Object.assign({}, s); });
    var single = series.length === 1, stacked = !!spec.stacked && !single;
    assignColors(series, null);
    var hl = spec.highlight ? [].concat(spec.highlight) : null;
    function colorOf(s, ci) { return hl && single ? (hl.indexOf(cats[ci]) >= 0 ? s._c : 'var(--rk-gray)') : s._c; }
    var vals = [];
    cats.forEach(function (_, ci) {
      if (stacked) {
        var pos = 0, neg = 0;
        series.forEach(function (s) { var v = s.values[ci]; if (isNum(v)) { if (v >= 0) pos += v; else neg += v; } });
        vals.push(pos, neg);
      } else series.forEach(function (s) { if (isNum(s.values[ci])) vals.push(s.values[ci]); });
    });
    var vmin = Math.min(0, Math.min.apply(null, vals)), vmax = Math.max(0, Math.max.apply(null, vals));
    if (spec.max != null) vmax = spec.max;
    var vf = spec.valueFormat || 'auto';
    var labelsOn = spec.labels != null ? spec.labels : (single || stacked) && cats.length <= 16;
    // orientation: vertical when every category label fits its band
    var horiz = spec.horizontal;
    var catW = cats.map(function (c) { return textWidth(c, 12); });
    if (horiz == null || horiz === 'auto') {
      var guessLeft = 40, band = (W - guessLeft - 10) / Math.max(1, cats.length);
      horiz = Math.max.apply(null, catW.concat([0])) > band - 8;
    }
    var nS = stacked ? 1 : series.length, gap = 2;
    legend(box, series.map(function (s) { return { name: s.name, color: s._c }; }), 'rect');
    var el, H, tip = tooltip(box), marks = [];
    if (!horiz) {
      H = spec.height || 280;
      var top = spec.yLabel ? 26 : 12, bottom = 28;
      var y = linear(vmin, vmax, H - bottom, top);
      var yf = tickFmt(y, spec.valueFormat);
      var left = Math.max.apply(null, y.ticks().map(function (t) { return textWidth(yf(t)); })) + 14;
      var x0 = left, x1 = W - 10, band = (x1 - x0) / cats.length;
      var bw = Math.min(24, (band * 0.72 - gap * (nS - 1)) / nS);
      el = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', role: 'img', tabindex: 0 }, box);
      el.style.height = 'auto';
      var g = svg('g', {}, el);
      axisY(g, y, x0, x1, yf, spec.yLabel, H);
      var mg = svg('g', {}, el), zero = y(0);
      cats.forEach(function (c, ci) {
        var cx = x0 + band * (ci + 0.5), gw = bw * nS + gap * (nS - 1), bx = cx - gw / 2;
        txt(g, r1(cx), H - bottom + 18, c, { 'text-anchor': 'middle', 'class': 'rk-cat' });
        var acc = { p: 0, n: 0 }, grp = [];
        series.forEach(function (s, si) {
          var v = s.values[ci]; if (!isNum(v)) return;
          var xx = stacked ? bx : bx + si * (bw + gap), a, b;
          if (stacked) { if (v >= 0) { a = acc.p; acc.p += v; b = acc.p; } else { a = acc.n; acc.n += v; b = acc.n; } } else { a = 0; b = v; }
          var ya = y(a), yb = y(b), top_ = Math.min(ya, yb), hh = Math.abs(ya - yb);
          var isEnd = !stacked || si === series.length - 1 || !series.slice(si + 1).some(function (t) { return isNum(t.values[ci]) && (t.values[ci] >= 0) === (v >= 0); });
          if (stacked && si > 0) { if (v >= 0) hh -= gap; else { top_ += gap; hh -= gap; } }
          var p = svg('path', { d: barPath(r1(xx), r1(top_), r1(bw), Math.max(0, r1(hh)), isEnd ? 4 : 0, v >= 0 ? 'top' : 'bottom'), style: 'fill:' + colorOf(s, ci), 'class': 'rk-mark' }, mg);
          grp.push(p);
          if (labelsOn && !stacked) txt(mg, r1(xx + bw / 2), v >= 0 ? yb - 6 : yb + 15, fmt(v, vf), { 'text-anchor': 'middle', 'class': 'rk-dl' });
        });
        if (labelsOn && stacked) txt(mg, r1(cx), y(acc.p) - 6, fmt(acc.p + acc.n, vf), { 'text-anchor': 'middle', 'class': 'rk-dl' });
        var hit = svg('rect', { x: x0 + band * ci, y: top, width: band, height: H - bottom - top, 'class': 'rk-hit' }, el);
        marks.push({ hit: hit, grp: grp, ci: ci, cx: cx });
      });
      svg('line', { x1: x0, x2: x1, y1: Math.round(zero) + 0.5, y2: Math.round(zero) + 0.5, 'class': 'rk-base' }, g);
    } else {
      var rowH = Math.max(30, nS * (Math.min(18, 24) + gap) + 12);
      var bwH = Math.min(single || stacked ? 18 : 14, (rowH - 12 - gap * (nS - 1)) / nS);
      var labW = Math.min(Math.max.apply(null, catW.concat([0])) + 16, W * 0.38);
      var topH = 8, bottomH = 26 + (spec.yLabel ? 16 : 0);
      H = topH + rowH * cats.length + bottomH;
      var tipW = labelsOn ? Math.max.apply(null, vals.map(function (v) { return textWidth(fmt(v, vf), 12, 600); })) + 10 : 10;
      var xa = labW, xb = W - tipW - 4;
      var xs = linear(vmin, vmax, xa, xb, { ticks: Math.max(2, Math.floor((xb - xa) / 90)) });
      var xf = tickFmt(xs, spec.valueFormat);
      el = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', role: 'img', tabindex: 0 }, box);
      el.style.height = 'auto';
      var g2 = svg('g', {}, el), y0 = topH + rowH * cats.length;
      xs.ticks().forEach(function (t) {
        var px = Math.round(xs(t)) + 0.5;
        svg('line', { x1: px, x2: px, y1: topH, y2: y0, 'class': 'rk-grid' }, g2);
        txt(g2, px, y0 + 17, xf(t), { 'text-anchor': 'middle' });
      });
      if (spec.yLabel) txt(g2, xb, H - 2, spec.yLabel, { 'class': 'rk-ax-title', 'text-anchor': 'end' });
      var mg2 = svg('g', {}, el), z = xs(0);
      cats.forEach(function (c, ci) {
        var cy = topH + rowH * (ci + 0.5), gh = bwH * nS + gap * (nS - 1), by = cy - gh / 2;
        txt(g2, labW - 10, cy + 4, ellipsize(c, labW - 12, 12), { 'text-anchor': 'end', 'class': 'rk-cat' });
        var acc = { p: 0, n: 0 }, grp = [];
        series.forEach(function (s, si) {
          var v = s.values[ci]; if (!isNum(v)) return;
          var yy = stacked ? by : by + si * (bwH + gap), a, b;
          if (stacked) { if (v >= 0) { a = acc.p; acc.p += v; b = acc.p; } else { a = acc.n; acc.n += v; b = acc.n; } } else { a = 0; b = v; }
          var xa_ = xs(a), xb_ = xs(b), l = Math.min(xa_, xb_), w = Math.abs(xb_ - xa_);
          var isEnd = !stacked || !series.slice(si + 1).some(function (t2) { return isNum(t2.values[ci]) && (t2.values[ci] >= 0) === (v >= 0); });
          if (stacked && si > 0) { if (v >= 0) { l += gap; w -= gap; } else w -= gap; }
          grp.push(svg('path', { d: barPath(r1(l), r1(yy), Math.max(0, r1(w)), r1(bwH), isEnd ? 4 : 0, v >= 0 ? 'right' : 'left'), style: 'fill:' + colorOf(s, ci), 'class': 'rk-mark' }, mg2));
          if (labelsOn && !stacked) txt(mg2, v >= 0 ? xb_ + 6 : xb_ - 6, yy + bwH / 2 + 4, fmt(v, vf), { 'class': 'rk-dl', 'text-anchor': v >= 0 ? 'start' : 'end' });
        });
        if (labelsOn && stacked) txt(mg2, xs(acc.p) + 6, cy + 4, fmt(acc.p + acc.n, vf), { 'class': 'rk-dl' });
        var hit = svg('rect', { x: 0, y: topH + rowH * ci, width: W, height: rowH, 'class': 'rk-hit' }, el);
        marks.push({ hit: hit, grp: grp, ci: ci, cy: cy });
      });
      svg('line', { x1: Math.round(z) + 0.5, x2: Math.round(z) + 0.5, y1: topH, y2: y0, 'class': 'rk-base' }, g2);
    }
    function show(m) {
      el.classList.add('rk-hovering');
      marks.forEach(function (o) { o.grp.forEach(function (p) { p.classList.toggle('rk-dimmed', o !== m); p.classList.toggle('rk-lift', o === m); }); });
      var rows = series.map(function (s) { return { color: colorOf(s, m.ci), rect: true, value: fmt(s.values[m.ci], vf), name: single ? null : s.name }; });
      if (stacked) rows.push({ value: fmt(series.reduce(function (a, s) { return a + (isNum(s.values[m.ci]) ? s.values[m.ci] : 0); }, 0), vf), name: '合计' });
      var k = el.getBoundingClientRect().width / W;
      tip.show((m.cx != null ? m.cx : W * 0.6) * k, (m.cy != null ? m.cy : H * 0.4) * k, String(cats[m.ci]), rows);
    }
    function clear() { tip.hide(); el.classList.remove('rk-hovering'); marks.forEach(function (o) { o.grp.forEach(function (p) { p.classList.remove('rk-dimmed', 'rk-lift'); }); }); }
    marks.forEach(function (m) { m.hit.addEventListener('pointerenter', function () { show(m); }); });
    el.addEventListener('pointerleave', clear);
    var cur = -1;
    el.addEventListener('keydown', function (e) {
      var fwd = e.key === 'ArrowRight' || e.key === 'ArrowDown', back = e.key === 'ArrowLeft' || e.key === 'ArrowUp';
      if (!fwd && !back) return;
      e.preventDefault(); cur = clamp(cur + (fwd ? 1 : -1), 0, marks.length - 1); show(marks[cur]);
    });
    el.addEventListener('blur', clear);
  }

  /* ---------------- histogram ---------------- */
  function renderHist(box, spec, W) {
    var edges = spec.edges, series = (spec.series || []).map(function (s) { return Object.assign({}, s); });
    assignColors(series, null);
    var nb = edges.length - 1, H = spec.height || 260, vals = [];
    series.forEach(function (s) { s.counts.forEach(function (v) { if (isNum(v)) vals.push(v); }); });
    var top = spec.yLabel ? 26 : 10, bottom = 26 + (spec.xLabel ? 18 : 0);
    var y = linear(0, Math.max.apply(null, vals), H - bottom, top);
    var yf = tickFmt(y, spec.yFormat);
    var left = Math.max.apply(null, y.ticks().map(function (t) { return textWidth(yf(t)); })) + 14;
    var x0 = left, x1 = W - 12;
    var x = spec.xLog ? logScale(edges[0], edges[nb], x0, x1) : linear(edges[0], edges[nb], x0, x1, { nice: false, ticks: Math.max(2, Math.floor((x1 - x0) / 80)) });
    if (spec.xLog) { x0 = x(edges[0]); x1 = x(edges[nb]); }
    legend(box, series.map(function (s) { return { name: s.name, color: s._c, kind: series.length > 1 ? 'line' : 'rect' }; }), 'rect');
    var el = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', role: 'img', tabindex: 0 }, box);
    el.style.height = 'auto';
    var g = svg('g', {}, el);
    axisY(g, y, left, W - 12, yf, spec.yLabel, H);
    axisX(g, x, x.ticks(), H - bottom, tickFmt(x, spec.xFormat), left, W - 12);
    if (spec.xLabel) txt(g, W - 12, H - 2, spec.xLabel, { 'class': 'rk-ax-title', 'text-anchor': 'end' });
    var mg = svg('g', {}, el), y0 = y(0);
    series.forEach(function (s, si) {
      if (series.length === 1) {
        s.counts.forEach(function (v, i) {
          if (!isNum(v) || v <= 0) return;
          var a = x(edges[i]) + 1, b = x(edges[i + 1]) - 1, w = Math.max(0.5, b - a);
          svg('path', { d: barPath(r1(a), r1(y(v)), r1(w), r1(y0 - y(v)), 2, 'top'), style: 'fill:' + s._c, 'class': 'rk-mark' }, mg);
        });
      } else {
        var d = 'M' + r1(x(edges[0])) + ',' + r1(y0);
        s.counts.forEach(function (v, i) { var yy = r1(y(isNum(v) ? v : 0)); d += 'V' + yy + 'H' + r1(x(edges[i + 1])); });
        d += 'V' + r1(y0);
        svg('path', { d: d + 'Z', style: 'fill:' + s._c, 'fill-opacity': 0.1, 'class': 'rk-mark' }, mg);
        svg('path', { d: d, fill: 'none', style: 'stroke:' + s._c, 'stroke-width': 2, 'stroke-linejoin': 'round', 'class': 'rk-mark' }, mg);
      }
    });
    svg('line', { x1: left, x2: W - 12, y1: Math.round(y0) + 0.5, y2: Math.round(y0) + 0.5, 'class': 'rk-base' }, g);
    (spec.refs || []).forEach(function (r) {
      if (!isNum(r.x)) return;
      var px = x(r.x);
      svg('line', { x1: px, x2: px, y1: top, y2: y0, 'class': 'rk-refl' }, mg);
      if (r.label) txt(mg, px + 5, top + 10, r.label, {});
    });
    var tip = tooltip(box), hov = svg('g', { 'pointer-events': 'none' }, el), xf = spec.xFormat || 'auto';
    function show(i) {
      hov.textContent = '';
      var a = x(edges[i]), b = x(edges[i + 1]);
      svg('rect', { x: a, y: top, width: Math.max(1, b - a), height: y0 - top, style: 'fill:var(--rk-ink)', 'fill-opacity': 0.05 }, hov);
      var k = el.getBoundingClientRect().width / W;
      tip.show((a + b) / 2 * k, (top + (y0 - top) * 0.3) * k, '[' + fmt(edges[i], xf) + ', ' + fmt(edges[i + 1], xf) + ')',
        series.map(function (s) { return { color: s._c, rect: series.length === 1, value: fmt(s.counts[i], spec.yFormat), name: series.length > 1 ? s.name : (spec.yLabel || null) }; }));
    }
    function clear() { tip.hide(); hov.textContent = ''; }
    var cur = -1;
    el.addEventListener('pointermove', function (e) {
      var q = pointer(e, el, W), v = x.invert(q[0]);
      if (v < edges[0] || v >= edges[nb]) { clear(); return; }
      for (var i = 0; i < nb; i++) if (v < edges[i + 1]) { cur = i; show(i); return; }
    });
    el.addEventListener('pointerleave', clear);
    el.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault(); cur = clamp(cur + (e.key === 'ArrowRight' ? 1 : -1), 0, nb - 1); show(cur);
    });
    el.addEventListener('blur', clear);
  }

  /* ---------------- heatmap ---------------- */
  function renderHeat(box, spec, W) {
    var rows = spec.rows, cols = spec.cols, M = spec.values, kind = spec.scale || 'diverging';
    var all = []; M.forEach(function (r) { r.forEach(function (v) { if (isNum(v)) all.push(v); }); });
    var dom = spec.domain || (kind === 'diverging' ? (function () { var a = Math.max.apply(null, all.map(Math.abs)); return [-a, a]; })() : [Math.min.apply(null, all), Math.max.apply(null, all)]);
    var col = heatScale(kind), vf = spec.valueFormat || '.2f';
    var t = function (v) { return (v - dom[0]) / (dom[1] - dom[0]); };
    var labW = Math.min(Math.max.apply(null, rows.map(function (r) { return textWidth(r, 12); })) + 16, W * 0.32);
    var avail = W - labW - 4, cw = Math.min(spec.cell || 64, avail / cols.length), ch = Math.min(cw, spec.cellHeight || 34);
    if (ch < 22) ch = 22;
    var colW = Math.max.apply(null, cols.map(function (c) { return textWidth(c, 12); }));
    var rot = colW > cw - 6;
    // rotated labels run up and to the right: shorten the ones that would leave the chart
    var colLab = cols.map(function (c, j) {
      if (!rot) return c;
      var cx = labW + cw * (j + 0.5);
      return ellipsize(c, Math.min(120, (W - cx) / 0.7071 - 4), 12);
    });
    var topH = rot ? Math.max.apply(null, colLab.map(function (c) { return textWidth(c, 12); })) * 0.7071 + 16 : 22;
    var H = topH + ch * rows.length + 4;
    var el = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', role: 'img', tabindex: 0 }, box);
    el.style.height = 'auto';
    var g = svg('g', {}, el), mg = svg('g', {}, el), cells = [];
    cols.forEach(function (c, j) {
      var cx = labW + cw * (j + 0.5);
      if (rot) txt(g, cx - 2, topH - 8, colLab[j], { 'text-anchor': 'start', 'class': 'rk-cat', transform: 'rotate(-45 ' + (cx - 2) + ' ' + (topH - 8) + ')' });
      else txt(g, cx, topH - 8, c, { 'text-anchor': 'middle', 'class': 'rk-cat' });
    });
    // cell labels: all or none, so a missing number never looks like a missing value
    var labMax = Math.max.apply(null, all.map(function (v) { return textWidth(fmt(v, vf), 11, 600); }));
    var showLab = spec.labels !== false && labMax < cw - 8 && ch >= 20;
    rows.forEach(function (r, i) {
      var cy = topH + ch * i;
      txt(g, labW - 10, cy + ch / 2 + 4, ellipsize(r, labW - 12, 12), { 'text-anchor': 'end', 'class': 'rk-cat' });
      cols.forEach(function (c, j) {
        var v = M[i][j], fill = isNum(v) ? col(t(v)) : 'transparent';
        var cell = svg('rect', { x: r1(labW + cw * j + 1), y: r1(cy + 1), width: r1(cw - 2), height: r1(ch - 2), rx: 2, fill: fill, 'class': 'rk-mark' }, mg);
        if (!isNum(v)) cell.setAttribute('style', 'stroke:var(--rk-line);fill:none');
        if (showLab && isNum(v)) txt(mg, labW + cw * (j + 0.5), cy + ch / 2 + 4, fmt(v, vf), { 'text-anchor': 'middle', 'class': 'rk-dl-in', style: 'fill:' + inkOn(fill) });
        cells.push({ el: cell, i: i, j: j, x: labW + cw * (j + 0.5), y: cy + ch / 2 });
      });
    });
    // scale legend
    var sc = h('div', 'rk-scale', box), stops = [];
    for (var k = 0; k <= 10; k++) stops.push(col(k / 10) + ' ' + (k * 10) + '%');
    h('span', '', sc, fmt(dom[0], vf));
    var rp = h('span', 'rk-ramp', sc); rp.style.background = 'linear-gradient(90deg,' + stops.join(',') + ')';
    h('span', '', sc, fmt(dom[1], vf));
    if (spec.scaleLabel) h('span', '', sc, spec.scaleLabel);
    var tip = tooltip(box), cur = -1;
    function show(c) {
      cells.forEach(function (o) { o.el.classList.toggle('rk-lift', o === c); });
      var kk = el.getBoundingClientRect().width / W;
      tip.show(c.x * kk, c.y * kk, rows[c.i] + ' × ' + cols[c.j], [{ value: fmt(M[c.i][c.j], vf), name: spec.valueName || null }]);
    }
    function clear() { tip.hide(); cells.forEach(function (o) { o.el.classList.remove('rk-lift'); }); }
    cells.forEach(function (c, n) { c.el.addEventListener('pointerenter', function () { cur = n; show(c); }); });
    el.addEventListener('pointerleave', clear);
    el.addEventListener('keydown', function (e) {
      var d = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols.length, ArrowUp: -cols.length }[e.key];
      if (!d) return; e.preventDefault(); cur = clamp(cur + d, 0, cells.length - 1); show(cells[cur]);
    });
    el.addEventListener('blur', clear);
  }

  /* ---------------- table view (the accessible twin of every chart) ---------------- */
  function dataTable(spec) {
    var head = [], body = [];
    if (spec.type === 'line') {
      head = [spec.xName || spec.xLabel || 'x'].concat(spec.series.map(function (s) { return s.name; }));
      spec.x.forEach(function (xv, i) { body.push([typeof xv === 'string' ? xv : fmt(xv, spec.xFormat)].concat(spec.series.map(function (s) { return fmt(s.values[i], spec.yFormat); }))); });
    } else if (spec.type === 'bar') {
      head = [spec.categoryName || '类别'].concat(spec.series.map(function (s) { return s.name; }));
      spec.categories.forEach(function (c, i) { body.push([c].concat(spec.series.map(function (s) { return fmt(s.values[i], spec.valueFormat); }))); });
    } else if (spec.type === 'hist') {
      head = ['区间'].concat(spec.series.map(function (s) { return s.name; }));
      for (var i = 0; i < spec.edges.length - 1; i++) body.push(['[' + fmt(spec.edges[i], spec.xFormat) + ', ' + fmt(spec.edges[i + 1], spec.xFormat) + ')'].concat(spec.series.map(function (s) { return fmt(s.counts[i], spec.yFormat); })));
    } else if (spec.type === 'heatmap') {
      head = [''].concat(spec.cols);
      spec.rows.forEach(function (r, i) { body.push([r].concat(spec.values[i].map(function (v) { return fmt(v, spec.valueFormat || '.2f'); }))); });
    } else if (spec.type === 'scatter') {
      head = ['序列', spec.xLabel || 'x', spec.yLabel || 'y'];
      spec.series.forEach(function (s) { s.x.forEach(function (xv, i) { if (body.length < 500) body.push([s.name, fmt(xv, spec.xFormat), fmt(s.y[i], spec.yFormat)]); }); });
    }
    var wrap = h('div', 'rk-tablewrap'), tb = h('table', 'rk-table', wrap), tr = h('tr', '', h('thead', '', tb));
    head.forEach(function (c, j) { h('th', j ? 'rk-r' : '', tr, c); });
    var tbody = h('tbody', '', tb);
    body.forEach(function (r) { var row = h('tr', '', tbody); r.forEach(function (c, j) { h('td', j ? 'rk-r' : '', row, c); }); });
    return wrap;
  }

  /* ---------------- chart driver ---------------- */
  var PRINT_W = 660;
  var RENDER = { line: renderXY, scatter: renderXY, bar: renderBar, hist: renderHist, heatmap: renderHeat };
  var charts = [];
  function readSpec(box) {
    if (box._spec) return box._spec;
    var s = box.querySelector('script[type="application/json"]');
    try { box._spec = JSON.parse(s ? s.textContent : box.getAttribute('data-spec')); } catch (e) { box._spec = null; }
    return box._spec;
  }
  function draw(box) {
    var spec = readSpec(box);
    var keep = box.querySelector('script[type="application/json"]');
    Array.prototype.slice.call(box.childNodes).forEach(function (n) { if (n !== keep) box.removeChild(n); });
    if (!spec || !RENDER[spec.type]) { h('p', 'rk-muted', box, '图表数据无法读取。'); return; }
    var W = Math.max(240, Math.floor(box.clientWidth));
    box._w = W;
    if (root.getAttribute('data-print') === '1') {
      // print: draw at the A4 text width (182 mm = 688 px) so fonts keep their size after scaling
      var paper = box.closest('.rk-paper');
      if (paper) {
        var cs = getComputedStyle(paper), inner = paper.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
        W = Math.max(240, Math.floor(PRINT_W * box.clientWidth / inner));
      }
    }
    try { RENDER[spec.type](box, spec, W); }
    catch (e) { h('p', 'rk-muted', box, '图表渲染失败：' + e.message); if (window.console) console.error(e); }
    var fig = box.closest('.rk-fig'), cap = fig && fig.querySelector('.rk-capt'), s2 = box.querySelector('svg');
    if (s2 && cap) s2.setAttribute('aria-label', cap.textContent);
  }
  function setupFigure(box) {
    var fig = box.closest('.rk-fig');
    if (!fig || fig._rk) return;
    fig._rk = true;
    var tools = fig.querySelector('.rk-figtools');
    if (!tools) {
      var head = fig.querySelector('.rk-fighead');
      if (!head) return;
      tools = h('div', 'rk-figtools', head);
    }
    var btn = h('button', 'rk-btn', tools, '表格');
    btn.type = 'button'; btn.setAttribute('aria-pressed', 'false');
    var view = h('div', 'rk-dataview'); view.hidden = true;
    var note = fig.querySelector('.rk-source');
    fig.insertBefore(view, note || null);
    btn.onclick = function () {
      var on = view.hidden;
      if (on && !view.firstChild) view.appendChild(dataTable(readSpec(box)));
      view.hidden = !on; btn.setAttribute('aria-pressed', String(on));
    };
  }
  function spark(el) {
    var v; try { v = JSON.parse(el.getAttribute('data-spark')); } catch (e) { return; }
    el.textContent = '';
    var W = Math.max(60, el.clientWidth), H = 30, vals = v.filter(isNum);
    if (vals.length < 2) return;
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var y = function (t) { return hi === lo ? H / 2 : 3 + (1 - (t - lo) / (hi - lo)) * (H - 6); };
    var x = function (i) { return 2 + i / (v.length - 1) * (W - 6); };
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, 'aria-hidden': 'true' }, el), d = '', pen = false, last = null;
    v.forEach(function (t, i) { if (!isNum(t)) { pen = false; return; } d += (pen ? 'L' : 'M') + r1(x(i)) + ',' + r1(y(t)); pen = true; last = [x(i), y(t)]; });
    svg('path', { d: d, fill: 'none', style: 'stroke:var(--rk-gray)', 'stroke-width': 1.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, s);
    if (last) svg('circle', { cx: r1(last[0]), cy: r1(last[1]), r: 3, style: 'fill:var(--rk-accent);stroke:var(--rk-paper);stroke-width:1.5' }, s);
  }
  function renderAll(force) {
    registry = {}; nextSlot = 1;
    doc.querySelectorAll('.rk-chart').forEach(function (box) {
      if (force || box._w !== Math.floor(box.clientWidth)) draw(box); else { var sp = readSpec(box); if (sp && sp.series) assignColors(sp.series.map(function (s) { return Object.assign({}, s); }), sp.emphasis == null ? null : [].concat(sp.emphasis)); }
    });
    doc.querySelectorAll('.rk-spark[data-spark]').forEach(spark);
  }

  /* ---------------- TOC, copy, math, tools ---------------- */
  function buildToc() {
    var nav = doc.querySelector('.rk-toc');
    if (!nav || nav.getAttribute('data-static') != null) return;
    var hs = doc.querySelectorAll('.rk-paper h2[id], .rk-paper h3[id]');
    if (!hs.length) return;
    nav.textContent = '';
    h('div', 'rk-toc-label', nav, '目录');
    var ol = h('ol', '', nav), links = [];
    hs.forEach(function (hd) {
      if (hd.closest('.rk-foot')) return;
      var li = h('li', '', ol), a = h('a', hd.tagName === 'H3' ? 'rk-sub' : '', li);
      a.href = '#' + hd.id;
      var n = hd.querySelector('.rk-n');
      if (n) h('span', 'rk-toc-n', a, n.textContent);
      var label = Array.prototype.filter.call(hd.childNodes, function (c) { return !(c.classList && c.classList.contains('rk-n')); }).map(function (c) { return c.textContent; }).join('').trim();
      h('span', '', a, label);
      links.push([hd, a]);
    });
    if (!('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function () {
      var cur = links[0];
      links.forEach(function (l) { if (l[0].getBoundingClientRect().top < 120) cur = l; });
      links.forEach(function (l) { l[1].classList.toggle('on', l === cur); });
    }, { rootMargin: '0px 0px -60% 0px', threshold: [0, 1] });
    links.forEach(function (l) { io.observe(l[0]); });
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
    doc.querySelectorAll('.rk-im[data-tex]').forEach(function (t) {
      try { window.katex.render(t.getAttribute('data-tex'), t, { throwOnError: false }); } catch (e) { }
    });
  }
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
      var mq = matchMedia('(prefers-color-scheme: dark)');
      var on = function () { if (!root.getAttribute('data-theme')) renderAll(true); };
      if (mq.addEventListener) mq.addEventListener('change', on); else if (mq.addListener) mq.addListener(on);
    }
    window.addEventListener('beforeprint', function () { root.setAttribute('data-print', '1'); renderAll(true); });
    window.addEventListener('afterprint', function () { root.removeAttribute('data-print'); renderAll(true); });
  }

  function init() {
    buildToc(); copyButtons(); math(); tools();
    doc.querySelectorAll('.rk-chart').forEach(setupFigure);
    renderAll(true);
    var t = null;
    var again = function () { clearTimeout(t); t = setTimeout(function () { renderAll(false); }, 120); };
    if ('ResizeObserver' in window) { var ro = new ResizeObserver(again); doc.querySelectorAll('.rk-chart,.rk-spark').forEach(function (b) { ro.observe(b); }); }
    else window.addEventListener('resize', again);
  }
  window.reportkit = { render: renderAll, format: fmt };
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', init); else init();
})();
