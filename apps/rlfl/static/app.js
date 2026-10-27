/* FedNet Edu - antarmuka simulator Federated Learning untuk optimasi jaringan */
'use strict';

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
const SVGNS = 'http://www.w3.org/2000/svg';
const GLOBAL_COLOR = '#d6006f';

let ST = null;                                  // state dari server
const RES = { supervised: null, qrouting: null, rl: null };
let MODE = 'supervised';
const VIEW = { supervised: {}, qrouting: {} };  // pilihan src/dst/ronde per mode

// ------------------------------------------------------------------ util
async function api(method, url, body, isForm) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opt.body = body;
    else { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  }
  const r = await fetch(url, opt);
  let data = null;
  try { data = await r.json(); } catch (e) { /* noop */ }
  if (!r.ok) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}
function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
}
function sv(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toFixed(d);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function status(id, msg, isErr) {
  const e = document.getElementById(id);
  if (!e) return;
  e.textContent = msg || '';
  e.style.color = isErr ? 'var(--critical)' : '';
}
function bytesFmt(b) {
  if (b == null) return '–';
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(2)} MB`;
}
function lerpColor(a, b, t) {
  const pa = a.match(/\w\w/g).map(h => parseInt(h, 16));
  const pb = b.match(/\w\w/g).map(h => parseInt(h, 16));
  return '#' + pa.map((x, i) => Math.round(x + (pb[i] - x) * t).toString(16).padStart(2, '0')).join('');
}
function utilColor(u) {            // 0 hijau -> 0.6 kuning -> 0.95 merah
  if (u <= 0.6) return lerpColor('1baf7a', 'eda100', u / 0.6);
  return lerpColor('eda100', 'e34948', Math.min(1, (u - 0.6) / 0.35));
}
function delayColor(v, lo, hi) {   // skala log biru -> kuning -> merah
  const t = Math.max(0, Math.min(1, (Math.log(v) - Math.log(lo)) / ((Math.log(hi) - Math.log(lo)) || 1)));
  return t < 0.5 ? lerpColor('3b6fd8', 'f2c230', t * 2) : lerpColor('f2c230', 'd7263d', (t - 0.5) * 2);
}
const regionColor = rid => (ST.net.clients.find(c => c.client_id === rid) || {}).color || '#888888';

// ------------------------------------------------------------------ parameter FL
const SUP_PARAMS = [
  { key: 'model', label: 'Model', type: 'select', options: [['mlp', 'MLP (1 hidden layer, ReLU)'], ['linear', 'Regresi linear']], help: 'Model prediksi delay link. Linear tidak mampu menangkap lonjakan antrian (non-linear).' },
  { key: 'hidden_units', label: 'Neuron hidden', type: 'number', min: 2, max: 128, step: 1, help: 'Kapasitas MLP.', dep: c => c.model === 'mlp' },
  { key: 'rounds', label: 'Ronde komunikasi', type: 'number', min: 1, max: 200, step: 1, help: 'Jumlah siklus kirim model global → latih lokal → agregasi.' },
  { key: 'local_epochs', label: 'Epoch lokal / ronde', type: 'number', min: 1, max: 50, step: 1, help: 'Banyak epoch pelatihan di tiap klien sebelum dikirim.' },
  { key: 'batch_size', label: 'Batch size', type: 'number', min: 1, max: 1024, step: 1, help: 'Ukuran mini-batch SGD lokal.' },
  { key: 'learning_rate', label: 'Learning rate', type: 'number', min: 0.0001, max: 1, step: 0.005, help: 'Laju belajar SGD.' },
  { key: 'client_fraction', label: 'Fraksi klien / ronde', type: 'number', min: 0.1, max: 1, step: 0.05, help: 'Proporsi klien yang dipilih acak tiap ronde (C pada FedAvg).' },
  { key: 'dropout_prob', label: 'Peluang klien gagal kirim', type: 'number', min: 0, max: 0.9, step: 0.05, help: 'Simulasi straggler / koneksi putus.' },
  { key: 'aggregation', label: 'Agregasi server', type: 'select', options: [['fedavg', 'FedAvg (rata-rata berbobot)'], ['fedprox', 'FedProx (term proximal)'], ['fedmedian', 'FedMedian (median per koordinat)'], ['trimmed_mean', 'Trimmed Mean (robust)']], help: 'Cara server menggabungkan bobot model klien.' },
  { key: 'prox_mu', label: 'μ FedProx', type: 'number', min: 0, max: 1, step: 0.005, help: 'Kekuatan penarik model lokal ke model global.', dep: c => c.aggregation === 'fedprox' },
  { key: 'trim_ratio', label: 'Rasio trim', type: 'number', min: 0, max: 0.45, step: 0.05, help: 'Proporsi nilai ekstrem dibuang tiap sisi.', dep: c => c.aggregation === 'trimmed_mean' },
  { key: 'weighted', label: 'Bobot = jumlah sampel', type: 'bool', help: 'FedAvg berbobot n_k/n, atau rata-rata biasa.', dep: c => c.aggregation === 'fedavg' || c.aggregation === 'fedprox' },
  { key: 'dp_clip', label: 'DP: clip norma update', type: 'number', min: 0, max: 10, step: 0.1, help: '0 = mati. Membatasi besar update tiap klien (privasi diferensial).' },
  { key: 'dp_noise', label: 'DP: noise σ (× clip)', type: 'number', min: 0, max: 2, step: 0.01, help: 'Noise Gaussian ditambahkan ke update klien.', dep: c => Number(c.dp_clip) > 0 },
  { key: 'accuracy_tolerance_pct', label: 'Toleransi akurasi (%)', type: 'number', min: 1, max: 100, step: 1, help: 'Prediksi dianggap benar bila |error| ≤ toleransi × delay sebenarnya.' },
  { key: 'train_centralized', label: 'Latih baseline terpusat', type: 'bool', help: 'Pembanding: semua data digabung di satu server (melanggar privasi).' },
];
const Q_PARAMS = [
  { key: 'rounds', label: 'Ronde komunikasi', type: 'number', min: 1, max: 200, step: 1, help: 'Jumlah siklus tabel-Q global → eksplorasi lokal → agregasi.' },
  { key: 'episodes_per_round', label: 'Episode (paket) / klien / ronde', type: 'number', min: 10, max: 5000, step: 10, help: 'Banyak paket yang dirutekan agen tiap ronde.' },
  { key: 'alpha', label: 'α learning rate', type: 'number', min: 0.01, max: 1, step: 0.05, help: 'Laju update Q-value.' },
  { key: 'epsilon', label: 'ε eksplorasi awal', type: 'number', min: 0, max: 1, step: 0.05, help: 'Peluang memilih tetangga acak (ε-greedy).' },
  { key: 'epsilon_decay', label: 'Peluruhan ε / ronde', type: 'number', min: 0.5, max: 1, step: 0.01, help: 'ε dikalikan faktor ini tiap ronde.' },
  { key: 'epsilon_min', label: 'ε minimum', type: 'number', min: 0, max: 0.5, step: 0.01, help: 'Batas bawah eksplorasi.' },
  { key: 'max_hops_factor', label: 'Batas hop (× jumlah node)', type: 'number', min: 0.5, max: 5, step: 0.5, help: 'Paket dibuang bila melebihi batas hop (loop).' },
  { key: 'cost_noise_pct', label: 'Variasi delay dialami (%)', type: 'number', min: 0, max: 50, step: 1, help: 'Fluktuasi delay tiap paket (lingkungan stokastik).' },
  { key: 'client_fraction', label: 'Fraksi klien / ronde', type: 'number', min: 0.1, max: 1, step: 0.05, help: 'Proporsi agen yang ikut tiap ronde.' },
  { key: 'dropout_prob', label: 'Peluang klien gagal kirim', type: 'number', min: 0, max: 0.9, step: 0.05, help: 'Simulasi straggler.' },
  { key: 'aggregation', label: 'Agregasi tabel-Q', type: 'select', options: [['visit_weighted', 'Rata-rata berbobot kunjungan'], ['fedavg', 'FedAvg (rata-rata biasa)']], help: 'Berbobot kunjungan: entri hanya dipengaruhi agen yang pernah mengalaminya.' },
  { key: 'train_centralized', label: 'Latih baseline terpusat', type: 'bool', help: 'Satu agen yang melihat paket dari semua region.' },
];

function renderParams(container, specs, values) {
  container.innerHTML = '';
  for (const p of specs) {
    let input;
    if (p.type === 'select') {
      input = el('select', { 'data-key': p.key }, p.options.map(([v, l]) => el('option', { value: v, text: l })));
      input.value = values[p.key];
    } else if (p.type === 'bool') {
      input = el('select', { 'data-key': p.key }, [el('option', { value: 'true', text: 'Ya' }), el('option', { value: 'false', text: 'Tidak' })]);
      input.value = values[p.key] ? 'true' : 'false';
    } else {
      input = el('input', { type: 'number', 'data-key': p.key, min: p.min, max: p.max, step: p.step, value: values[p.key] });
    }
    input.addEventListener('change', () => updateDeps(container, specs));
    const lab = el('label', {}, p.sign ? [el('span', { class: `sign ${p.sign}`, html: { plus: '+', minus: '&minus;', pm: '&plusmn;' }[p.sign] }), p.label] : [p.label]);
    container.append(el('div', { class: `param${p.sign ? ' s-' + p.sign : ''}`, 'data-param': p.key }, [lab, input, el('small', { text: p.help })]));
  }
  updateDeps(container, specs);
}
function readParams(container, specs) {
  const out = {};
  for (const p of specs) {
    const v = $(`[data-key="${p.key}"]`, container).value;
    out[p.key] = p.type === 'number' ? Number(v) : p.type === 'bool' ? v === 'true' : v;
  }
  return out;
}
function updateDeps(container, specs) {
  const cur = readParams(container, specs);
  for (const p of specs) {
    if (!p.dep) continue;
    $(`[data-param="${p.key}"]`, container).classList.toggle('disabled', !p.dep(cur));
  }
}

// ------------------------------------------------------------------ state & sidebar
async function loadState(payload) {
  ST = payload || await api('GET', '/api/state');
  renderSidebar();
  renderTopology();
  renderDataPanel();
  if (!ST.results.rl) { RES.rl = null; $('#rl-results').hidden = true; }
  for (const mode of ['supervised', 'qrouting']) {
    if (!ST.results[mode]) { RES[mode] = null; $(mode === 'supervised' ? '#sup-results' : '#q-results').hidden = true; }
  }
  if (MODE === 'download') loadFiles();
}

function renderSidebar() {
  const net = ST.net, cfg = ST.cfg;
  $('#topo-meta').textContent = `${net.nodes.length} node · ${net.links.length} link · ${net.clients.length} klien · area ${net.area.width}×${net.area.height} m`;
  const eb = $('#net-errors');
  eb.hidden = !ST.errors.length;
  eb.innerHTML = ST.errors.map(esc).join('<br>');
  const t = cfg.topology;
  $('#g-nodes').value = net.nodes.length; $('#g-clients').value = net.clients.length;
  $('#g-w').value = net.area.width; $('#g-h').value = net.area.height;
  $('#g-k').value = t.k_neighbors || 3; $('#g-seed').value = t.seed ?? 42;

  const cb = $('#client-body'); cb.innerHTML = '';
  for (const c of net.clients) {
    const n = net.nodes.filter(x => x.region === c.client_id).length;
    cb.append(el('tr', { 'data-cid': c.client_id }, [
      el('td', {}, [el('span', { class: 'dot', style: `background:${c.color}` }), el('b', { text: c.client_id }), el('br'),
        el('input', { class: 'client-name', 'data-f': 'name', value: c.name, title: `${n} node` })]),
      el('td', {}, el('input', { type: 'number', 'data-f': 'traffic_intensity', min: 0.05, max: 1.2, step: 0.05, value: c.traffic_intensity })),
      el('td', {}, el('input', { type: 'number', 'data-f': 'burstiness', min: 0, max: 0.4, step: 0.01, value: c.burstiness })),
      el('td', {}, el('input', { type: 'number', 'data-f': 'peak_hour', min: 0, max: 23, step: 1, value: c.peak_hour })),
      el('td', {}, el('input', { type: 'number', 'data-f': 'dest_locality', min: 0, max: 1, step: 0.05, value: c.dest_locality })),
    ]));
  }
  const regOpts = () => net.clients.map(c => el('option', { value: c.client_id, text: `${c.client_id} ${c.name}` }));
  const nr = $('#new-node-region'); nr.innerHTML = ''; nr.append(el('option', { value: '', text: 'region terdekat' }), ...regOpts());
  const ids = net.nodes.map(n => n.node_id);
  for (const id of ['#e-src', '#e-dst']) {
    const s = $(id), prev = s.value; s.innerHTML = '';
    ids.forEach(n => s.append(el('option', { value: n, text: n })));
    if (ids.includes(prev)) s.value = prev;
  }
  if (!$('#e-dst').value || $('#e-dst').value === $('#e-src').value) $('#e-dst').value = ids[1];

  const nb = $('#node-body'); nb.innerHTML = '';
  for (const n of net.nodes) {
    const sel = el('select', { onchange: e => mutate('POST', '/api/node/region', { node_id: n.node_id, region: e.target.value }) }, regOpts());
    sel.value = n.region;
    nb.append(el('tr', {}, [el('td', {}, [el('span', { class: 'dot', style: `background:${regionColor(n.region)}` }), n.node_id]), el('td', {}, sel),
      el('td', {}, el('button', { class: 'icon-btn', title: 'Hapus node', text: '✕', onclick: () => mutate('DELETE', `/api/node/${n.node_id}`) }))]));
  }
  const lb = $('#link-body'); lb.innerHTML = '';
  for (const l of net.links) {
    lb.append(el('tr', { title: `${l.link_id} · jarak ${l.distance_m} m`, onclick: () => { $('#e-src').value = l.source; $('#e-dst').value = l.target; $('#e-bw').value = l.bandwidth_mbps; $('#e-loss').value = l.loss; $('#e-tw').value = l.traffic_weight; $('#e-delay').value = l.base_delay_ms; } }, [
      el('td', { text: `${l.source}–${l.target}` }), el('td', { text: fmt(l.base_delay_ms, 1) }), el('td', { text: l.bandwidth_mbps }),
      el('td', { text: l.loss }), el('td', { text: l.traffic_weight }),
      el('td', {}, el('button', { class: 'icon-btn', title: 'Hapus link', text: '✕', onclick: e => { e.stopPropagation(); mutate('DELETE', `/api/link/${l.link_id}`); } }))]));
  }
  const s = cfg.simulation;
  $('#s-steps').value = s.time_steps; $('#s-eval').value = s.eval_hour; $('#s-test').value = s.test_ratio;
  $('#s-noise').value = s.noise_pct; $('#s-seed').value = s.seed;
  $('#sup-pairs').value = cfg.evaluation.num_pairs; $('#q-pairs').value = cfg.evaluation.num_pairs;
  if (!$('#sup-params').children.length) renderParams($('#sup-params'), SUP_PARAMS, cfg.supervised);
  if (!$('#q-params').children.length) renderParams($('#q-params'), Q_PARAMS, cfg.qrouting);
  if (!$('#rl-params').children.length) renderParams($('#rl-params'), RL_PARAMS, cfg.rl);
  if (!$('#rl-reward-params').children.length) renderParams($('#rl-reward-params'), RL_REWARD_PARAMS, cfg.rl);
  for (const [id, key] of [['#rl-src', 'source'], ['#rl-dst', 'target']]) {
    const sel = $(id); sel.innerHTML = '';
    ids.forEach(n => sel.append(el('option', { value: n, text: n })));
    sel.value = cfg.rl[key];
  }
}

async function mutate(method, url, body) {
  try {
    status('topo-status', 'Memproses...');
    const st = await api(method, url, body);
    await loadState(st);
    status('topo-status', 'Topologi diperbarui — data trafik & hasil lama direset.');
  } catch (e) { status('topo-status', e.message, true); }
}

// ------------------------------------------------------------------ topologi
function projector(W, H, pad) {
  const a = ST.net.area;
  const s = Math.min((W - 2 * pad) / a.width, (H - 2 * pad) / a.height);
  const ox = (W - a.width * s) / 2, oy = (H - a.height * s) / 2;
  return { x: v => ox + v * s, y: v => oy + v * s, ix: px => (px - ox) / s, iy: py => (py - oy) / s };
}

function drawNetwork(svg, opt = {}) {
  const W = opt.W || 640, H = opt.H || 480, pad = opt.pad || 22;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';
  const P = projector(W, H, pad);
  const pos = {};
  ST.net.nodes.forEach(n => { pos[n.node_id] = [P.x(n.x), P.y(n.y)]; });
  const gE = sv('g'), gP = sv('g'), gN = sv('g');
  svg.append(gE, gP, gN);
  ST.net.links.forEach((l, i) => {
    const [x1, y1] = pos[l.source], [x2, y2] = pos[l.target];
    let stroke = 'var(--border)', w = opt.small ? 1 : 1.5;
    if (opt.edgeColors) { stroke = opt.edgeColors[i]; w = opt.small ? 2 : 3; }
    const line = sv('line', { x1, y1, x2, y2, stroke, 'stroke-width': w, 'stroke-linecap': 'round' });
    if (opt.edgeTitles) line.append(Object.assign(sv('title'), { textContent: opt.edgeTitles[i] }));
    gE.append(line);
  });
  if (opt.path && opt.path.length > 1) {
    const mid = `arr${Math.random().toString(36).slice(2, 8)}`;
    const defs = sv('defs');
    const mk = sv('marker', { id: mid, viewBox: '0 0 10 10', refX: 5, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto-start-reverse' });
    mk.append(sv('path', { d: 'M0,0 L10,5 L0,10 z', fill: opt.pathColor }));
    defs.append(mk); svg.prepend(defs);
    for (let i = 0; i < opt.path.length - 1; i++) {
      const [x1, y1] = pos[opt.path[i]], [x2, y2] = pos[opt.path[i + 1]];
      gP.append(sv('line', { x1, y1, x2, y2, stroke: opt.pathColor, 'stroke-width': opt.small ? 4.5 : 6, 'stroke-linecap': 'round', opacity: 0.9 }));
      gP.append(sv('line', { x1, y1, x2: (x1 + x2) / 2, y2: (y1 + y2) / 2, stroke: 'transparent', 'stroke-width': 1, 'marker-end': `url(#${mid})` }));
    }
  }
  const r = opt.small ? 5 : 8;
  const showLabels = !opt.small || ST.net.nodes.length <= 40;
  ST.net.nodes.forEach(n => {
    const [x, y] = pos[n.node_id];
    const g = sv('g', { class: 'topo-node', 'data-id': n.node_id, transform: `translate(${x},${y})` });
    g.append(sv('circle', { r, fill: regionColor(n.region), stroke: '#ffffff', 'stroke-width': 1.3 }));
    if (showLabels) {
      const t = sv('text', { y: -r - 3, 'text-anchor': 'middle', 'font-size': opt.small ? 8.5 : 10, fill: 'var(--text-primary)' });
      t.textContent = n.node_id; g.append(t);
    }
    gN.append(g);
  });
  if (opt.path && opt.path.length) {
    const [sx, sy] = pos[opt.path[0]], [tx, ty] = pos[opt.path[opt.path.length - 1]];
    const k = r + 4;
    gN.append(sv('rect', { x: sx - k, y: sy - k, width: 2 * k, height: 2 * k, fill: 'none', stroke: 'var(--text-primary)', 'stroke-width': 2 }));
    gN.append(sv('polygon', { points: star(tx, ty, k + 3, (k + 3) / 2.3), fill: 'var(--text-primary)', opacity: 0.85 }));
  } else if (opt.endpoints) {
    const [s, t] = opt.endpoints; const k = r + 4;
    if (pos[s]) gN.append(sv('rect', { x: pos[s][0] - k, y: pos[s][1] - k, width: 2 * k, height: 2 * k, fill: 'none', stroke: 'var(--critical)', 'stroke-width': 2, 'stroke-dasharray': '3 2' }));
    if (pos[t]) gN.append(sv('polygon', { points: star(pos[t][0], pos[t][1], k + 3, (k + 3) / 2.3), fill: 'var(--critical)', opacity: 0.7 }));
  }
  if (opt.interactive) enableDrag(svg, P);
  return P;
}
function star(cx, cy, R, r) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const a = Math.PI / 5 * i - Math.PI / 2, rr = i % 2 ? r : R;
    pts.push(`${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`);
  }
  return pts.join(' ');
}
function enableDrag(svg, P) {
  let drag = null;
  const pt = e => { const m = svg.getScreenCTM().inverse(); const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m); return [p.x, p.y]; };
  $$('.topo-node', svg).forEach(g => {
    g.addEventListener('pointerdown', e => { drag = { g, id: g.dataset.id, moved: false }; g.setPointerCapture(e.pointerId); });
    g.addEventListener('pointermove', e => {
      if (!drag || drag.g !== g) return;
      const [x, y] = pt(e); drag.moved = true; drag.x = x; drag.y = y;
      g.setAttribute('transform', `translate(${x},${y})`);
    });
    g.addEventListener('pointerup', () => {
      if (drag && drag.moved) mutate('POST', '/api/node/move', { node_id: drag.id, x: P.ix(drag.x), y: P.iy(drag.y) });
      drag = null;
    });
  });
}

function renderTopology() {
  const snap = ST.data && ST.data.snapshot;
  const useUtil = snap && $('#show-util').checked;
  const opt = { interactive: true };
  if (useUtil) {
    opt.edgeColors = snap.map(s => utilColor(s.utilization));
    opt.edgeTitles = snap.map(s => `${s.source}–${s.target}: utilisasi ${(s.utilization * 100).toFixed(0)}%, delay ${s.true_delay_ms.toFixed(2)} ms`);
  }
  const ev = ST.cfg.evaluation;
  opt.endpoints = [ev.viz_source, ev.viz_target];
  drawNetwork($('#topo-svg'), opt);
  const lg = $('#region-legend'); lg.innerHTML = '';
  ST.net.clients.forEach(c => lg.append(el('span', {}, [el('span', { class: 'dot', style: `background:${c.color}` }), `${c.client_id} ${c.name} (intensitas ${c.traffic_intensity})`])));
  if (useUtil) lg.append(el('span', { html: '&nbsp;| link: <b style="color:#1baf7a">sepi</b> → <b style="color:#eda100">sibuk</b> → <b style="color:#e34948">macet</b>' }));
}

// ------------------------------------------------------------------ data panel
function renderDataPanel() {
  const d = ST.data;
  $('#data-stats').hidden = !d;
  if (!d) { status('data-status', 'Data trafik belum dibangkitkan (akan dibangkitkan otomatis saat training).'); return; }
  status('data-status', `${d.total} baris pengukuran lokal · snapshot jam ${d.sim.eval_hour}`);
  const tb = $('#data-body'); tb.innerHTML = '';
  for (const s of d.stats) {
    const mx = Math.max(1, ...s.util_hist);
    const hist = el('span', { class: 'mini-hist', title: 'histogram utilisasi 0..1' },
      s.util_hist.map((v, i) => el('i', { style: `height:${Math.max(1, v / mx * 22)}px;background:${utilColor(i / 10 + 0.05)}` })));
    tb.append(el('tr', {}, [el('td', {}, [el('span', { class: 'dot', style: `background:${s.color}` }), `${s.client_id} ${s.name}`]),
      el('td', { text: `${s.samples} (${s.train}/${s.test})` }), el('td', { text: s.links }),
      el('td', { text: `${fmt(s.util_min)} – ${fmt(s.util_max)}` }), el('td', { text: fmt(s.util_mean) }),
      el('td', { text: fmt(s.delay_max, 1) }), el('td', {}, hist)]));
  }
}
function readSim() {
  return { time_steps: Number($('#s-steps').value), eval_hour: Number($('#s-eval').value), test_ratio: Number($('#s-test').value),
    noise_pct: Number($('#s-noise').value), seed: Number($('#s-seed').value) };
}

// ------------------------------------------------------------------ routing helpers
function edgeIndex() {
  const m = {};
  ST.net.links.forEach((l, i) => { m[`${l.source}|${l.target}`] = i; m[`${l.target}|${l.source}`] = i; });
  return m;
}
function followPath(table, ids, s, d) {
  const idx = Object.fromEntries(ids.map((n, i) => [n, i]));
  let x = idx[s]; const t = idx[d];
  const path = [s], seen = new Set([x]);
  for (let h = 0; h < 2 * ids.length; h++) {
    if (x === t) return path;
    const y = table[t][x];
    if (y == null || y < 0 || seen.has(y)) return null;
    path.push(ids[y]); seen.add(y); x = y;
  }
  return x === t ? path : null;
}
function pathDelay(path, trueW, EI) {
  if (!path) return null;
  let d = 0;
  for (let i = 0; i < path.length - 1; i++) d += trueW[EI[`${path[i]}|${path[i + 1]}`]];
  return d;
}
function gapBadge(g) {
  if (g == null) return el('span', { class: 'badge bad', text: 'gagal' });
  const cls = g < 0.5 ? 'ok' : g < 10 ? 'mid' : 'bad';
  return el('span', { class: `badge ${cls}`, text: g < 0.005 ? 'optimal' : `gap +${g.toFixed(1)}%` });
}

// ------------------------------------------------------------------ training
async function runMode(mode) {
  const isSup = mode === 'supervised';
  const sid = isSup ? 'sup-status' : 'q-status';
  const btn = $(isSup ? '#sup-run' : '#q-run');
  const body = { simulation: readSim(), evaluation: { num_pairs: Number($(isSup ? '#sup-pairs' : '#q-pairs').value) } };
  body[mode] = readParams($(isSup ? '#sup-params' : '#q-params'), isSup ? SUP_PARAMS : Q_PARAMS);
  btn.disabled = true;
  status(sid, 'Melatih model lokal & global... (beberapa detik)');
  const t0 = performance.now();
  try {
    const st = await api('POST', `/api/run/${mode}`, body);
    RES[mode] = st.result;
    delete st.result;
    await loadState(st);
    VIEW[mode] = { src: st.cfg.evaluation.viz_source, dst: st.cfg.evaluation.viz_target, round: RES[mode].rounds.length };
    renderResults(mode);
    status(sid, `Selesai dalam ${((performance.now() - t0) / 1000).toFixed(1)} s · ${RES[mode].rounds.length} ronde · ${RES[mode].n_params} parameter per model`);
  } catch (e) { status(sid, e.message, true); }
  btn.disabled = false;
}

// ------------------------------------------------------------------ hasil
function renderResults(mode) {
  const res = RES[mode];
  const box = $(mode === 'supervised' ? '#sup-results' : '#q-results');
  box.innerHTML = '';
  box.append($('#results-template').content.cloneNode(true));
  box.hidden = false;
  const R = s => $(`[data-role="${s}"]`, box);
  const isSup = mode === 'supervised';
  const models = res.models;
  const glob = models.find(m => m.key === 'global');
  const cent = models.find(m => m.key === 'centralized');
  const locals = models.filter(m => m.kind === 'local');
  const avg = (arr, f) => { const v = arr.map(f).filter(x => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const lastRound = res.rounds[res.rounds.length - 1];

  // KPI
  const kpis = R('kpis');
  const kpi = (cls, label, value, sub) => kpis.append(el('div', { class: `kpi ${cls}` }, [el('div', { class: 'k-label', text: label }), el('div', { class: 'k-value', text: value }), el('div', { class: 'k-sub', text: sub })]));
  if (isSup) {
    kpi('global', 'Akurasi model global', `${fmt(glob.metrics.acc, 1)}%`, `MAE ${fmt(glob.metrics.mae, 3)} ms · R² ${fmt(glob.metrics.r2, 3)}`);
    kpi('local', 'Rata² akurasi lokal-saja', `${fmt(avg(locals, m => m.metrics.acc), 1)}%`, `MAE ${fmt(avg(locals, m => m.metrics.mae), 3)} ms (uji gabungan)`);
    if (cent) kpi('cent', 'Akurasi terpusat', `${fmt(cent.metrics.acc, 1)}%`, `MAE ${fmt(cent.metrics.mae, 3)} ms · batas atas`);
  } else {
    kpi('global', 'Jalur berhasil (global)', `${fmt(glob.routing.success_pct, 1)}%`, `cakupan state ${fmt(glob.coverage_pct, 1)}%`);
    kpi('local', 'Rata² jalur berhasil lokal', `${fmt(avg(locals, m => m.routing.success_pct), 1)}%`, `cakupan ${fmt(avg(locals, m => m.coverage_pct), 1)}%`);
    if (cent) kpi('cent', 'Jalur berhasil terpusat', `${fmt(cent.routing.success_pct, 1)}%`, 'satu agen, semua trafik');
  }
  kpi('global', 'Gap jalur global', `${fmt(glob.routing.gap_pct, 2)}%`, `${fmt(glob.routing.match_pct, 1)}% jalur = optimal`);
  kpi('local', 'Rata² gap jalur lokal', `${fmt(avg(locals, m => m.routing.gap_pct), 2)}%`, `${fmt(avg(locals, m => m.routing.match_pct), 1)}% jalur = optimal`);
  kpi('cent', 'Komunikasi total', bytesFmt(lastRound.cum_bytes), `${res.n_params} parameter × ${res.rounds.length} ronde`);

  // tabel
  const th = R('thead'), tb = R('tbody');
  if (isSup) {
    th.innerHTML = '<tr><th>Model</th><th class="num">MAE (ms)</th><th class="num">RMSE</th><th class="num">R²</th><th class="num">Akurasi</th><th class="num">Akurasi data sendiri</th><th class="num">MAE beban rendah / sedang / tinggi</th><th class="num">Gap jalur</th><th class="num">Jalur optimal</th><th class="num">Rata² delay jalur</th></tr>';
    for (const m of models) {
      const mb = m.mae_by_load;
      tb.append(el('tr', { class: m.key === 'global' ? 'row-global' : m.key === 'oracle' ? 'row-oracle' : '' }, [
        el('td', {}, [el('span', { class: 'dot', style: `background:${m.color}` }), m.label]),
        el('td', { class: 'num', text: fmt(m.metrics.mae, 3) }), el('td', { class: 'num', text: fmt(m.metrics.rmse, 3) }),
        el('td', { class: 'num', text: fmt(m.metrics.r2, 3) }), el('td', { class: 'num', text: `${fmt(m.metrics.acc, 1)}%` }),
        el('td', { class: 'num', text: m.metrics_own ? `${fmt(m.metrics_own.acc, 1)}%` : '–' }),
        el('td', { class: 'num', text: mb ? `${fmt(mb.rendah)} / ${fmt(mb.sedang)} / ${fmt(mb.tinggi)}` : '–' }),
        el('td', { class: 'num' }, gapBadge(m.routing.gap_pct)), el('td', { class: 'num', text: `${fmt(m.routing.match_pct, 1)}%` }),
        el('td', { class: 'num', text: `${fmt(m.routing.avg_delay, 2)} ms` })]));
    }
    R('table-hint').innerHTML = 'Akurasi = % prediksi delay yang error-nya ≤ toleransi, diuji pada data uji <b>gabungan semua klien</b>. "Akurasi data sendiri" = diuji pada data uji klien itu saja — model lokal tampak bagus di datanya sendiri tetapi gagal pada kondisi beban tinggi yang tidak pernah ia lihat (lihat kolom MAE beban tinggi). Gap jalur = rata-rata selisih delay jalur pilihan model terhadap jalur optimal pada ' + res.pairs.length + ' pasangan uji.';
  } else {
    th.innerHTML = '<tr><th>Model</th><th class="num">Jalur berhasil</th><th class="num">Gap jalur</th><th class="num">Jalur optimal</th><th class="num">Rata² delay (berhasil)</th><th class="num">Rata² hop</th><th class="num">Cakupan state</th></tr>';
    for (const m of models) {
      tb.append(el('tr', { class: m.key === 'global' ? 'row-global' : m.key === 'oracle' ? 'row-oracle' : '' }, [
        el('td', {}, [el('span', { class: 'dot', style: `background:${m.color}` }), m.label]),
        el('td', { class: 'num', text: `${fmt(m.routing.success_pct, 1)}%` }), el('td', { class: 'num' }, gapBadge(m.routing.gap_pct)),
        el('td', { class: 'num', text: `${fmt(m.routing.match_pct, 1)}%` }), el('td', { class: 'num', text: `${fmt(m.routing.avg_delay, 2)} ms` }),
        el('td', { class: 'num', text: fmt(m.routing.avg_hops, 2) }), el('td', { class: 'num', text: m.coverage_pct != null ? `${fmt(m.coverage_pct, 1)}%` : '–' })]));
    }
    R('table-hint').innerHTML = 'Jalur berhasil = paket sampai tujuan dengan mengikuti tabel next-hop greedy (tanpa loop). Cakupan state = proporsi pasangan (node, tujuan) yang pernah dialami agen. Agen lokal hanya mengalami paket dari region-nya sehingga banyak rute "asing" gagal; agregasi federated membagikan pengalaman semua agen.';
  }

  // jalur
  const ids = res.node_ids;
  const v = VIEW[mode];
  for (const role of ['src', 'dst']) {
    const s = R(role); ids.forEach(n => s.append(el('option', { value: n, text: n })));
    s.value = role === 'src' ? v.src : v.dst;
    s.addEventListener('change', () => { v[role] = s.value; drawPaths(mode, box); });
  }
  const rg = R('round');
  rg.max = res.rounds.length; rg.value = v.round;
  rg.addEventListener('input', () => { v.round = Number(rg.value); drawPaths(mode, box); });
  R('play').addEventListener('click', () => {
    if (v.timer) { clearInterval(v.timer); v.timer = null; R('play').innerHTML = '&#9654; Putar'; return; }
    v.round = 1; R('play').innerHTML = '&#10074;&#10074; Stop';
    v.timer = setInterval(() => {
      rg.value = v.round; drawPaths(mode, box);
      if (++v.round > res.rounds.length) { clearInterval(v.timer); v.timer = null; v.round = res.rounds.length; R('play').innerHTML = '&#9654; Putar'; }
    }, Math.max(120, 2400 / res.rounds.length));
  });
  R('weights-wrap').hidden = !isSup;
  R('weights').addEventListener('change', () => drawPaths(mode, box));
  drawPaths(mode, box);

  // gambar matplotlib
  const figLink = (role, kind, extra) => { const a = R(role); if (!a) return; if (!isSup && (kind === 'link_weights' || kind === 'data_distribution')) { a.hidden = true; return; } a.href = `/api/figure/${mode}/${kind}.png?dl=1${extra || ''}`; };
  const pairQ = () => `&source=${v.src}&target=${v.dst}`;
  figLink('fig-curves', 'learning_curves'); figLink('fig-compare', 'model_comparison');
  figLink('fig-weights', 'link_weights'); figLink('fig-data', 'data_distribution');
  for (const [role, kind] of [['fig-paths', 'paths'], ['fig-evo', 'global_evolution']]) {
    R(role).addEventListener('click', e => { e.currentTarget.href = `/api/figure/${mode}/${kind}.png?dl=1${pairQ()}`; });
  }

  // grafik
  const charts = R('charts');
  const rounds = res.rounds.map(r => r.round);
  const refs = key => [...locals.map(m => ({ label: `Lokal ${m.client_id}`, color: m.color, value: key(m), dash: '5 4' })),
    ...(cent ? [{ label: 'Terpusat', color: '#85847e', value: key(cent), dash: '2 3' }] : [])];
  if (isSup) {
    lineChart(charts, 'Akurasi (±toleransi, %) — lebih tinggi lebih baik', rounds, res.rounds.map(r => r.acc), refs(m => m.metrics.acc));
    lineChart(charts, 'MAE prediksi delay (ms) — lebih rendah lebih baik', rounds, res.rounds.map(r => r.mae), refs(m => m.metrics.mae));
    lineChart(charts, 'R² — lebih tinggi lebih baik', rounds, res.rounds.map(r => r.r2), refs(m => m.metrics.r2));
    lineChart(charts, 'Gap jalur vs optimal (%) — lebih rendah lebih baik', rounds, res.rounds.map(r => r.gap_pct), refs(m => m.routing.gap_pct));
    groupedBars(charts, 'MAE per tingkat beban link (ms) — model lokal gagal di beban tinggi', models.filter(m => m.mae_by_load),
      [['rendah', '<0.5'], ['sedang', '0.5–0.8'], ['tinggi', '>0.8']]);
  } else {
    lineChart(charts, 'Jalur berhasil (%) — lebih tinggi lebih baik', rounds, res.rounds.map(r => r.success_pct), refs(m => m.routing.success_pct));
    lineChart(charts, 'Gap jalur vs optimal (%) — lebih rendah lebih baik', rounds, res.rounds.map(r => r.gap_pct), refs(m => m.routing.gap_pct));
    lineChart(charts, 'Jalur = optimal (%) — lebih tinggi lebih baik', rounds, res.rounds.map(r => r.match_pct), refs(m => m.routing.match_pct));
    lineChart(charts, 'Paket terkirim saat eksplorasi (%) & ε', rounds, res.rounds.map(r => r.delivered_pct), [], { second: res.rounds.map(r => r.epsilon * 100), secondLabel: 'ε × 100' });
  }
  lineChart(charts, 'Komunikasi kumulatif (KB) — biaya federated', rounds, res.rounds.map(r => r.cum_bytes / 1024), [], { color: '#2a78d6', label: 'Upload + download' });

  // log ronde
  const lh = R('log-head'), lbd = R('log-body');
  const cols = isSup
    ? [['round', 'Ronde'], ['selected', 'Klien terpilih'], ['dropped', 'Gagal kirim'], ['train_loss', 'Loss latih'], ['mae', 'MAE'], ['r2', 'R²'], ['acc', 'Akurasi %'], ['gap_pct', 'Gap %'], ['match_pct', 'Optimal %'], ['cum_bytes', 'Komunikasi']]
    : [['round', 'Ronde'], ['selected', 'Klien terpilih'], ['dropped', 'Gagal kirim'], ['epsilon', 'ε'], ['episodes', 'Episode'], ['delivered_pct', 'Terkirim %'], ['success_pct', 'Berhasil %'], ['gap_pct', 'Gap %'], ['match_pct', 'Optimal %'], ['cum_bytes', 'Komunikasi']];
  lh.innerHTML = '<tr>' + cols.map(c => `<th>${c[1]}</th>`).join('') + '</tr>';
  for (const r of res.rounds) {
    lbd.append(el('tr', {}, cols.map(([k]) => el('td', { text: k === 'cum_bytes' ? bytesFmt(r[k]) : (typeof r[k] === 'number' && !Number.isInteger(r[k]) ? fmt(r[k], 3) : (r[k] === '' ? '–' : r[k] ?? '–')) }))));
  }
}

function drawPaths(mode, box) {
  const res = RES[mode];
  const R = s => $(`[data-role="${s}"]`, box);
  const v = VIEW[mode];
  const ids = res.node_ids;
  const EI = edgeIndex();
  const trueW = res.true_weights;
  const grid = R('path-grid');
  grid.innerHTML = '';
  R('round-label').textContent = `${v.round} / ${res.rounds.length}`;
  if (v.src === v.dst) { grid.append(el('p', { class: 'empty-note', text: 'Pilih source dan target yang berbeda.' })); return; }
  const oracle = res.models.find(m => m.key === 'oracle');
  const optPath = followPath(oracle.table, ids, v.src, v.dst);
  const optD = pathDelay(optPath, trueW, EI);
  const showW = mode === 'supervised' && R('weights').checked;
  const allW = showW ? trueW.concat(...res.models.filter(m => m.weights).map(m => m.weights)) : null;
  const lo = showW ? Math.max(0.05, Math.min(...allW)) : 0, hi = showW ? Math.max(...allW) : 1;
  for (const m of res.models) {
    const isG = m.key === 'global';
    const table = isG ? res.global_round_tables[v.round - 1] : m.table;
    const path = followPath(table, ids, v.src, v.dst);
    const d = pathDelay(path, trueW, EI);
    const gap = path && optD ? (d - optD) / optD * 100 : null;
    const color = m.key === 'oracle' ? '#111111' : m.color;
    const card = el('div', { class: `path-card${isG ? ' global' : ''}`, style: `border-top-color:${color}` });
    const title = isG ? `${m.label} — ronde ${v.round}` : m.label;
    card.append(el('h4', {}, [el('span', { text: title }), m.key === 'oracle' ? el('span', { class: 'badge ok', text: 'acuan' }) : gapBadge(gap)]));
    const svg = sv('svg');
    let edgeColors = null, edgeTitles = null;
    if (showW) {
      const w = isG ? res.global_round_weights[v.round - 1] : m.weights;
      edgeColors = w.map(x => delayColor(Math.max(x, 0.05), lo, hi));
      edgeTitles = w.map((x, i) => `${ST.net.links[i].source}–${ST.net.links[i].target}: model ${x.toFixed(2)} ms / sebenarnya ${trueW[i].toFixed(2)} ms`);
    }
    drawNetwork(svg, { W: 420, H: 320, pad: 14, small: true, path, pathColor: color, edgeColors, edgeTitles, endpoints: [v.src, v.dst] });
    card.append(el('div', { class: 'p-meta', text: path ? `Delay sebenarnya ${d.toFixed(2)} ms · ${path.length - 1} hop` + (m.key === 'oracle' ? ' (minimum)' : '') : 'Model tidak menemukan jalur (loop / belum pernah dijelajahi)' }));
    card.append(svg);
    card.append(el('div', { class: 'p-route', text: path ? path.join(' → ') : '' }));
    grid.append(card);
  }
}

// ------------------------------------------------------------------ grafik SVG
function lineChart(container, title, xs, ys, refs, opt = {}) {
  const W = 460, H = 220, L = 42, Rr = 10, T = 10, B = 26;
  const card = el('div', { class: 'chart-card' }, [el('div', { class: 'chart-card-head' }, el('h3', { text: title }))]);
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}` });
  const vals = ys.concat(refs.map(r => r.value), opt.second || []).filter(v => v != null && !Number.isNaN(v));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const padv = (hi - lo) * 0.08; lo -= padv; hi += padv;
  const X = i => L + (xs.length > 1 ? (i / (xs.length - 1)) * (W - L - Rr) : (W - L - Rr) / 2);
  const Y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    svg.append(sv('line', { x1: L, x2: W - Rr, y1: y, y2: y, class: 'gridline' }));
    const t = sv('text', { x: L - 4, y: y + 3, 'text-anchor': 'end', class: 'axis' }); t.textContent = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2); svg.append(t);
  }
  const step = Math.max(1, Math.ceil(xs.length / 8));
  xs.forEach((x, i) => { const last = i === xs.length - 1; if ((i % step === 0 && (last || xs.length - 1 - i >= step / 2)) || last) { const t = sv('text', { x: X(i), y: H - 8, 'text-anchor': 'middle', class: 'axis' }); t.textContent = x; svg.append(t); } });
  for (const r of refs) {
    if (r.value == null) continue;
    svg.append(sv('line', { x1: L, x2: W - Rr, y1: Y(r.value), y2: Y(r.value), stroke: r.color, 'stroke-width': 1.4, 'stroke-dasharray': r.dash }));
  }
  const poly = (arr, color, w) => {
    const pts = arr.map((v, i) => v == null ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(' ');
    svg.append(sv('polyline', { points: pts, fill: 'none', stroke: color, 'stroke-width': w, 'stroke-linejoin': 'round' }));
    arr.forEach((v, i) => { if (v != null) { const c = sv('circle', { cx: X(i), cy: Y(v), r: 2.4, fill: color }); c.append(Object.assign(sv('title'), { textContent: `ronde ${xs[i]}: ${fmt(v, 3)}` })); svg.append(c); } });
  };
  if (opt.second) poly(opt.second, '#85847e', 1.4);
  poly(ys, opt.color || GLOBAL_COLOR, 2.6);
  card.append(svg);
  const lg = el('div', { class: 'chart-legend' }, [el('span', {}, [el('i', { style: `background:${opt.color || GLOBAL_COLOR}` }), opt.label || 'Global (federated)'])]);
  if (opt.second) lg.append(el('span', {}, [el('i', { style: 'background:#85847e' }), opt.secondLabel]));
  refs.forEach(r => lg.append(el('span', {}, [el('i', { style: `background:${r.color};height:2px` }), `${r.label}: ${fmt(r.value, 2)}`])));
  card.append(lg);
  container.append(card);
}

function groupedBars(container, title, models, groups) {
  const W = 920, H = 230, L = 42, T = 10, B = 40;
  const card = el('div', { class: 'chart-card full-span' }, [el('div', { class: 'chart-card-head' }, el('h3', { text: title }))]);
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}` });
  const vals = models.flatMap(m => groups.map(g => m.mae_by_load[g[0]] || 0));
  const hi = Math.max(...vals, 1) * 1.1;
  const gw = (W - L - 10) / groups.length, bw = Math.min(26, (gw - 20) / models.length);
  const Y = v => T + (1 - v / hi) * (H - T - B);
  for (let k = 0; k <= 4; k++) {
    const v = hi * k / 4; svg.append(sv('line', { x1: L, x2: W - 10, y1: Y(v), y2: Y(v), class: 'gridline' }));
    const t = sv('text', { x: L - 4, y: Y(v) + 3, 'text-anchor': 'end', class: 'axis' }); t.textContent = v.toFixed(1); svg.append(t);
  }
  groups.forEach(([g, lab], gi) => {
    const x0 = L + gi * gw + (gw - bw * models.length) / 2;
    models.forEach((m, mi) => {
      const v = m.mae_by_load[g];
      if (v == null) return;
      const r = sv('rect', { x: x0 + mi * bw, y: Y(v), width: bw - 2, height: Y(0) - Y(v), fill: m.color, rx: 2 });
      r.append(Object.assign(sv('title'), { textContent: `${m.label}: ${v.toFixed(3)} ms` }));
      svg.append(r);
    });
    const t = sv('text', { x: L + gi * gw + gw / 2, y: H - 16, 'text-anchor': 'middle', class: 'axis', 'font-size': 11 });
    t.textContent = `Utilisasi ${g} (${lab})`; svg.append(t);
  });
  card.append(svg);
  card.append(el('div', { class: 'chart-legend' }, models.map(m => el('span', {}, [el('i', { style: `background:${m.color};height:8px` }), m.label]))));
  container.append(card);
}

// ------------------------------------------------------------------ unduhan
const GROUPS = [['data', 'Data konfigurasi jaringan (CSV terpisah)'], ['config', 'Konfigurasi federated learning'],
  ['script', 'Script Python standalone'], ['result', 'Hasil (CSV / JSON) — tersedia setelah simulasi dijalankan'], ['figure', 'Output visualisasi (PNG, matplotlib)']];
async function loadFiles() {
  const box = $('#file-groups');
  box.innerHTML = '<p class="empty-note">Memuat daftar file...</p>';
  try {
    const files = await api('GET', '/api/files');
    box.innerHTML = '';
    for (const [g, label] of GROUPS) {
      const fs = files.filter(f => f.group === g);
      const grp = el('div', { class: 'file-group' }, [el('h3', { text: label })]);
      if (!fs.length) grp.append(el('p', { class: 'empty-note', text: g === 'result' || g === 'figure' ? 'Belum ada — jalankan tab 1 / tab 2 terlebih dahulu.' : 'Tidak ada.' }));
      for (const f of fs) {
        const href = f.figure ? `${f.figure}?dl=1` : `/api/download/${f.path}`;
        grp.append(el('div', { class: 'file-row' }, [
          el('code', { text: f.path }), el('span', { class: 'f-desc', text: f.desc }),
          el('span', { class: 'f-size', text: f.size != null ? bytesFmt(f.size) : 'PNG' }),
          el('span', { class: 'f-actions' }, [
            el('button', { class: 'btn-secondary small', text: 'Pratinjau', onclick: () => preview(f) }),
            el('a', { class: 'btn-primary', style: 'padding:4px 10px;font-size:12px', href, text: 'Unduh' })])]));
      }
      box.append(grp);
    }
  } catch (e) { box.innerHTML = `<p class="empty-note">${esc(e.message)}</p>`; }
}
async function preview(f) {
  $('#preview-panel').hidden = false;
  $('#preview-title').textContent = f.path;
  const img = $('#preview-img'), pre = $('#preview-body');
  if (f.figure) {
    pre.hidden = true; img.hidden = false; img.src = `${f.figure}?t=${Date.now()}`;
  } else {
    img.hidden = true; pre.hidden = false; pre.textContent = 'Memuat...';
    const r = await fetch(`/api/download/${f.path}?preview=1`);
    let txt = await r.text();
    if (txt.length > 200000) txt = txt.slice(0, 200000) + '\n... (dipotong, unduh untuk melihat seluruhnya)';
    pre.textContent = txt;
  }
  $('#preview-panel').scrollIntoView({ behavior: 'smooth' });
}

// ------------------------------------------------------------------ tab & event
function setMode(mode) {
  MODE = mode;
  $$('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  $('#tab-supervised').hidden = mode !== 'supervised';
  $('#tab-qrouting').hidden = mode !== 'qrouting';
  $('#tab-download').hidden = mode !== 'download';
  $('#tab-rl').hidden = mode !== 'rl';
  $('#data-panel').hidden = mode === 'download';
  $('#topo-panel').hidden = mode === 'download';
  if (mode === 'download') loadFiles();
}

function bind() {
  $$('.mode-btn').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $$('[data-goto]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.goto)));
  $('#generate-btn').addEventListener('click', () => mutate('POST', '/api/topology/generate', {
    num_nodes: Number($('#g-nodes').value), k_neighbors: Number($('#g-k').value), area_width: Number($('#g-w').value),
    area_height: Number($('#g-h').value), num_clients: Number($('#g-clients').value), seed: Number($('#g-seed').value) }));
  $('#partition-btn').addEventListener('click', () => mutate('POST', '/api/topology/partition', { num_clients: Number($('#g-clients').value), seed: Number($('#g-seed').value) }));
  $('#save-clients-btn').addEventListener('click', () => {
    const clients = $$('#client-body tr').map(tr => {
      const o = { client_id: tr.dataset.cid };
      $$('[data-f]', tr).forEach(i => { o[i.dataset.f] = i.value; });
      return o;
    });
    mutate('POST', '/api/clients', { clients });
  });
  $('#add-node-btn').addEventListener('click', () => mutate('POST', '/api/node/add', { node_id: $('#new-node-id').value, region: $('#new-node-region').value }));
  $('#save-link-btn').addEventListener('click', () => mutate('POST', '/api/link', {
    source: $('#e-src').value, target: $('#e-dst').value, bandwidth_mbps: $('#e-bw').value, loss: $('#e-loss').value,
    traffic_weight: $('#e-tw').value, base_delay_ms: $('#e-delay').value }));
  $('#show-util').addEventListener('change', renderTopology);
  $('#data-btn').addEventListener('click', async () => {
    try {
      status('data-status', 'Membangkitkan data...');
      await loadState(await api('POST', '/api/data/generate', { simulation: readSim() }));
    } catch (e) { status('data-status', e.message, true); }
  });
  $('#sup-run').addEventListener('click', () => runMode('supervised'));
  $('#q-run').addEventListener('click', () => runMode('qrouting'));
  $('#preview-close').addEventListener('click', () => { $('#preview-panel').hidden = true; });
  bindRL();

  $('#import-config-btn').addEventListener('click', () => $('#import-config-input').click());
  $('#import-config-input').addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const st = await api('POST', '/api/config/import', JSON.parse(await f.text()));
      $('#sup-params').innerHTML = ''; $('#q-params').innerHTML = '';
      await loadState(st);
      status('global-status', `Konfigurasi ${f.name} dimuat.`);
    } catch (err) { status('global-status', err.message, true); }
    e.target.value = '';
  });
  $('#import-csv-btn').addEventListener('click', () => $('#import-csv-input').click());
  $('#import-csv-input').addEventListener('change', async e => {
    const fd = new FormData();
    for (const f of e.target.files) {
      const n = f.name.toLowerCase();
      if (n.includes('node')) fd.append('nodes', f);
      else if (n.includes('link')) fd.append('links', f);
      else if (n.includes('client')) fd.append('clients', f);
    }
    try {
      await loadState(await api('POST', '/api/import/csv', fd, true));
      status('global-status', 'CSV jaringan berhasil diimpor.');
    } catch (err) { status('global-status', err.message + ' (pilih ketiga file sekaligus)', true); }
    e.target.value = '';
  });
}

// ================================================================== TAB RL: REWARD PER LANGKAH
const RL_REWARD_PARAMS = [
  { key: 'reward_goal', label: 'Hadiah tiba di target', sign: 'plus', type: 'number', min: 0, max: 1000, step: 5, help: 'Diberikan sekali saat paket sampai di target.' },
  { key: 'w_progress', label: 'Bobot kemajuan (per hop)', sign: 'pm', type: 'number', min: 0, max: 50, step: 0.5, help: 'Positif bila langkah mendekatkan ke target (jarak hop berkurang), negatif bila menjauh.' },
  { key: 'w_delay', label: 'Bobot delay (per ms)', sign: 'minus', type: 'number', min: 0, max: 10, step: 0.1, help: 'Hukuman = bobot × delay link yang dilewati.' },
  { key: 'hop_penalty', label: 'Biaya per hop', sign: 'minus', type: 'number', min: 0, max: 50, step: 0.5, help: 'Hukuman tetap setiap langkah → mendorong jalur pendek.' },
  { key: 'loop_penalty', label: 'Hukuman loop', sign: 'minus', type: 'number', min: 0, max: 500, step: 5, help: 'Bila agen kembali ke node yang sudah dikunjungi pada episode yang sama.' },
  { key: 'congestion_penalty', label: 'Hukuman kongesti', sign: 'minus', type: 'number', min: 0, max: 500, step: 5, help: 'Bila utilisasi link > ambang; makin padat makin besar.' },
  { key: 'congestion_threshold', label: 'Ambang utilisasi kongesti', sign: 'minus', type: 'number', min: 0.1, max: 0.99, step: 0.05, help: 'Utilisasi di atas nilai ini dianggap padat.' },
  { key: 'timeout_penalty', label: 'Hukuman timeout', sign: 'minus', type: 'number', min: 0, max: 1000, step: 5, help: 'Bila batas hop habis sebelum paket tiba.' },
];
const RL_PARAMS = [
  { key: 'episodes', label: 'Jumlah episode', type: 'number', min: 10, max: 3000, step: 10, help: 'Satu episode = satu paket dari source hingga tiba / timeout.' },
  { key: 'alpha', label: 'α learning rate', type: 'number', min: 0.01, max: 1, step: 0.05, help: 'Seberapa besar Q digeser ke target TD.' },
  { key: 'gamma', label: 'γ discount factor', type: 'number', min: 0, max: 1, step: 0.01, help: 'Bobot reward masa depan (0 = hanya reward langsung).' },
  { key: 'epsilon', label: 'ε eksplorasi awal', type: 'number', min: 0, max: 1, step: 0.05, help: 'Peluang memilih aksi acak.' },
  { key: 'epsilon_decay', label: 'Peluruhan ε / episode', type: 'number', min: 0.8, max: 1, step: 0.005, help: 'ε dikali faktor ini setiap episode.' },
  { key: 'epsilon_min', label: 'ε minimum', type: 'number', min: 0, max: 0.5, step: 0.01, help: 'Batas bawah eksplorasi.' },
  { key: 'max_hops_factor', label: 'Batas hop (× jumlah node)', type: 'number', min: 0.5, max: 5, step: 0.25, help: 'Episode berakhir (timeout) bila melewati batas ini.' },
  { key: 'cost_noise_pct', label: 'Variasi delay (%)', type: 'number', min: 0, max: 50, step: 1, help: 'Delay yang dialami paket berfluktuasi acak.' },
  { key: 'q_init', label: 'Nilai Q awal', type: 'number', min: -100, max: 100, step: 1, help: 'Q awal tinggi (optimis) mendorong eksplorasi.' },
];
const COMP_COLORS = { goal: '#1baf7a', progress: '#2a78d6', delay: '#e8743b', hop: '#c98500', loop: '#a259d9', congestion: '#e34948', timeout: '#52514e' };
const COMP_INFO = {
  goal: ['+', 'Tiba di target'], progress: ['±', 'Kemajuan menuju target'], delay: ['−', 'Delay link'],
  hop: ['−', 'Biaya per hop'], loop: ['−', 'Mengulang node (loop)'], congestion: ['−', 'Link padat (kongesti)'], timeout: ['−', 'Batas hop habis'],
};
const RLV = { view: 'greedy', ep: 1, step: 0, timer: null };

async function runRL() {
  const btn = $('#rl-run');
  const rl = Object.assign(readParams($('#rl-params'), RL_PARAMS), readParams($('#rl-reward-params'), RL_REWARD_PARAMS),
    { source: $('#rl-src').value, target: $('#rl-dst').value });
  if (rl.source === rl.target) { status('rl-status', 'Source dan target harus berbeda.', true); return; }
  btn.disabled = true; status('rl-status', 'Melatih agen...');
  const t0 = performance.now();
  try {
    const st = await api('POST', '/api/rl/run', { rl, simulation: readSim() });
    RES.rl = st.result; delete st.result;
    await loadState(st);
    RLV.view = 'greedy'; RLV.ep = RES.rl.episodes.length; RLV.step = 0;
    renderRL();
    status('rl-status', `Selesai dalam ${((performance.now() - t0) / 1000).toFixed(1)} s · ${RES.rl.episodes.length} episode`);
  } catch (e) { status('rl-status', e.message, true); }
  btn.disabled = false;
}

function rlSteps() {
  const r = RES.rl;
  if (RLV.view === 'greedy') return r.greedy.steps;
  if (RLV.view === 'optimal') return r.optimal.steps;
  return r.steps[RLV.ep - 1] || [];
}
const sumSign = (comp, pos) => Object.values(comp).reduce((a, v) => a + ((pos ? v > 0 : v < 0) ? v : 0), 0);

function renderRL() {
  const r = RES.rl;
  $('#rl-results').hidden = false;
  // KPI
  const k = $('#rl-kpis'); k.innerHTML = '';
  const last = r.episodes.slice(-Math.max(1, Math.floor(r.episodes.length / 10)));
  const avgLast = last.reduce((a, e) => a + e.total_reward, 0) / last.length;
  const reachedPct = r.episodes.filter(e => e.reached).length / r.episodes.length * 100;
  const kpi = (cls, label, value, sub) => k.append(el('div', { class: `kpi ${cls}` }, [el('div', { class: 'k-label', text: label }), el('div', { class: 'k-value', text: value }), el('div', { class: 'k-sub', text: sub })]));
  const g = r.greedy, o = r.optimal;
  kpi('global', 'Jalur agen (greedy)', g.reached ? `${g.path.length - 1} hop` : 'GAGAL', g.reached ? `delay ${fmt(g.delay)} ms · reward ${fmt(g.total_reward)}` : 'kebijakan masih loop / buntu');
  kpi('cent', 'Jalur delay minimum', `${o.path.length - 1} hop`, `delay ${fmt(o.delay)} ms · reward ${fmt(o.total_reward)}`);
  const gap = g.reached && o.delay ? (g.delay - o.delay) / o.delay * 100 : null;
  kpi(gap !== null && gap < 0.5 ? 'global' : 'local', 'Selisih delay vs optimal', gap === null ? '–' : `${gap >= 0 ? '+' : ''}${gap.toFixed(2)}%`, gap !== null && gap < 0.005 ? 'jalur agen = jalur optimal' : 'positif = lebih lambat');
  kpi('local', 'Episode pertama tiba', r.first_reach ? `#${r.first_reach}` : 'belum', `${reachedPct.toFixed(1)}% episode tiba di target`);
  kpi('cent', `Rata² reward ${last.length} ep. terakhir`, fmt(avgLast), `episode 1: ${fmt(r.episodes[0].total_reward)}`);
  // tabel pembanding
  const keys = Object.keys(COMP_INFO);
  const ct = $('#rl-compare');
  ct.innerHTML = '<thead><tr><th>Jalur</th><th>Rute</th><th class="num">Hop</th><th class="num">Delay (ms)</th>' +
    keys.map(c => `<th class="num">${COMP_INFO[c][0]} ${c}</th>`).join('') + '<th class="num">Reward +</th><th class="num">Reward −</th><th class="num">Total</th></tr></thead>';
  const tb = el('tbody');
  for (const [lab, p] of [['Agen (greedy)', g], ['Delay minimum', o]]) {
    const tot = {}; keys.forEach(c => { tot[c] = p.steps.reduce((a, s) => a + s.comp[c], 0); });
    const plus = p.steps.reduce((a, s) => a + sumSign(s.comp, true), 0), minus = p.steps.reduce((a, s) => a + sumSign(s.comp, false), 0);
    tb.append(el('tr', {}, [el('td', { html: `<b>${lab}</b>` }), el('td', { class: 'path-cell', text: p.path.join(' → ') }),
      el('td', { class: 'num', text: p.path.length - 1 }), el('td', { class: 'num', text: fmt(p.delay) }),
      ...keys.map(c => el('td', { class: `num ${tot[c] > 0 ? 'c-plus' : tot[c] < 0 ? 'c-minus' : ''}`, text: tot[c] ? fmt(tot[c]) : '0' })),
      el('td', { class: 'num c-plus', text: `+${fmt(plus)}` }), el('td', { class: 'num c-minus', text: fmt(minus) }),
      el('td', { class: 'num', html: `<b>${fmt(p.total_reward)}</b>` })]));
  }
  ct.append(tb);
  $('#rl-fig-curve').href = '/api/figure/rl/reward_curve.png?dl=1';
  $('#rl-fig-policy').href = '/api/figure/rl/policy.png?dl=1';
  rewardCurve();
  const epr = $('#rl-ep'); epr.max = r.episodes.length; $('#rl-ep-num').max = r.episodes.length;
  syncRLControls();
  drawRLStep();
  drawPolicy();
}

function syncRLControls() {
  $('#rl-view').value = RLV.view;
  $('#rl-ep-wrap').style.opacity = RLV.view === 'episode' ? 1 : 0.4;
  $('#rl-ep').value = RLV.ep; $('#rl-ep-num').value = RLV.ep;
  const e = RES.rl.episodes[RLV.ep - 1];
  $('#rl-ep-label').textContent = `${RLV.ep} / ${RES.rl.episodes.length} · ε=${e.epsilon} · reward ${fmt(e.total_reward)} · ${e.reached ? 'tiba' : 'tidak tiba'}`;
  const kind = RLV.view === 'episode' ? `steps_episode.png?episode=${RLV.ep}&` : `steps_${RLV.view}.png?`;
  $('#rl-fig-steps').href = `/api/figure/rl/${kind}dl=1`;
}

function rewardCurve() {
  const eps = RES.rl.episodes, box = $('#rl-curve');
  box.innerHTML = '';
  const W = 920, H = 300, L = 52, R = 12, T = 12, B = 30;
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}` });
  const lo = Math.min(0, ...eps.map(e => e.minus)), hi = Math.max(0, ...eps.map(e => e.plus));
  const n = eps.length, bw = (W - L - R) / n;
  const X = i => L + i * bw, Y = v => T + (1 - (v - lo) / ((hi - lo) || 1)) * (H - T - B);
  for (let k = 0; k <= 5; k++) {
    const v = lo + (hi - lo) * k / 5;
    svg.append(sv('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: 'gridline' }));
    const t = sv('text', { x: L - 4, y: Y(v) + 3, 'text-anchor': 'end', class: 'axis' }); t.textContent = v.toFixed(0); svg.append(t);
  }
  eps.forEach((e, i) => {
    svg.append(sv('rect', { x: X(i), y: Y(e.plus), width: Math.max(bw - 0.3, 0.6), height: Y(0) - Y(e.plus), fill: '#1baf7a', opacity: 0.45 }));
    svg.append(sv('rect', { x: X(i), y: Y(0), width: Math.max(bw - 0.3, 0.6), height: Y(e.minus) - Y(0), fill: '#e34948', opacity: 0.45 }));
  });
  svg.append(sv('line', { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: 'var(--text-muted)', 'stroke-width': 1 }));
  const line = (vals, color, w) => svg.append(sv('polyline', { points: vals.map((v, i) => `${(X(i) + bw / 2).toFixed(1)},${Y(v).toFixed(1)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': w }));
  const tot = eps.map(e => e.total_reward);
  line(tot, 'var(--text-primary)', 1);
  const w = Math.max(1, Math.floor(n / 20));
  line(tot.map((_, i) => { const a = tot.slice(Math.max(0, i - w + 1), i + 1); return a.reduce((x, y) => x + y, 0) / a.length; }), GLOBAL_COLOR, 2.6);
  const step = Math.ceil(n / 10);
  for (let i = 0; i < n; i += step) { const t = sv('text', { x: X(i) + bw / 2, y: H - 10, 'text-anchor': 'middle', class: 'axis' }); t.textContent = i + 1; svg.append(t); }
  const marker = sv('line', { x1: 0, x2: 0, y1: T, y2: H - B, stroke: GLOBAL_COLOR, 'stroke-width': 1.5, 'stroke-dasharray': '4 3', visibility: 'hidden' });
  svg.append(marker);
  const tip = sv('text', { x: L + 6, y: T + 12, class: 'axis', 'font-size': 11 }); svg.append(tip);
  svg.addEventListener('mousemove', ev => {
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const i = Math.min(n - 1, Math.max(0, Math.floor((p.x - L) / bw)));
    marker.setAttribute('x1', X(i) + bw / 2); marker.setAttribute('x2', X(i) + bw / 2); marker.setAttribute('visibility', 'visible');
    const e = eps[i];
    tip.textContent = `Episode ${e.episode}: total ${fmt(e.total_reward)} (+${fmt(e.plus)} / ${fmt(e.minus)}), ${e.steps} langkah, ${e.reached ? 'tiba' : 'tidak tiba'}, ε=${e.epsilon} — klik untuk melihat langkahnya`;
  });
  svg.addEventListener('click', ev => {
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    RLV.ep = Math.min(n, Math.max(1, Math.floor((p.x - L) / bw) + 1)); RLV.view = 'episode'; RLV.step = 0;
    syncRLControls(); drawRLStep(); $('#rl-player').scrollIntoView({ behavior: 'smooth' });
  });
  box.append(svg);
  box.append(el('div', { class: 'chart-legend' }, [el('span', {}, [el('i', { style: 'background:#1baf7a;height:8px' }), 'reward + per episode']),
    el('span', {}, [el('i', { style: 'background:#e34948;height:8px' }), 'reward − per episode']),
    el('span', {}, [el('i', { style: 'background:var(--text-primary)' }), 'total reward']),
    el('span', {}, [el('i', { style: `background:${GLOBAL_COLOR}` }), `rata-rata bergerak (${w} ep)`])]));
}

function nodePos(P) { const pos = {}; ST.net.nodes.forEach(n => { pos[n.node_id] = [P.x(n.x), P.y(n.y)]; }); return pos; }

function drawRLStep() {
  const r = RES.rl, steps = rlSteps();
  RLV.step = Math.max(0, Math.min(RLV.step, steps.length));
  $('#rl-step-label').textContent = `${RLV.step} / ${steps.length}`;
  const svg = $('#rl-map');
  const P = drawNetwork(svg, { endpoints: [r.source, r.target] });
  const pos = nodePos(P);
  const g = sv('g'); svg.append(g);
  // segmen langkah yang sudah dilalui
  steps.slice(0, RLV.step).forEach((s, i) => {
    const [x1, y1] = pos[s.x], [x2, y2] = pos[s.y];
    const cur = i === RLV.step - 1;
    g.append(sv('line', { x1, y1, x2, y2, stroke: s.r >= 0 ? '#1baf7a' : '#e34948', 'stroke-width': cur ? 8 : 5, 'stroke-linecap': 'round', opacity: cur ? 1 : 0.75 }));
  });
  steps.slice(0, RLV.step).forEach(s => {
    const [x1, y1] = pos[s.x], [x2, y2] = pos[s.y], mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    g.append(sv('circle', { cx: mx, cy: my, r: 8, fill: 'var(--surface-1)', stroke: s.r >= 0 ? '#1baf7a' : '#e34948', 'stroke-width': 1.5 }));
    const t = sv('text', { x: mx, y: my + 3.5, 'text-anchor': 'middle', 'font-size': 9, 'font-weight': 700, fill: 'var(--text-primary)' }); t.textContent = s.t; g.append(t);
  });
  // node saat ini & pilihan berikutnya (nilai Q)
  const curNode = RLV.step === 0 ? r.source : steps[RLV.step - 1].y;
  const next = steps[RLV.step];
  if (next && next.options) {
    const [cx, cy] = pos[curNode];
    next.options.forEach(o => {
      const [x2, y2] = pos[o.y];
      const chosen = o.y === next.y;
      g.append(sv('line', { x1: cx, y1: cy, x2: cx + (x2 - cx) * 0.8, y2: cy + (y2 - cy) * 0.8, stroke: chosen ? GLOBAL_COLOR : '#2a78d6', 'stroke-width': chosen ? 2.6 : 1.6, 'stroke-dasharray': '5 4' }));
      const lx = cx + (x2 - cx) * 0.55, ly = cy + (y2 - cy) * 0.55;
      const bg = sv('rect', { x: lx - 22, y: ly - 9, width: 44, height: 16, rx: 4, fill: chosen ? GLOBAL_COLOR : 'var(--surface-1)', stroke: '#2a78d6', 'stroke-width': chosen ? 0 : 1 });
      const t = sv('text', { x: lx, y: ly + 3, 'text-anchor': 'middle', 'font-size': 9.5, 'font-weight': 600, fill: chosen ? '#fff' : 'var(--text-primary)' });
      t.textContent = `Q ${o.q.toFixed(1)}`;
      g.append(bg, t);
    });
  }
  const [px, py] = pos[curNode];
  g.append(sv('circle', { cx: px, cy: py, r: 14, fill: 'none', stroke: GLOBAL_COLOR, 'stroke-width': 3 }));
  stepCard(steps, curNode);
  stepChart(steps);
  stepTable(steps);
}

function stepCard(steps, curNode) {
  const r = RES.rl, card = $('#rl-step-card'), c = r.config;
  card.innerHTML = '';
  const k = RLV.step;
  if (k === 0) {
    const next = steps[0];
    card.append(el('h4', { text: `Langkah 0 — paket di ${r.source}` }));
    card.append(el('div', { class: 's-sub', html: `Tujuan: <b>${r.target}</b> (jarak ${r.hop_dist[r.node_ids.indexOf(r.source)]} hop). ` +
      (RLV.view === 'episode' ? `Episode ${RLV.ep}, ε = ${r.episodes[RLV.ep - 1].epsilon}: agen memilih aksi acak dengan peluang ε, selain itu memilih tetangga dengan Q tertinggi.` : RLV.view === 'greedy' ? 'Kebijakan akhir: selalu memilih tetangga dengan Q tertinggi.' : 'Jalur pembanding hasil Dijkstra pada delay sebenarnya, dinilai dengan fungsi reward yang sama.') }));
    if (next && next.options) card.append(optionsTable(next));
    card.append(el('p', { class: 'hint', style: 'margin-top:8px', text: 'Tekan ▶ untuk maju satu langkah dan melihat rincian reward-nya.' }));
    return;
  }
  const s = steps[k - 1];
  card.append(el('h4', {}, [`Langkah ${s.t}: ${s.x} → ${s.y}`, el('span', { class: `mode-badge ${s.mode}`, text: { explore: 'eksplorasi (acak)', exploit: 'eksploitasi (Q tertinggi)', greedy: RLV.view === 'optimal' ? 'jalur pembanding' : 'greedy' }[s.mode] })]));
  card.append(el('div', { class: 's-sub', text: `Delay link ${fmt(s.delay)} ms · utilisasi ${(s.util * 100).toFixed(0)}% · sisa jarak ke target ${s.hop_to_target} hop` }));
  const maxAbs = Math.max(1, ...Object.values(s.comp).map(Math.abs));
  for (const key of Object.keys(COMP_INFO)) {
    const v = s.comp[key];
    const bar = el('div', { class: 'cbar' });
    const wpct = Math.abs(v) / maxAbs * 50;
    bar.append(el('i', { style: `background:${COMP_COLORS[key]};width:${wpct}%;${v >= 0 ? 'left:50%' : `left:${50 - wpct}%`}` }));
    bar.append(el('i', { style: 'left:50%;width:1px;background:var(--text-muted)' }));
    card.append(el('div', { class: `comp-row${v === 0 ? ' zero' : ''}` }, [
      el('span', { class: `sign ${COMP_INFO[key][0] === '+' ? 'plus' : COMP_INFO[key][0] === '±' ? 'pm' : 'minus'}`, text: COMP_INFO[key][0] }),
      el('span', { text: COMP_INFO[key][1] + compExplain(key, s, c) }), bar,
      el('span', { class: `cval ${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}`, text: v === 0 ? '0' : (v > 0 ? '+' : '') + v.toFixed(2) })]));
  }
  card.append(el('div', { class: 'r-total' }, [el('span', { text: 'Reward langkah ini (r)' }), el('span', { class: s.r >= 0 ? 'c-plus' : 'c-minus', text: (s.r >= 0 ? '+' : '') + s.r.toFixed(2) })]));
  card.append(el('div', { class: 'r-total', style: 'font-weight:600;font-size:13px' }, [el('span', { text: 'Return kumulatif (Σr)' }), el('span', { text: s.ret.toFixed(2) })]));
  if (s.td_target !== undefined) {
    card.append(el('div', { class: 'q-calc', html:
      `Q(${s.x}→${s.y}) ← Q + α·[r + γ·max Q(${s.y},·) − Q]<br>` +
      `= ${s.q_before.toFixed(2)} + ${c.alpha}·[${s.r.toFixed(2)} + ${c.gamma}·${s.max_next.toFixed(2)} − ${s.q_before.toFixed(2)}]<br>` +
      `= ${s.q_before.toFixed(2)} + ${c.alpha}·(${(s.td_target - s.q_before).toFixed(2)}) = <b>${s.q_after.toFixed(2)}</b>` +
      (s.y === r.target ? '<br><i>terminal: tidak ada reward masa depan (max Q = 0)</i>' : '') }));
  } else {
    card.append(el('div', { class: 'q-calc', html: 'Evaluasi kebijakan (tanpa update Q, tanpa noise delay).' }));
  }
  if (s.options) card.append(optionsTable(s));
}

function compExplain(key, s, c) {
  switch (key) {
    case 'progress': return ` (${c.w_progress} × perubahan jarak hop)`;
    case 'delay': return ` (${c.w_delay} × ${fmt(s.delay)} ms)`;
    case 'congestion': return s.util > c.congestion_threshold ? ` (util ${(s.util * 100).toFixed(0)}% > ${(c.congestion_threshold * 100).toFixed(0)}%)` : '';
    default: return '';
  }
}

function optionsTable(s) {
  const t = el('table', { class: 'opt-table' });
  t.innerHTML = `<thead><tr><th>Pilihan dari ${s.x}</th><th class="num">Q sebelum</th><th></th></tr></thead>`;
  const best = Math.max(...s.options.map(o => o.q));
  const tb = el('tbody');
  [...s.options].sort((a, b) => b.q - a.q).forEach(o => {
    tb.append(el('tr', { class: o.y === s.y ? 'chosen' : '' }, [el('td', { text: `→ ${o.y}` }), el('td', { class: 'num', text: o.q.toFixed(2) }),
      el('td', { text: (o.y === s.y ? 'dipilih' : '') + (o.q === best ? ' · Q tertinggi' : '') })]));
  });
  t.append(tb);
  return t;
}

function stepChart(steps) {
  const box = $('#rl-step-chart'); box.innerHTML = '';
  if (!steps.length) { box.append(el('p', { class: 'empty-note', text: 'Episode tanpa langkah.' })); return; }
  const W = 920, H = 260, L = 50, R = 12, T = 14, B = 40;
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}` });
  const keys = Object.keys(COMP_COLORS);
  let hi = 0, lo = 0;
  steps.forEach(s => { hi = Math.max(hi, sumSign(s.comp, true), s.ret); lo = Math.min(lo, sumSign(s.comp, false), s.ret); });
  const n = steps.length, cw = (W - L - R) / n, bw = Math.min(40, cw * 0.7);
  const X = i => L + i * cw + cw / 2, Y = v => T + (1 - (v - lo) / ((hi - lo) || 1)) * (H - T - B);
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4; svg.append(sv('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: 'gridline' }));
    const t = sv('text', { x: L - 4, y: Y(v) + 3, 'text-anchor': 'end', class: 'axis' }); t.textContent = v.toFixed(0); svg.append(t);
  }
  steps.forEach((s, i) => {
    if (i === RLV.step - 1) svg.append(sv('rect', { x: X(i) - cw / 2, y: T, width: cw, height: H - T - B, fill: GLOBAL_COLOR, opacity: 0.08 }));
    let pb = 0, nb = 0;
    for (const k of keys) {
      const v = s.comp[k]; if (!v) continue;
      const y0 = v > 0 ? pb : nb, y1 = y0 + v;
      const rct = sv('rect', { x: X(i) - bw / 2, y: Y(Math.max(y0, y1)), width: bw, height: Math.abs(Y(y0) - Y(y1)), fill: COMP_COLORS[k] });
      rct.append(Object.assign(sv('title'), { textContent: `Langkah ${s.t} ${s.x}→${s.y}: ${COMP_INFO[k][1]} ${v > 0 ? '+' : ''}${v.toFixed(2)}` }));
      svg.append(rct);
      if (v > 0) pb = y1; else nb = y1;
    }
    if (n <= 40 || i % Math.ceil(n / 40) === 0) {
      const t = sv('text', { x: X(i), y: H - 24, 'text-anchor': 'middle', class: 'axis' }); t.textContent = s.t; svg.append(t);
      if (n <= 16) { const t2 = sv('text', { x: X(i), y: H - 12, 'text-anchor': 'middle', class: 'axis' }); t2.textContent = `${s.x}→${s.y}`; svg.append(t2); }
    }
  });
  svg.append(sv('line', { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: 'var(--text-muted)' }));
  svg.append(sv('polyline', { points: steps.map((s, i) => `${X(i).toFixed(1)},${Y(s.ret).toFixed(1)}`).join(' '), fill: 'none', stroke: 'var(--text-primary)', 'stroke-width': 2 }));
  steps.forEach((s, i) => svg.append(sv('circle', { cx: X(i), cy: Y(s.ret), r: i === RLV.step - 1 ? 5 : 3, fill: i < RLV.step ? 'var(--text-primary)' : 'var(--text-muted)' })));
  svg.addEventListener('click', ev => {
    const p = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(svg.getScreenCTM().inverse());
    RLV.step = Math.min(n, Math.max(1, Math.floor((p.x - L) / cw) + 1)); drawRLStep();
  });
  box.append(svg);
  box.append(el('div', { class: 'chart-legend' }, [...keys.map(k => el('span', {}, [el('i', { style: `background:${COMP_COLORS[k]};height:8px` }), `${COMP_INFO[k][0]} ${COMP_INFO[k][1]}`])),
    el('span', {}, [el('i', { style: 'background:var(--text-primary)' }), 'return kumulatif'])]));
}

function stepTable(steps) {
  const t = $('#rl-step-table');
  const keys = Object.keys(COMP_INFO);
  t.innerHTML = '<thead><tr><th>#</th><th>Langkah</th><th>Mode</th><th class="num">Delay</th><th class="num">Util</th>' +
    keys.map(k => `<th class="num">${COMP_INFO[k][0]} ${k}</th>`).join('') + '<th class="num">r</th><th class="num">Return</th><th class="num">Q sebelum → sesudah</th></tr></thead>';
  const tb = el('tbody');
  steps.forEach((s, i) => {
    const tr = el('tr', { class: i === RLV.step - 1 ? 'cur' : '', onclick: () => { RLV.step = i + 1; drawRLStep(); } }, [
      el('td', { text: s.t }), el('td', { text: `${s.x} → ${s.y}` }), el('td', { text: s.mode }),
      el('td', { class: 'num', text: fmt(s.delay) }), el('td', { class: 'num', text: `${(s.util * 100).toFixed(0)}%` }),
      ...keys.map(k => el('td', { class: `num ${s.comp[k] > 0 ? 'pos' : s.comp[k] < 0 ? 'neg' : ''}`, text: s.comp[k] ? fmt(s.comp[k]) : '·' })),
      el('td', { class: `num ${s.r >= 0 ? 'pos' : 'neg'}`, html: `<b>${fmt(s.r)}</b>` }), el('td', { class: 'num', text: fmt(s.ret) }),
      el('td', { class: 'num', text: s.td_target !== undefined ? `${fmt(s.q_before)} → ${fmt(s.q_after)}` : '–' })]);
    tb.append(tr);
  });
  t.append(tb);
  const cur = t.querySelector('tr.cur');
  if (cur) { const wrap = t.parentElement; wrap.scrollTop = Math.max(0, cur.offsetTop - wrap.clientHeight / 2); }
}

function drawPolicy() {
  const r = RES.rl, svg = $('#rl-policy');
  const P = drawNetwork(svg, { path: r.optimal.path, pathColor: '#111111' });
  const pos = nodePos(P);
  const id = 'pol-arrow';
  const defs = sv('defs'); const mk = sv('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto' });
  mk.append(sv('path', { d: 'M0,0 L10,5 L0,10 z', fill: GLOBAL_COLOR })); defs.append(mk); svg.prepend(defs);
  r.node_ids.forEach((n, i) => {
    const nx = r.policy[i]; if (!nx) return;
    const [x1, y1] = pos[n], [x2, y2] = pos[nx];
    svg.append(sv('line', { x1, y1, x2: x1 + (x2 - x1) * 0.6, y2: y1 + (y2 - y1) * 0.6, stroke: GLOBAL_COLOR, 'stroke-width': 2.4, 'marker-end': `url(#${id})` }));
  });
}

function bindRL() {
  $('#rl-run').addEventListener('click', runRL);
  $('#rl-view').addEventListener('change', e => { RLV.view = e.target.value; RLV.step = 0; syncRLControls(); drawRLStep(); });
  const setEp = v => { RLV.ep = Math.max(1, Math.min(RES.rl.episodes.length, Number(v) || 1)); RLV.view = 'episode'; RLV.step = 0; syncRLControls(); drawRLStep(); };
  $('#rl-ep').addEventListener('input', e => setEp(e.target.value));
  $('#rl-ep-num').addEventListener('change', e => setEp(e.target.value));
  const stop = () => { if (RLV.timer) { clearInterval(RLV.timer); RLV.timer = null; $('#rl-play').textContent = 'Putar'; } };
  $('#rl-first').addEventListener('click', () => { stop(); RLV.step = 0; drawRLStep(); });
  $('#rl-prev').addEventListener('click', () => { stop(); RLV.step--; drawRLStep(); });
  $('#rl-next').addEventListener('click', () => { stop(); RLV.step++; drawRLStep(); });
  $('#rl-last').addEventListener('click', () => { stop(); RLV.step = rlSteps().length; drawRLStep(); });
  $('#rl-play').addEventListener('click', () => {
    if (RLV.timer) { stop(); return; }
    if (RLV.step >= rlSteps().length) RLV.step = 0;
    $('#rl-play').textContent = 'Stop';
    RLV.timer = setInterval(() => { RLV.step++; drawRLStep(); if (RLV.step >= rlSteps().length) stop(); }, 900);
  });
}

bind();
loadState().catch(e => status('global-status', e.message, true));
