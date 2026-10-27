const el = id => document.getElementById(id);

let TOPOLOGY = null;
let ALGORITHMS = [];
let LAST_RESULTS = [];
let LAST_HISTORY_ROWS = [];

// ------------------------------------------------------- history / compare --
// Menyimpan snapshot terakhir (params + metrics) tiap algoritma agar run
// berikutnya bisa dibandingkan (sebelum vs sesudah perubahan parameter).
const HISTORY_KEY = 'sdn_prev_run_by_algo_v1';

function loadHistoryStore() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || {};
  } catch (e) {
    return {};
  }
}

function saveHistoryStore(store) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(store));
  } catch (e) { /* ignore quota errors */ }
}

function clearHistoryStore() {
  localStorage.removeItem(HISTORY_KEY);
  el('history-body').innerHTML = '';
}

function fmtTs(ts) {
  if (!ts) return '—';
  const d = new Date(ts);
  return d.toLocaleString();
}

function diffParams(prevParams, curParams) {
  const keys = new Set([...Object.keys(prevParams || {}), ...Object.keys(curParams || {})]);
  const changed = [];
  keys.forEach(k => {
    const pv = (prevParams || {})[k];
    const cv = (curParams || {})[k];
    if (pv !== cv) changed.push(`${k}: ${pv === undefined ? '—' : pv} → ${cv === undefined ? '—' : cv}`);
  });
  return changed;
}

function deltaCell(prevVal, curVal, digits, lowerIsBetter) {
  if (prevVal === undefined || prevVal === null || curVal === undefined || curVal === null) {
    return '<span class="delta delta-same">—</span>';
  }
  const diff = curVal - prevVal;
  if (Math.abs(diff) < Math.pow(10, -digits) / 2) {
    return `<span class="delta delta-same">0${digits ? '.' + '0'.repeat(digits) : ''}</span>`;
  }
  const better = lowerIsBetter ? diff < 0 : diff > 0;
  const cls = better ? 'delta-better' : 'delta-worse';
  const sign = diff > 0 ? '+' : '';
  return `<span class="delta ${cls}">${sign}${fmt(diff, digits)}</span>`;
}

function rawDelta(prevVal, curVal) {
  if (prevVal === undefined || prevVal === null || curVal === undefined || curVal === null) return null;
  return curVal - prevVal;
}

function renderHistoryComparison(results) {
  const store = loadHistoryStore();
  const body = el('history-body');
  body.innerHTML = '';
  let anyRow = false;
  LAST_HISTORY_ROWS = [];

  results.forEach(r => {
    if (r.error || !r.metrics || !r.metrics.valid) return;
    const prev = store[r.algorithm];
    anyRow = true;
    const tr = document.createElement('tr');
    if (!prev) {
      tr.innerHTML = `
        <td>${r.label}</td>
        <td>—</td>
        <td class="params-changed"><em>Run pertama, belum ada pembanding.</em></td>
        <td colspan="5"></td>`;
      LAST_HISTORY_ROWS.push({
        label: r.label, prev_ts: null, changed_params: 'Run pertama, belum ada pembanding.',
        delta_delay: null, delta_hop: null, delta_throughput: null, delta_loss: null, delta_qos: null,
      });
    } else {
      const changedParams = diffParams(prev.params, r.params);
      const paramsStr = changedParams.length ? changedParams.join('; ') : 'Tidak ada perubahan';
      tr.innerHTML = `
        <td>${r.label}</td>
        <td>${fmtTs(prev.ts)}</td>
        <td class="params-changed">${paramsStr}</td>
        <td>${deltaCell(prev.metrics.delay_ms, r.metrics.delay_ms, 2, true)}</td>
        <td>${deltaCell(prev.metrics.hop_count, r.metrics.hop_count, 0, true)}</td>
        <td>${deltaCell(prev.metrics.throughput_mbps, r.metrics.throughput_mbps, 2, false)}</td>
        <td>${deltaCell(prev.metrics.packet_loss_pct, r.metrics.packet_loss_pct, 3, true)}</td>
        <td>${deltaCell((prev.qos || {}).score, (r.qos || {}).score, 1, false)}</td>`;
      LAST_HISTORY_ROWS.push({
        label: r.label, prev_ts: fmtTs(prev.ts), changed_params: paramsStr,
        delta_delay: rawDelta(prev.metrics.delay_ms, r.metrics.delay_ms),
        delta_hop: rawDelta(prev.metrics.hop_count, r.metrics.hop_count),
        delta_throughput: rawDelta(prev.metrics.throughput_mbps, r.metrics.throughput_mbps),
        delta_loss: rawDelta(prev.metrics.packet_loss_pct, r.metrics.packet_loss_pct),
        delta_qos: rawDelta((prev.qos || {}).score, (r.qos || {}).score),
      });
    }
    body.appendChild(tr);

    // Simpan snapshot run ini sebagai pembanding untuk run berikutnya.
    store[r.algorithm] = {
      ts: Date.now(),
      params: r.params,
      metrics: {
        delay_ms: r.metrics.delay_ms,
        hop_count: r.metrics.hop_count,
        throughput_mbps: r.metrics.throughput_mbps,
        packet_loss_pct: r.metrics.packet_loss_pct,
      },
      qos: r.qos,
    };
  });

  if (!anyRow) {
    body.innerHTML = '<tr class="history-empty-row"><td colspan="8">Belum ada hasil valid untuk dibandingkan.</td></tr>';
  }
  saveHistoryStore(store);
}

// ---------------------------------------------------------------- helpers --
function fmt(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

function qosCssClass(cls) {
  if (!cls) return '';
  return 'qos-' + cls.toLowerCase();
}

async function api(path, method = 'GET', body) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---------------------------------------------------------- topology CRUD --
async function loadTopology() {
  TOPOLOGY = await api('/api/topology');
  renderNodeSelectors();
  renderNodeChips();
  renderEdgeTable();
  renderTopoMeta();
  drawTopology(TOPOLOGY, []);
}

function renderTopoMeta() {
  const modeLabel = TOPOLOGY.mode === 'generated' ? 'Hasil simulasi' : 'Topologi bawaan/manual';
  const area = TOPOLOGY.area || {};
  const areaStr = TOPOLOGY.mode === 'generated'
    ? ` &middot; area ${fmt(area.width, 0)}m &times; ${fmt(area.height, 0)}m`
    : '';
  el('topo-meta-info').innerHTML =
    `${modeLabel} &middot; ${TOPOLOGY.nodes.length} node, ${TOPOLOGY.edges.length} link${areaStr}`;
}

function renderNodeSelectors() {
  const ids = TOPOLOGY.nodes.map(n => n.id).sort();
  const opts = ids.map(id => `<option value="${id}">${id}</option>`).join('');
  ['source', 'target', 'edge-source', 'edge-target'].forEach(sel => {
    const node = el(sel);
    const prev = node.value;
    node.innerHTML = opts;
    if (ids.includes(prev)) node.value = prev;
  });
  if (ids.length > 1) el('target').value = ids[ids.length - 1];
}

function renderNodeChips() {
  const container = el('node-list');
  container.innerHTML = '';
  TOPOLOGY.nodes.map(n => n.id).sort().forEach(id => {
    const chip = document.createElement('span');
    chip.className = 'node-chip';
    chip.innerHTML = `${id} <button title="Hapus node">&times;</button>`;
    chip.querySelector('button').addEventListener('click', () => deleteNode(id));
    container.appendChild(chip);
  });
}

function renderEdgeTable() {
  const body = el('edge-table-body');
  body.innerHTML = '';
  TOPOLOGY.edges.forEach(e => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${e.source}-${e.target}</td>
      <td>${fmt(e.delay, 1)}</td>
      <td>${fmt(e.bandwidth, 0)}</td>
      <td>${fmt(e.loss, 3)}</td>
      <td><button class="icon-btn" title="Hapus link">&times;</button></td>`;
    tr.querySelector('button').addEventListener('click', () => deleteEdge(e.source, e.target));
    body.appendChild(tr);
  });
}

async function refreshTopology(data) {
  TOPOLOGY = data;
  renderNodeSelectors();
  renderNodeChips();
  renderEdgeTable();
  renderTopoMeta();
  drawTopology(TOPOLOGY, []);
  el('results-section').hidden = true;
}

async function generateTopology() {
  const payload = {
    num_nodes: parseInt(el('gen-num-nodes').value || '12', 10),
    k_neighbors: parseInt(el('gen-k-neighbors').value || '3', 10),
    area_width: parseFloat(el('gen-area-width').value || '1000'),
    area_height: parseFloat(el('gen-area-height').value || '1000'),
    seed: parseInt(el('gen-seed').value || '42', 10),
  };
  el('topo-status').textContent = 'Membangkitkan topologi acak...';
  try {
    const data = await api('/api/topology/generate', 'POST', payload);
    await refreshTopology(data);
    el('topo-status').textContent =
      `Topologi disimulasikan: ${payload.num_nodes} node pada area ${payload.area_width}m x ${payload.area_height}m.`;
  } catch (err) {
    el('topo-status').textContent = 'Gagal: ' + err.message;
  }
}

async function addNode() {
  const id = el('new-node-id').value.trim();
  if (!id) return;
  el('topo-status').textContent = 'Menambahkan node...';
  try {
    const data = await api('/api/topology/node', 'POST', { id });
    el('new-node-id').value = '';
    await refreshTopology(data);
    el('topo-status').textContent = `Node ${id} ditambahkan.`;
  } catch (err) {
    el('topo-status').textContent = 'Gagal: ' + err.message;
  }
}

async function deleteNode(id) {
  el('topo-status').textContent = 'Menghapus node...';
  try {
    const data = await api(`/api/topology/node/${encodeURIComponent(id)}`, 'DELETE');
    await refreshTopology(data);
    el('topo-status').textContent = `Node ${id} dihapus.`;
  } catch (err) {
    el('topo-status').textContent = 'Gagal: ' + err.message;
  }
}

async function upsertEdge() {
  const payload = {
    source: el('edge-source').value,
    target: el('edge-target').value,
    delay: parseFloat(el('edge-delay').value || '1'),
    bandwidth: parseFloat(el('edge-bandwidth').value || '100'),
    loss: parseFloat(el('edge-loss').value || '0.001'),
  };
  el('topo-status').textContent = 'Menyimpan link...';
  try {
    const data = await api('/api/topology/edge', 'POST', payload);
    await refreshTopology(data);
    el('topo-status').textContent = `Link ${payload.source}-${payload.target} disimpan.`;
  } catch (err) {
    el('topo-status').textContent = 'Gagal: ' + err.message;
  }
}

async function deleteEdge(source, target) {
  el('topo-status').textContent = 'Menghapus link...';
  try {
    const data = await api('/api/topology/edge', 'DELETE', { source, target });
    await refreshTopology(data);
    el('topo-status').textContent = `Link ${source}-${target} dihapus.`;
  } catch (err) {
    el('topo-status').textContent = 'Gagal: ' + err.message;
  }
}

async function resetTopology() {
  el('topo-status').textContent = 'Mereset topologi...';
  try {
    const data = await api('/api/topology/reset', 'POST');
    await refreshTopology(data);
    el('topo-status').textContent = 'Topologi direset ke bawaan.';
  } catch (err) {
    el('topo-status').textContent = 'Gagal: ' + err.message;
  }
}

// -------------------------------------------------------- topology drawing --
// Node bisa digeser langsung (drag & drop) pada visualisasi; posisi baru
// disimpan ke backend (mempengaruhi heuristik jarak A* & ikut tersimpan pada
// export/import konfigurasi JSON). Lihat initTopoDrag().
let TOPO_VIEW = null;
let TOPO_LAST_HIGHLIGHT = [];
let topoDrag = null;

function cssEscape(s) {
  if (window.CSS && CSS.escape) return CSS.escape(String(s));
  return String(s).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
}

function svgPoint(svgEl, evt) {
  const pt = svgEl.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const ctm = svgEl.getScreenCTM();
  if (!ctm) return { x: 0, y: 0 };
  const loc = pt.matrixTransform(ctm.inverse());
  return { x: loc.x, y: loc.y };
}

function viewToData(view, px, py) {
  if (!view) return { x: px, y: py };
  const spanX = (view.maxX - view.minX) || 1;
  const spanY = (view.maxY - view.minY) || 1;
  return {
    x: view.minX + (px - view.pad) / (view.W - 2 * view.pad) * spanX,
    y: view.minY + (py - view.pad) / (view.H - 2 * view.pad) * spanY,
  };
}

function drawTopology(topo, highlightEdges) {
  TOPO_LAST_HIGHLIGHT = highlightEdges || [];
  const W = 640, H = 480, pad = 50;
  if (!topo.nodes.length) { el('topo-svg').innerHTML = ''; TOPO_VIEW = null; return; }
  const xs = topo.nodes.map(n => n.x), ys = topo.nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  TOPO_VIEW = { minX, maxX, minY, maxY, W, H, pad };
  const sx = x => pad + (x - minX) / (maxX - minX || 1) * (W - 2 * pad);
  const sy = y => pad + (y - minY) / (maxY - minY || 1) * (H - 2 * pad);
  const pos = {};
  topo.nodes.forEach(n => { pos[n.id] = [sx(n.x), sy(n.y)]; });

  const hSet = new Set((highlightEdges || []).map(e => e.join('|')));
  const hSetRev = new Set((highlightEdges || []).map(e => [...e].reverse().join('|')));

  let svg = '';
  topo.edges.forEach(e => {
    const [x1, y1] = pos[e.source], [x2, y2] = pos[e.target];
    const isH = hSet.has(`${e.source}|${e.target}`) || hSetRev.has(`${e.source}|${e.target}`);
    svg += `<line class="topo-edge ${isH ? 'highlight' : ''}" data-source="${e.source}" data-target="${e.target}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"><title>${e.source}-${e.target} delay=${e.delay}ms bw=${e.bandwidth}Mbps loss=${e.loss}</title></line>`;
  });
  topo.nodes.forEach(n => {
    const [x, y] = pos[n.id];
    svg += `<g class="topo-node" data-id="${n.id}"><circle cx="${x}" cy="${y}" r="14"><title>${n.id} (seret utk memindah posisi)</title></circle><text x="${x}" y="${y + 4}">${n.id}</text></g>`;
  });
  el('topo-svg').innerHTML = svg;
}

// Drag & drop node topologi. Listener dipasang SEKALI (delegasi event pada
// elemen <svg> yang persisten) sehingga tidak terduplikasi tiap redraw.
function initTopoDrag() {
  const svg = el('topo-svg');

  svg.addEventListener('pointerdown', (evt) => {
    const g = evt.target.closest('.topo-node');
    if (!g) return;
    topoDrag = { id: g.dataset.id };
    svg.setPointerCapture(evt.pointerId);
    g.classList.add('dragging');
    evt.preventDefault();
  });

  svg.addEventListener('pointermove', (evt) => {
    if (!topoDrag) return;
    const pt = svgPoint(svg, evt);
    const g = svg.querySelector(`.topo-node[data-id="${cssEscape(topoDrag.id)}"]`);
    if (g) {
      g.querySelector('circle').setAttribute('cx', pt.x);
      g.querySelector('circle').setAttribute('cy', pt.y);
      g.querySelector('text').setAttribute('x', pt.x);
      g.querySelector('text').setAttribute('y', pt.y + 4);
    }
    svg.querySelectorAll(`.topo-edge[data-source="${cssEscape(topoDrag.id)}"]`).forEach(l => {
      l.setAttribute('x1', pt.x); l.setAttribute('y1', pt.y);
    });
    svg.querySelectorAll(`.topo-edge[data-target="${cssEscape(topoDrag.id)}"]`).forEach(l => {
      l.setAttribute('x2', pt.x); l.setAttribute('y2', pt.y);
    });
  });

  const finishDrag = async (evt) => {
    if (!topoDrag) return;
    const id = topoDrag.id;
    topoDrag = null;
    const g = svg.querySelector(`.topo-node[data-id="${cssEscape(id)}"]`);
    if (g) g.classList.remove('dragging');
    const pt = svgPoint(svg, evt);
    const { x, y } = viewToData(TOPO_VIEW, pt.x, pt.y);
    const node = (TOPOLOGY.nodes || []).find(n => n.id === id);
    if (node) { node.x = x; node.y = y; }
    drawTopology(TOPOLOGY, TOPO_LAST_HIGHLIGHT);
    try {
      const data = await api('/api/topology/positions', 'POST', { positions: { [id]: [x, y] } });
      TOPOLOGY = data;
      drawTopology(TOPOLOGY, TOPO_LAST_HIGHLIGHT);
      el('topo-status').textContent = `Posisi node ${id} disimpan.`;
    } catch (err) {
      el('topo-status').textContent = 'Gagal menyimpan posisi: ' + err.message;
    }
  };
  svg.addEventListener('pointerup', finishDrag);
  svg.addEventListener('pointercancel', () => { topoDrag = null; });
}

// -------------------------------------------------------- algorithm forms --
async function loadAlgorithms() {
  ALGORITHMS = await api('/api/algorithms');
  renderAlgoForms();
}

// Anchor pada halaman /tutorial untuk tiap algoritma routing (lihat templates/tutorial.html)
const ROUTING_ALGO_ANCHOR = {
  dijkstra: 'algo-dijkstra', astar: 'algo-astar', greedy: 'algo-greedy', mst: 'algo-mst',
  aco: 'algo-aco', genetic: 'algo-genetic', pso: 'algo-pso',
  simulated_annealing: 'algo-sa', tabu_search: 'algo-tabu', qlearning: 'algo-qlearning',
};

function renderAlgoForms() {
  const grid = el('algo-grid');
  grid.innerHTML = '';
  ALGORITHMS.forEach(a => {
    const card = document.createElement('div');
    card.className = 'algo-card';
    const paramsHtml = a.params.length
      ? a.params.map(p => `
          <div class="param-row">
            <label for="param-${a.key}-${p.name}">${p.label}</label>
            <input type="number" id="param-${a.key}-${p.name}" min="${p.min}" max="${p.max}"
                   step="${p.step}" value="${p.default}">
          </div>`).join('')
      : '<div class="no-params">Tidak ada parameter yang perlu diatur.</div>';
    const anchor = ROUTING_ALGO_ANCHOR[a.key] || 'algoritma-routing';

    card.innerHTML = `
      <div class="algo-card-head">
        <input type="checkbox" id="chk-${a.key}" value="${a.key}" checked>
        <label for="chk-${a.key}">${a.label}</label>
        <a class="help-link" href="/tutorial#${anchor}" target="_blank" rel="noopener" title="Bantuan: ${a.label}">?</a>
      </div>
      ${paramsHtml}`;
    grid.appendChild(card);
  });
}

function selectedAlgorithms() {
  return ALGORITHMS.filter(a => el(`chk-${a.key}`).checked).map(a => a.key);
}

function gatherAllParams() {
  const out = {};
  ALGORITHMS.forEach(a => {
    const p = {};
    a.params.forEach(spec => {
      const input = el(`param-${a.key}-${spec.name}`);
      if (input) p[spec.name] = parseFloat(input.value);
    });
    out[a.key] = p;
  });
  return out;
}

// -------------------------------------------------------------- run & UI --
function renderTable(results) {
  const body = el('results-body');
  body.innerHTML = '';
  results.forEach((r, idx) => {
    const tr = document.createElement('tr');
    if (r.error || !r.metrics || !r.metrics.valid) {
      tr.classList.add('error-row');
      tr.innerHTML = `<td>${r.label || r.algorithm}</td><td colspan="9">${r.error || 'Jalur tidak ditemukan'}</td>`;
      body.appendChild(tr);
      return;
    }
    const pathStr = r.type === 'tree'
      ? (r.metrics.edges || []).map(e => e.join('-')).join(', ')
      : (r.path || []).join(' → ');
    const qos = r.qos || {};
    tr.classList.add('result-row');
    tr.dataset.idx = idx;
    tr.innerHTML = `
      <td>${r.label}</td>
      <td class="path-cell">${pathStr}</td>
      <td>${r.metrics.hop_count}</td>
      <td>${fmt(r.metrics.delay_ms)}</td>
      <td>${fmt(r.metrics.throughput_mbps)}</td>
      <td>${fmt(r.metrics.packet_loss_pct)}</td>
      <td>${fmt(qos.score, 1)}</td>
      <td><span class="qos-badge ${qosCssClass(qos.class)}">${qos.class || '—'}</span></td>
      <td>${fmt(r.runtime_ms, 3)}</td>
      <td><button class="detail-btn" type="button">Lihat</button></td>`;
    tr.addEventListener('click', (evt) => {
      if (evt.target.closest('.detail-btn')) { openDetailModal(idx); return; }
      selectRow(idx);
    });
    body.appendChild(tr);
  });
}

function kvRows(obj, labelMap) {
  const entries = Object.entries(obj || {});
  if (!entries.length) return '<div class="kv-empty">Tidak ada data.</div>';
  return entries.map(([k, v]) => {
    const label = (labelMap && labelMap[k]) || k;
    const value = v === null || v === undefined ? '—' : (typeof v === 'number' ? fmt(v, Number.isInteger(v) ? 0 : 3) : v);
    return `<div class="kv-row"><span>${label}</span><span>${value}</span></div>`;
  }).join('');
}

function openDetailModal(idx) {
  const r = LAST_RESULTS[idx];
  if (!r) return;
  el('detail-modal-title').textContent = `Detail — ${r.label}`;
  el('detail-params').innerHTML = kvRows(r.params);
  el('detail-meta').innerHTML = kvRows(r.meta);
  el('detail-modal').hidden = false;
}

function closeDetailModal() {
  el('detail-modal').hidden = true;
}

function selectRow(idx) {
  document.querySelectorAll('tr.result-row').forEach(tr => tr.classList.toggle('selected', Number(tr.dataset.idx) === idx));
  const r = LAST_RESULTS[idx];
  if (!r || !r.metrics || !r.metrics.valid) return;
  const edges = r.type === 'tree'
    ? (r.metrics.edges || [])
    : (r.path || []).slice(0, -1).map((n, i) => [n, r.path[i + 1]]);
  drawTopology(TOPOLOGY, edges);
}

function renderBarChart(containerId, results, valueFn, digits, unit) {
  const container = el(containerId);
  const valid = results.filter(r => r.metrics && r.metrics.valid);
  const maxVal = Math.max(1e-9, ...valid.map(r => valueFn(r) ?? 0));
  container.innerHTML = '';
  results.forEach(r => {
    const val = r.metrics && r.metrics.valid ? valueFn(r) : null;
    const pct = val === null || val === undefined ? 0 : Math.max(2, (val / maxVal) * 100);
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-label">${r.label || r.algorithm}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div class="bar-value">${val === null || val === undefined ? '—' : fmt(val, digits) + (unit || '')}</div>`;
    container.appendChild(row);
  });
}

async function runComparison() {
  const source = el('source').value, target = el('target').value;
  const congestion = parseFloat(el('congestion').value || '0');
  const seed = parseInt(el('seed').value || '42', 10);
  const algorithms = selectedAlgorithms();
  const params = gatherAllParams();

  if (!algorithms.length) {
    el('status').textContent = 'Pilih minimal satu algoritma.';
    return;
  }

  el('status').textContent = 'Menjalankan algoritma...';
  el('run-btn').disabled = true;
  try {
    const data = await api('/api/compare', 'POST', { source, target, congestion, seed, algorithms, params });
    LAST_RESULTS = data.results;
    el('results-section').hidden = false;
    renderTable(data.results);
    renderHistoryComparison(data.results);
    renderBarChart('chart-delay', data.results, r => r.metrics.delay_ms, 2, ' ms');
    renderBarChart('chart-hop', data.results, r => r.metrics.hop_count, 0, '');
    renderBarChart('chart-throughput', data.results, r => r.metrics.throughput_mbps, 2, ' Mbps');
    renderBarChart('chart-loss', data.results, r => r.metrics.packet_loss_pct, 3, ' %');
    renderBarChart('chart-qos', data.results, r => (r.qos || {}).score, 1, '');

    const firstValid = data.results.findIndex(r => r.metrics && r.metrics.valid);
    if (firstValid >= 0) selectRow(firstValid);
    el('status').textContent = `Selesai — ${data.results.length} algoritma dievaluasi.`;
  } catch (err) {
    el('status').textContent = 'Gagal menjalankan: ' + err.message;
  } finally {
    el('run-btn').disabled = false;
  }
}

// ============================================================ mode switch =
function setMode(mode) {
  ['routing', 'lb', 'ids'].forEach(m => {
    const active = m === mode;
    el(`view-${m}`).hidden = !active;
    el(`mode-btn-${m}`).classList.toggle('active', active);
    el(`mode-btn-${m}`).setAttribute('aria-selected', String(active));
  });
  if (mode === 'lb' && !LB_INITIALIZED) {
    LB_INITIALIZED = true;
    Promise.all([loadLbCluster(), loadLbAlgorithms()]);
  }
  if (mode === 'ids' && !IDS_INITIALIZED) {
    IDS_INITIALIZED = true;
    Promise.all([loadIdsTopology(), loadIdsAttackTypes()]);
  }
}

// ======================================================== load balancing =
let LB_CLUSTER = null;
let LB_ALGORITHMS = [];
let LB_LAST_RESULTS = [];
let LB_INITIALIZED = false;

async function loadLbCluster() {
  LB_CLUSTER = await api('/api/loadbalance/topology');
  renderLbMeta();
  renderLbServerTable();
  renderLbClientChips();
  drawLbCluster(LB_CLUSTER, null);
}

function renderLbMeta() {
  const c = LB_CLUSTER;
  el('lb-meta-info').innerHTML =
    `${c.servers.length} server &middot; ${c.clients.length} klien &middot; ${c.num_requests} request &middot; area ${fmt(c.area.width, 0)}m &times; ${fmt(c.area.height, 0)}m`;
}

function renderLbServerTable() {
  const body = el('lb-server-table-body');
  body.innerHTML = '';
  LB_CLUSTER.servers.forEach(s => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${s.id}</td><td>${fmt(s.capacity, 1)}</td>`;
    body.appendChild(tr);
  });
}

function renderLbClientChips() {
  const container = el('lb-client-list');
  container.innerHTML = '';
  LB_CLUSTER.clients.forEach(c => {
    const chip = document.createElement('span');
    chip.className = 'node-chip';
    chip.textContent = c.id;
    container.appendChild(chip);
  });
}

async function generateLbCluster() {
  const payload = {
    num_servers: parseInt(el('lb-num-servers').value || '6', 10),
    num_clients: parseInt(el('lb-num-clients').value || '8', 10),
    requests_per_client: parseInt(el('lb-req-per-client').value || '4', 10),
    seed: parseInt(el('lb-seed').value || '42', 10),
    capacity_min: parseFloat(el('lb-capacity-min').value || '60'),
    capacity_max: parseFloat(el('lb-capacity-max').value || '220'),
    demand_min: parseFloat(el('lb-demand-min').value || '1'),
    demand_max: parseFloat(el('lb-demand-max').value || '20'),
    area_width: parseFloat(el('lb-area-width').value || '1000'),
    area_height: parseFloat(el('lb-area-height').value || '1000'),
  };
  el('lb-topo-status').textContent = 'Membangkitkan cluster...';
  try {
    LB_CLUSTER = await api('/api/loadbalance/generate', 'POST', payload);
    renderLbMeta();
    renderLbServerTable();
    renderLbClientChips();
    drawLbCluster(LB_CLUSTER, null);
    el('lb-results-section').hidden = true;
    el('lb-log-section').hidden = true;
    el('lb-scaling-section').hidden = true;
    el('lb-topo-status').textContent = `Cluster dibangkitkan: ${payload.num_servers} server, ${payload.num_clients} klien.`;
  } catch (err) {
    el('lb-topo-status').textContent = 'Gagal: ' + err.message;
  }
}

async function loadLbAlgorithms() {
  LB_ALGORITHMS = await api('/api/loadbalance/algorithms');
  renderLbAlgoForms();
}

// Anchor pada halaman /tutorial untuk tiap algoritma load balancing
const LB_ALGO_ANCHOR = {
  round_robin: 'lb-algo-rr', least_connections: 'lb-algo-lc', weighted_response_time: 'lb-algo-wrt',
  genetic: 'lb-algo-ga', pso: 'lb-algo-pso', aco: 'lb-algo-aco',
  simulated_annealing: 'lb-algo-sa', tabu_search: 'lb-algo-tabu',
};

function renderLbAlgoForms() {
  const grid = el('lb-algo-grid');
  grid.innerHTML = '';
  LB_ALGORITHMS.forEach(a => {
    const card = document.createElement('div');
    card.className = 'algo-card';
    const paramsHtml = a.params.length
      ? a.params.map(p => `
          <div class="param-row">
            <label for="lbparam-${a.key}-${p.name}">${p.label}</label>
            <input type="number" id="lbparam-${a.key}-${p.name}" min="${p.min}" max="${p.max}"
                   step="${p.step}" value="${p.default}">
          </div>`).join('')
      : '<div class="no-params">Tidak ada parameter yang perlu diatur.</div>';
    const anchor = LB_ALGO_ANCHOR[a.key] || 'algoritma-lb';

    card.innerHTML = `
      <div class="algo-card-head">
        <input type="checkbox" id="lbchk-${a.key}" value="${a.key}" checked>
        <label for="lbchk-${a.key}">${a.label}</label>
        <a class="help-link" href="/tutorial#${anchor}" target="_blank" rel="noopener" title="Bantuan: ${a.label}">?</a>
      </div>
      ${paramsHtml}`;
    grid.appendChild(card);
  });
}

function lbSelectedAlgorithms() {
  return LB_ALGORITHMS.filter(a => el(`lbchk-${a.key}`).checked).map(a => a.key);
}

function lbGatherAllParams() {
  const out = {};
  LB_ALGORITHMS.forEach(a => {
    const p = {};
    a.params.forEach(spec => {
      const input = el(`lbparam-${a.key}-${spec.name}`);
      if (input) p[spec.name] = parseFloat(input.value);
    });
    out[a.key] = p;
  });
  return out;
}

function lbUtilClass(pct) {
  if (pct >= 100) return 'lb-util-crit';
  if (pct >= 75) return 'lb-util-warn';
  return 'lb-util-ok';
}

// Server & klien juga bisa digeser (drag & drop); posisi baru dipakai ulang
// untuk menghitung latensi jaringan klien->server (mempengaruhi hasil
// algoritma load balancing) dan ikut tersimpan pada export/import JSON.
let LB_VIEW = null;
let LB_LAST_PERSERVER = null;
let lbDrag = null;

function drawLbCluster(topo, perServer) {
  LB_LAST_PERSERVER = perServer || null;
  const W = 640, H = 480, pad = 50;
  if (!topo.servers.length) { el('lb-svg').innerHTML = ''; LB_VIEW = null; return; }
  const allX = topo.servers.map(s => s.x).concat(topo.clients.map(c => c.x));
  const allY = topo.servers.map(s => s.y).concat(topo.clients.map(c => c.y));
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const minY = Math.min(...allY), maxY = Math.max(...allY);
  LB_VIEW = { minX, maxX, minY, maxY, W, H, pad };
  const sx = x => pad + (x - minX) / (maxX - minX || 1) * (W - 2 * pad);
  const sy = y => pad + (y - minY) / (maxY - minY || 1) * (H - 2 * pad);
  const utilById = {};
  (perServer || []).forEach(s => { utilById[s.id] = s; });

  let svg = '';
  topo.clients.forEach(c => {
    svg += `<g class="lb-client" data-id="${c.id}"><circle cx="${sx(c.x)}" cy="${sy(c.y)}" r="6"><title>${c.id} (seret utk memindah posisi)</title></circle><text x="${sx(c.x)}" y="${sy(c.y) - 9}">${c.id}</text></g>`;
  });
  topo.servers.forEach(s => {
    const cx = sx(s.x), cy = sy(s.y);
    const r = 14 + Math.min(16, s.capacity / 12);
    const info = utilById[s.id];
    const cls = info ? lbUtilClass(info.utilization_pct) : 'lb-util-ok';
    const label = info ? `${s.id} (${fmt(info.utilization_pct, 0)}%)` : s.id;
    const warnMark = info && info.utilization_pct >= 80 ? `<text class="lb-warn-mark" x="${cx + r - 2}" y="${cy - r + 6}">&#9888;</text>` : '';
    svg += `<g class="lb-server" data-id="${s.id}"><circle class="${cls}" cx="${cx}" cy="${cy}" r="${r}"><title>${s.id} kapasitas=${s.capacity}${info ? ' beban=' + fmt(info.load, 1) + ' util=' + fmt(info.utilization_pct, 1) + '%' : ''} (seret utk memindah posisi)</title></circle><text x="${cx}" y="${cy + 4}">${label}</text><text class="lb-cap-label" x="${cx}" y="${cy + r + 11}">cap ${fmt(s.capacity, 0)}</text>${warnMark}</g>`;
  });
  el('lb-svg').innerHTML = svg;
}

function initLbDrag() {
  const svg = el('lb-svg');

  svg.addEventListener('pointerdown', (evt) => {
    const g = evt.target.closest('.lb-server, .lb-client');
    if (!g) return;
    lbDrag = { id: g.dataset.id, type: g.classList.contains('lb-server') ? 'server' : 'client' };
    svg.setPointerCapture(evt.pointerId);
    g.classList.add('dragging');
    evt.preventDefault();
  });

  svg.addEventListener('pointermove', (evt) => {
    if (!lbDrag) return;
    const pt = svgPoint(svg, evt);
    const sel = lbDrag.type === 'server' ? '.lb-server' : '.lb-client';
    const g = svg.querySelector(`${sel}[data-id="${cssEscape(lbDrag.id)}"]`);
    if (!g) return;
    g.querySelector('circle').setAttribute('cx', pt.x);
    g.querySelector('circle').setAttribute('cy', pt.y);
    const yOff = lbDrag.type === 'server' ? 4 : -9;
    g.querySelector('text').setAttribute('x', pt.x);
    g.querySelector('text').setAttribute('y', pt.y + yOff);
  });

  const finishDrag = async (evt) => {
    if (!lbDrag) return;
    const { id, type } = lbDrag;
    lbDrag = null;
    const sel = type === 'server' ? '.lb-server' : '.lb-client';
    const g = svg.querySelector(`${sel}[data-id="${cssEscape(id)}"]`);
    if (g) g.classList.remove('dragging');
    const pt = svgPoint(svg, evt);
    const { x, y } = viewToData(LB_VIEW, pt.x, pt.y);
    const list = type === 'server' ? LB_CLUSTER.servers : LB_CLUSTER.clients;
    const entry = (list || []).find(e => e.id === id);
    if (entry) { entry.x = x; entry.y = y; }
    drawLbCluster(LB_CLUSTER, LB_LAST_PERSERVER);
    try {
      const payload = type === 'server' ? { servers: { [id]: [x, y] } } : { clients: { [id]: [x, y] } };
      const data = await api('/api/loadbalance/positions', 'POST', payload);
      LB_CLUSTER = data;
      drawLbCluster(LB_CLUSTER, LB_LAST_PERSERVER);
      el('lb-status').textContent = `Posisi ${id} disimpan.`;
    } catch (err) {
      el('lb-status').textContent = 'Gagal menyimpan posisi: ' + err.message;
    }
  };
  svg.addEventListener('pointerup', finishDrag);
  svg.addEventListener('pointercancel', () => { lbDrag = null; });
}

function lbQosCssClass(cls) {
  if (!cls) return '';
  return 'qos-' + cls.toLowerCase();
}

function renderLbTable(results) {
  const body = el('lb-results-body');
  body.innerHTML = '';
  results.forEach((r, idx) => {
    const tr = document.createElement('tr');
    if (r.error || !r.metrics || !r.metrics.valid) {
      tr.classList.add('error-row');
      tr.innerHTML = `<td>${r.label || r.algorithm}</td><td colspan="9">${r.error || 'Tidak ada hasil valid'}</td>`;
      body.appendChild(tr);
      return;
    }
    const m = r.metrics;
    const qos = r.qos || {};
    tr.classList.add('result-row');
    tr.dataset.idx = idx;
    tr.innerHTML = `
      <td>${r.label}</td>
      <td>${fmt(m.max_utilization_pct, 1)}</td>
      <td>${fmt(m.avg_response_ms, 2)}</td>
      <td>${fmt(m.fairness_index, 3)}</td>
      <td>${m.served} / ${m.total}</td>
      <td>${fmt(m.dropped_pct, 2)}</td>
      <td>${fmt(qos.score, 1)}</td>
      <td><span class="qos-badge ${lbQosCssClass(qos.class)}">${qos.class || '—'}</span></td>
      <td>${fmt(r.runtime_ms, 3)}</td>
      <td><button class="detail-btn" type="button">Lihat</button></td>`;
    tr.addEventListener('click', (evt) => {
      if (evt.target.closest('.detail-btn')) { openLbDetailModal(idx); return; }
      selectLbRow(idx);
    });
    body.appendChild(tr);
  });
}

function openLbDetailModal(idx) {
  const r = LB_LAST_RESULTS[idx];
  if (!r) return;
  el('detail-modal-title').textContent = `Detail — ${r.label}`;
  el('detail-params').innerHTML = kvRows(r.params);
  const metaWithServers = { ...(r.meta || {}) };
  if (r.metrics && r.metrics.per_server) {
    r.metrics.per_server.forEach(s => {
      metaWithServers[`server_${s.id}`] = `beban ${fmt(s.load, 1)} / kap ${fmt(s.capacity, 1)} (${fmt(s.utilization_pct, 1)}%), ${s.requests_count} req`;
    });
  }
  el('detail-meta').innerHTML = kvRows(metaWithServers);
  el('detail-modal').hidden = false;
}

function renderLbClientLog(clientLog) {
  const container = el('lb-log-list');
  el('lb-log-section').hidden = false;
  if (!clientLog || !clientLog.length) {
    container.innerHTML = '<div class="lb-log-empty">Tidak ada data log untuk hasil ini.</div>';
    return;
  }
  container.innerHTML = clientLog.map(c => {
    const servers = c.servers.length
      ? c.servers.map(s => `<span class="server-chip"><b>${s.server_id}</b>: ${s.requests} req &middot; ${fmt(s.demand, 1)} beban</span>`).join('')
      : '<span class="lb-log-empty">Tidak ada request terlayani.</span>';
    const droppedStr = c.dropped_requests > 0 ? ` &middot; <span class="lb-log-dropped">${c.dropped_requests} request ditolak</span>` : '';
    return `
      <div class="lb-log-row">
        <div class="lb-log-head">
          <strong>${c.client_id}</strong>
          <span>${c.total_requests} request &middot; total beban ${fmt(c.total_demand, 1)}${droppedStr}</span>
        </div>
        <div class="lb-log-servers">${servers}</div>
      </div>`;
  }).join('');
}

// -------------------------------------------------- saran auto-scaling lb --
// Server dgn utilisasi >=80% (per hasil algoritma terpilih) diberi saran
// scale up: SCALE OUT/horizontal (tambah instance server) bila sudah ada
// request yang di-drop (util >=100%), atau SCALE UP/vertical (naikkan
// kapasitas) bila baru mendekati penuh -- lihat scaling_suggestions() di
// algorithms/load_balancing.py.
function renderLbScalingSuggestions(suggestions) {
  const section = el('lb-scaling-section');
  const container = el('lb-scaling-list');
  if (!section || !container) return;
  if (!suggestions || !suggestions.length) {
    section.hidden = true;
    container.innerHTML = '';
    return;
  }
  section.hidden = false;
  container.innerHTML = suggestions.map(s => `
    <div class="lb-scaling-row lb-scaling-${s.level}">
      <div class="lb-scaling-head">
        <strong>${s.server_id}</strong>
        <span class="lb-scaling-badge lb-scaling-badge-${s.level}">${s.level === 'critical' ? 'Overload' : 'Sibuk'}</span>
        <span class="lb-scaling-badge lb-scaling-type">${s.recommendation === 'horizontal' ? 'Scale Out (Horizontal)' : 'Scale Up (Vertical)'}</span>
        <span class="lb-scaling-util">${fmt(s.utilization_pct, 1)}% utilisasi</span>
      </div>
      <div class="lb-scaling-note">${s.note}</div>
    </div>`).join('');
}

function selectLbRow(idx) {
  document.querySelectorAll('#lb-results-body tr.result-row').forEach(tr => tr.classList.toggle('selected', Number(tr.dataset.idx) === idx));
  const r = LB_LAST_RESULTS[idx];
  if (!r || !r.metrics || !r.metrics.valid) return;
  drawLbCluster(LB_CLUSTER, r.metrics.per_server);
  renderLbClientLog(r.client_log);
  renderLbScalingSuggestions(r.scaling_suggestions);
}

async function runLbComparison() {
  const seed = parseInt(el('lb-run-seed').value || '42', 10);
  const algorithms = lbSelectedAlgorithms();
  const params = lbGatherAllParams();

  if (!algorithms.length) {
    el('lb-status').textContent = 'Pilih minimal satu algoritma.';
    return;
  }

  el('lb-status').textContent = 'Menjalankan algoritma load balancing...';
  el('lb-run-btn').disabled = true;
  try {
    const data = await api('/api/loadbalance/compare', 'POST', { seed, algorithms, params });
    LB_LAST_RESULTS = data.results;
    el('lb-results-section').hidden = false;
    renderLbTable(data.results);
    renderBarChart('lb-chart-util', data.results, r => r.metrics.max_utilization_pct, 1, ' %');
    renderBarChart('lb-chart-rt', data.results, r => r.metrics.avg_response_ms, 2, ' ms');
    renderBarChart('lb-chart-fair', data.results, r => r.metrics.fairness_index, 3, '');
    renderBarChart('lb-chart-drop', data.results, r => r.metrics.dropped_pct, 2, ' %');
    renderBarChart('lb-chart-qos', data.results, r => (r.qos || {}).score, 1, '');

    const firstValid = data.results.findIndex(r => r.metrics && r.metrics.valid);
    if (firstValid >= 0) selectLbRow(firstValid);
    el('lb-status').textContent = `Selesai — ${data.results.length} algoritma dievaluasi.`;
  } catch (err) {
    el('lb-status').textContent = 'Gagal menjalankan: ' + err.message;
  } finally {
    el('lb-run-btn').disabled = false;
  }
}

// ===================================================== export grafik (png) =
// Peta konfigurasi tiap chart-card (id kontainer -> sumber data, key metrik,
// judul, format angka) agar bisa digambar ulang ke <canvas> lepas dari DOM
// on-screen (yang berupa div/bar, bukan elemen yang bisa "discreenshot"
// langsung) lalu diunduh sebagai PNG.
const CHART_CONFIGS = {
  'chart-delay': { source: 'routing', key: 'delay_ms', title: 'Delay (ms) — lebih rendah lebih baik', digits: 2, unit: ' ms' },
  'chart-hop': { source: 'routing', key: 'hop_count', title: 'Hop Count — lebih rendah lebih baik', digits: 0, unit: '' },
  'chart-throughput': { source: 'routing', key: 'throughput_mbps', title: 'Throughput (Mbps, simulasi) — lebih tinggi lebih baik', digits: 2, unit: ' Mbps' },
  'chart-loss': { source: 'routing', key: 'packet_loss_pct', title: 'Packet Loss (%, simulasi) — lebih rendah lebih baik', digits: 3, unit: ' %' },
  'chart-qos': { source: 'routing', key: 'qos_score', title: 'Skor QoS (0-100) — lebih tinggi lebih baik', digits: 1, unit: '' },
  'lb-chart-util': { source: 'lb', key: 'max_utilization_pct', title: 'Beban Maks Server (%) — lebih rendah lebih baik', digits: 1, unit: ' %' },
  'lb-chart-rt': { source: 'lb', key: 'avg_response_ms', title: 'Rata-rata Response Time (ms) — lebih rendah lebih baik', digits: 2, unit: ' ms' },
  'lb-chart-fair': { source: 'lb', key: 'fairness_index', title: 'Fairness Index (Jain, 0-1) — lebih tinggi lebih baik', digits: 3, unit: '' },
  'lb-chart-drop': { source: 'lb', key: 'dropped_pct', title: 'Request Loss (%) — lebih rendah lebih baik', digits: 2, unit: ' %' },
  'lb-chart-qos': { source: 'lb', key: 'qos_score', title: 'Skor QoS (0-100) — lebih tinggi lebih baik', digits: 1, unit: '' },
};

function chartMetricValue(r, key) {
  if (key === 'qos_score') return (r.qos || {}).score;
  return (r.metrics || {})[key];
}

function roundRectPath(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function truncateToWidth(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let out = text;
  while (out.length > 1 && ctx.measureText(out + '…').width > maxWidth) {
    out = out.slice(0, -1);
  }
  return out + '…';
}

function buildBarChartCanvas(title, results, key, digits, unit) {
  const rows = results.map(r => ({
    label: r.label || r.algorithm,
    value: (r.metrics && r.metrics.valid) ? chartMetricValue(r, key) : null,
  }));

  const width = 960;
  const rowH = 38, gap = 8;
  const padTop = 60, padBottom = 22, padX = 26;
  const labelW = 230, valueW = 100;
  const trackX = padX + labelW;
  const trackW = width - padX - trackX - valueW;
  const height = padTop + padBottom + rows.length * (rowH + gap) - gap;
  const scale = 2; // render 2x lalu discale via canvas.width utk PNG yang tajam

  const canvas = document.createElement('canvas');
  canvas.width = width * scale;
  canvas.height = height * scale;
  const ctx = canvas.getContext('2d');
  ctx.scale(scale, scale);

  ctx.fillStyle = '#fcfcfb';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#0b0b0b';
  ctx.font = '600 16px -apple-system, "Segoe UI", Roboto, sans-serif';
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(truncateToWidth(ctx, title, width - padX * 2), padX, 28);

  ctx.strokeStyle = '#dcdcd8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padX, 40.5);
  ctx.lineTo(width - padX, 40.5);
  ctx.stroke();

  const validValues = rows.map(r => r.value).filter(v => v !== null && v !== undefined);
  const maxVal = Math.max(1e-9, ...(validValues.length ? validValues : [0]));

  rows.forEach((row, i) => {
    const y = padTop + i * (rowH + gap);

    ctx.fillStyle = '#0b0b0b';
    ctx.font = '13px -apple-system, "Segoe UI", Roboto, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.fillText(truncateToWidth(ctx, row.label, labelW - 10), trackX - 10, y + rowH / 2);
    ctx.textAlign = 'left';

    ctx.fillStyle = '#eef0ee';
    roundRectPath(ctx, trackX, y + 6, trackW, rowH - 12, 4);
    ctx.fill();

    if (row.value !== null && row.value !== undefined) {
      const pct = Math.max(0.02, row.value / maxVal);
      ctx.fillStyle = '#2a78d6';
      roundRectPath(ctx, trackX, y + 6, Math.max(4, trackW * pct), rowH - 12, 4);
      ctx.fill();
    }

    ctx.fillStyle = '#52514e';
    ctx.font = '12px -apple-system, "Segoe UI", Roboto, sans-serif';
    const valStr = row.value === null || row.value === undefined ? '—' : fmt(row.value, digits) + (unit || '');
    ctx.fillText(valStr, trackX + trackW + 10, y + rowH / 2);
  });

  return canvas;
}

function statusElForSource(source) {
  return source === 'lb' ? el('lb-status') : el('status');
}

function downloadChartPng(chartId) {
  const cfg = CHART_CONFIGS[chartId];
  if (!cfg) return;
  const results = cfg.source === 'lb' ? LB_LAST_RESULTS : LAST_RESULTS;
  if (!results.length) {
    statusElForSource(cfg.source).textContent = 'Jalankan perbandingan terlebih dahulu sebelum mengunduh grafik.';
    return;
  }
  const canvas = buildBarChartCanvas(cfg.title, results, cfg.key, cfg.digits, cfg.unit);
  canvas.toBlob(blob => {
    if (blob) downloadBlob(blob, `sdn_${chartId}_${tsFileName()}.png`);
  }, 'image/png');
}

async function downloadAllCharts(source) {
  const ids = Object.keys(CHART_CONFIGS).filter(id => CHART_CONFIGS[id].source === source);
  const results = source === 'lb' ? LB_LAST_RESULTS : LAST_RESULTS;
  if (!results.length) {
    statusElForSource(source).textContent = 'Jalankan perbandingan terlebih dahulu sebelum mengunduh grafik.';
    return;
  }
  for (let i = 0; i < ids.length; i++) {
    downloadChartPng(ids[i]);
    // jeda kecil antar unduhan supaya browser tidak memblokir multi-download beruntun
    await new Promise(resolve => setTimeout(resolve, 350));
  }
}

// ============================================ export visualisasi svg (png) =
// Style tiap elemen SVG topologi/cluster diambil dari class CSS di style.css
// (var(--...) tema light/dark). Saat SVG diserialisasi berdiri sendiri untuk
// dirender ke <img>/<canvas>, ia tidak lagi mewarisi stylesheet halaman --
// jadi style dituliskan ulang di sini sebagai warna literal (skema terang)
// agar PNG yang dihasilkan konsisten apa pun tema browser yang aktif.
const SVG_EXPORT_CSS = `
  .topo-edge { stroke: #dcdcd8; stroke-width: 1.5; }
  .topo-edge.highlight { stroke: #2a78d6; stroke-width: 3; }
  .topo-node circle { fill: #cde2fb; stroke: #2a78d6; stroke-width: 1.5; }
  .topo-node text { fill: #0b0b0b; font-size: 10px; text-anchor: middle; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  .lb-client circle { fill: #fcfcfb; stroke: #85847e; stroke-width: 1.2; }
  .lb-client text { fill: #85847e; font-size: 8px; text-anchor: middle; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  .lb-server text { fill: #0b0b0b; font-size: 10px; text-anchor: middle; font-weight: 600; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  .lb-server .lb-cap-label { fill: #52514e; font-size: 8px; font-weight: 400; }
  .lb-util-ok { fill: #d7f3e8; stroke: #1baf7a; stroke-width: 1.5; }
  .lb-util-warn { fill: #fbe9c6; stroke: #eda100; stroke-width: 1.5; }
  .lb-util-crit { fill: #fbdada; stroke: #e34948; stroke-width: 2; }
`;
const SVG_EXPORT_BG = '#eef0ee';

function exportSvgToPng(svgId, filenamePrefix, statusEl) {
  const svgEl = el(svgId);
  if (!svgEl || !svgEl.childNodes.length) {
    if (statusEl) statusEl.textContent = 'Belum ada visualisasi untuk diunduh (jalankan simulasi/perbandingan terlebih dahulu).';
    return;
  }

  const viewBoxAttr = svgEl.getAttribute('viewBox') || '0 0 640 480';
  const parts = viewBoxAttr.split(/\s+/).map(Number);
  const vbW = parts[2] || 640, vbH = parts[3] || 480;

  const clone = svgEl.cloneNode(true);
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('width', String(vbW));
  clone.setAttribute('height', String(vbH));

  const styleEl = document.createElementNS('http://www.w3.org/2000/svg', 'style');
  styleEl.textContent = SVG_EXPORT_CSS;
  clone.insertBefore(styleEl, clone.firstChild);

  const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bgRect.setAttribute('x', '0');
  bgRect.setAttribute('y', '0');
  bgRect.setAttribute('width', String(vbW));
  bgRect.setAttribute('height', String(vbH));
  bgRect.setAttribute('fill', SVG_EXPORT_BG);
  clone.insertBefore(bgRect, styleEl.nextSibling);

  const svgStr = new XMLSerializer().serializeToString(clone);
  const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  const img = new Image();
  img.onload = () => {
    const scale = 2; // render 2x utk PNG yang tajam
    const canvas = document.createElement('canvas');
    canvas.width = vbW * scale;
    canvas.height = vbH * scale;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = SVG_EXPORT_BG;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    canvas.toBlob(blob => {
      if (blob) downloadBlob(blob, `${filenamePrefix}_${tsFileName()}.png`);
    }, 'image/png');
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    if (statusEl) statusEl.textContent = 'Gagal membuat PNG dari visualisasi.';
  };
  img.src = url;
}

// ============================================== konfigurasi export/import =
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadJson(obj, filename) {
  downloadBlob(new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' }), filename);
}

function tsFileName() {
  const d = new Date();
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function exportConfig() {
  el('config-status').textContent = 'Menyiapkan konfigurasi...';
  try {
    const backend = await api('/api/config/export');
    const config = {
      version: 1,
      exported_at: new Date().toISOString(),
      topology: backend.topology,
      load_balancing: backend.load_balancing,
      ui: {
        routing: {
          source: el('source').value,
          target: el('target').value,
          congestion: el('congestion').value,
          seed: el('seed').value,
          selected_algorithms: ALGORITHMS.length ? selectedAlgorithms() : [],
          params: ALGORITHMS.length ? gatherAllParams() : {},
        },
        load_balancing: {
          seed: el('lb-run-seed').value,
          selected_algorithms: LB_ALGORITHMS.length ? lbSelectedAlgorithms() : [],
          params: LB_ALGORITHMS.length ? lbGatherAllParams() : {},
          generate_form: {
            num_servers: el('lb-num-servers').value,
            num_clients: el('lb-num-clients').value,
            requests_per_client: el('lb-req-per-client').value,
            seed: el('lb-seed').value,
            capacity_min: el('lb-capacity-min').value,
            capacity_max: el('lb-capacity-max').value,
            demand_min: el('lb-demand-min').value,
            demand_max: el('lb-demand-max').value,
            area_width: el('lb-area-width').value,
            area_height: el('lb-area-height').value,
          },
        },
      },
    };
    downloadJson(config, `sdn_config_${tsFileName()}.json`);
    el('config-status').textContent = 'Konfigurasi berhasil diekspor.';
  } catch (err) {
    el('config-status').textContent = 'Gagal ekspor konfigurasi: ' + err.message;
  }
}

function applyRoutingUi(ui) {
  if (!ui || !ALGORITHMS.length) return;
  if (ui.source) el('source').value = ui.source;
  if (ui.target) el('target').value = ui.target;
  if (ui.congestion !== undefined && ui.congestion !== '') el('congestion').value = ui.congestion;
  if (ui.seed !== undefined && ui.seed !== '') el('seed').value = ui.seed;
  ALGORITHMS.forEach(a => {
    const chk = el(`chk-${a.key}`);
    if (chk) chk.checked = !ui.selected_algorithms || ui.selected_algorithms.includes(a.key);
    const p = (ui.params || {})[a.key] || {};
    a.params.forEach(spec => {
      const input = el(`param-${a.key}-${spec.name}`);
      if (input && p[spec.name] !== undefined && p[spec.name] !== null) input.value = p[spec.name];
    });
  });
}

function applyLbGenerateForm(gf) {
  if (!gf) return;
  const map = {
    'lb-num-servers': gf.num_servers, 'lb-num-clients': gf.num_clients,
    'lb-req-per-client': gf.requests_per_client, 'lb-seed': gf.seed,
    'lb-capacity-min': gf.capacity_min, 'lb-capacity-max': gf.capacity_max,
    'lb-demand-min': gf.demand_min, 'lb-demand-max': gf.demand_max,
    'lb-area-width': gf.area_width, 'lb-area-height': gf.area_height,
  };
  Object.entries(map).forEach(([id, val]) => {
    if (val !== undefined && val !== null && val !== '') el(id).value = val;
  });
}

function applyLbAlgoUi(ui) {
  if (!ui || !LB_ALGORITHMS.length) return;
  if (ui.seed !== undefined && ui.seed !== '') el('lb-run-seed').value = ui.seed;
  LB_ALGORITHMS.forEach(a => {
    const chk = el(`lbchk-${a.key}`);
    if (chk) chk.checked = !ui.selected_algorithms || ui.selected_algorithms.includes(a.key);
    const p = (ui.params || {})[a.key] || {};
    a.params.forEach(spec => {
      const input = el(`lbparam-${a.key}-${spec.name}`);
      if (input && p[spec.name] !== undefined && p[spec.name] !== null) input.value = p[spec.name];
    });
  });
}

async function importConfigFile(file) {
  el('config-status').textContent = 'Mengimpor konfigurasi...';
  try {
    const text = await file.text();
    let config;
    try {
      config = JSON.parse(text);
    } catch (e) {
      throw new Error('File bukan JSON yang valid');
    }
    if (!config || (!config.topology && !config.load_balancing)) {
      throw new Error("File konfigurasi tidak valid (butuh 'topology' dan/atau 'load_balancing')");
    }

    const resp = await api('/api/config/import', 'POST', {
      topology: config.topology,
      load_balancing: config.load_balancing,
    });

    if (resp.topology) {
      await refreshTopology({ ...resp.topology, mode: resp.topology_mode, area: resp.area });
    }

    const ui = config.ui || {};
    applyRoutingUi(ui.routing);
    applyLbGenerateForm((ui.load_balancing || {}).generate_form);

    if (resp.load_balancing) {
      LB_CLUSTER = resp.load_balancing;
      if (!LB_INITIALIZED) {
        LB_INITIALIZED = true;
        await loadLbAlgorithms();
      }
      renderLbMeta();
      renderLbServerTable();
      renderLbClientChips();
      drawLbCluster(LB_CLUSTER, null);
      el('lb-results-section').hidden = true;
      el('lb-log-section').hidden = true;
      el('lb-scaling-section').hidden = true;
      applyLbAlgoUi(ui.load_balancing);
    }

    el('config-status').textContent = 'Konfigurasi berhasil diimpor.';
  } catch (err) {
    el('config-status').textContent = 'Gagal impor konfigurasi: ' + err.message;
  }
}

// ================================================ intrusion detection (IDS) =
let IDS_CLUSTER = null;
let IDS_ATTACK_TYPES = [];
let IDS_LAST_FLOWS = [];
let IDS_INITIALIZED = false;

const IDS_LABEL_COLORS = {
  normal: 'var(--good)',
  dos_ddos: 'var(--critical)',
  probing: 'var(--warning)',
  r2l: '#8855e0',
  u2r: '#b3261e',
};
const IDS_LABEL_NAMES = { normal: 'Normal' };

async function loadIdsTopology() {
  IDS_CLUSTER = await api('/api/ids/topology');
  renderIdsMeta();
  renderIdsServerList();
  renderIdsClientList();
  drawIdsNetwork(IDS_CLUSTER, null);
}

function renderIdsMeta() {
  const c = IDS_CLUSTER;
  el('ids-meta-info').innerHTML =
    `${c.servers.length} server &middot; ${c.clients.length} client (${c.num_normal} normal, ${c.num_attackers} penyerang) &middot; area ${fmt(c.area.width, 0)}m &times; ${fmt(c.area.height, 0)}m`;
}

function renderIdsServerList() {
  const container = el('ids-server-list');
  container.innerHTML = '';
  IDS_CLUSTER.servers.forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'node-chip';
    chip.textContent = `${s.id} (kap ${fmt(s.capacity, 0)})`;
    container.appendChild(chip);
  });
}

function renderIdsClientList() {
  const container = el('ids-client-list');
  container.innerHTML = '';
  IDS_CLUSTER.clients.forEach(c => {
    const chip = document.createElement('span');
    chip.className = 'node-chip' + (c.role === 'attacker' ? ' chip-attacker' : '');
    chip.textContent = c.id + (c.role === 'attacker' ? ' ⚠' : '');
    container.appendChild(chip);
  });
}

async function generateIdsTopology() {
  const payload = {
    num_servers: parseInt(el('ids-num-servers').value || '4', 10),
    num_clients: parseInt(el('ids-num-clients').value || '20', 10),
    attacker_ratio: parseFloat(el('ids-attacker-ratio').value || '25') / 100,
    area_width: parseFloat(el('ids-area-width').value || '1000'),
    area_height: parseFloat(el('ids-area-height').value || '1000'),
    seed: parseInt(el('ids-topo-seed').value || '42', 10),
  };
  el('ids-topo-status').textContent = 'Membangkitkan topologi...';
  try {
    IDS_CLUSTER = await api('/api/ids/generate', 'POST', payload);
    renderIdsMeta();
    renderIdsServerList();
    renderIdsClientList();
    drawIdsNetwork(IDS_CLUSTER, null);
    el('ids-results-section').hidden = true;
    el('ids-log-section').hidden = true;
    el('ids-topo-status').textContent =
      `Topologi dibangkitkan: ${IDS_CLUSTER.servers.length} server, ${IDS_CLUSTER.clients.length} client (${IDS_CLUSTER.num_attackers} penyerang).`;
  } catch (err) {
    el('ids-topo-status').textContent = 'Gagal: ' + err.message;
  }
}

async function loadIdsAttackTypes() {
  IDS_ATTACK_TYPES = await api('/api/ids/attack-types');
  IDS_ATTACK_TYPES.forEach(a => { IDS_LABEL_NAMES[a.key] = a.label; });
  renderIdsAttackGrid();
}

function renderIdsAttackGrid() {
  const grid = el('ids-attack-grid');
  grid.innerHTML = '';
  IDS_ATTACK_TYPES.forEach(a => {
    const card = document.createElement('div');
    card.className = 'algo-card';
    card.innerHTML = `
      <div class="algo-card-head">
        <input type="checkbox" id="ids-chk-${a.key}" value="${a.key}" checked>
        <label for="ids-chk-${a.key}">${a.label}</label>
      </div>
      <p class="hint ids-attack-desc">${a.description}</p>
      <div class="param-row">
        <label for="ids-potensi-${a.key}">Potensi / intensitas (%)</label>
        <input type="number" id="ids-potensi-${a.key}" min="0" max="100" step="5" value="${Math.round(a.default_potensi * 100)}">
      </div>`;
    grid.appendChild(card);
  });
}

function idsSelectedAttacks() {
  return IDS_ATTACK_TYPES.filter(a => el(`ids-chk-${a.key}`).checked).map(a => a.key);
}

function idsGatherPotensi() {
  const out = {};
  IDS_ATTACK_TYPES.forEach(a => {
    const input = el(`ids-potensi-${a.key}`);
    out[a.key] = input ? parseFloat(input.value) / 100 : a.default_potensi;
  });
  return out;
}

function idsLabelColor(label) {
  return IDS_LABEL_COLORS[label] || 'var(--text-muted)';
}

function idsLabelText(label) {
  return label === 'normal' ? 'Normal' : (IDS_LABEL_NAMES[label] || label);
}

function renderIdsLegend(selectedLabels) {
  const container = el('ids-legend');
  const labels = ['normal', ...selectedLabels];
  container.innerHTML = labels.map(l =>
    `<span class="ids-legend-item"><span class="ids-legend-dot" style="background:${idsLabelColor(l)}"></span>${idsLabelText(l)}</span>`
  ).join('');
}

const IDS_FEATURE_LABELS = {
  duration: 'Duration', protocol_type: 'Protocol Type', service: 'Service',
  src_bytes: 'Src Bytes', dst_bytes: 'Dst Bytes', wrong_fragment: 'Wrong Fragment',
  count: 'Count', srv_count: 'Srv Count', serror_rate: 'Serror Rate', rerror_rate: 'Rerror Rate',
  dst_host_diff_srv_rate: 'Dst Host Diff Srv Rate', dst_host_same_src_port_rate: 'Dst Host Same Src Port Rate',
  num_failed_logins: 'Num Failed Logins', is_guest_login: 'Is Guest Login',
  root_shell: 'Root Shell', su_attempted: 'Su Attempted',
  num_file_creations: 'Num File Creations', num_root: 'Num Root',
};

function renderIdsKeyFeaturesInline(keyFeaturesAvg) {
  if (!keyFeaturesAvg || !keyFeaturesAvg.length) return '—';
  return keyFeaturesAvg.map(kf =>
    `<span class="ids-feat-chip">${IDS_FEATURE_LABELS[kf.feature] || kf.feature}=${fmt(kf.avg, Number.isInteger(kf.avg) ? 0 : 3)}</span>`
  ).join(' ');
}

function renderIdsSummaryTable(perLabel) {
  const body = el('ids-summary-body');
  body.innerHTML = '';
  perLabel.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="ids-legend-dot" style="background:${idsLabelColor(row.label)}"></span> ${row.label_text}</td>
      <td>${row.count}</td>
      <td>${row.detected}</td>
      <td>${row.detection_rate_pct === null ? '—' : fmt(row.detection_rate_pct, 1)}</td>
      <td>${row.avg_severity === null ? '—' : fmt(row.avg_severity, 3)}</td>
      <td class="ids-feat-cell">${renderIdsKeyFeaturesInline(row.key_features_avg)}</td>`;
    body.appendChild(tr);
  });
}

function renderIdsConfusion(summary) {
  const cm = summary.confusion;
  el('ids-cm-tp').textContent = cm.tp;
  el('ids-cm-fn').textContent = cm.fn;
  el('ids-cm-fp').textContent = cm.fp;
  el('ids-cm-tn').textContent = cm.tn;
  const kv = {
    precision: summary.precision, recall: summary.recall, f1_score: summary.f1_score,
    accuracy: summary.accuracy, false_alarm_rate_pct: summary.false_alarm_rate_pct,
    skor_deteksi: summary.score,
  };
  const kvLabels = {
    precision: 'Precision', recall: 'Recall', f1_score: 'F1-score', accuracy: 'Akurasi',
    false_alarm_rate_pct: 'False Alarm Rate (%)', skor_deteksi: 'Skor Deteksi (0-100)',
  };
  el('ids-score-kv').innerHTML = kvRows(kv, kvLabels) +
    (summary.qos_class ? `<div class="kv-row"><span>Kelas</span><span><span class="qos-badge ${qosCssClass(summary.qos_class)}">${summary.qos_class}</span></span></div>` : '');
}

function renderSimpleBarChart(containerId, items, valueFn, digits, unit, labelFn) {
  const container = el(containerId);
  const vals = items.map(r => valueFn(r)).filter(v => v !== null && v !== undefined);
  const maxVal = Math.max(1e-9, ...vals);
  container.innerHTML = '';
  items.forEach(r => {
    const val = valueFn(r);
    const pct = val === null || val === undefined ? 0 : Math.max(2, (val / maxVal) * 100);
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-label">${labelFn(r)}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div class="bar-value">${val === null || val === undefined ? '—' : fmt(val, digits) + (unit || '')}</div>`;
    container.appendChild(row);
  });
}

function renderIdsBarCharts(perLabel) {
  renderSimpleBarChart('ids-chart-count', perLabel, r => r.count, 0, '', r => r.label_text);
  renderSimpleBarChart('ids-chart-detect', perLabel, r => r.detection_rate_pct, 1, ' %', r => r.label_text);
  renderSimpleBarChart('ids-chart-severity', perLabel, r => r.avg_severity, 3, '', r => r.label_text);
}

// ------------------------------- perbandingan variabel: normal vs serangan --
// Per jenis intrusion, tampilkan kurva sebaran (nilai diurutkan naik, sumbu-x
// = persentil sampel) untuk fitur kunci -- membandingkan kondisi normal vs
// serangan langsung berdampingan (small multiples), konsisten dgn variabel
// yang disebut pada deskripsi tiap ATTACK_TYPES di intrusion_detection.py.
const IDS_LINECHART_SPECS = {
  dos_ddos: {
    title: 'DoS & DDoS (Banjir Trafik)',
    features: [
      { key: 'count', label: 'Count' },
      { key: 'srv_count', label: 'Srv Count' },
      { key: 'serror_rate', label: 'Serror Rate' },
      { key: 'rerror_rate', label: 'Rerror Rate' },
      { key: 'src_bytes', label: 'Src Bytes' },
    ],
  },
  probing: {
    title: 'Reconnaissance / Probing (Pengintaian)',
    features: [
      { key: 'dst_host_diff_srv_rate', label: 'Dst Host Diff Srv Rate' },
      { key: 'dst_host_same_src_port_rate', label: 'Dst Host Same Src Port Rate' },
    ],
    categorical: ['protocol_type', 'service'],
  },
  r2l: {
    title: 'Eksploitasi Akses Masuk (Remote to Local / R2L)',
    features: [
      { key: 'num_failed_logins', label: 'Num Failed Logins' },
      { key: 'is_guest_login', label: 'Is Guest Login (proporsi)' },
      { key: 'duration', label: 'Duration' },
    ],
  },
};

function idsQuantilePoints(values, numPoints) {
  numPoints = numPoints || 21;
  const sorted = (values || []).filter(v => typeof v === 'number').slice().sort((a, b) => a - b);
  const n = sorted.length;
  if (!n) return [];
  const pts = [];
  for (let i = 0; i < numPoints; i++) {
    const pct = (i / (numPoints - 1)) * 100;
    const idx = Math.round((pct / 100) * (n - 1));
    pts.push({ x: pct, y: sorted[idx] });
  }
  return pts;
}

function renderIdsLineChartSVG(container, seriesList) {
  const W = 320, H = 150, padL = 46, padR = 10, padT = 10, padB = 22;
  const allY = seriesList.flatMap(s => s.points.map(p => p.y));
  if (!allY.length) { container.innerHTML = '<div class="kv-empty">Tidak ada data.</div>'; return; }
  let minY = Math.min(...allY), maxY = Math.max(...allY);
  if (minY === maxY) { minY -= Math.abs(minY) * 0.1 || 1; maxY += Math.abs(maxY) * 0.1 || 1; }
  const sx = x => padL + (x / 100) * (W - padL - padR);
  const sy = y => H - padB - ((y - minY) / (maxY - minY)) * (H - padT - padB);
  const decimals = maxY >= 5 ? 0 : (maxY >= 1 ? 2 : 3);

  const yTicks = 4;
  let gridSvg = '';
  for (let i = 0; i <= yTicks; i++) {
    const val = minY + (i / yTicks) * (maxY - minY);
    const y = sy(val);
    gridSvg += `<line class="ids-grid-line" x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}"></line>`;
    gridSvg += `<text class="ids-axis-label" x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end">${fmt(val, decimals)}</text>`;
  }
  [0, 25, 50, 75, 100].forEach(pct => {
    gridSvg += `<text class="ids-axis-label" x="${sx(pct).toFixed(1)}" y="${H - 6}" text-anchor="middle">${pct}%</text>`;
  });

  let linesSvg = '';
  seriesList.forEach(s => {
    if (!s.points.length) return;
    const d = s.points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
    linesSvg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"></path>`;
  });

  container.innerHTML = `
    <svg class="ids-linechart-svg" viewBox="0 0 ${W} ${H}">${gridSvg}${linesSvg}</svg>
    <div class="ids-legend">${seriesList.map(s =>
      `<span class="ids-legend-item"><span class="ids-legend-dot" style="background:${s.color}"></span>${s.name}</span>`
    ).join('')}</div>`;
}

function idsCategoryCounts(flows, key) {
  const counts = {};
  flows.forEach(f => {
    const v = f.features ? f.features[key] : undefined;
    if (v === undefined || v === null) return;
    counts[v] = (counts[v] || 0) + 1;
  });
  return counts;
}

function renderIdsCategoricalCompare(container, metricLabel, normalFlows, attackFlows, key, attackColor, attackName) {
  const normalCounts = idsCategoryCounts(normalFlows, key);
  const attackCounts = idsCategoryCounts(attackFlows, key);
  const categories = Array.from(new Set([...Object.keys(normalCounts), ...Object.keys(attackCounts)])).sort();
  const nTotal = normalFlows.length || 1;
  const aTotal = attackFlows.length || 1;
  const rowsHtml = categories.map(cat => {
    const nPct = ((normalCounts[cat] || 0) / nTotal) * 100;
    const aPct = ((attackCounts[cat] || 0) / aTotal) * 100;
    return `
      <div class="ids-cat-cat-label">${cat}</div>
      <div class="bar-row"><div class="bar-label">Normal</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, nPct)}%;background:${idsLabelColor('normal')}"></div></div><div class="bar-value">${fmt(nPct, 1)} %</div></div>
      <div class="bar-row"><div class="bar-label">${attackName}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, aPct)}%;background:${attackColor}"></div></div><div class="bar-value">${fmt(aPct, 1)} %</div></div>`;
  }).join('');
  container.innerHTML = `<div class="ids-cat-title">${metricLabel}</div><div class="bars">${rowsHtml || '<div class="kv-empty">Tidak ada data.</div>'}</div>`;
}

function renderIdsFeatureComparisons(flows, selectedAttacks) {
  const section = el('ids-linechart-section');
  const body = el('ids-linechart-body');
  const normalFlows = flows.filter(f => f.label === 'normal');
  const activeTypes = Object.keys(IDS_LINECHART_SPECS).filter(a =>
    selectedAttacks.includes(a) && flows.some(f => f.label === a)
  );
  if (!activeTypes.length || !normalFlows.length) {
    section.hidden = true;
    body.innerHTML = '';
    return;
  }
  section.hidden = false;

  let html = '';
  activeTypes.forEach(atype => {
    const spec = IDS_LINECHART_SPECS[atype];
    html += `<div class="ids-linechart-group">
      <h3 class="ids-linechart-heading"><span class="ids-legend-dot" style="background:${idsLabelColor(atype)}"></span>${spec.title}</h3>
      <div class="chart-grid">`;
    spec.features.forEach(feat => {
      html += `<div class="chart-card"><h3>${feat.label}</h3><div id="ids-lc-${atype}-${feat.key}"></div></div>`;
    });
    if (spec.categorical) {
      html += `<div class="chart-card full-span"><h3>Distribusi Protocol Type &amp; Service (%)</h3><div id="ids-lc-${atype}-categorical"></div></div>`;
    }
    html += `</div></div>`;
  });
  body.innerHTML = html;

  activeTypes.forEach(atype => {
    const spec = IDS_LINECHART_SPECS[atype];
    const attackFlows = flows.filter(f => f.label === atype);
    const color = idsLabelColor(atype);
    const attackName = idsLabelText(atype);
    spec.features.forEach(feat => {
      const nPts = idsQuantilePoints(normalFlows.map(f => f.features[feat.key]));
      const aPts = idsQuantilePoints(attackFlows.map(f => f.features[feat.key]));
      renderIdsLineChartSVG(el(`ids-lc-${atype}-${feat.key}`), [
        { name: 'Normal', color: idsLabelColor('normal'), points: nPts },
        { name: attackName, color, points: aPts },
      ]);
    });
    if (spec.categorical) {
      const container = el(`ids-lc-${atype}-categorical`);
      container.innerHTML = '';
      spec.categorical.forEach(key => {
        const sub = document.createElement('div');
        sub.className = 'ids-cat-block';
        container.appendChild(sub);
        renderIdsCategoricalCompare(sub, IDS_FEATURE_LABELS[key] || key, normalFlows, attackFlows, key, color, attackName);
      });
    }
  });
}

function drawIdsNetwork(topo, flows) {
  const W = 640, H = 480, pad = 50;
  if (!topo.servers.length) { el('ids-svg').innerHTML = ''; return; }
  const allX = topo.servers.map(s => s.x).concat(topo.clients.map(c => c.x));
  const allY = topo.servers.map(s => s.y).concat(topo.clients.map(c => c.y));
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const minY = Math.min(...allY), maxY = Math.max(...allY);
  const sx = x => pad + (x - minX) / (maxX - minX || 1) * (W - 2 * pad);
  const sy = y => pad + (y - minY) / (maxY - minY || 1) * (H - 2 * pad);

  // Agregasi per client (server tujuan terbanyak, label dominan) & tekanan tiap server.
  const clientAgg = {};
  const serverLoad = {};
  topo.servers.forEach(s => { serverLoad[s.id] = { attack: 0, total: 0 }; });

  (flows || []).forEach(f => {
    if (!clientAgg[f.client_id]) clientAgg[f.client_id] = { counts: {}, servers: {} };
    const agg = clientAgg[f.client_id];
    agg.counts[f.label] = (agg.counts[f.label] || 0) + 1;
    agg.servers[f.server_id] = (agg.servers[f.server_id] || 0) + 1;
    if (serverLoad[f.server_id]) {
      serverLoad[f.server_id].total += 1;
      if (f.label !== 'normal') serverLoad[f.server_id].attack += 1;
    }
  });

  function dominantAttackLabel(agg) {
    let domLabel = null, domCount = -1;
    Object.entries(agg.counts).forEach(([lab, cnt]) => {
      if (lab !== 'normal' && cnt > domCount) { domCount = cnt; domLabel = lab; }
    });
    return domLabel;
  }

  let svg = '';
  topo.clients.forEach(c => {
    const agg = clientAgg[c.id];
    if (!agg) return;
    let bestServer = null, bestCount = -1;
    Object.entries(agg.servers).forEach(([sid, cnt]) => { if (cnt > bestCount) { bestCount = cnt; bestServer = sid; } });
    const server = topo.servers.find(s => s.id === bestServer);
    if (!server) return;
    const domLabel = dominantAttackLabel(agg) || 'normal';
    const [x1, y1] = [sx(c.x), sy(c.y)];
    const [x2, y2] = [sx(server.x), sy(server.y)];
    svg += `<line class="ids-edge" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${idsLabelColor(domLabel)}" stroke-opacity="0.4"></line>`;
  });

  topo.servers.forEach(s => {
    const cx = sx(s.x), cy = sy(s.y);
    const load = serverLoad[s.id] || { attack: 0, total: 0 };
    const pressure = load.total ? load.attack / load.total : 0;
    const fill = pressure > 0.5 ? 'var(--critical-soft)' : pressure > 0.15 ? 'var(--warning-soft)' : 'var(--good-soft)';
    const stroke = pressure > 0.5 ? 'var(--critical)' : pressure > 0.15 ? 'var(--warning)' : 'var(--good)';
    const size = 20 + Math.min(16, s.capacity / 14);
    svg += `<g class="ids-server"><rect x="${cx - size / 2}" y="${cy - size / 2}" width="${size}" height="${size}" rx="4" fill="${fill}" stroke="${stroke}" stroke-width="1.5"><title>${s.id} kapasitas=${s.capacity}${load.total ? ' traffic=' + load.total + ' (berbahaya ' + load.attack + ')' : ''}</title></rect><text x="${cx}" y="${cy + size / 2 + 12}">${s.id}</text></g>`;
  });

  topo.clients.forEach(c => {
    const cx = sx(c.x), cy = sy(c.y);
    const agg = clientAgg[c.id];
    let color = c.role === 'attacker' ? 'var(--critical)' : 'var(--series-1)';
    let fillColor = c.role === 'attacker' ? 'var(--critical-soft)' : 'var(--series-1-soft)';
    if (agg) {
      const domLabel = dominantAttackLabel(agg);
      if (domLabel) { color = idsLabelColor(domLabel); fillColor = color; }
    }
    svg += `<g class="ids-client"><circle cx="${cx}" cy="${cy}" r="7" fill="${fillColor}" stroke="${color}" stroke-width="1.3"><title>${c.id} (${c.role === 'attacker' ? 'penyerang' : 'normal'})</title></circle></g>`;
  });

  el('ids-svg').innerHTML = svg;
}

function renderIdsLog(flows, total) {
  const body = el('ids-log-body');
  body.innerHTML = '';
  const sample = flows.slice(0, 60);
  sample.forEach((f, idx) => {
    const isAttack = f.label !== 'normal';
    const correct = (isAttack && f.detected) || (!isAttack && !f.detected);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${f.client_id}</td>
      <td>${f.client_role === 'attacker' ? 'Penyerang' : 'Normal'}</td>
      <td>${f.server_id}</td>
      <td><span class="ids-legend-dot" style="background:${idsLabelColor(f.label)}"></span> ${idsLabelText(f.label)}</td>
      <td>${fmt(f.severity, 3)}</td>
      <td><span class="qos-badge ${correct ? 'qos-good' : 'qos-poor'}">${f.detected ? 'Ya' : 'Tidak'}</span></td>
      <td>${fmt(f.confidence, 3)}</td>
      <td><button class="detail-btn ids-detail-btn" type="button" data-idx="${idx}">Lihat</button></td>`;
    body.appendChild(tr);
  });
  el('ids-log-hint').textContent = `Menampilkan ${sample.length} dari total ${total} flow (kolom "Terdeteksi" hijau = hasil deteksi sesuai label sebenarnya, merah = keliru/lolos).`;
  el('ids-log-section').hidden = false;
}

function openIdsDetailModal(idx) {
  const f = IDS_LAST_FLOWS[idx];
  if (!f) return;
  const isAttack = f.label !== 'normal';
  const correct = (isAttack && f.detected) || (!isAttack && !f.detected);
  el('ids-detail-modal-title').textContent = `Detail Flow — ${f.id} (${idsLabelText(f.label)})`;
  el('ids-detail-summary').innerHTML = kvRows({
    client: f.client_id,
    peran_client: f.client_role === 'attacker' ? 'Penyerang' : 'Normal',
    server: f.server_id,
    label_sebenarnya: idsLabelText(f.label),
    severity: f.severity,
    terdeteksi: f.detected ? 'Ya' : 'Tidak',
    confidence: f.confidence,
    hasil: correct ? 'Sesuai' : 'Keliru/lolos',
  }, {
    client: 'Client', peran_client: 'Peran Client', server: 'Server',
    label_sebenarnya: 'Label Sebenarnya', severity: 'Severity', terdeteksi: 'Terdeteksi',
    confidence: 'Confidence', hasil: 'Hasil Deteksi',
  });
  el('ids-detail-features').innerHTML = kvRows(f.features, IDS_FEATURE_LABELS);
  el('ids-detail-modal').hidden = false;
}

function closeIdsDetailModal() {
  el('ids-detail-modal').hidden = true;
}

async function runIdsSimulation() {
  const attacks = idsSelectedAttacks();
  const potensi = idsGatherPotensi();
  const flowsPerClient = parseInt(el('ids-flows-per-client').value || '6', 10);
  const seed = parseInt(el('ids-run-seed').value || '42', 10);

  el('ids-status').textContent = 'Menjalankan simulasi traffic & deteksi...';
  el('ids-run-btn').disabled = true;
  try {
    const data = await api('/api/ids/simulate', 'POST', {
      attacks, potensi, flows_per_client: flowsPerClient, seed,
    });
    IDS_LAST_FLOWS = data.flows;
    el('ids-results-section').hidden = false;
    renderIdsLegend(attacks);
    renderIdsSummaryTable(data.summary.per_label);
    renderIdsConfusion(data.summary);
    renderIdsBarCharts(data.summary.per_label);
    drawIdsNetwork(IDS_CLUSTER, data.flows);
    renderIdsLog(data.flows, data.summary.total_flows);
    renderIdsFeatureComparisons(data.flows, attacks);
    el('ids-status').textContent =
      `Selesai — ${data.summary.total_flows} flow (${data.summary.total_attack_flows} serangan, ${data.summary.total_normal_flows} normal).`;
  } catch (err) {
    el('ids-status').textContent = 'Gagal menjalankan: ' + err.message;
  } finally {
    el('ids-run-btn').disabled = false;
  }
}

// ===================================================== export hasil (excel) =
async function exportResultsExcel() {
  if (!LAST_RESULTS.length && !LB_LAST_RESULTS.length) {
    el('export-status').textContent = 'Jalankan minimal satu perbandingan (routing atau load balancing) sebelum ekspor.';
    return;
  }
  el('export-status').textContent = 'Menyiapkan file Excel...';
  try {
    const payload = {
      routing: LAST_RESULTS.length ? {
        source: el('source').value,
        target: el('target').value,
        congestion: parseFloat(el('congestion').value || '0'),
        seed: parseInt(el('seed').value || '42', 10),
        results: LAST_RESULTS,
        history: LAST_HISTORY_ROWS,
        topology: TOPOLOGY,
        topology_meta: {
          mode: TOPOLOGY.mode,
          num_nodes: TOPOLOGY.nodes.length,
          num_edges: TOPOLOGY.edges.length,
          area: TOPOLOGY.area,
          generation_form: {
            num_nodes: el('gen-num-nodes').value,
            k_neighbors: el('gen-k-neighbors').value,
            area_width: el('gen-area-width').value,
            area_height: el('gen-area-height').value,
            seed: el('gen-seed').value,
          },
        },
      } : {},
      load_balancing: LB_LAST_RESULTS.length ? {
        seed: parseInt(el('lb-run-seed').value || '42', 10),
        results: LB_LAST_RESULTS,
        cluster: LB_CLUSTER,
        cluster_meta: {
          num_servers: (LB_CLUSTER.servers || []).length,
          num_clients: (LB_CLUSTER.clients || []).length,
          requests_per_client: (LB_CLUSTER.clients || []).length
            ? Math.round((LB_CLUSTER.num_requests || 0) / LB_CLUSTER.clients.length)
            : LB_CLUSTER.num_requests,
          area: LB_CLUSTER.area,
          generation_form: {
            num_servers: el('lb-num-servers').value,
            num_clients: el('lb-num-clients').value,
            requests_per_client: el('lb-req-per-client').value,
            seed: el('lb-seed').value,
            capacity_min: el('lb-capacity-min').value,
            capacity_max: el('lb-capacity-max').value,
            demand_min: el('lb-demand-min').value,
            demand_max: el('lb-demand-max').value,
            area_width: el('lb-area-width').value,
            area_height: el('lb-area-height').value,
          },
        },
      } : {},
    };
    const res = await fetch('/api/export/excel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const cd = res.headers.get('Content-Disposition') || '';
    const match = cd.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : `sdn_simulator_export_${tsFileName()}.xlsx`;
    downloadBlob(blob, filename);
    el('export-status').textContent = 'File Excel berhasil diunduh.';
  } catch (err) {
    el('export-status').textContent = 'Gagal ekspor Excel: ' + err.message;
  }
}

// ------------------------------------------------------------------ init --
el('run-btn').addEventListener('click', runComparison);
el('add-node-btn').addEventListener('click', addNode);
el('add-edge-btn').addEventListener('click', upsertEdge);
el('reset-topo-btn').addEventListener('click', resetTopology);
el('generate-topo-btn').addEventListener('click', generateTopology);
el('detail-modal-close').addEventListener('click', closeDetailModal);
el('clear-history-btn').addEventListener('click', clearHistoryStore);
el('detail-modal').addEventListener('click', (evt) => {
  if (evt.target.id === 'detail-modal') closeDetailModal();
});
el('mode-btn-routing').addEventListener('click', () => setMode('routing'));
el('mode-btn-lb').addEventListener('click', () => setMode('lb'));
el('mode-btn-ids').addEventListener('click', () => setMode('ids'));
el('lb-generate-btn').addEventListener('click', generateLbCluster);
el('lb-run-btn').addEventListener('click', runLbComparison);
el('ids-generate-btn').addEventListener('click', generateIdsTopology);
el('ids-run-btn').addEventListener('click', runIdsSimulation);
el('ids-log-body').addEventListener('click', (evt) => {
  const btn = evt.target.closest('.ids-detail-btn');
  if (btn) openIdsDetailModal(Number(btn.dataset.idx));
});
el('ids-detail-modal-close').addEventListener('click', closeIdsDetailModal);
el('ids-detail-modal').addEventListener('click', (evt) => {
  if (evt.target.id === 'ids-detail-modal') closeIdsDetailModal();
});
el('export-config-btn').addEventListener('click', exportConfig);
el('import-config-btn').addEventListener('click', () => el('import-config-input').click());
el('import-config-input').addEventListener('change', (evt) => {
  const file = evt.target.files[0];
  if (file) importConfigFile(file);
  evt.target.value = '';
});
el('export-excel-btn').addEventListener('click', exportResultsExcel);
document.querySelectorAll('.chart-png-btn').forEach(btn => {
  btn.addEventListener('click', () => downloadChartPng(btn.dataset.chart));
});
el('export-charts-routing-btn').addEventListener('click', () => downloadAllCharts('routing'));
el('export-charts-lb-btn').addEventListener('click', () => downloadAllCharts('lb'));
el('export-topo-png-btn').addEventListener('click', () => exportSvgToPng('topo-svg', 'sdn_topologi', el('topo-status')));
el('export-lb-png-btn').addEventListener('click', () => exportSvgToPng('lb-svg', 'sdn_cluster', el('lb-topo-status')));

(async function init() {
  initTopoDrag();
  initLbDrag();
  await Promise.all([loadTopology(), loadAlgorithms()]);
})();
