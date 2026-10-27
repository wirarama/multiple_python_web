/* Grafik SVG sederhana (tanpa pustaka luar): garis/area & batang horizontal.
   Warna mengikuti palet tervalidasi: satu seri per grafik, tanpa legenda. */
const VIZ = {
  light: { s1:"#2a78d6", s2:"#eb6834", grid:"#e2e8f0", ink:"#0f172a", mut:"#64748b", surf:"#ffffff" },
  dark:  { s1:"#3987e5", s2:"#d95926", grid:"#25314a", ink:"#e6edf7", mut:"#93a4bf", surf:"#121a2b" }
};
function vizPal(){
  return document.documentElement.getAttribute("data-theme") === "dark" ? VIZ.dark : VIZ.light;
}
function tip(show, x, y, html){
  const el = document.getElementById("vizTip"); if(!el) return;
  if(!show){ el.style.display = "none"; return; }
  el.innerHTML = html; el.style.display = "block";
  el.style.left = Math.min(window.innerWidth - el.offsetWidth - 12, x + 14) + "px";
  el.style.top = (y - 10) + "px";
}
function svgEsc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;"); }

/* Grafik garis + area, satu seri. data: [{label, value}] */
function lineChart(host, data, o){
  o = o || {};
  const P = vizPal(), W = 620, H = o.height || 190;
  const m = { t:12, r:14, b:24, l:48 };
  const el = document.getElementById(host);
  if(!data.length){ el.innerHTML = "<p class='muted small'>" + t("no_data") + "</p>"; return; }
  const vals = data.map(d => d.value);
  const max = Math.max.apply(null, vals) || 1, min = o.zero === false ? Math.min.apply(null, vals) : 0;
  const span = (max - min) || 1;
  const X = i => m.l + (i / Math.max(1, data.length - 1)) * (W - m.l - m.r);
  const Y = v => H - m.b - ((v - min) / span) * (H - m.t - m.b);
  const color = o.color === "s2" ? P.s2 : P.s1;

  let grid = "", ticks = 4;
  for(let i = 0; i <= ticks; i++){
    const v = min + (span * i / ticks), y = Y(v);
    grid += "<line x1='" + m.l + "' y1='" + y + "' x2='" + (W - m.r) + "' y2='" + y +
      "' stroke='" + P.grid + "' stroke-width='1'/>" +
      "<text x='" + (m.l - 7) + "' y='" + (y + 3.5) + "' text-anchor='end' font-size='10' fill='" +
      P.mut + "'>" + (o.fmt ? o.fmt(v) : Math.round(v)) + "</text>";
  }
  const step = Math.max(1, Math.floor(data.length / 6));
  for(let i = 0; i < data.length; i += step){
    grid += "<text x='" + X(i) + "' y='" + (H - 7) + "' text-anchor='middle' font-size='10' fill='" +
      P.mut + "'>" + svgEsc(data[i].label) + "</text>";
  }
  const line = data.map((d, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(d.value).toFixed(1)).join(" ");
  const area = line + " L" + X(data.length - 1).toFixed(1) + " " + Y(min) + " L" + X(0) + " " + Y(min) + " Z";
  const gid = "g_" + host;

  el.innerHTML =
    "<svg viewBox='0 0 " + W + " " + H + "' width='100%' style='display:block' id='svg_" + host + "'>" +
    "<defs><linearGradient id='" + gid + "' x1='0' y1='0' x2='0' y2='1'>" +
    "<stop offset='0%' stop-color='" + color + "' stop-opacity='.28'/>" +
    "<stop offset='100%' stop-color='" + color + "' stop-opacity='.02'/></linearGradient></defs>" +
    grid + "<path d='" + area + "' fill='url(#" + gid + ")'/>" +
    "<path d='" + line + "' fill='none' stroke='" + color + "' stroke-width='2' " +
    "stroke-linejoin='round' stroke-linecap='round'/>" +
    "<line id='ch_" + host + "' y1='" + m.t + "' y2='" + (H - m.b) + "' stroke='" + P.mut +
    "' stroke-width='1' stroke-dasharray='3 3' style='display:none'/>" +
    "<circle id='cd_" + host + "' r='4.5' fill='" + color + "' stroke='" + P.surf +
    "' stroke-width='2' style='display:none'/>" +
    "<rect x='" + m.l + "' y='0' width='" + (W - m.l - m.r) + "' height='" + H +
    "' fill='transparent' id='hit_" + host + "'/></svg>";

  const svg = document.getElementById("svg_" + host);
  const hit = document.getElementById("hit_" + host);
  const ch = document.getElementById("ch_" + host), cd = document.getElementById("cd_" + host);
  hit.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect();
    const sx = (e.clientX - r.left) / r.width * W;
    let i = Math.round((sx - m.l) / ((W - m.l - m.r) / Math.max(1, data.length - 1)));
    i = Math.max(0, Math.min(data.length - 1, i));
    ch.setAttribute("x1", X(i)); ch.setAttribute("x2", X(i)); ch.style.display = "";
    cd.setAttribute("cx", X(i)); cd.setAttribute("cy", Y(data[i].value)); cd.style.display = "";
    tip(true, e.clientX, e.clientY, "<b>" + svgEsc(data[i].label) + "</b><br>" +
      (o.fmt ? o.fmt(data[i].value) : Math.round(data[i].value)) + " " + (o.unit || ""));
  });
  hit.addEventListener("pointerleave", () => {
    ch.style.display = "none"; cd.style.display = "none"; tip(false);
  });
}

/* Batang horizontal, satu ukuran. data: [{label, value}] */
function barChart(host, data, o){
  o = o || {};
  const P = vizPal(), W = 620, rowH = 26, m = { t:6, r:70, b:6, l:150 };
  const el = document.getElementById(host);
  data = data.filter(d => d.value > 0).sort((a, b) => b.value - a.value).slice(0, o.top || 10);
  if(!data.length){ el.innerHTML = "<p class='muted small'>" + t("no_data") + "</p>"; return; }
  const H = m.t + m.b + data.length * rowH;
  const max = Math.max.apply(null, data.map(d => d.value)) || 1;
  const bw = W - m.l - m.r;
  const color = o.color === "s2" ? P.s2 : P.s1;

  const rows = data.map((d, i) => {
    const y = m.t + i * rowH + 4, h = rowH - 10;
    const w = Math.max(2, (d.value / max) * bw);
    return "<g class='bar' data-i='" + i + "'>" +
      "<text x='" + (m.l - 9) + "' y='" + (y + h - 3) + "' text-anchor='end' font-size='11' fill='" +
      P.ink + "'>" + svgEsc(d.label.length > 22 ? d.label.slice(0, 21) + "…" : d.label) + "</text>" +
      "<rect x='" + m.l + "' y='" + y + "' width='" + bw + "' height='" + h +
      "' rx='4' fill='" + P.grid + "' opacity='.45'/>" +
      "<rect x='" + m.l + "' y='" + y + "' width='" + w + "' height='" + h +
      "' rx='4' fill='" + color + "'/>" +
      "<text x='" + (m.l + w + 8) + "' y='" + (y + h - 3) + "' font-size='11' fill='" + P.mut +
      "'>" + (o.fmt ? o.fmt(d.value) : Math.round(d.value)) + " " + (o.unit || "") + "</text>" +
      "<rect x='0' y='" + (y - 3) + "' width='" + W + "' height='" + rowH +
      "' fill='transparent' data-hit='" + i + "'/></g>";
  }).join("");

  el.innerHTML = "<svg viewBox='0 0 " + W + " " + H + "' width='100%' style='display:block'>" + rows + "</svg>";
  el.querySelectorAll("[data-hit]").forEach(r => {
    const d = data[+r.dataset.hit];
    r.addEventListener("pointermove", e => tip(true, e.clientX, e.clientY,
      "<b>" + svgEsc(d.label) + "</b><br>" + (o.fmt ? o.fmt(d.value) : Math.round(d.value)) + " " + (o.unit || "")));
    r.addEventListener("pointerleave", () => tip(false));
  });
}
