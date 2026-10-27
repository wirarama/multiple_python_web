/* Pembantu umum: pemanggilan API, notifikasi, format angka. */
async function api(method, url, body){
  const opt = { method, headers: {"Content-Type":"application/json"} };
  if(body !== undefined) opt.body = JSON.stringify(body);
  const res = await fetch(url, opt);
  let data = null;
  try { data = await res.json(); } catch(e){ data = null; }
  if(!res.ok){ toast((data && data.error) || ("HTTP "+res.status)); throw new Error(res.status); }
  return data;
}
const apiGet = u => api("GET", u);
const apiPost = (u,b) => api("POST", u, b || {});
const apiPut = (u,b) => api("PUT", u, b || {});
const apiDel = u => api("DELETE", u);

function toast(msg){
  const box = document.getElementById("toast"); if(!box) return;
  const el = document.createElement("div"); el.textContent = msg; box.appendChild(el);
  setTimeout(()=>{ el.style.opacity = 0; setTimeout(()=>el.remove(), 400); }, 2600);
}
function num(v, d){ const n = Number(v); return isNaN(n) ? (d||0) : n; }
function fmt(v, dec){ return Number(v||0).toLocaleString(LANG==="id"?"id-ID":"en-US",
  {minimumFractionDigits:dec||0, maximumFractionDigits:dec||0}); }
function rupiah(v){ return "Rp " + fmt(v, 0); }
function esc(s){ return String(s==null?"":s).replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function el(tag, attrs, html){
  const e = document.createElement(tag);
  if(attrs) for(const k in attrs){ if(k==="class") e.className=attrs[k]; else e.setAttribute(k, attrs[k]); }
  if(html!==undefined) e.innerHTML = html;
  return e;
}
function opt(value, label, sel){
  return '<option value="'+esc(value)+'"'+(String(sel)===String(value)?' selected':'')+'>'+esc(label)+'</option>';
}

/* --------------------------------------------------- tema & bahasa global */
function initChrome(){
  const saved = localStorage.getItem("sh_theme") || "light";
  document.documentElement.setAttribute("data-theme", saved);
  const tb = document.getElementById("themeBtn");
  if(tb) tb.onclick = () => {
    const now = document.documentElement.getAttribute("data-theme")==="dark" ? "light":"dark";
    document.documentElement.setAttribute("data-theme", now);
    localStorage.setItem("sh_theme", now);
    document.dispatchEvent(new CustomEvent("themechange"));
  };
  const lb = document.getElementById("langBtn");
  if(lb) lb.onclick = () => setLang(LANG==="id" ? "en" : "id");
  applyI18n();
}
function headerEnv(env){
  const c = document.getElementById("clockBadge");
  if(c) c.textContent = env.clock + "  D" + (env.day||1);
  const d = document.getElementById("todBadge");
  if(d) d.textContent = t(env.time_of_day);
  const w = document.getElementById("wxBadge");
  if(w) w.textContent = t(env.weather) + "  " + Math.round(env.outdoor_temp) + "°C";
}
document.addEventListener("DOMContentLoaded", initChrome);
