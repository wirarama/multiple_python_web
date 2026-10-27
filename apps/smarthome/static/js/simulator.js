/* Halaman Simulator: denah, kontrol lingkungan, properti, ringkasan listrik. */
let ST = null, CAT = null, timer = null, running = false;

const OPS = ["==", "!=", ">", ">=", "<", "<="];
const TOD = ["pagi", "siang", "sore", "malam"];

document.addEventListener("DOMContentLoaded", async () => {
  Plan.init(document.getElementById("plan"));
  Plan.onSelect = (type, id) => renderProps(type, id);
  Plan.onChange = async (type, id, patch) => {
    const url = { room:"/api/rooms/", component:"/api/components/", occupant:"/api/occupants/" }[type];
    await apiPut(url + id, patch);
    await refresh();
  };
  Plan.onToggle = async (id) => {
    const c = ST.components.find(x => x.id === id);
    if(!c) return;
    apply(await apiPost("/api/components/" + id + "/toggle"));
  };
  Plan.onPlaced = async (x, y, room) => {
    const p = Plan.pending; if(!p) return;
    await apiPost("/api/components", Object.assign({}, p,
      { x: Math.round(x), y: Math.round(y), room_id: room ? room.id : null }));
    Plan.pending = null; setMode("view");
    document.getElementById("placeHint").style.display = "none";
    toast(t("added")); await refresh();
  };

  CAT = await apiGet("/api/catalog");
  buildEnvControls();
  bindControls();
  await refresh();
  document.addEventListener("langchange", () => { buildEnvControls(); paint(); });
});

// ---------------------------------------------------------------- data
async function refresh(){ apply(await apiGet("/api/state")); }
function apply(state){ ST = state; paint(); }

function paint(){
  if(!ST) return;
  headerEnv(ST.env);
  Plan.render(ST);
  renderEnv(); renderSummary(); renderCircuits(); renderSources();
  renderOccupants(); renderLogs(); renderLegend();
  if(Plan.sel.type) renderProps(Plan.sel.type, Plan.sel.id);
}

// --------------------------------------------------------- kontrol umum
function buildEnvControls(){
  const chips = document.getElementById("todChips");
  chips.innerHTML = TOD.map(k => '<span class="chip" data-tod="' + k + '">' + t(k) + "</span>").join("");
  chips.querySelectorAll("[data-tod]").forEach(c => c.onclick = async () => {
    apply(await apiPost("/api/sim/env", { time_of_day: c.dataset.tod }));
  });
  const ws = document.getElementById("weatherSel");
  ws.innerHTML = CAT.weathers.map(w => opt(w, t(w), ST ? ST.env.weather : "")).join("");
}

function bindControls(){
  const $ = id => document.getElementById(id);

  $("btnRun").onclick = () => setRunning(!running);
  $("btnStep").onclick = async () => apply((await apiPost("/api/sim/tick", { steps:1 })).state);

  $("stepMin").oninput = e => $("stepVal").textContent = e.target.value;
  $("stepMin").onchange = async e => apply(await apiPost("/api/sim/env",
    { step_minutes: +e.target.value }));
  $("speed").oninput = e => { $("spdVal").textContent = (+e.target.value / 4).toFixed(1);
    if(running) setRunning(true); };

  $("hourRange").oninput = e => {
    const m = +e.target.value;
    $("clockVal").textContent = String(Math.floor(m/60)).padStart(2,"0") + ":" +
      String(m%60).padStart(2,"0");
  };
  $("hourRange").onchange = async e => apply(await apiPost("/api/sim/env", { hour: +e.target.value / 60 }));
  $("weatherSel").onchange = async e => apply(await apiPost("/api/sim/env", { weather: e.target.value }));
  $("tempOff").oninput = e => $("offVal").textContent = e.target.value;
  $("tempOff").onchange = async e => apply(await apiPost("/api/sim/env",
    { outdoor_temp_offset: +e.target.value }));
  $("autoWx").onchange = async e => apply(await apiPost("/api/sim/env", { auto_weather: e.target.checked }));

  $("btnResetTrip").onclick = async () => apply(await apiPost("/api/sim/reset-trip"));
  $("btnResetEnergy").onclick = async () => apply(await apiPost("/api/sim/reset-energy"));
  $("btnDemo").onclick = async () => { if(confirm("Muat ulang data contoh? Data saat ini hilang."))
    apply(await apiPost("/api/sim/reset-demo")); };
  $("btnClear").onclick = async () => { if(confirm("Kosongkan seluruh isi rumah?"))
    apply(await apiPost("/api/sim/clear")); };
  $("btnClearLog").onclick = async () => { await apiDel("/api/logs"); await refresh(); };

  document.querySelectorAll("#modeSeg button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
  ["optWire","optCtrl","optLabel"].forEach((id, i) => {
    $(id).onchange = e => { Plan.opts[["wire","ctrl","label"][i]] = e.target.checked; Plan.render(ST); };
  });

  $("btnTidy").onclick = async () => { apply(await apiPost("/api/layout")); toast(t("saved")); };
  $("btnExport").onclick = () => { window.location.href = "/api/export"; toast(t("export_ok")); };
  $("btnImport").onclick = () => $("fileImport").click();
  $("fileImport").onchange = async (ev) => {
    const f = ev.target.files && ev.target.files[0];
    ev.target.value = "";
    if(!f) return;
    if(!confirm(t("import_confirm"))) return;
    let doc;
    try { doc = JSON.parse(await f.text()); }
    catch(e){ return toast(t("import_bad")); }
    const res = await apiPost("/api/import", doc);
    (res.notes || []).forEach(n => toast(n));
    toast(t("import_ok"));
    apply(res.state);
  };
  $("btnAddRoom").onclick = dlgAddRoom;
  $("btnAddComp").onclick = dlgAddComponent;
  $("btnAddCirc").onclick = dlgAddCircuit;
  $("btnAddOcc").onclick = dlgAddOccupant;
  $("btnCancelPlace").onclick = () => { Plan.pending = null; setMode("view");
    $("placeHint").style.display = "none"; };

  $("btnSaveHouse").onclick = async () => {
    await apiPut("/api/house", { voltage:+$("hVolt").value, power_factor:+$("hPf").value,
      main_mcb_a:+$("hMcb").value, tariff:+$("hTariff").value });
    toast(t("saved")); await refresh();
  };
}

function setMode(m){
  Plan.mode = m;
  document.querySelectorAll("#modeSeg button").forEach(b =>
    b.classList.toggle("on", b.dataset.mode === m));
  Plan.render(ST);
}
function setRunning(on){
  running = on;
  document.getElementById("runLbl").textContent = on ? t("pause") : t("play");
  document.getElementById("btnRun").classList.toggle("primary", !on);
  document.getElementById("btnRun").classList.toggle("danger", on);
  if(timer) clearInterval(timer);
  if(on){
    const spd = +document.getElementById("speed").value;
    timer = setInterval(async () => {
      try { apply((await apiPost("/api/sim/tick", { steps:1 })).state); }
      catch(e){ setRunning(false); }
    }, Math.max(250, 2000 / spd));
  }
}

// ------------------------------------------------------------- panel kiri
function renderEnv(){
  const e = ST.env, h = ST.house;
  document.getElementById("hourRange").value = h.sim_minute % 1440;
  document.getElementById("clockVal").textContent = e.clock;
  document.getElementById("weatherSel").value = e.weather;
  document.getElementById("tempOff").value = h.outdoor_temp_offset || 0;
  document.getElementById("offVal").textContent = h.outdoor_temp_offset || 0;
  document.getElementById("autoWx").checked = !!h.auto_weather;
  document.getElementById("stepMin").value = h.step_minutes;
  document.getElementById("stepVal").textContent = h.step_minutes;
  document.getElementById("oTemp").textContent = e.outdoor_temp.toFixed(1);
  document.getElementById("oLux").textContent = fmt(e.outdoor_lux);
  document.querySelectorAll("#todChips .chip").forEach(c =>
    c.classList.toggle("on", c.dataset.tod === e.time_of_day));
}

function renderOccupants(){
  const box = document.getElementById("occList");
  const rooms = ST.rooms;
  box.innerHTML = ST.occupants.map(o => {
    const rn = o.away ? t("away") : ((rooms.find(r => r.id === o.room_id) || {}).name || "-");
    return '<div class="item"><span class="dot" style="background:' + o.color + '"></span>' +
      '<span class="nm">' + esc(o.name) + '<div class="sub">' + esc(rn) + ' · ' + esc(o.activity) + '</div></span>' +
      '<button class="sm" data-occ="' + o.id + '">…</button></div>';
  }).join("") || '<div class="item muted small">-</div>';
  box.querySelectorAll("[data-occ]").forEach(b => b.onclick = () => {
    Plan.select("person", +b.dataset.occ);
  });
}

function renderSources(){
  const box = document.getElementById("srcList");
  const src = ST.components.filter(c => c.kind === "source");
  box.innerHTML = src.map(s => {
    const load = ST.circuits.filter(c => c.source_id === s.id && !c.tripped)
      .reduce((a, c) => a + c.load_w, 0);
    return '<div class="item">' + iconSvg(s.subtype, 17) +
      '<span class="nm">' + esc(s.name) + '<div class="sub">' + fmt(load,0) + ' / ' +
      fmt(s.capacity_w,0) + ' W</div></span>' +
      '<button class="sm ' + (s.enabled ? "ok" : "") + '" data-src="' + s.id + '">' +
      (s.enabled ? "ON" : "OFF") + '</button></div>';
  }).join("") || '<div class="item muted small">-</div>';
  box.querySelectorAll("[data-src]").forEach(b => b.onclick = async () => {
    const s = ST.components.find(c => c.id === +b.dataset.src);
    await apiPut("/api/components/" + s.id, { enabled: !s.enabled });
    apply(await apiPost("/api/sim/recompute"));
  });
}

// ------------------------------------------------------------ panel kanan
function renderSummary(){
  const T = ST.totals, h = ST.house;
  document.getElementById("mPower").textContent = fmt(T.power_w, 0);
  document.getElementById("mCurr").textContent = T.current_a.toFixed(2);
  document.getElementById("mKwh").textContent = T.energy_kwh.toFixed(3);
  document.getElementById("mCost").textContent = rupiah(T.cost);
  document.getElementById("hVolt").value = h.voltage;
  document.getElementById("hPf").value = h.power_factor;
  document.getElementById("hMcb").value = h.main_mcb_a;
  document.getElementById("hTariff").value = h.tariff;
  const b = document.getElementById("mcbBadge");
  const anyTrip = h.main_tripped || ST.circuits.some(c => c.tripped);
  b.className = "badge " + (anyTrip ? "danger" : "ok");
  b.textContent = (h.main_tripped ? "MAIN " : "") + (anyTrip ? t("tripped") : t("normal")) +
    " · " + T.devices_on + "/" + T.devices_total;
}

function renderCircuits(){
  const box = document.getElementById("circList");
  box.innerHTML = ST.circuits.map(c => {
    const pct = Math.min(100, (c.current_a / (c.mcb_a || 1)) * 100);
    return '<div class="item"><span class="dot" style="background:' + c.color + '"></span>' +
      '<span class="nm">' + esc(c.name) +
      '<div class="sub">' + fmt(c.load_w,0) + ' W · ' + c.current_a.toFixed(2) + ' / ' + c.mcb_a + ' A' +
      '<div style="height:4px;background:var(--line);border-radius:3px;margin-top:3px">' +
      '<div style="height:4px;width:' + pct + '%;border-radius:3px;background:' +
      (pct > 90 ? "var(--danger)" : pct > 70 ? "var(--warn)" : c.color) + '"></div></div></div></span>' +
      (c.tripped ? '<span class="badge danger">' + t("tripped") + '</span>' : '') +
      '<button class="sm" data-circ="' + c.id + '">…</button></div>';
  }).join("") || '<div class="item muted small">-</div>';
  box.querySelectorAll("[data-circ]").forEach(b => b.onclick = () => renderProps("circuit", +b.dataset.circ));
}

function renderLegend(){
  document.getElementById("legendCirc").innerHTML = ST.circuits.map(c =>
    '<span style="margin-right:10px"><i style="background:' + c.color + '"></i>' + esc(c.name) + '</span>').join("");
}

function renderLogs(){
  document.getElementById("logBox").innerHTML = ST.logs.map(l =>
    '<div class="logline ' + l.level + '"><span class="t">' + esc(l.sim_time) + '</span>' +
    '<span>' + esc(l.message) + '</span></div>').join("") ||
    '<div class="logline muted">-</div>';
}

// ------------------------------------------------------------- properti
function roomOptions(sel){
  return opt("", "—", sel) + ST.rooms.map(r => opt(r.id, r.name, sel)).join("");
}
function circOptions(sel){
  return opt("", "—", sel) + ST.circuits.map(c => opt(c.id, c.name, sel)).join("");
}

function renderProps(type, id){
  Plan.sel = { type, id };
  const box = document.getElementById("propBox");
  if(!type){ box.innerHTML = '<p class="muted small">' + t("nothing_sel") + '</p>'; return; }

  if(type === "room"){
    const r = ST.rooms.find(x => x.id === id); if(!r) return;
    box.innerHTML =
      '<label class="f"><span>' + t("name") + '</span><input id="p_name" value="' + esc(r.name) + '"></label>' +
      '<div class="grid2">' +
      '<label class="f"><span>X</span><input type="number" id="p_x" value="' + r.x + '"></label>' +
      '<label class="f"><span>Y</span><input type="number" id="p_y" value="' + r.y + '"></label>' +
      '<label class="f"><span>W</span><input type="number" id="p_w" value="' + r.w + '"></label>' +
      '<label class="f"><span>H</span><input type="number" id="p_h" value="' + r.h + '"></label>' +
      '<label class="f"><span>' + t("color") + '</span><input type="text" id="p_color" value="' + esc(r.color) + '"></label>' +
      '<label class="f"><span>' + t("window") + '</span><input type="number" step="0.05" id="p_wf" value="' + r.window_factor + '"></label>' +
      '</div>' +
      '<div class="kv small"><b>' + t("temp") + '</b><span>' + r.temp_c.toFixed(1) + ' °C</span>' +
      '<b>' + t("lux") + '</b><span>' + fmt(r.lux) + ' lx</span>' +
      '<b>' + t("humidity") + '</b><span>' + r.humidity.toFixed(0) + ' %</span></div>' +
      propButtons();
    bindSave(async () => apiPut("/api/rooms/" + id, {
      name: val("p_name"), x:+val("p_x"), y:+val("p_y"), w:+val("p_w"), h:+val("p_h"),
      color: val("p_color"), window_factor: +val("p_wf") }),
      () => apiDel("/api/rooms/" + id));
    return;
  }

  if(type === "comp"){
    const c = ST.components.find(x => x.id === id); if(!c) return;
    const list = c.kind === "source" ? CAT.source_types
               : c.kind === "switch" ? CAT.switch_types : CAT.device_types;
    let extra = "";
    if(c.kind === "device"){
      extra =
        '<div class="grid2">' +
        '<label class="f"><span>' + t("power_w") + '</span><input type="number" id="p_pw" value="' + c.power_w + '"></label>' +
        '<label class="f"><span>' + t("standby_w") + '</span><input type="number" step="0.1" id="p_sb" value="' + c.standby_w + '"></label></div>' +
        '<label class="f"><span>' + t("combine") + '</span><select id="p_comb">' +
        opt("parallel", t("parallel"), c.combine) + opt("series", t("series"), c.combine) + '</select></label>';
      if(c.subtype === "ac"){
        extra += '<label class="f"><span>' + t("setpoint") + '</span><input type="number" id="p_sp" value="' +
          ((c.params && c.params.setpoint) || 24) + '"></label>';
      }
      const ctrl = ST.links.filter(l => l.device_id === c.id);
      extra += '<div class="small muted" style="margin:6px 0 3px">' + t("controlled_by") + ':</div>' +
        (ctrl.length ? ctrl.map(l => {
          const s = ST.components.find(x => x.id === l.switch_id) || {};
          return '<span class="badge ' + (s.state ? "on" : "") + '">' + esc(s.name || "?") +
            (l.inverted ? " (NC)" : "") + '</span> ';
        }).join("") : '<span class="small muted">' + t("no_link") + '</span>');
    }
    if(c.kind === "switch"){
      const p = c.params || {};
      if(["ldr","suhu","kelembaban"].includes(c.subtype)){
        extra += '<div class="grid2">' +
          '<label class="f"><span>' + t("operator") + '</span><select id="p_op">' +
          OPS.map(o => opt(o, o, p.op || ">")).join("") + '</select></label>' +
          '<label class="f"><span>' + t("threshold") + '</span><input type="number" step="0.5" id="p_thr" value="' +
          (p.threshold != null ? p.threshold : 0) + '"></label></div>';
      }
      if(c.subtype === "timer"){
        extra += '<div class="grid2">' +
          '<label class="f"><span>' + t("on_hour") + '</span><input type="number" id="p_onh" value="' + (p.on_hour != null ? p.on_hour : 18) + '"></label>' +
          '<label class="f"><span>' + t("off_hour") + '</span><input type="number" id="p_offh" value="' + (p.off_hour != null ? p.off_hour : 6) + '"></label></div>';
      }
      if(c.subtype === "tombol"){
        extra += '<label class="row small"><input type="checkbox" id="p_mom" style="width:auto"' +
          (p.momentary ? " checked" : "") + '> ' + t("momentary") + '</label>';
      }
      const ctrl = ST.links.filter(l => l.switch_id === c.id);
      extra += '<div class="small muted" style="margin:6px 0 3px">' + t("controls") + ':</div>' +
        (ctrl.length ? ctrl.map(l => {
          const d = ST.components.find(x => x.id === l.device_id) || {};
          return '<span class="badge ' + (d.state ? "on" : "") + '">' + esc(d.name || "?") + '</span> ';
        }).join("") : '<span class="small muted">' + t("no_link") + '</span>');
    }
    if(c.kind === "source"){
      extra = '<label class="f"><span>' + t("capacity") + ' (W)</span><input type="number" id="p_cap" value="' +
        c.capacity_w + '"></label>';
    }

    box.innerHTML =
      '<div class="row" style="margin-bottom:8px">' + iconSvg(c.subtype, 20) +
      '<b>' + esc(c.name) + '</b><span class="spacer"></span>' +
      '<span class="badge ' + (c.state ? "on" : "off") + '">' + (c.state ? t("on") : t("offs")) + '</span></div>' +
      '<label class="f"><span>' + t("name") + '</span><input id="p_name" value="' + esc(c.name) + '"></label>' +
      '<div class="grid2">' +
      '<label class="f"><span>' + t("type") + '</span><select id="p_sub">' +
      list.map(s => opt(s, tt(s), c.subtype)).join("") + '</select></label>' +
      '<label class="f"><span>' + t("room") + '</span><select id="p_room">' + roomOptions(c.room_id) + '</select></label>' +
      '<label class="f"><span>' + t("circuit") + '</span><select id="p_circ">' + circOptions(c.circuit_id) + '</select></label>' +
      '<label class="f"><span>' + t("mode") + '</span><select id="p_mode">' +
      opt("auto", t("auto"), c.mode) + opt("manual", t("manual"), c.mode) + opt("off", t("off"), c.mode) +
      '</select></label></div>' + extra +
      '<div class="row" style="margin-top:10px">' +
      '<button class="sm ' + (c.manual_state ? "ok" : "") + '" id="p_toggle">' + t("toggle") + '</button>' +
      '<label class="row small"><input type="checkbox" id="p_en" style="width:auto"' +
      (c.enabled ? " checked" : "") + '> ' + t("enabled") + '</label></div>' +
      propButtons();

    document.getElementById("p_toggle").onclick = async () =>
      apply(await apiPost("/api/components/" + id + "/toggle"));

    bindSave(async () => {
      const params = Object.assign({}, c.params || {});
      if(exists("p_op")) params.op = val("p_op");
      if(exists("p_thr")) params.threshold = +val("p_thr");
      if(exists("p_onh")) params.on_hour = +val("p_onh");
      if(exists("p_offh")) params.off_hour = +val("p_offh");
      if(exists("p_sp")) params.setpoint = +val("p_sp");
      if(exists("p_mom")) params.momentary = document.getElementById("p_mom").checked;
      const body = { name: val("p_name"), subtype: val("p_sub"),
        room_id: val("p_room") ? +val("p_room") : null,
        circuit_id: val("p_circ") ? +val("p_circ") : null,
        mode: val("p_mode"), enabled: document.getElementById("p_en").checked, params };
      if(exists("p_pw")) { body.power_w = +val("p_pw"); body.standby_w = +val("p_sb");
        body.combine = val("p_comb"); }
      if(exists("p_cap")) body.capacity_w = +val("p_cap");
      return apiPut("/api/components/" + id, body);
    }, () => apiDel("/api/components/" + id));
    return;
  }

  if(type === "circuit"){
    const c = ST.circuits.find(x => x.id === id); if(!c) return;
    const srcs = ST.components.filter(x => x.kind === "source");
    box.innerHTML =
      '<label class="f"><span>' + t("name") + '</span><input id="p_name" value="' + esc(c.name) + '"></label>' +
      '<div class="grid2">' +
      '<label class="f"><span>MCB (A)</span><input type="number" step="0.5" id="p_mcb" value="' + c.mcb_a + '"></label>' +
      '<label class="f"><span>' + t("color") + '</span><input id="p_color" value="' + esc(c.color) + '"></label></div>' +
      '<label class="f"><span>' + t("source") + '</span><select id="p_src">' +
      srcs.map(s => opt(s.id, s.name, c.source_id)).join("") + '</select></label>' +
      '<div class="kv small"><b>' + t("load") + '</b><span>' + fmt(c.load_w,0) + ' W · ' +
      c.current_a.toFixed(2) + ' A</span></div>' +
      '<label class="row small"><input type="checkbox" id="p_trip" style="width:auto"' +
      (c.tripped ? " checked" : "") + '> ' + t("tripped") + '</label>' + propButtons();
    bindSave(() => apiPut("/api/circuits/" + id, { name: val("p_name"), mcb_a: +val("p_mcb"),
      color: val("p_color"), source_id: +val("p_src"),
      tripped: document.getElementById("p_trip").checked }),
      () => apiDel("/api/circuits/" + id));
    return;
  }

  if(type === "person"){
    const o = ST.occupants.find(x => x.id === id); if(!o) return;
    let sched = '<div class="grid3" style="max-height:190px;overflow:auto">';
    for(let hh = 0; hh < 24; hh++){
      const v = (o.schedule || {})[String(hh)];
      sched += '<label class="f" style="margin:0"><span>' + String(hh).padStart(2,"0") + ':00</span>' +
        '<select data-sched="' + hh + '">' + opt("", "—", v) + opt("away", t("away"), v) +
        ST.rooms.map(r => opt(r.id, r.name, v)).join("") + '</select></label>';
    }
    sched += "</div>";
    box.innerHTML =
      '<label class="f"><span>' + t("name") + '</span><input id="p_name" value="' + esc(o.name) + '"></label>' +
      '<div class="grid2">' +
      '<label class="f"><span>' + t("room") + '</span><select id="p_room">' + roomOptions(o.room_id) + '</select></label>' +
      '<label class="f"><span>' + t("activity") + '</span><input id="p_act" value="' + esc(o.activity) + '"></label>' +
      '<label class="f"><span>' + t("color") + '</span><input id="p_color" value="' + esc(o.color) + '"></label></div>' +
      '<div class="row small"><label class="row"><input type="checkbox" id="p_away" style="width:auto"' +
      (o.away ? " checked" : "") + '> ' + t("away") + '</label>' +
      '<label class="row"><input type="checkbox" id="p_auto" style="width:auto"' +
      (o.auto_move ? " checked" : "") + '> ' + t("schedule") + '</label></div>' +
      '<div class="small muted" style="margin:8px 0 4px">' + t("schedule") + '</div>' + sched +
      propButtons();
    bindSave(() => {
      const sc = {};
      box.querySelectorAll("[data-sched]").forEach(s => { if(s.value) sc[s.dataset.sched] = s.value; });
      return apiPut("/api/occupants/" + id, { name: val("p_name"),
        room_id: val("p_room") ? +val("p_room") : null, activity: val("p_act"),
        color: val("p_color"), away: document.getElementById("p_away").checked,
        auto_move: document.getElementById("p_auto").checked, schedule: sc });
    }, () => apiDel("/api/occupants/" + id));
  }
}

function propButtons(){
  return '<div class="row" style="margin-top:12px">' +
    '<button class="primary sm" id="p_save">' + t("save") + '</button>' +
    '<span class="spacer"></span>' +
    '<button class="danger sm" id="p_del">' + t("del") + '</button></div>';
}
function val(id){ const e = document.getElementById(id); return e ? e.value : ""; }
function exists(id){ return !!document.getElementById(id); }
function bindSave(saveFn, delFn){
  document.getElementById("p_save").onclick = async () => {
    await saveFn(); toast(t("saved")); apply(await apiPost("/api/sim/recompute"));
  };
  document.getElementById("p_del").onclick = async () => {
    if(!confirm(t("confirm_del"))) return;
    await delFn(); Plan.sel = { type:null, id:null }; toast(t("deleted"));
    renderProps(null); await refresh();
  };
}

// ------------------------------------------------------------------ dialog
function openDlg(title, bodyHtml, okLabel, onOk){
  const d = document.getElementById("dlg");
  document.getElementById("dlgTitle").textContent = title;
  document.getElementById("dlgBody").innerHTML = bodyHtml;
  document.getElementById("dlgOk").textContent = okLabel || t("add");
  d.returnValue = "";
  d.onclose = async () => { if(d.returnValue === "ok") await onOk(); };
  d.showModal();
  return d;
}

function dlgAddRoom(){
  openDlg(t("add_room"),
    '<div class="grid2">' +
    '<label class="f"><span>' + t("name") + '</span><input id="d_name" value="Kamar Baru"></label>' +
    '<label class="f"><span>' + t("color") + '</span><input id="d_color" value="#e0f2fe"></label>' +
    '<label class="f"><span>W</span><input type="number" id="d_w" value="220"></label>' +
    '<label class="f"><span>H</span><input type="number" id="d_h" value="180"></label>' +
    '<label class="f"><span>' + t("window") + '</span><input type="number" step="0.05" id="d_wf" value="0.25"></label>' +
    '</div>', t("add"), async () => {
      await apiPost("/api/rooms", { name: val("d_name"), color: val("d_color"),
        w:+val("d_w"), h:+val("d_h"), window_factor:+val("d_wf"),
        x: 30 + (ST.rooms.length * 25) % 300, y: 30 + (ST.rooms.length * 25) % 300 });
      toast(t("added")); await refresh();
    });
}

function dlgAddCircuit(){
  const srcs = ST.components.filter(c => c.kind === "source");
  openDlg(t("circuits"),
    '<label class="f"><span>' + t("name") + '</span><input id="d_name" value="Grup Baru"></label>' +
    '<div class="grid2">' +
    '<label class="f"><span>MCB (A)</span><input type="number" step="0.5" id="d_mcb" value="6"></label>' +
    '<label class="f"><span>' + t("color") + '</span><input id="d_color" value="#8b5cf6"></label></div>' +
    '<label class="f"><span>' + t("source") + '</span><select id="d_src">' +
    srcs.map(s => opt(s.id, s.name)).join("") + '</select></label>', t("add"), async () => {
      await apiPost("/api/circuits", { name: val("d_name"), mcb_a:+val("d_mcb"),
        color: val("d_color"), source_id: +val("d_src") });
      toast(t("added")); await refresh();
    });
}

function dlgAddOccupant(){
  openDlg(t("occupants"),
    '<div class="grid2">' +
    '<label class="f"><span>' + t("name") + '</span><input id="d_name" value="Penghuni"></label>' +
    '<label class="f"><span>' + t("activity") + '</span><input id="d_act" value="santai"></label>' +
    '<label class="f"><span>' + t("room") + '</span><select id="d_room">' + roomOptions() + '</select></label>' +
    '<label class="f"><span>' + t("color") + '</span><input id="d_color" value="#0ea5e9"></label></div>',
    t("add"), async () => {
      await apiPost("/api/occupants", { name: val("d_name"), activity: val("d_act"),
        room_id: val("d_room") ? +val("d_room") : null, color: val("d_color"), auto_move:false });
      toast(t("added")); await refresh();
    });
}

function dlgAddComponent(){
  const kindSel = '<div class="seg" id="d_kind">' +
    '<button type="button" data-k="device" class="on">' + t("comp_devices") + '</button>' +
    '<button type="button" data-k="switch">' + t("comp_switches") + '</button>' +
    '<button type="button" data-k="source">' + t("comp_sources") + '</button></div>';
  const d = openDlg(t("add_comp"),
    kindSel + '<div id="d_types" class="chips" style="margin:12px 0"></div>' +
    '<div class="grid2">' +
    '<label class="f"><span>' + t("name") + '</span><input id="d_name"></label>' +
    '<label class="f"><span>' + t("power_w") + '</span><input type="number" id="d_pw" value="0"></label>' +
    '<label class="f"><span>' + t("circuit") + '</span><select id="d_circ">' + circOptions() + '</select></label>' +
    '<label class="f"><span>' + t("mode") + '</span><select id="d_mode">' +
    opt("auto", t("auto")) + opt("manual", t("manual")) + '</select></label></div>' +
    '<p class="small muted">' + t("click_place") + '</p>',
    t("add"), async () => {
      if(!picked.sub) return;
      Plan.pending = { kind: picked.kind, subtype: picked.sub,
        name: val("d_name") || tt(picked.sub), power_w: +val("d_pw"),
        circuit_id: val("d_circ") ? +val("d_circ") : null, mode: val("d_mode") };
      document.getElementById("placeName").textContent = Plan.pending.name;
      document.getElementById("placeHint").style.display = "flex";
      setMode("place");
    });

  const picked = { kind:"device", sub:null };
  function fillTypes(){
    const list = picked.kind === "source" ? CAT.source_types
               : picked.kind === "switch" ? CAT.switch_types : CAT.device_types;
    picked.sub = list[0];
    document.getElementById("d_types").innerHTML = list.map((s, i) =>
      '<span class="chip' + (i === 0 ? " on" : "") + '" data-sub="' + s + '">' +
      iconSvg(s, 15) + tt(s) + "</span>").join("");
    document.getElementById("d_types").querySelectorAll("[data-sub]").forEach(c => c.onclick = () => {
      picked.sub = c.dataset.sub;
      document.getElementById("d_types").querySelectorAll(".chip").forEach(x => x.classList.remove("on"));
      c.classList.add("on");
      document.getElementById("d_name").value = tt(picked.sub);
      document.getElementById("d_pw").value = CAT.default_power[picked.sub] || 0;
    });
    document.getElementById("d_name").value = tt(picked.sub);
    document.getElementById("d_pw").value = CAT.default_power[picked.sub] || 0;
  }
  d.querySelectorAll("#d_kind button").forEach(b => b.onclick = () => {
    picked.kind = b.dataset.k;
    d.querySelectorAll("#d_kind button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); fillTypes();
  });
  fillTypes();
}
