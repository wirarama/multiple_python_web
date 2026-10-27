/* Halaman Komponen: tabel perangkat/sensor/sumber + pengkabelan. */
let ST = null, CAT = null;

document.addEventListener("DOMContentLoaded", async () => {
  CAT = await apiGet("/api/catalog");
  await refresh();
  document.getElementById("wDev").onchange = renderWiring;
  document.getElementById("wComb").onchange = async e => {
    const id = +document.getElementById("wDev").value;
    await apiPut("/api/components/" + id, { combine: e.target.value });
    toast(t("saved")); await refresh();
  };
  document.getElementById("wAdd").onclick = async () => {
    const dev = +document.getElementById("wDev").value;
    const sw = +document.getElementById("wSw").value;
    if(!dev || !sw) return;
    await apiPost("/api/links", { device_id: dev, switch_id: sw,
      inverted: document.getElementById("wInv").checked });
    toast(t("added")); await refresh();
  };
  document.addEventListener("langchange", paint);
});

async function refresh(){ ST = await apiGet("/api/state"); paint(); }

function roomName(id){ const r = ST.rooms.find(x => x.id === id); return r ? r.name : "—"; }
function circName(id){ const c = ST.circuits.find(x => x.id === id); return c ? c.name : "—"; }
function selRoom(v){ return opt("", "—", v) + ST.rooms.map(r => opt(r.id, r.name, v)).join(""); }
function selCirc(v){ return opt("", "—", v) + ST.circuits.map(c => opt(c.id, c.name, v)).join(""); }

function paint(){
  headerEnv(ST.env);
  const dev = ST.components.filter(c => c.kind === "device");
  const sw = ST.components.filter(c => c.kind === "switch");
  const src = ST.components.filter(c => c.kind === "source");
  document.getElementById("cntDev").textContent = dev.length;
  document.getElementById("cntSw").textContent = sw.length;

  // ------------------------------------------------------------ perangkat
  document.getElementById("tblDev").innerHTML =
    "<thead><tr><th></th><th>" + t("name") + "</th><th>" + t("type") + "</th><th>" + t("room") +
    "</th><th>" + t("circuit") + "</th><th>" + t("power_w") + "</th><th>" + t("mode") +
    "</th><th>" + t("combine") + "</th><th>" + t("state") + "</th><th></th></tr></thead><tbody>" +
    dev.map(c =>
      "<tr data-id='" + c.id + "'><td>" + iconSvg(c.subtype, 16) + "</td>" +
      "<td><input value='" + esc(c.name) + "' data-f='name'></td>" +
      "<td><select data-f='subtype'>" + CAT.device_types.map(s => opt(s, tt(s), c.subtype)).join("") + "</select></td>" +
      "<td><select data-f='room_id'>" + selRoom(c.room_id) + "</select></td>" +
      "<td><select data-f='circuit_id'>" + selCirc(c.circuit_id) + "</select></td>" +
      "<td><input type='number' style='width:80px' value='" + c.power_w + "' data-f='power_w'></td>" +
      "<td><select data-f='mode'>" + opt("auto", t("auto"), c.mode) + opt("manual", t("manual"), c.mode) +
      opt("off", t("off"), c.mode) + "</select></td>" +
      "<td><select data-f='combine'>" + opt("parallel", t("parallel"), c.combine) +
      opt("series", t("series"), c.combine) + "</select></td>" +
      "<td><span class='badge " + (c.state ? "on" : "off") + "'>" + (c.state ? "ON" : "OFF") + "</span></td>" +
      "<td class='act'><button class='sm' data-act='toggle'>⏻</button>" +
      "<button class='sm' data-act='save'>✓</button>" +
      "<button class='sm danger' data-act='del'>✕</button></td></tr>").join("") + "</tbody>";

  // --------------------------------------------------------- saklar/sensor
  document.getElementById("tblSw").innerHTML =
    "<thead><tr><th></th><th>" + t("name") + "</th><th>" + t("type") + "</th><th>" + t("room") +
    "</th><th>" + t("operator") + "</th><th>" + t("threshold") + "</th><th>" + t("mode") +
    "</th><th>" + t("state") + "</th><th></th></tr></thead><tbody>" +
    sw.map(c => {
      const p = c.params || {};
      const thr = ["ldr","suhu","kelembaban"].includes(c.subtype);
      const tim = c.subtype === "timer";
      return "<tr data-id='" + c.id + "'><td>" + iconSvg(c.subtype, 16) + "</td>" +
      "<td><input value='" + esc(c.name) + "' data-f='name'></td>" +
      "<td><select data-f='subtype'>" + CAT.switch_types.map(s => opt(s, tt(s), c.subtype)).join("") + "</select></td>" +
      "<td><select data-f='room_id'>" + selRoom(c.room_id) + "</select></td>" +
      "<td>" + (thr ? "<select data-p='op'>" + ["<","<=",">",">=","==","!="].map(o => opt(o, o, p.op)).join("") + "</select>" : "—") + "</td>" +
      "<td>" + (thr ? "<input type='number' step='0.5' style='width:80px' data-p='threshold' value='" + (p.threshold != null ? p.threshold : 0) + "'>"
                    : tim ? "<input type='number' style='width:46px' data-p='on_hour' value='" + (p.on_hour != null ? p.on_hour : 18) + "'>" +
                            "<input type='number' style='width:46px' data-p='off_hour' value='" + (p.off_hour != null ? p.off_hour : 6) + "'>"
                    : "—") + "</td>" +
      "<td><select data-f='mode'>" + opt("auto", t("auto"), c.mode) + opt("manual", t("manual"), c.mode) +
      opt("off", t("off"), c.mode) + "</select></td>" +
      "<td><span class='badge " + (c.state ? "on" : "off") + "'>" + (c.state ? "ON" : "OFF") + "</span></td>" +
      "<td class='act'><button class='sm' data-act='toggle'>⏻</button>" +
      "<button class='sm' data-act='save'>✓</button>" +
      "<button class='sm danger' data-act='del'>✕</button></td></tr>";
    }).join("") + "</tbody>";

  // ------------------------------------------------------------- sumber
  document.getElementById("tblSrc").innerHTML =
    "<thead><tr><th></th><th>" + t("name") + "</th><th>" + t("type") + "</th><th>" + t("capacity") +
    " (W)</th><th>" + t("enabled") + "</th><th></th></tr></thead><tbody>" +
    src.map(c =>
      "<tr data-id='" + c.id + "'><td>" + iconSvg(c.subtype, 16) + "</td>" +
      "<td><input value='" + esc(c.name) + "' data-f='name'></td>" +
      "<td><select data-f='subtype'>" + CAT.source_types.map(s => opt(s, tt(s), c.subtype)).join("") + "</select></td>" +
      "<td><input type='number' style='width:100px' value='" + c.capacity_w + "' data-f='capacity_w'></td>" +
      "<td><input type='checkbox' data-f='enabled' " + (c.enabled ? "checked" : "") + " style='width:auto'></td>" +
      "<td class='act'><button class='sm' data-act='save'>✓</button>" +
      "<button class='sm danger' data-act='del'>✕</button></td></tr>").join("") + "</tbody>";

  // ------------------------------------------------------------ sirkuit
  document.getElementById("tblCirc").innerHTML =
    "<thead><tr><th>" + t("name") + "</th><th>MCB (A)</th><th>" + t("source") + "</th><th>" +
    t("load") + "</th><th>" + t("state") + "</th></tr></thead><tbody>" +
    ST.circuits.map(c =>
      "<tr><td>" + esc(c.name) + "</td><td class='mono'>" + c.mcb_a + "</td><td>" +
      esc((ST.components.find(x => x.id === c.source_id) || {}).name || "—") + "</td>" +
      "<td class='mono'>" + fmt(c.load_w, 0) + " W / " + c.current_a.toFixed(2) + " A</td>" +
      "<td>" + (c.tripped ? "<span class='badge danger'>" + t("tripped") + "</span>"
                          : "<span class='badge ok'>" + t("normal") + "</span>") + "</td></tr>").join("") + "</tbody>";

  // -------------------------------------------------------------- kamar
  document.getElementById("tblRoom").innerHTML =
    "<thead><tr><th>" + t("name") + "</th><th>" + t("temp") + "</th><th>" + t("lux") +
    "</th><th>" + t("humidity") + "</th></tr></thead><tbody>" +
    ST.rooms.map(r => "<tr><td>" + esc(r.name) + "</td><td class='mono'>" + r.temp_c.toFixed(1) +
      " °C</td><td class='mono'>" + fmt(r.lux, 0) + "</td><td class='mono'>" +
      r.humidity.toFixed(0) + " %</td></tr>").join("") + "</tbody>";

  bindRows();
  fillWiringSelects();
  renderWiring();
}

function bindRows(){
  document.querySelectorAll("tr[data-id]").forEach(tr => {
    const id = +tr.dataset.id;
    tr.querySelectorAll("[data-act]").forEach(b => b.onclick = async () => {
      if(b.dataset.act === "del"){
        if(!confirm(t("confirm_del"))) return;
        await apiDel("/api/components/" + id); toast(t("deleted")); return refresh();
      }
      if(b.dataset.act === "toggle"){
        await apiPost("/api/components/" + id + "/toggle"); return refresh();
      }
      const body = {}, params = {};
      tr.querySelectorAll("[data-f]").forEach(i => {
        const f = i.dataset.f;
        let v = i.type === "checkbox" ? i.checked : i.value;
        if(["room_id","circuit_id"].includes(f)) v = v ? +v : null;
        if(["power_w","capacity_w"].includes(f)) v = +v;
        body[f] = v;
      });
      tr.querySelectorAll("[data-p]").forEach(i => {
        params[i.dataset.p] = i.type === "number" ? +i.value : i.value;
      });
      const cur = ST.components.find(c => c.id === id) || {};
      if(Object.keys(params).length) body.params = Object.assign({}, cur.params || {}, params);
      await apiPut("/api/components/" + id, body);
      toast(t("saved")); await refresh();
    });
  });
}

function fillWiringSelects(){
  const dsel = document.getElementById("wDev");
  const keep = dsel.value;
  const devs = ST.components.filter(c => c.kind === "device");
  dsel.innerHTML = devs.map(c => opt(c.id, c.name + " (" + roomName(c.room_id) + ")", keep)).join("");
  if(keep && devs.some(c => String(c.id) === keep)) dsel.value = keep;
  document.getElementById("wSw").innerHTML = ST.components.filter(c => c.kind === "switch")
    .map(c => opt(c.id, c.name + " — " + tt(c.subtype))).join("");
  document.getElementById("wComb").innerHTML =
    opt("parallel", t("parallel")) + opt("series", t("series"));
}

function renderWiring(){
  const id = +document.getElementById("wDev").value;
  const dev = ST.components.find(c => c.id === id);
  const box = document.getElementById("wList");
  if(!dev){ box.innerHTML = ""; return; }
  document.getElementById("wComb").value = dev.combine;
  const links = ST.links.filter(l => l.device_id === id);
  box.innerHTML = links.length ? links.map(l => {
    const s = ST.components.find(c => c.id === l.switch_id) || {};
    return "<div class='item' style='border:1px solid var(--line);border-radius:9px;margin-bottom:6px;padding:6px 9px'>" +
      iconSvg(s.subtype, 16) + "<span class='nm'>" + esc(s.name) +
      "<div class='sub'>" + tt(s.subtype) + " · " + (s.state ? "ON" : "OFF") + "</div></span>" +
      "<label class='row small'><input type='checkbox' data-inv='" + l.id + "' style='width:auto'" +
      (l.inverted ? " checked" : "") + "> NC</label>" +
      "<button class='sm danger' data-rm='" + l.id + "'>✕</button></div>";
  }).join("") : "<p class='small muted'>" + t("no_link") + "</p>";
  box.querySelectorAll("[data-rm]").forEach(b => b.onclick = async () => {
    await apiDel("/api/links/" + b.dataset.rm); toast(t("deleted")); await refresh();
  });
  box.querySelectorAll("[data-inv]").forEach(b => b.onchange = async () => {
    await apiPut("/api/links/" + b.dataset.inv, { inverted: b.checked }); await refresh();
  });
}
