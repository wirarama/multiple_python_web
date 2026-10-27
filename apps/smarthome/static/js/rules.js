/* Halaman Rule: pembangun syarat (IF) dan aksi (THEN). */
let ST = null, cur = null;

const SUBJECTS = [
  { k:"time_of_day",   ref:null,     vt:"tod"   },
  { k:"hour",          ref:null,     vt:"num"   },
  { k:"weather",       ref:null,     vt:"wx"    },
  { k:"outdoor_temp",  ref:null,     vt:"num"   },
  { k:"outdoor_lux",   ref:null,     vt:"num"   },
  { k:"room_temp",     ref:"room",   vt:"num"   },
  { k:"room_lux",      ref:"room",   vt:"num"   },
  { k:"room_humidity", ref:"room",   vt:"num"   },
  { k:"occupancy",     ref:"room",   vt:"num"   },
  { k:"occupant_total",ref:null,     vt:"num"   },
  { k:"sensor",        ref:"switch", vt:"bool"  },
  { k:"device",        ref:"device", vt:"bool"  },
  { k:"total_power",   ref:null,     vt:"num"   },
  { k:"energy_kwh",    ref:null,     vt:"num"   },
];
const OPS = ["==", "!=", ">", ">=", "<", "<="];
const TODS = ["pagi","siang","sore","malam"];
const WXS = ["cerah","berawan","hujan","badai","kabut"];

document.addEventListener("DOMContentLoaded", async () => {
  await refresh();
  document.getElementById("btnNew").onclick = async () => {
    const r = await apiPost("/api/rules", { name: t("rule_new"), logic:"AND",
      conditions:[{subject:"time_of_day", ref:null, op:"==", value:"malam"}],
      actions:[] });
    await refresh(); edit(r.id);
  };
  document.addEventListener("langchange", () => { paint(); if(cur) edit(cur.id); });
});

async function refresh(){ ST = await apiGet("/api/state"); paint(); }

function paint(){
  headerEnv(ST.env);
  document.getElementById("ruleList").innerHTML = ST.rules.map(r =>
    "<div class='item'><input type='checkbox' data-en='" + r.id + "' " + (r.enabled ? "checked" : "") +
    " style='width:auto'><span class='nm' data-open='" + r.id + "' style='cursor:pointer'>" +
    esc(r.name) + "<div class='sub'>" + (r.conditions || []).length + " syarat · " +
    (r.actions || []).length + " aksi · P" + r.priority +
    (r.last_fired ? " · " + t("fired") + " " + r.last_fired : "") + "</div></span>" +
    "<span class='badge'>" + r.fire_count + "×</span></div>").join("") ||
    "<div class='item muted small'>-</div>";
  document.querySelectorAll("[data-open]").forEach(e => e.onclick = () => edit(+e.dataset.open));
  document.querySelectorAll("[data-en]").forEach(e => e.onchange = async () => {
    await apiPut("/api/rules/" + e.dataset.en, { enabled: e.checked }); await refresh();
  });
  document.getElementById("logBox").innerHTML = ST.logs.map(l =>
    "<div class='logline " + l.level + "'><span class='t'>" + esc(l.sim_time) + "</span><span>" +
    esc(l.message) + "</span></div>").join("");
}

function refOptions(kind, sel){
  if(kind === "room") return ST.rooms.map(r => opt(r.id, r.name, sel)).join("");
  const list = ST.components.filter(c => c.kind === (kind === "switch" ? "switch" : "device"));
  return list.map(c => opt(c.id, c.name, sel)).join("");
}
function valueField(vt, v, idx, tag){
  const id = tag + "_v" + idx;
  if(vt === "tod") return "<select id='" + id + "'>" + TODS.map(x => opt(x, t(x), v)).join("") + "</select>";
  if(vt === "wx")  return "<select id='" + id + "'>" + WXS.map(x => opt(x, t(x), v)).join("") + "</select>";
  if(vt === "bool")return "<select id='" + id + "'>" + opt("true", "ON", v) + opt("false", "OFF", v) + "</select>";
  return "<input type='number' step='0.1' id='" + id + "' value='" + (v != null ? v : 0) + "'>";
}

function edit(id){
  cur = ST.rules.find(r => r.id === id);
  if(!cur) return;
  document.getElementById("edTitle").textContent = cur.name;
  const box = document.getElementById("edBox");

  const conds = (cur.conditions || []).map((c, i) => condRow(c, i)).join("");
  const acts = (cur.actions || []).map((a, i) => actRow(a, i)).join("");

  box.innerHTML =
    "<div class='grid3'>" +
    "<label class='f'><span>" + t("rule_name") + "</span><input id='r_name' value='" + esc(cur.name) + "'></label>" +
    "<label class='f'><span>" + t("logic") + "</span><select id='r_logic'>" +
    opt("AND", "AND (semua)", cur.logic) + opt("OR", "OR (salah satu)", cur.logic) + "</select></label>" +
    "<label class='f'><span>" + t("priority") + "</span><input type='number' id='r_prio' value='" + cur.priority + "'></label>" +
    "</div>" +
    "<h4 style='margin:10px 0 6px'>" + t("conditions") + "</h4><div id='condBox'>" + conds + "</div>" +
    "<button class='sm' id='addCond'>+ " + t("add_cond") + "</button>" +
    "<h4 style='margin:14px 0 6px'>" + t("actions") + "</h4><div id='actBox'>" + acts + "</div>" +
    "<button class='sm' id='addAct'>+ " + t("add_act") + "</button>" +
    "<div class='row' style='margin-top:16px'>" +
    "<button class='primary' id='r_save'>" + t("save") + "</button>" +
    "<button id='r_test'>" + t("test_rule") + "</button>" +
    "<span class='spacer'></span><button class='danger' id='r_del'>" + t("del") + "</button></div>" +
    "<div id='testOut' class='small' style='margin-top:10px'></div>";

  bindCond(); bindAct();
  document.getElementById("addCond").onclick = () => {
    cur.conditions = readConds(); cur.conditions.push({ subject:"hour", ref:null, op:">", value:18 });
    edit(id);
  };
  document.getElementById("addAct").onclick = () => {
    cur.actions = readActs(); cur.actions.push({ target:"device", ref:null, act:"on" });
    edit(id);
  };
  document.getElementById("r_save").onclick = async () => {
    await apiPut("/api/rules/" + id, { name: document.getElementById("r_name").value,
      logic: document.getElementById("r_logic").value,
      priority: +document.getElementById("r_prio").value,
      conditions: readConds(), actions: readActs() });
    toast(t("saved")); await refresh(); edit(id);
  };
  document.getElementById("r_del").onclick = async () => {
    if(!confirm(t("confirm_del"))) return;
    await apiDel("/api/rules/" + id); cur = null; toast(t("deleted"));
    document.getElementById("edBox").innerHTML = "<p class='muted small'>" + t("pick_rule") + "</p>";
    document.getElementById("edTitle").textContent = "—";
    await refresh();
  };
  document.getElementById("r_test").onclick = async () => {
    await apiPut("/api/rules/" + id, { conditions: readConds(), actions: readActs(),
      logic: document.getElementById("r_logic").value });
    const res = await apiPost("/api/rules/" + id + "/test");
    document.getElementById("testOut").innerHTML =
      "<div class='badge " + (res.match ? "ok" : "") + "'>" +
      (res.match ? t("rule_match") : t("rule_nomatch")) + "</div><div style='margin-top:6px'>" +
      res.detail.map(d => "<div class='logline'><span class='t'>" +
        (d.pass ? "✔" : "✘") + "</span><span>" + t("s_" + d.cond.subject) + " = <b>" +
        esc(String(d.value)) + "</b> " + esc(d.cond.op) + " " + esc(String(d.cond.value)) +
        "</span></div>").join("") + "</div>";
    await refresh(); 
  };
}

function condRow(c, i){
  const s = SUBJECTS.find(x => x.k === c.subject) || SUBJECTS[0];
  return "<div class='row' data-cond='" + i + "' style='margin-bottom:6px'>" +
    "<select id='c_s" + i + "' style='width:190px'>" +
    SUBJECTS.map(x => opt(x.k, t("s_" + x.k), c.subject)).join("") + "</select>" +
    (s.ref ? "<select id='c_r" + i + "' style='width:170px'>" + refOptions(s.ref, c.ref) + "</select>" : "") +
    "<select id='c_o" + i + "' style='width:72px'>" + OPS.map(o => opt(o, o, c.op)).join("") + "</select>" +
    "<span style='width:150px'>" + valueField(s.vt, c.value, i, "c") + "</span>" +
    "<button class='sm danger' data-rmc='" + i + "'>✕</button></div>";
}
function actRow(a, i){
  const targets = [["device", t("device")], ["switch", t("switch")],
                   ["circuit", t("circuit")], ["all_devices", t("all_devices")]];
  const refKind = a.target === "switch" ? "switch" : a.target === "circuit" ? "circuit" : "device";
  let refSel = "";
  if(a.target === "circuit") refSel = "<select id='a_r" + i + "' style='width:190px'>" +
    ST.circuits.map(c => opt(c.id, c.name, a.ref)).join("") + "</select>";
  else if(a.target !== "all_devices") refSel = "<select id='a_r" + i + "' style='width:210px'>" +
    refOptions(refKind, a.ref) + "</select>";
  return "<div class='row' data-act='" + i + "' style='margin-bottom:6px'>" +
    "<select id='a_t" + i + "' style='width:150px'>" +
    targets.map(x => opt(x[0], x[1], a.target)).join("") + "</select>" + refSel +
    "<select id='a_a" + i + "' style='width:130px'>" +
    opt("on", t("act_on"), a.act) + opt("off", t("act_off"), a.act) +
    opt("toggle", t("act_toggle"), a.act) + "</select>" +
    "<button class='sm danger' data-rma='" + i + "'>✕</button></div>";
}
function bindCond(){
  document.querySelectorAll("[data-cond]").forEach(row => {
    const i = +row.dataset.cond;
    document.getElementById("c_s" + i).onchange = () => { cur.conditions = readConds(); edit(cur.id); };
    row.querySelector("[data-rmc]").onclick = () => {
      cur.conditions = readConds(); cur.conditions.splice(i, 1); edit(cur.id);
    };
  });
}
function bindAct(){
  document.querySelectorAll("[data-act]").forEach(row => {
    const i = +row.dataset.act;
    document.getElementById("a_t" + i).onchange = () => { cur.actions = readActs(); edit(cur.id); };
    row.querySelector("[data-rma]").onclick = () => {
      cur.actions = readActs(); cur.actions.splice(i, 1); edit(cur.id);
    };
  });
}
function readConds(){
  const out = [];
  document.querySelectorAll("[data-cond]").forEach(row => {
    const i = +row.dataset.cond;
    const subj = document.getElementById("c_s" + i).value;
    const s = SUBJECTS.find(x => x.k === subj);
    const refEl = document.getElementById("c_r" + i);
    const vEl = document.getElementById("c_v" + i);
    let v = vEl ? vEl.value : null;
    if(s.vt === "num") v = parseFloat(v);
    if(s.vt === "bool") v = (v === "true");
    out.push({ subject: subj, ref: refEl ? +refEl.value : null,
               op: document.getElementById("c_o" + i).value, value: v });
  });
  return out;
}
function readActs(){
  const out = [];
  document.querySelectorAll("[data-act]").forEach(row => {
    const i = +row.dataset.act;
    const tg = document.getElementById("a_t" + i).value;
    const refEl = document.getElementById("a_r" + i);
    out.push({ target: tg, ref: refEl ? +refEl.value : null,
               act: document.getElementById("a_a" + i).value });
  });
  return out;
}
