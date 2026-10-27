/* Halaman Studi Kasus: daftar kasus contoh, pemuatan, serta ekspor/impor JSON. */
let CASES = [];

document.addEventListener("DOMContentLoaded", async () => {
  await load();
  document.getElementById("btnExport").onclick = doExport;
  document.getElementById("btnImport").onclick = () =>
    document.getElementById("fileImport").click();
  document.getElementById("fileImport").onchange = doImport;
  document.addEventListener("langchange", paint);
  try { headerEnv((await apiGet("/api/state")).env); } catch (e) {}
});

async function load(){
  CASES = await apiGet("/api/scenarios");
  paint();
}

function list(title, items, cls){
  if(!items || !items.length) return "";
  return "<div style='margin-top:10px'><div class='small' style='font-weight:700;color:var(--muted);" +
    "text-transform:uppercase;letter-spacing:.04em;font-size:11px;margin-bottom:4px'>" + esc(title) + "</div>" +
    "<ul class='" + (cls || "") + "' style='margin:0;padding-left:18px'>" +
    items.map(i => "<li class='small' style='margin-bottom:3px'>" + esc(i) + "</li>").join("") +
    "</ul></div>";
}

function paint(){
  document.getElementById("caseCount").textContent =
    CASES.length + " " + t("cases_available");
  document.getElementById("caseGrid").innerHTML = CASES.map(c => {
    const k = c.case || {};
    return "<div class='card'>" +
      "<h3><span class='badge ok'>" + esc(k.kode || "") + "</span>" +
      "<span style='margin-left:8px;text-transform:none;font-size:13.5px;color:var(--ink)'>" +
      esc(k.judul || c.title) + "</span></h3>" +
      "<div class='body'>" +
      "<div class='chips' style='margin-bottom:9px'>" +
      "<span class='chip'>" + t("case_time") + ": " + esc(k.waktu || "-") + "</span>" +
      "<span class='chip'>" + esc(t(k.periode || "") || k.periode || "") + "</span>" +
      "<span class='chip'>" + t("weather") + ": " + esc(t(k.cuaca || "") || k.cuaca || "") + "</span>" +
      "<span class='chip'>" + c.counts.components + " " + t("components_short") + "</span>" +
      "</div>" +
      "<p class='small' style='margin:0'>" + esc(k.ringkasan || c.description) + "</p>" +
      "<div class='small' style='margin-top:9px'><b>" + t("case_source") + ":</b> " +
      esc(k.sumber || "-") + "</div>" +
      list(t("case_goals"), k.tujuan) +
      list(t("case_steps"), k.langkah) +
      list(t("case_questions"), k.pertanyaan) +
      list(t("case_expected"), k.hasil) +
      list(t("case_note"), k.catatan) +
      "<div class='row' style='margin-top:14px'>" +
      "<button class='primary sm' data-load='" + esc(c.slug) + "'>" + t("case_load") + "</button>" +
      "<button class='sm' data-dl='" + esc(c.slug) + "'>" + t("case_download") + "</button>" +
      "</div></div></div>";
  }).join("") || "<p class='muted small'>" + t("no_data") + "</p>";

  document.querySelectorAll("[data-load]").forEach(b => b.onclick = async () => {
    if(!confirm(t("case_confirm"))) return;
    b.disabled = true;
    try {
      const res = await apiPost("/api/scenarios/" + b.dataset.load);
      (res.notes || []).forEach(n => toast(n));
      toast(t("case_loaded"));
      headerEnv(res.state.env);
      setTimeout(() => { window.location.href = "/"; }, 700);
    } finally { b.disabled = false; }
  });
  document.querySelectorAll("[data-dl]").forEach(b => b.onclick = () => {
    window.location.href = "/api/scenarios/" + b.dataset.dl + "/file";
  });
}

function doExport(){
  window.location.href = "/api/export";
  toast(t("export_ok"));
}

async function doImport(ev){
  const f = ev.target.files && ev.target.files[0];
  ev.target.value = "";
  if(!f) return;
  if(!confirm(t("import_confirm"))) return;
  let doc;
  try { doc = JSON.parse(await f.text()); }
  catch(e){ return toast(t("import_bad")); }
  try {
    const res = await apiPost("/api/import", doc);
    (res.notes || []).forEach(n => toast(n));
    toast(t("import_ok"));
    headerEnv(res.state.env);
    setTimeout(() => { window.location.href = "/"; }, 700);
  } catch(e){ /* pesan galat sudah ditampilkan oleh api() */ }
}
