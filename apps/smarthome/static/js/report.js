/* Halaman Laporan: grafik daya/energi, konsumsi per kamar & grup, tabel perangkat. */
let ST = null;

document.addEventListener("DOMContentLoaded", async () => {
  await load();
  setInterval(load, 5000);
  document.addEventListener("themechange", draw);
  document.addEventListener("langchange", draw);
  document.getElementById("csvPower").onclick = exportCsv;
});

async function load(){ ST = await apiGet("/api/state"); draw(); }

function draw(){
  if(!ST) return;
  headerEnv(ST.env);
  const s = ST.samples || [];

  lineChart("chPower", s.map(x => ({ label:x.sim_time, value:x.total_w })),
    { unit:"W", color:"s1", fmt:v => fmt(v, 0) });
  lineChart("chEnergy", s.map(x => ({ label:x.sim_time, value:x.kwh })),
    { unit:"kWh", color:"s2", fmt:v => v.toFixed(2) });

  const perRoom = ST.rooms.map(r => ({
    label: r.name,
    value: ST.components.filter(c => c.kind === "device" && c.room_id === r.id && c.state)
      .reduce((a, c) => a + c.power_w, 0) }));
  barChart("chRoom", perRoom, { unit:"W", fmt:v => fmt(v, 0) });

  barChart("chCirc", ST.circuits.map(c => ({ label:c.name, value:c.load_w })),
    { unit:"W", color:"s2", fmt:v => fmt(v, 0) });

  const dev = ST.components.filter(c => c.kind === "device")
    .sort((a, b) => (b.state ? b.power_w : 0) - (a.state ? a.power_w : 0));
  document.getElementById("tblDev").innerHTML =
    "<thead><tr><th></th><th>" + t("name") + "</th><th>" + t("room") + "</th><th>" +
    t("circuit") + "</th><th>" + t("power_w") + "</th><th>" + t("state") + "</th></tr></thead><tbody>" +
    dev.map(c => {
      const rm = ST.rooms.find(r => r.id === c.room_id);
      const ci = ST.circuits.find(x => x.id === c.circuit_id);
      return "<tr><td>" + iconSvg(c.subtype, 15) + "</td><td>" + esc(c.name) + "</td><td>" +
        esc(rm ? rm.name : "—") + "</td><td>" + esc(ci ? ci.name : "—") + "</td>" +
        "<td class='mono'>" + fmt(c.power_w, 0) + "</td><td><span class='badge " +
        (c.state ? "on" : "off") + "'>" + (c.state ? "ON" : "OFF") + "</span></td></tr>";
    }).join("") + "</tbody>";

  const T = ST.totals;
  document.getElementById("mets").innerHTML =
    met(t("power"), fmt(T.power_w, 0), "W") + met(t("current"), T.current_a.toFixed(2), "A") +
    met(t("energy"), T.energy_kwh.toFixed(3), "kWh") + met(t("cost"), rupiah(T.cost), "") +
    met(t("devices_on"), T.devices_on + " / " + T.devices_total, "") +
    met(t("main_mcb"), ST.house.main_mcb_a + " A", ST.house.main_tripped ? t("tripped") : t("normal"));
}
function met(k, v, u){
  return "<div class='metric'><div class='k'>" + esc(k) + "</div><div class='v' style='font-size:16px'>" +
    esc(v) + " <span class='u'>" + esc(u) + "</span></div></div>";
}

function exportCsv(){
  const rows = [["sim_time","minute","total_w","current_a","kwh"]]
    .concat((ST.samples || []).map(s => [s.sim_time, s.minute, s.total_w, s.current_a, s.kwh]));
  const csv = rows.map(r => r.join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type:"text/csv" }));
  a.download = "simulasi_energi.csv"; a.click();
}
