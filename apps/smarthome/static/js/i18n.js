/* Dwibahasa: Indonesia (id) / English (en) */
const DICT = {
  id: {
    app: "Simulator Smart Home", app_sub: "Denah rumah, instalasi listrik & otomasi",
    nav_sim: "Simulator", nav_comp: "Komponen", nav_rule: "Rule", nav_rep: "Laporan",
    nav_case: "Studi Kasus", nav_help: "Panduan",
    transfer: "Ekspor & Impor Konfigurasi",
    transfer_help: "Seluruh isi rumah — kamar, komponen, grup instalasi, pengkabelan, aturan, dan penghuni — dapat disimpan ke satu berkas JSON dan dimuat kembali kapan saja.",
    export_json: "Ekspor JSON", import_json: "Impor JSON",
    export_ok: "Berkas konfigurasi sedang diunduh",
    import_ok: "Konfigurasi berhasil diimpor",
    import_bad: "Berkas bukan JSON yang sah",
    import_confirm: "Impor akan mengganti seluruh isi rumah saat ini. Lanjutkan?",
    case_confirm: "Memuat kasus ini akan mengganti seluruh isi rumah saat ini. Lanjutkan?",
    case_loaded: "Kasus berhasil dimuat",
    cases_available: "kasus tersedia", components_short: "komponen",
    case_time: "Waktu", case_source: "Sumber listrik", case_goals: "Tujuan pembelajaran",
    case_steps: "Langkah pengamatan", case_questions: "Pertanyaan pemandu",
    case_expected: "Hasil yang diharapkan", case_note: "Catatan",
    case_load: "Muat Kasus", case_download: "Unduh JSON",
    lang: "Bahasa", theme: "Tema",

    env: "Kondisi Lingkungan", sim: "Kontrol Simulasi", time: "Waktu", weather: "Cuaca",
    tod: "Periode", pagi: "Pagi", siang: "Siang", sore: "Sore", malam: "Malam",
    cerah: "Cerah", berawan: "Berawan", hujan: "Hujan", badai: "Badai", kabut: "Kabut",
    outdoor: "Luar rumah", temp: "Suhu", lux: "Cahaya", humidity: "Kelembaban",
    temp_offset: "Koreksi suhu (°C)", auto_weather: "Cuaca acak tiap hari",
    play: "Jalan", pause: "Jeda", step: "Maju 1 langkah", step_min: "Menit / langkah",
    speed: "Kecepatan", reset_energy: "Reset energi", reset_trip: "Reset MCB",
    reset_demo: "Muat data contoh", clear_house: "Kosongkan rumah", day: "Hari",

    plan: "Denah Rumah", mode_view: "Mode Lihat", mode_edit: "Edit Denah", mode_place: "Tempel Komponen",
    show_wire: "Kabel daya", show_ctrl: "Jalur kendali", show_label: "Nama komponen",
    add_room: "Tambah Kamar", add_comp: "Tambah Komponen", tidy: "Rapikan", legend_on: "Menyala",
    legend_off: "Mati", legend_person: "Penghuni", legend_trip: "Grup trip",
    click_place: "Klik pada denah untuk menempatkan:",

    summary: "Ringkasan Listrik", power: "Daya", current: "Arus", energy: "Energi", cost: "Biaya",
    devices_on: "Perangkat aktif", main_mcb: "MCB utama", voltage: "Tegangan", pf: "Faktor daya",
    tariff: "Tarif (Rp/kWh)", tripped: "TRIP", normal: "Normal",
    circuits: "Grup Instalasi (MCB)", load: "Beban", source: "Sumber", capacity: "Kapasitas",
    sources: "Sumber Listrik", occupants: "Penghuni", rooms: "Kamar", log: "Catatan Kejadian",
    clear_log: "Bersihkan",

    pick_rule: "Pilih rule di daftar kiri, atau tekan Rule Baru untuk membuat aturan.",
    properties: "Properti", nothing_sel: "Belum ada objek dipilih. Klik kamar, komponen, atau penghuni pada denah.",
    name: "Nama", type: "Jenis", room: "Kamar", circuit: "Grup instalasi", pos: "Posisi",
    size: "Ukuran", color: "Warna", window: "Faktor jendela", power_w: "Daya (W)",
    standby_w: "Daya standby (W)", mode: "Mode", auto: "Otomatis", manual: "Manual", off: "Nonaktif",
    combine: "Gabungan saklar", parallel: "Paralel (salah satu aktif)", series: "Seri (semua harus aktif)",
    state: "Status", on: "ON", offs: "OFF", enabled: "Diaktifkan", threshold: "Ambang",
    operator: "Operator", setpoint: "Set point (°C)", on_hour: "Jam nyala", off_hour: "Jam mati",
    momentary: "Tombol sesaat", save: "Simpan", del: "Hapus", cancel: "Batal", add: "Tambah",
    toggle: "Ubah status", confirm_del: "Hapus objek ini?",

    comp_sources: "Sumber Listrik", comp_switches: "Saklar & Sensor", comp_devices: "Perangkat Elektronik",
    wiring: "Pengkabelan Saklar → Perangkat", controlled_by: "Dikendalikan oleh", controls: "Mengendalikan",
    inverted: "Logika terbalik (NC)", link_add: "Hubungkan", no_link: "Belum ada hubungan",
    switch: "Saklar/Sensor", device: "Perangkat",

    rules: "Aturan Otomasi (Rule)", rule_new: "Rule Baru", rule_name: "Nama rule",
    logic: "Logika syarat", priority: "Prioritas", conditions: "Syarat (IF)", actions: "Aksi (THEN)",
    add_cond: "Tambah syarat", add_act: "Tambah aksi", subject: "Objek", value: "Nilai",
    target: "Sasaran", action: "Aksi", test_rule: "Uji sekarang", fired: "Terakhir aktif",
    times: "kali", rule_match: "Syarat TERPENUHI", rule_nomatch: "Syarat belum terpenuhi",
    all_devices: "Semua perangkat", act_on: "Nyalakan", act_off: "Matikan", act_toggle: "Balik status",

    s_time_of_day: "Periode waktu", s_hour: "Jam (0-24)", s_weather: "Cuaca",
    s_outdoor_temp: "Suhu luar (°C)", s_outdoor_lux: "Cahaya luar (lux)",
    s_room_temp: "Suhu kamar (°C)", s_room_lux: "Cahaya kamar (lux)",
    s_room_humidity: "Kelembaban kamar (%)", s_occupancy: "Jumlah orang di kamar",
    s_occupant_total: "Total orang di rumah", s_sensor: "Status sensor/saklar",
    s_device: "Status perangkat", s_total_power: "Total daya (W)", s_energy_kwh: "Energi (kWh)",

    report: "Laporan & Analisis", chart_power: "Grafik Daya (W)", chart_energy: "Grafik Energi (kWh)",
    per_room: "Konsumsi per Kamar", per_circuit: "Konsumsi per Grup", per_device: "Perangkat Teraktif",
    export_csv: "Unduh CSV", no_data: "Belum ada data. Jalankan simulasi terlebih dahulu.",
    activity: "Aktivitas", away: "Di luar rumah", schedule: "Jadwal per jam", present: "Di rumah",
    saved: "Tersimpan", deleted: "Dihapus", added: "Ditambahkan",
  },
  en: {
    app: "Smart Home Simulator", app_sub: "Floor plan, electrical wiring & automation",
    nav_sim: "Simulator", nav_comp: "Components", nav_rule: "Rules", nav_rep: "Report",
    nav_case: "Case Studies", nav_help: "Guide",
    transfer: "Export & Import Configuration",
    transfer_help: "The whole house — rooms, components, circuits, wiring, rules and occupants — can be saved to a single JSON file and loaded back at any time.",
    export_json: "Export JSON", import_json: "Import JSON",
    export_ok: "The configuration file is being downloaded",
    import_ok: "Configuration imported successfully",
    import_bad: "The file is not valid JSON",
    import_confirm: "Importing will replace the entire current house. Continue?",
    case_confirm: "Loading this case will replace the entire current house. Continue?",
    case_loaded: "Case loaded",
    cases_available: "cases available", components_short: "components",
    case_time: "Time", case_source: "Power source", case_goals: "Learning goals",
    case_steps: "Observation steps", case_questions: "Guiding questions",
    case_expected: "Expected results", case_note: "Notes",
    case_load: "Load case", case_download: "Download JSON",
    lang: "Language", theme: "Theme",

    env: "Environment", sim: "Simulation Control", time: "Time", weather: "Weather",
    tod: "Period", pagi: "Morning", siang: "Midday", sore: "Afternoon", malam: "Night",
    cerah: "Clear", berawan: "Cloudy", hujan: "Rain", badai: "Storm", kabut: "Fog",
    outdoor: "Outdoor", temp: "Temp", lux: "Light", humidity: "Humidity",
    temp_offset: "Temp offset (°C)", auto_weather: "Random weather each day",
    play: "Run", pause: "Pause", step: "Step once", step_min: "Minutes / step",
    speed: "Speed", reset_energy: "Reset energy", reset_trip: "Reset breaker",
    reset_demo: "Load demo data", clear_house: "Empty the house", day: "Day",

    plan: "Floor Plan", mode_view: "View mode", mode_edit: "Edit plan", mode_place: "Place component",
    show_wire: "Power wiring", show_ctrl: "Control links", show_label: "Component names",
    add_room: "Add Room", add_comp: "Add Component", tidy: "Tidy up", legend_on: "On",
    legend_off: "Off", legend_person: "Occupant", legend_trip: "Tripped circuit",
    click_place: "Click on the plan to place:",

    summary: "Electrical Summary", power: "Power", current: "Current", energy: "Energy", cost: "Cost",
    devices_on: "Active devices", main_mcb: "Main breaker", voltage: "Voltage", pf: "Power factor",
    tariff: "Tariff (Rp/kWh)", tripped: "TRIPPED", normal: "Normal",
    circuits: "Circuits (MCB)", load: "Load", source: "Source", capacity: "Capacity",
    sources: "Power Sources", occupants: "Occupants", rooms: "Rooms", log: "Event Log",
    clear_log: "Clear",

    pick_rule: "Pick a rule from the list, or press New Rule to create one.",
    properties: "Properties", nothing_sel: "Nothing selected. Click a room, component or occupant on the plan.",
    name: "Name", type: "Type", room: "Room", circuit: "Circuit", pos: "Position",
    size: "Size", color: "Color", window: "Window factor", power_w: "Power (W)",
    standby_w: "Standby (W)", mode: "Mode", auto: "Automatic", manual: "Manual", off: "Disabled",
    combine: "Switch combination", parallel: "Parallel (any ON)", series: "Series (all ON)",
    state: "State", on: "ON", offs: "OFF", enabled: "Enabled", threshold: "Threshold",
    operator: "Operator", setpoint: "Set point (°C)", on_hour: "On hour", off_hour: "Off hour",
    momentary: "Momentary button", save: "Save", del: "Delete", cancel: "Cancel", add: "Add",
    toggle: "Toggle", confirm_del: "Delete this item?",

    comp_sources: "Power Sources", comp_switches: "Switches & Sensors", comp_devices: "Electrical Devices",
    wiring: "Wiring Switch → Device", controlled_by: "Controlled by", controls: "Controls",
    inverted: "Inverted logic (NC)", link_add: "Connect", no_link: "No connection yet",
    switch: "Switch/Sensor", device: "Device",

    rules: "Automation Rules", rule_new: "New Rule", rule_name: "Rule name",
    logic: "Condition logic", priority: "Priority", conditions: "Conditions (IF)", actions: "Actions (THEN)",
    add_cond: "Add condition", add_act: "Add action", subject: "Subject", value: "Value",
    target: "Target", action: "Action", test_rule: "Test now", fired: "Last fired",
    times: "times", rule_match: "Conditions MET", rule_nomatch: "Conditions not met",
    all_devices: "All devices", act_on: "Turn on", act_off: "Turn off", act_toggle: "Toggle",

    s_time_of_day: "Time period", s_hour: "Hour (0-24)", s_weather: "Weather",
    s_outdoor_temp: "Outdoor temp (°C)", s_outdoor_lux: "Outdoor light (lux)",
    s_room_temp: "Room temp (°C)", s_room_lux: "Room light (lux)",
    s_room_humidity: "Room humidity (%)", s_occupancy: "People in room",
    s_occupant_total: "People at home", s_sensor: "Sensor/switch state",
    s_device: "Device state", s_total_power: "Total power (W)", s_energy_kwh: "Energy (kWh)",

    report: "Report & Analysis", chart_power: "Power chart (W)", chart_energy: "Energy chart (kWh)",
    per_room: "Consumption per Room", per_circuit: "Consumption per Circuit", per_device: "Busiest Devices",
    export_csv: "Download CSV", no_data: "No data yet. Run the simulation first.",
    activity: "Activity", away: "Away", schedule: "Hourly schedule", present: "At home",
    saved: "Saved", deleted: "Deleted", added: "Added",
  }
};

const TYPE_LABEL = {
  id: {
    pln:"PLN", genset:"Genset", solar:"Panel Surya", baterai:"Baterai",
    tombol:"Tombol", saklar:"Saklar", pir:"Sensor Gerak (PIR)", ldr:"Sensor Cahaya (LDR)",
    infrared:"Infrared / Remote", suhu:"Sensor Suhu", kelembaban:"Sensor Kelembaban",
    hujan:"Sensor Hujan", timer:"Timer", asap:"Sensor Asap", pintu:"Sensor Pintu",
    lampu:"Lampu", tv:"TV", ac:"AC", komputer:"Komputer", mesin_cuci:"Mesin Cuci",
    kipas:"Kipas Angin", kulkas:"Kulkas", pompa_air:"Pompa Air", microwave:"Microwave",
    rice_cooker:"Rice Cooker", water_heater:"Water Heater", router:"Router", cctv:"CCTV",
    charger:"Charger", setrika:"Setrika", dispenser:"Dispenser", exhaust:"Exhaust Fan",
    lainnya:"Lainnya"
  },
  en: {
    pln:"Grid (PLN)", genset:"Generator", solar:"Solar Panel", baterai:"Battery",
    tombol:"Push Button", saklar:"Switch", pir:"Motion Sensor (PIR)", ldr:"Light Sensor (LDR)",
    infrared:"Infrared / Remote", suhu:"Temperature Sensor", kelembaban:"Humidity Sensor",
    hujan:"Rain Sensor", timer:"Timer", asap:"Smoke Sensor", pintu:"Door Sensor",
    lampu:"Lamp", tv:"TV", ac:"Air Conditioner", komputer:"Computer", mesin_cuci:"Washing Machine",
    kipas:"Fan", kulkas:"Refrigerator", pompa_air:"Water Pump", microwave:"Microwave",
    rice_cooker:"Rice Cooker", water_heater:"Water Heater", router:"Router", cctv:"CCTV",
    charger:"Charger", setrika:"Iron", dispenser:"Water Dispenser", exhaust:"Exhaust Fan",
    lainnya:"Other"
  }
};

let LANG = localStorage.getItem("sh_lang") || "id";
function t(k){ return (DICT[LANG] && DICT[LANG][k]) || (DICT.id[k]) || k; }
function tt(sub){ return (TYPE_LABEL[LANG] && TYPE_LABEL[LANG][sub]) || sub; }
function setLang(l){ LANG = l; localStorage.setItem("sh_lang", l); applyI18n();
  document.dispatchEvent(new CustomEvent("langchange")); }
function applyI18n(){
  document.querySelectorAll("[data-i18n]").forEach(function(el){
    el.textContent = t(el.getAttribute("data-i18n"));
  });
  document.querySelectorAll("[data-i18n-title]").forEach(function(el){
    el.title = t(el.getAttribute("data-i18n-title"));
  });
  document.documentElement.lang = LANG;
  var b = document.getElementById("langBtn"); if(b) b.textContent = LANG.toUpperCase();
}
