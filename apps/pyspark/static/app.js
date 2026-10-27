// PySpark MLlib Script Generator - frontend logic
// Semua orkestrasi form dinamis (task -> data source -> algoritma -> generate).

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  const state = {
    task: null,
    dataSource: "csv",
    uploadedFile: null, // {file_id, filename, columns: [{name, dtype}], preview}
    algorithms: [],     // dari /api/algorithms/<task>
    selectedAlgo: null,
    lastScript: null,
    lastFilename: null,
  };

  // Spesifikasi parameter data acak per task (dipakai untuk render form & payload)
  const RANDOM_PARAM_SPECS = {
    classification: [
      { name: "n_samples", label: "Jumlah Sampel", type: "int", default: 1000, min: 10, max: 200000 },
      { name: "n_features", label: "Jumlah Fitur", type: "int", default: 5, min: 1, max: 100 },
      { name: "n_classes", label: "Jumlah Kelas", type: "int", default: 2, min: 2, max: 20 },
      { name: "class_sep", label: "Separasi Antar Kelas", type: "float", default: 2.0, min: 0.1, max: 20, step: 0.1 },
      { name: "seed", label: "Random Seed", type: "int", default: 42, min: 0, max: 999999 },
    ],
    regression: [
      { name: "n_samples", label: "Jumlah Sampel", type: "int", default: 1000, min: 10, max: 200000 },
      { name: "n_features", label: "Jumlah Fitur", type: "int", default: 5, min: 1, max: 100 },
      { name: "noise", label: "Noise (Std Dev)", type: "float", default: 1.0, min: 0, max: 100, step: 0.1 },
      { name: "seed", label: "Random Seed", type: "int", default: 42, min: 0, max: 999999 },
    ],
    clustering: [
      { name: "n_samples", label: "Jumlah Sampel", type: "int", default: 1000, min: 10, max: 200000 },
      { name: "n_features", label: "Jumlah Fitur", type: "int", default: 2, min: 1, max: 100 },
      { name: "n_clusters", label: "Jumlah Cluster", type: "int", default: 3, min: 2, max: 30 },
      { name: "cluster_std", label: "Std Dev Cluster", type: "float", default: 1.0, min: 0.1, max: 20, step: 0.1 },
      { name: "seed", label: "Random Seed", type: "int", default: 42, min: 0, max: 999999 },
    ],
    association: [
      { name: "n_transactions", label: "Jumlah Transaksi", type: "int", default: 1000, min: 10, max: 200000 },
      { name: "n_items", label: "Jumlah Item Unik", type: "int", default: 20, min: 2, max: 1000 },
      { name: "min_items", label: "Min Item / Transaksi", type: "int", default: 1, min: 1, max: 50 },
      { name: "max_items", label: "Max Item / Transaksi", type: "int", default: 5, min: 1, max: 50 },
      { name: "seed", label: "Random Seed", type: "int", default: 42, min: 0, max: 999999 },
    ],
  };

  const NEEDS_TARGET = (task) => task === "classification" || task === "regression";
  const NEEDS_SPLIT_SETTINGS = (task) => task === "classification" || task === "regression";
  const IS_ASSOCIATION = (task) => task === "association";

  // ------------------------------------------------------------------
  // Elemen DOM
  // ------------------------------------------------------------------
  const el = (id) => document.getElementById(id);

  const dataSourceSection = el("dataSourceSection");
  const algoSection = el("algoSection");
  const settingsSection = el("settingsSection");
  const generateSection = el("generateSection");
  const resultSection = el("resultSection");
  const splitSettings = el("splitSettings");

  // ------------------------------------------------------------------
  // 1. Pemilihan Task
  // ------------------------------------------------------------------
  document.querySelectorAll(".task-card").forEach((btn) => {
    btn.addEventListener("click", () => selectTask(btn.dataset.task, btn));
  });

  function selectTask(task, btnEl) {
    state.task = task;
    state.uploadedFile = null;
    state.selectedAlgo = null;

    document.querySelectorAll(".task-card").forEach((b) => b.classList.remove("active"));
    if (btnEl) btnEl.classList.add("active");

    resetUploadUI();
    renderRandomParamFields();
    toggleAssociationVsFeatureUI();

    dataSourceSection.hidden = false;
    settingsSection.hidden = false;
    generateSection.hidden = false;
    resultSection.hidden = true;

    splitSettings.style.display = NEEDS_SPLIT_SETTINGS(task) ? "" : "none";

    loadAlgorithms(task);
    applyDataSourceVisibility();
  }

  // ------------------------------------------------------------------
  // 2. Sumber Data: toggle CSV vs Random
  // ------------------------------------------------------------------
  document.querySelectorAll('input[name="dataSource"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      state.dataSource = e.target.value;
      applyDataSourceVisibility();
    });
  });

  function applyDataSourceVisibility() {
    el("csvPanel").hidden = state.dataSource !== "csv";
    el("randomPanel").hidden = state.dataSource !== "random";
  }

  function toggleAssociationVsFeatureUI() {
    const isAssoc = IS_ASSOCIATION(state.task);
    el("itemsColumnWrap").hidden = !isAssoc;
    el("featureColumnsWrap").hidden = isAssoc;
    el("targetColumnWrap").hidden = !NEEDS_TARGET(state.task);
  }

  // ------------------------------------------------------------------
  // CSV Upload
  // ------------------------------------------------------------------
  function resetUploadUI() {
    state.uploadedFile = null;
    el("uploadStatus").textContent = "";
    el("uploadStatus").className = "status-msg";
    el("columnConfig").hidden = true;
    el("featureColumnsList").innerHTML = "";
    el("targetColumn").innerHTML = "";
    el("itemsColumn").innerHTML = "";
    el("previewTable").innerHTML = "";
  }

  el("uploadBtn").addEventListener("click", async () => {
    const fileInput = el("csvFile");
    if (!fileInput.files || fileInput.files.length === 0) {
      setStatus("uploadStatus", "Pilih file CSV terlebih dahulu.", "error");
      return;
    }
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    setStatus("uploadStatus", "Mengunggah dan menganalisis file...", "");
    el("uploadBtn").disabled = true;

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload gagal.");

      state.uploadedFile = data;
      setStatus(
        "uploadStatus",
        `Berhasil: ${data.filename} (${data.columns.length} kolom, ${data.n_rows_total ?? "?"} baris).`,
        "success"
      );
      renderColumnConfig(data);
      el("columnConfig").hidden = false;
    } catch (err) {
      setStatus("uploadStatus", err.message, "error");
    } finally {
      el("uploadBtn").disabled = false;
    }
  });

  function renderColumnConfig(data) {
    toggleAssociationVsFeatureUI();

    // Target column select
    const targetSelect = el("targetColumn");
    targetSelect.innerHTML = "";
    data.columns.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = `${c.name} (${c.dtype})`;
      targetSelect.appendChild(opt);
    });

    // Items column select (association)
    const itemsSelect = el("itemsColumn");
    itemsSelect.innerHTML = "";
    data.columns.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.name;
      opt.textContent = c.name;
      itemsSelect.appendChild(opt);
    });

    // Feature checkboxes
    const listWrap = el("featureColumnsList");
    listWrap.innerHTML = "";
    data.columns.forEach((c, idx) => {
      const item = document.createElement("label");
      item.className = "checkbox-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = c.name;
      checkbox.className = "feature-checkbox";
      // default: centang semua kecuali kolom target pertama (heuristik: kolom terakhir)
      checkbox.checked = idx !== data.columns.length - 1;
      item.appendChild(checkbox);
      const span = document.createElement("span");
      span.textContent = c.name + " ";
      item.appendChild(span);
      const badge = document.createElement("span");
      badge.className = "dtype-badge " + c.dtype;
      badge.textContent = c.dtype === "numeric" ? "num" : "cat";
      item.appendChild(badge);
      listWrap.appendChild(item);
    });

    // default target = kolom terakhir
    if (data.columns.length > 0) {
      targetSelect.value = data.columns[data.columns.length - 1].name;
    }

    // Preview table
    renderPreviewTable(data.columns.map((c) => c.name), data.preview);
  }

  function renderPreviewTable(columns, rows) {
    const table = el("previewTable");
    table.innerHTML = "";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach((c) => {
      const th = document.createElement("th");
      th.textContent = c;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((c) => {
        const td = document.createElement("td");
        const v = row[c];
        td.textContent = v === null || v === undefined ? "" : v;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  // ------------------------------------------------------------------
  // Random data param fields
  // ------------------------------------------------------------------
  function renderRandomParamFields() {
    const wrap = el("randomParamsFields");
    wrap.innerHTML = "";
    const specs = RANDOM_PARAM_SPECS[state.task] || [];
    specs.forEach((spec) => {
      wrap.appendChild(buildNumberField("random_" + spec.name, spec));
    });
  }

  function buildNumberField(id, spec) {
    const block = document.createElement("div");
    block.className = "field-block";
    const label = document.createElement("label");
    label.setAttribute("for", id);
    label.textContent = spec.label;
    block.appendChild(label);
    const input = document.createElement("input");
    input.type = "number";
    input.id = id;
    input.dataset.paramName = spec.name;
    input.dataset.paramType = spec.type;
    if (spec.min !== undefined) input.min = spec.min;
    if (spec.max !== undefined) input.max = spec.max;
    input.step = spec.type === "float" ? (spec.step || 0.01) : 1;
    input.value = spec.default;
    block.appendChild(input);
    return block;
  }

  // ------------------------------------------------------------------
  // 3. Algoritma & Hyperparameter
  // ------------------------------------------------------------------
  async function loadAlgorithms(task) {
    algoSection.hidden = false;
    const select = el("algoSelect");
    select.innerHTML = "<option>Memuat...</option>";
    try {
      const res = await fetch(`/api/algorithms/${task}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Gagal memuat algoritma.");
      state.algorithms = data.algorithms;
      select.innerHTML = "";
      data.algorithms.forEach((a) => {
        const opt = document.createElement("option");
        opt.value = a.key;
        opt.textContent = a.label;
        select.appendChild(opt);
      });
      if (data.algorithms.length > 0) {
        selectAlgorithm(data.algorithms[0].key);
      }
    } catch (err) {
      select.innerHTML = `<option>Error: ${err.message}</option>`;
    }
  }

  el("algoSelect").addEventListener("change", (e) => selectAlgorithm(e.target.value));

  function selectAlgorithm(key) {
    const algo = state.algorithms.find((a) => a.key === key);
    if (!algo) return;
    state.selectedAlgo = algo;
    el("algoSelect").value = key;

    const notesEl = el("algoNotes");
    if (algo.notes) {
      notesEl.textContent = algo.notes;
      notesEl.classList.add("visible");
    } else {
      notesEl.textContent = "";
      notesEl.classList.remove("visible");
    }

    const wrap = el("hyperparamFields");
    wrap.innerHTML = "";
    (algo.params || []).forEach((spec) => {
      wrap.appendChild(buildNumberField("hp_" + spec.name, spec));
    });
  }

  // ------------------------------------------------------------------
  // 4. Generate script
  // ------------------------------------------------------------------
  el("generateBtn").addEventListener("click", generateScript);

  function collectRandomParams() {
    const specs = RANDOM_PARAM_SPECS[state.task] || [];
    const result = {};
    specs.forEach((spec) => {
      const input = el("random_" + spec.name);
      result[spec.name] = spec.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
    });
    return result;
  }

  function collectHyperparams() {
    const algo = state.selectedAlgo;
    const result = {};
    if (!algo) return result;
    (algo.params || []).forEach((spec) => {
      const input = el("hp_" + spec.name);
      if (!input) return;
      result[spec.name] = spec.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
    });
    return result;
  }

  function collectFeatureColumns() {
    return Array.from(document.querySelectorAll(".feature-checkbox"))
      .filter((cb) => cb.checked)
      .map((cb) => cb.value);
  }

  function collectColumnTypes() {
    const types = {};
    if (state.uploadedFile) {
      state.uploadedFile.columns.forEach((c) => {
        types[c.name] = c.dtype;
      });
    }
    return types;
  }

  async function generateScript() {
    if (!state.task) {
      setStatus("generateStatus", "Pilih jenis task terlebih dahulu.", "error");
      return;
    }
    if (!state.selectedAlgo) {
      setStatus("generateStatus", "Pilih algoritma terlebih dahulu.", "error");
      return;
    }

    const payload = {
      task: state.task,
      algorithm: state.selectedAlgo.key,
      params: collectHyperparams(),
      data_source: state.dataSource,
      app_name: el("appName").value || undefined,
      train_ratio: parseFloat(el("trainRatio").value || "0.8"),
      split_seed: parseInt(el("splitSeed").value || "42", 10),
    };

    if (state.dataSource === "csv") {
      if (!state.uploadedFile) {
        setStatus("generateStatus", "Upload file CSV terlebih dahulu.", "error");
        return;
      }
      payload.file_id = state.uploadedFile.file_id;

      if (IS_ASSOCIATION(state.task)) {
        payload.items_column = el("itemsColumn").value;
        payload.delimiter = el("itemsDelimiter").value || ",";
      } else {
        const featureColumns = collectFeatureColumns();
        if (featureColumns.length === 0) {
          setStatus("generateStatus", "Pilih minimal satu kolom fitur.", "error");
          return;
        }
        payload.feature_columns = featureColumns;
        payload.column_types = collectColumnTypes();
        if (NEEDS_TARGET(state.task)) {
          payload.target_column = el("targetColumn").value;
        }
      }
    } else {
      payload.random_params = collectRandomParams();
    }

    setStatus("generateStatus", "Membuat script...", "");
    el("generateBtn").disabled = true;

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Gagal membuat script.");

      state.lastScript = data.script;
      state.lastFilename = data.filename;

      el("scriptOutput").textContent = data.script;
      el("resultMeta").textContent = `Task: ${data.task_label} | Algoritma: ${data.algorithm_label} | File: ${data.filename}`;
      resultSection.hidden = false;
      resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
      setStatus("generateStatus", "", "");
    } catch (err) {
      setStatus("generateStatus", err.message, "error");
    } finally {
      el("generateBtn").disabled = false;
    }
  }

  // ------------------------------------------------------------------
  // Copy / Download
  // ------------------------------------------------------------------
  el("copyBtn").addEventListener("click", async () => {
    if (!state.lastScript) return;
    try {
      await navigator.clipboard.writeText(state.lastScript);
      flashButton(el("copyBtn"), "Tersalin!");
    } catch (err) {
      // fallback untuk browser lama
      const textarea = document.createElement("textarea");
      textarea.value = state.lastScript;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      flashButton(el("copyBtn"), "Tersalin!");
    }
  });

  el("downloadBtn").addEventListener("click", () => {
    if (!state.lastScript) return;
    const blob = new Blob([state.lastScript], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = state.lastFilename || "pyspark_script.py";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  function flashButton(btn, tempText) {
    const original = btn.textContent;
    btn.textContent = tempText;
    setTimeout(() => { btn.textContent = original; }, 1500);
  }

  // ------------------------------------------------------------------
  // Util
  // ------------------------------------------------------------------
  function setStatus(id, message, cls) {
    const node = el(id);
    node.textContent = message;
    node.className = "status-msg" + (cls ? " " + cls : "");
  }
})();
