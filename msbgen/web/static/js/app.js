// MSBGen front-end: decode, slice, analyse, then reconstruct one selected mode at a time.

const API_BITPLANES = "/api/bitplanes";
const API_ANALYSIS = "/api/analysis";
const API_RECONSTRUCT = "/api/reconstruct";
const API_RECONSTRUCT_MODEL = "/api/reconstruct/model";
const CHANNEL_COLORS = { R: "var(--ch-r)", G: "var(--ch-g)", B: "var(--ch-b)", L: "var(--ch-gray)" };
const FILL_SERIES = [
  { fill: "mid", name: "Midpoint fill", color: "var(--series-mid)", dashed: false },
  { fill: "random", name: "Random fill", color: "var(--series-random)", dashed: false },
  { fill: "zero", name: "Zero fill", color: "var(--series-zero)", dashed: true },
];
const METRICS = {
  psnr: { title: "PSNR (dB)", format: (v) => v.toFixed(1) },
  ssim: { title: "SSIM", format: (v) => v.toFixed(3) },
  mae: { title: "MAE (levels)", format: (v) => (v < 10 ? v.toFixed(2) : v.toFixed(1)) },
};

const $ = (sel) => document.querySelector(sel);
const el = {
  dropzone: $("#dropzone"),
  fileInput: $("#file-input"),
  preview: $("#preview"),
  meta: $("#file-meta"),
  sliceBtn: $("#slice-btn"),
  clearBtn: $("#clear-btn"),
  colorPick: $("#color-pick"),
  notes: $("#notes"),
  error: $("#error"),
  results: $("#results"),
  summary: $("#results-summary"),
  channels: $("#channels"),
  formula: $("#formula"),
  planeTpl: $("#plane-template"),
  analysis: $("#analysis"),
  analysisSummary: $("#analysis-summary"),
  analysisStatus: $("#analysis-status"),
  analysisBody: $("#analysis-body"),
  recMetric: $("#rec-metric"),
  recChart: $("#rec-chart"),
  recLegend: $("#rec-legend"),
  recGrid: $("#rec-grid"),
  recTable: $("#rec-table"),
  entropyChart: $("#entropy-chart"),
  entropyLegend: $("#entropy-legend"),
  entropyTable: $("#entropy-table"),
  corrChart: $("#corr-chart"),
  corrReadout: $("#corr-readout"),
  kControl: $("#k-control"),
  kPick: $("#k-pick"),
  models: $("#models"),
  modelsSummary: $("#models-summary"),
  modelsStatus: $("#models-status"),
  modelsBody: $("#models-body"),
  modelsView: $("#models-view"),
  modelsGrid: $("#models-grid"),
  modelsTable: $("#models-table"),
  lightbox: $("#lightbox"),
  lbTitle: $("#lightbox-title"),
  lbImg: $("#lightbox-img"),
  lbCaption: $("#lightbox-caption"),
  lbDownload: $("#lightbox-download"),
  themeToggle: $("#theme-toggle"),
};

let currentFile = null;
let previewUrl = null;
let requestId = 0; // ignore responses that arrive after the user picked another file
let selectedColorMode = "RGB";
let selectedK = 1;
let pipelineStarted = false;
let modeResults = createModeResults();
let workQueue = Promise.resolve();
let analysisData = null;
let recMetric = "psnr";
let modelTiles = []; // { tile, img, image, error } for the reconstruction / error toggle

function createModeResults() {
  const createMode = () => ({ bitplanes: null, analysis: null, reconstructions: new Map(), pending: null });
  return { L: createMode(), RGB: createMode() };
}

// ---------- Theme ----------
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("msbgen-theme"); } catch { /* storage unavailable */ }
  if (saved) document.documentElement.dataset.theme = saved;
  el.themeToggle.addEventListener("click", () => {
    const isDark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = isDark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("msbgen-theme", next); } catch { /* ignore */ }
  });
}

// ---------- Helpers ----------
const formatBytes = (n) =>
  n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(2)} MB`;
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const fmt = (v, digits = 3) => (v == null ? "—" : v.toFixed(digits));

function setMeta(entries) {
  el.meta.replaceChildren(
    ...entries.map(([k, v]) => {
      const div = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = k;
      dd.textContent = v;
      div.append(dt, dd);
      return div;
    }),
  );
  el.meta.hidden = entries.length === 0;
}

function showError(msg) {
  el.error.textContent = msg;
  el.error.hidden = !msg;
}

function showNotes(notes) {
  if (!notes?.length) { el.notes.hidden = true; return; }
  const ul = document.createElement("ul");
  for (const n of notes) {
    const li = document.createElement("li");
    li.textContent = n;
    ul.append(li);
  }
  el.notes.replaceChildren(ul);
  el.notes.hidden = false;
}

function setLoading(on) {
  el.sliceBtn.classList.toggle("loading", on);
  el.sliceBtn.disabled = on || !currentFile;
  el.sliceBtn.querySelector(".label").textContent = on ? "Running…" : "Slice & analyse";
}

function hideResults() {
  el.results.hidden = true;
  el.analysis.hidden = true;
  el.models.hidden = true;
  analysisData = null;
  modelTiles = [];
}

// ---------- File selection ----------
function selectFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    showError(`“${file.name}” is not an image.`);
    return;
  }
  requestId++;
  currentFile = file;
  modeResults = createModeResults();
  pipelineStarted = false;
  showError("");
  showNotes([]);
  hideResults();

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  el.preview.src = previewUrl;
  el.preview.hidden = false;
  el.dropzone.classList.add("has-image");

  setMeta([["File", file.name], ["Size", formatBytes(file.size)]]);
  el.preview.onload = () =>
    setMeta([
      ["File", file.name],
      ["Size", formatBytes(file.size)],
      ["Dimensions", `${el.preview.naturalWidth} × ${el.preview.naturalHeight}`],
    ]);
  // Browsers can't render some formats (e.g. TIFF); the server will still decode them.
  el.preview.onerror = () => { el.preview.hidden = true; el.dropzone.classList.remove("has-image"); };

  setLoading(false);
  el.clearBtn.disabled = false;
}

function clearAll() {
  requestId++;
  currentFile = null;
  modeResults = createModeResults();
  pipelineStarted = false;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  el.fileInput.value = "";
  el.preview.removeAttribute("src");
  el.preview.hidden = true;
  el.dropzone.classList.remove("has-image");
  setMeta([]);
  showError("");
  showNotes([]);
  hideResults();
  el.channels.replaceChildren();
  setLoading(false);
  el.clearBtn.disabled = true;
}

function initDropzone() {
  el.fileInput.addEventListener("change", () => selectFile(el.fileInput.files[0]));
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
  });
  ["dragenter", "dragover"].forEach((t) =>
    el.dropzone.addEventListener(t, (e) => { e.preventDefault(); el.dropzone.classList.add("dragover"); }),
  );
  ["dragleave", "drop"].forEach((t) =>
    el.dropzone.addEventListener(t, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragover"); }),
  );
  el.dropzone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));
  // Paste an image from the clipboard.
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
    if (item) selectFile(item.getAsFile());
  });
}

// ---------- API ----------
async function postImage(url, file, fields = {}) {
  const body = new FormData();
  body.append("file", file, file.name || "image.png");
  for (const [name, value] of Object.entries(fields)) body.append(name, String(value));
  const res = await fetch(url, { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

function enqueueWork(work) {
  const queued = workQueue.then(work);
  workQueue = queued.catch(() => {});
  return queued;
}

function ensureMode(mode, id = requestId) {
  const state = modeResults[mode];
  if (state.pending) return state.pending;
  const pending = enqueueWork(() => runMode(mode, id));
  state.pending = pending;
  pending.finally(() => { if (state.pending === pending) state.pending = null; });
  return pending;
}

async function slice() {
  if (!currentFile) return;
  pipelineStarted = true;
  showError("");
  setLoading(true);
  ensureMode(selectedColorMode);
}

async function runMode(mode, id) {
  if (id !== requestId || !currentFile) return;
  const file = currentFile;
  const state = modeResults[mode];
  const isVisible = () => id === requestId && mode === selectedColorMode;
  try {
    if (!state.bitplanes) {
      state.bitplanes = await postImage(API_BITPLANES, file, { color_mode: mode });
    }
    if (id !== requestId) return;
    if (isVisible()) {
      renderResults(state.bitplanes);
      showAnalysisStatus("Analysing bit-planes…", true);
    }

    if (!state.analysis) {
      state.analysis = await postImage(API_ANALYSIS, file, { color_mode: mode });
    }
    if (id !== requestId) return;
    if (isVisible()) renderAnalysis(state.analysis);

    while (isVisible()) {
      const k = selectedK;
      if (state.reconstructions.has(k)) {
        renderModels(state.reconstructions.get(k));
        break;
      }
      await runReconstruction(mode, k, id, file, state);
      if (k === selectedK) break;
    }
  } catch (err) {
    if (id === requestId && isVisible()) {
      if (!state.bitplanes) showError(err.message || String(err));
      else if (!state.analysis) showAnalysisStatus(`Analysis failed: ${err.message || err}`, false);
      else showModelsStatus(`Reconstruction unavailable: ${err.message || err}`, false);
    }
  } finally {
    if (isVisible()) setLoading(false);
  }
}

async function runReconstruction(mode, k, id, file, state) {
  if (id !== requestId) return;
  showModelsStatus(`Preparing ${mode === "L" ? "grayscale" : "RGB"} baselines for k=${k}…`, true);
  const data = await postImage(API_RECONSTRUCT, file, { color_mode: mode, k, run_models: false });
  if (id !== requestId) return;
  data.modelRuns = {}; // per model: { state: "running" | "done" | "error", result, error, note }
  state.reconstructions.set(k, data);
  if (mode === selectedColorMode && k === selectedK) renderModels(data);
}

function showModelsStatus(message, busy) {
  el.models.hidden = false;
  el.modelsBody.hidden = true;
  el.modelsSummary.textContent = "";
  el.modelsStatus.hidden = false;
  el.modelsStatus.classList.toggle("error-text", !busy);
  el.modelsStatus.querySelector(".spinner").hidden = !busy;
  el.modelsStatus.querySelector(".status-text").textContent = message;
}

function renderModels(data) {
  el.modelsStatus.hidden = true;
  el.modelsBody.hidden = false;
  const runs = data.modelRuns ?? (data.modelRuns = {});
  const candidates = data.candidates ?? [];
  const done = candidates.filter((c) => runs[c.name]?.state === "done").map((c) => runs[c.name].result);
  const all = [...data.results, ...done];
  const ratio = `${data.width} / ${data.height}`;
  const stem = fileStem(data.filename);
  const n = data.bit_depth;
  const bits = data.k === 1 ? `B${n - 1}` : `B${n - 1}…B${n - data.k}`;
  const models = all.filter((r) => r.kind === "model");
  const perceptual = (data.perceptual ?? []).filter((m) => m in PERCEPTUAL_TITLES);
  const scored = all.filter((r) => r.psnr != null);
  const best = scored.reduce((a, b) => (b.psnr > a.psnr ? b : a), scored[0]);
  const top = models.reduce((a, b) => (b.psnr > (a?.psnr ?? -Infinity) ? b : a), null);
  el.modelsSummary.textContent = top
    ? `Best model: ${top.label} (${top.name}) · PSNR ${top.psnr.toFixed(2)} dB, ${gainText(top.psnr_gain)} vs midpoint fill.`
    : candidates.length ? "Click a model card to run it." : "No compatible models for this color mode and k.";
  modelTiles = [];
  const showErrors = el.modelsView.querySelector('[aria-pressed="true"]').dataset.view === "error";
  const original = { name: "original", label: "Original", image: data.original, error_map: null };
  const known = new Set(data.results.map((r) => r.name));
  const entries = [original, ...data.results, ...candidates.filter((c) => !known.has(c.name))
    .map((c) => (runs[c.name]?.state === "done" ? runs[c.name].result : c))];
  const tiles = entries.map((r) => {
    if (!("image" in r)) return renderModelPlaceholder(data, r.name, runs[r.name], ratio);
    const isOriginal = r === original;
    const stats = isOriginal ? `${data.color_mode === "L" ? "grayscale" : "RGB"}, ${data.width} × ${data.height}` :
      `PSNR ${fmt(r.psnr, 2)} dB${r.kind === "model" ? ` (${gainText(r.psnr_gain)})` : ""} · SSIM ${fmt(r.ssim)}`;
    const tile = {
      image: showErrors && r.error_map ? r.error_map : r.image,
      label: r.label,
      side: isOriginal ? "reference" : r.kind === "model" ? r.name : `from ${bits}`,
      tag: r === best ? "best" : "",
      highlight: false,
      stats,
      title: isOriginal ? `Original (${data.color_mode === "L" ? "grayscale" : "RGB"})` : `${r.label} — from ${bits} only`,
      caption: isOriginal ? "The image every method is compared against." :
        `${stats} · MAE ${r.mae.toFixed(2)}${perceptual.map((m) => ` · ${PERCEPTUAL_TITLES[m]} ${fmt(r[m], 3)}`).join("")}` +
        ` · MSB consistency ${pct(r.msb_consistency)} · Time ${fmt(r.time_s, 3)} s.${runs[r.name]?.note ? ` ${runs[r.name].note}` : ""}`,
      download: `${stem}_${r.name}.png`,
    };
    const node = renderTile(tile, ratio);
    if (r === best) node.classList.add("best");
    modelTiles.push({ tile, img: node.querySelector("img"), image: r.image, error: r.error_map });
    return node;
  });
  el.modelsGrid.replaceChildren(...tiles);

  const rows = all.map((r) => {
    const gain = r.kind === "model" || r.name !== "mid" ? gainText(r.psnr_gain, true) : "—";
    return `<tr><td>${escapeHtml(r.label)}${r.kind === "model" ? ` <span class="hint">${escapeHtml(r.name)}</span>` : ""}</td>` +
      `<td>${r.kind === "model" ? "model" : "baseline"}</td><td>${fmt(r.psnr, 2)}</td><td>${gain}</td>` +
      `<td>${fmt(r.ssim, 4)}</td>${perceptual.map((m) => `<td>${fmt(r[m], 4)}</td>`).join("")}` +
      `<td>${r.mae.toFixed(2)}</td><td>${pct(r.msb_consistency)}</td><td>${fmt(r.time_s, 3)}</td></tr>`;
  });
  el.modelsTable.innerHTML = `<thead><tr><th>Method</th><th>Kind</th><th>PSNR (dB)</th><th>Δ vs midpoint</th>` +
    `<th>SSIM</th>${perceptual.map((m) => `<th>${PERCEPTUAL_TITLES[m]} ↓</th>`).join("")}` +
    `<th>MAE</th><th>MSB consistency</th><th>Time (s)</th></tr></thead><tbody>${rows.join("")}</tbody>`;
}

const PERCEPTUAL_TITLES = { lpips: "LPIPS", dists: "DISTS" };

function renderModelPlaceholder(data, name, run, ratio) {
  const state = run?.state ?? "idle";
  const node = el.planeTpl.content.firstElementChild.cloneNode(true);
  const thumb = node.querySelector(".plane-thumb");
  thumb.style.setProperty("--ratio", ratio);
  thumb.classList.add("model-slot", state);
  thumb.replaceChildren();
  thumb.textContent = { idle: "Click to run", running: "Running…", error: "Failed · click to retry" }[state];
  thumb.disabled = state === "running";
  node.querySelector(".plane-name").textContent = name;
  node.querySelector(".plane-weight").textContent = "model";
  const stats = node.querySelector(".plane-stats");
  stats.textContent = state === "error" ? run.error : state === "running" ? "Waiting for the server…" : "Not run yet";
  stats.classList.toggle("error-text", state === "error");
  thumb.addEventListener("click", () => runModelCard(data, name));
  return node;
}

function refreshModels(data) {
  if (data.color_mode === selectedColorMode && data.k === selectedK) renderModels(data);
}

function runModelCard(data, name) {
  const run = (data.modelRuns[name] ??= {});
  if (run.state === "running") return;
  const id = requestId;
  const file = currentFile;
  Object.assign(run, { state: "running", error: null });
  refreshModels(data);
  return enqueueWork(async () => {
    if (id !== requestId) return;
    try {
      const res = await postImage(API_RECONSTRUCT_MODEL, file, { color_mode: data.color_mode, k: data.k, model: name });
      Object.assign(run, { state: "done", result: res.result, note: res.note });
    } catch (err) {
      Object.assign(run, { state: "error", error: err.message || String(err) });
    }
    if (id === requestId) refreshModels(data);
  });
}

function gainText(g, html = false) {
  if (g == null) return "—";
  const text = `${g >= 0 ? "+" : "−"}${Math.abs(g).toFixed(2)} dB`;
  return html ? `<span class="${g >= 0 ? "gain-pos" : "gain-neg"}">${text}</span>` : text;
}

const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function initModelsView() {
  el.modelsView.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-view]");
    if (!btn) return;
    for (const b of el.modelsView.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
    const showError = btn.dataset.view === "error";
    for (const t of modelTiles) {
      const src = showError && t.error ? t.error : t.image;
      t.img.src = src;
      t.tile.image = src;
    }
  });
}

function initModeControls() {
  el.colorPick.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-color-mode]");
    if (!button || button.dataset.colorMode === selectedColorMode) return;
    selectedColorMode = button.dataset.colorMode;
    for (const item of el.colorPick.querySelectorAll("button")) {
      item.setAttribute("aria-pressed", String(item === button));
    }
    if (!pipelineStarted) return;
    showError("");
    displayModeResults(selectedColorMode);
    setLoading(true);
    ensureMode(selectedColorMode);
  });
  el.kPick.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-k]");
    if (!button) return;
    selectedK = Number(button.dataset.k);
    for (const item of el.kPick.querySelectorAll("button")) {
      item.setAttribute("aria-pressed", String(item === button));
    }
    if (pipelineStarted) {
      displayModeResults(selectedColorMode);
      ensureMode(selectedColorMode);
    }
  });
}

function displayModeResults(mode) {
  const state = modeResults[mode];
  el.results.hidden = !state.bitplanes;
  el.analysis.hidden = !state.bitplanes;
  el.models.hidden = !state.analysis;
  el.kControl.hidden = !state.analysis;
  if (!state.bitplanes) return;
  renderResults(state.bitplanes);
  if (!state.analysis) {
    showAnalysisStatus("Analysing bit-planes…", true);
    return;
  }
  renderAnalysis(state.analysis);
  const reconstruction = state.reconstructions.get(selectedK);
  if (reconstruction) renderModels(reconstruction);
  else showModelsStatus(`Waiting for ${mode === "L" ? "grayscale" : "RGB"} reconstruction, k=${selectedK}…`, true);
}

// ---------- Bit-plane rendering ----------
function renderResults(data) {
  showNotes(data.notes);
  // Show the image as the server decoded it (handles TIFF, alpha removal, etc.).
  if (el.preview.hidden) {
    el.preview.src = data.preview;
    el.preview.hidden = false;
    el.dropzone.classList.add("has-image");
  }
  const nch = data.channels.length;
  setMeta([
    ["File", data.filename ?? "—"],
    ["Size", formatBytes(currentFile.size)],
    ["Dimensions", `${data.width} × ${data.height}`],
    ["Format", data.original_format ?? "—"],
    ["Mode", `${data.color_mode} · ${data.bit_depth}-bit`],
    ["Channels", data.channels.map((c) => c.name).join(", ")],
  ]);

  el.summary.textContent =
    `${data.planes.length} planes · ${nch} channel${nch > 1 ? "s" : ""} × ${data.bit_depth} bit${data.bit_depth > 1 ? "s" : ""} · ` +
    `${data.width} × ${data.height} px. Click a plane to enlarge.`;
  renderFormula(data.bit_depth);

  const ratio = `${data.width} / ${data.height}`;
  const stem = fileStem(data.filename);
  const multi = nch > 1;
  const planeCount = `${data.bit_depth} plane${data.bit_depth > 1 ? "s" : ""}`;
  const byChannel = data.channels.map(() => []);
  for (const p of data.planes) byChannel[p.channel_index].push(p);

  const sections = data.channels.map((ch, i) => {
    const section = channelSection(ch.label, `${ch.name} · ${planeCount}`, CHANNEL_COLORS[ch.name]);
    const grid = section.querySelector(".plane-grid");
    for (const p of byChannel[i]) {
      const label = `${multi ? p.channel + " · " : ""}B${p.bit}`;
      const stats = `ones ${pct(p.ones_ratio)} · H ${p.entropy.toFixed(3)} b/px`;
      grid.append(renderTile({
        ...planeTileBase(p.bit, p.weight, data.bit_depth, label),
        image: p.image, stats,
        caption: `${stats}. White = 1, black = 0.`,
        download: `${stem}_${p.channel}_B${p.bit}.png`,
      }, ratio));
    }
    return section;
  });

  if (data.combined?.length) {
    const section = channelSection("Combined RGB", `R + G + B per bit · ${planeCount}`, "var(--ch-rgb)");
    section.classList.add("combined");
    section.querySelector(".channel-head").append(colourLegend());
    const grid = section.querySelector(".plane-grid");
    for (const c of data.combined) {
      const ones = byChannel.map((planes) => planes.find((p) => p.bit === c.bit).ones_ratio);
      const stats = `ones R ${pct(ones[0])} · G ${pct(ones[1])} · B ${pct(ones[2])}`;
      grid.append(renderTile({
        ...planeTileBase(c.bit, c.weight, data.bit_depth, `RGB · B${c.bit}`),
        image: c.image, stats,
        caption: `${stats}. Each channel is 255 where its bit ${c.bit} is 1, else 0 — ` +
          "e.g. yellow = R and G set, white = all three, black = none.",
        download: `${stem}_RGB_B${c.bit}.png`,
      }, ratio));
    }
    sections.push(section);
  }

  el.channels.replaceChildren(...sections);
  el.results.hidden = false;
  // el.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

const fileStem = (name) => (name || "image").replace(/\.[^.]+$/, "");

// Label, weight and MSB/LSB badge shared by single-channel and combined plane tiles.
function planeTileBase(bit, weight, bitDepth, label) {
  const msb = bitDepth - 1;
  return {
    label,
    side: `×${weight.toLocaleString()}`,
    tag: bitDepth > 1 ? (bit === msb ? "MSB" : bit === 0 ? "LSB" : "") : "",
    highlight: bit === msb,
    title: `${label} — weight 2^${bit} = ${weight.toLocaleString()}`,
  };
}

function channelSection(title, hint, dotColor) {
  const section = document.createElement("section");
  section.className = "channel card";
  section.innerHTML = `<div class="channel-head"><span class="channel-dot"></span><h3></h3><span class="hint"></span></div><div class="plane-grid"></div>`;
  section.querySelector(".channel-dot").style.background = dotColor;
  section.querySelector("h3").textContent = title;
  section.querySelector(".hint").textContent = hint;
  return section;
}

// The 8 colours a combined plane can contain, keyed by which channels have the bit set.
const RGB_BIT_COLOURS = [
  ["none", "#000"], ["R", "#f00"], ["G", "#0f0"], ["B", "#00f"],
  ["R+G", "#ff0"], ["R+B", "#f0f"], ["G+B", "#0ff"], ["R+G+B", "#fff"],
];

function colourLegend() {
  const legend = document.createElement("span");
  legend.className = "legend rgb-legend";
  for (const [name, colour] of RGB_BIT_COLOURS) {
    const item = document.createElement("span");
    item.innerHTML = `<i class="sw"></i>`;
    item.querySelector("i").style.background = colour;
    item.append(name);
    legend.append(item);
  }
  return legend;
}

// I = 2^(n-1) B_(n-1) + 2^(n-2) B_(n-2) + … + 2^0 B_0
function renderFormula(bitDepth) {
  const term = (k) => `2<sup>${k}</sup>B<sub>${k}</sub>`;
  const top = bitDepth - 1;
  const terms = top === 0 ? [term(0)] : top === 1 ? [term(1), term(0)] : [term(top), term(top - 1), "…", term(0)];
  el.formula.innerHTML = `I = ${terms.join(" + ")}`;
}

// tile: { image, label, side, tag, highlight, stats, title, caption, download }
function renderTile(tile, ratio) {
  const node = el.planeTpl.content.firstElementChild.cloneNode(true);
  if (tile.highlight) node.classList.add("msb");

  const thumb = node.querySelector(".plane-thumb");
  thumb.style.setProperty("--ratio", ratio);
  const img = thumb.querySelector("img");
  img.src = tile.image;
  img.alt = tile.label;

  const name = node.querySelector(".plane-name");
  name.textContent = tile.label;
  if (tile.tag) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = tile.tag;
    name.append(b);
  }
  node.querySelector(".plane-weight").textContent = tile.side;
  node.querySelector(".plane-stats").textContent = tile.stats;

  thumb.addEventListener("click", () => openLightbox(tile));
  return node;
}

function openLightbox(tile) {
  el.lbTitle.textContent = tile.title;
  el.lbImg.src = tile.image;
  el.lbImg.alt = tile.label;
  el.lbCaption.textContent = tile.caption;
  el.lbDownload.href = tile.image;
  el.lbDownload.download = tile.download;
  el.lightbox.showModal();
}

function initLightbox() {
  el.lightbox.addEventListener("click", (e) => {
    if (e.target === el.lightbox || e.target.closest("[data-close]")) el.lightbox.close();
  });
}

// ---------- Analysis rendering ----------
function showAnalysisStatus(message, busy) {
  el.analysis.hidden = false;
  el.analysisBody.hidden = true;
  el.analysisSummary.textContent = "";
  el.analysisStatus.hidden = false;
  el.analysisStatus.classList.toggle("error-text", !busy);
  el.analysisStatus.querySelector(".spinner").hidden = !busy;
  el.analysisStatus.lastChild.textContent = message;
}

function renderAnalysis(data) {
  analysisData = data;
  const first = data.reconstructions[0];
  const multi = data.channels.length > 1;
  const top = `B${data.bit_depth - 1}`;
  const quality = first.psnr.mid == null
    ? "lossless"
    : `PSNR ${first.psnr.mid.toFixed(1)} dB, SSIM ${fmt(first.ssim.mid)} (midpoint fill)`;
  el.analysisSummary.textContent =
    `${top} alone${multi ? " (all channels)" : ""}: ${first.bits_per_pixel} bpp raw, ` +
    `≈ ${first.coded_bpp.toFixed(2)} bpp with context coding · ${quality}.`;

  el.analysisStatus.hidden = true;
  el.analysisBody.hidden = false;
  el.kControl.hidden = false;
  renderReconstructions(data);
  renderEntropy(data);
  renderCorrelation(data);
}

function renderReconstructions(data) {
  const ratio = `${data.width} / ${data.height}`;
  const stem = fileStem(data.filename);
  const n = data.bit_depth;
  el.recGrid.replaceChildren(
    ...data.reconstructions.map((r) => {
      const lossless = r.psnr.mid == null;
      const bitsText = r.k === 1 ? `B${r.bits[0]}` : `B${r.bits[0]}…B${r.bits.at(-1)}`;
      const quality = lossless ? "lossless" : `PSNR ${r.psnr.mid.toFixed(1)} dB · SSIM ${fmt(r.ssim.mid)}`;
      const stats = `${bitsText} · ${quality}`;
      return renderTile({
        image: r.image,
        label: `Top ${r.k}`,
        side: `${r.bits_per_pixel} bpp`,
        tag: lossless ? "full" : r.k === 1 ? "MSB" : "",
        highlight: r.k === 1,
        stats,
        title: `Top ${r.k} plane${r.k > 1 ? "s" : ""} (${bitsText}) — midpoint fill`,
        caption: lossless
          ? "All planes kept: identical to the original."
          : `${quality} · MAE ${METRICS.mae.format(r.mae.mid)} levels. Missing ${n - r.k} bit${n - r.k > 1 ? "s" : ""} ` +
            `filled with ${((2 ** (n - r.k) - 1) / 2).toLocaleString()}. Raw ${r.bits_per_pixel} bpp, ` +
            `≈ ${r.coded_bpp.toFixed(2)} bpp with context coding.`,
        download: `${stem}_top${r.k}_mid.png`,
      }, ratio);
    }),
  );

  const perChannel = data.channels.length > 1 ? ` (${data.channels.length} channels)` : "";
  // One column group per metric, one column per fill (same order as the chart legend).
  const columns = [
    { metric: "psnr", title: "PSNR", format: (v) => (v == null ? "∞" : v.toFixed(2)) },
    { metric: "ssim", title: "SSIM", format: (v) => fmt(v, 4) },
    { metric: "mae", title: "MAE", format: (v) => v.toFixed(2) },
  ];
  const cells = (render) => columns.flatMap((c) =>
    FILL_SERIES.map((s, i) => `<${render.tag}${i === 0 ? ' class="group-start"' : ""}>${render.body(c, s)}</${render.tag}>`)).join("");
  const head = `<thead><tr><th>k</th><th>Planes</th><th>Raw bpp${perChannel}</th><th>≈ Coded bpp</th>` +
    `${cells({ tag: "th", body: (c, s) => `${c.title} ${s.fill}` })}</tr></thead>`;
  const rows = data.reconstructions.map((r) => {
    const bits = r.k === 1 ? `B${r.bits[0]}` : `B${r.bits[0]}–B${r.bits.at(-1)}`;
    return `<tr><td>${r.k}</td><td>${bits}</td><td>${r.bits_per_pixel}</td><td>${r.coded_bpp.toFixed(3)}</td>` +
      `${cells({ tag: "td", body: (c, s) => c.format(r[c.metric][s.fill]) })}</tr>`;
  });
  el.recTable.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;

  el.recLegend.replaceChildren(...FILL_SERIES.map((s) => legendItem(s.name, s.color, s.dashed)));
  drawRecChart();
}

function drawRecChart() {
  if (!analysisData) return;
  const metric = METRICS[recMetric];
  const recs = analysisData.reconstructions;
  const series = FILL_SERIES.map((s) => ({
    name: s.name, color: s.color, dashed: s.dashed,
    values: recs.map((r) => r[recMetric][s.fill]),
  }));
  const all = series.flatMap((s) => s.values).filter((v) => v != null);
  let yMin = 0;
  let yMax = recMetric === "ssim" ? 1 : niceCeil(Math.max(...all, 1));
  if (recMetric === "psnr") {
    yMin = Math.max(0, Math.floor(Math.min(...all) / 10) * 10);
    yMax = Math.max(yMin + 10, Math.ceil(Math.max(...all) / 10) * 10);
  }
  const lossless = recs.at(-1).psnr.mid == null && recMetric === "psnr";
  mountChart(el.recChart, {
    xs: recs.map((r) => r.k),
    xLabel: (k) => String(k),
    xTitle: "k (most-significant planes kept)" + (lossless ? " · all planes = ∞ dB" : ""),
    yTitle: metric.title,
    yMin, yMax,
    yFormat: recMetric === "ssim" ? (v) => v.toFixed(1) : (v) => String(Math.round(v)),
    pointLabel: (s, i, v) => `${s.name}, k = ${recs[i].k}: ${metric.format(v)}`,
    series,
  });
}

function initRecMetric() {
  el.recMetric.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-metric]");
    if (!btn) return;
    recMetric = btn.dataset.metric;
    for (const b of el.recMetric.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === btn));
    drawRecChart();
  });
}

function renderEntropy(data) {
  const n = data.bit_depth;
  const bits = Array.from({ length: n }, (_, i) => n - 1 - i); // MSB → LSB
  const byChannel = data.channels.map((ch, c) => ({
    ch, planes: bits.map((b) => data.planes.find((p) => p.channel_index === c && p.bit === b)),
  }));
  const multi = data.channels.length > 1;

  const series = byChannel.flatMap(({ ch, planes }) => [
    { name: `${multi ? ch.name + " " : ""}H | ctx`, color: CHANNEL_COLORS[ch.name], dashed: false, values: planes.map((p) => p.context_entropy) },
    { name: `${multi ? ch.name + " " : ""}H i.i.d.`, color: CHANNEL_COLORS[ch.name], dashed: true, values: planes.map((p) => p.entropy) },
  ]);
  mountChart(el.entropyChart, {
    xs: bits,
    xLabel: (b) => `B${b}`,
    xTitle: "bit-plane (MSB → LSB)",
    yTitle: "bits / pixel",
    yMin: 0, yMax: 1,
    yFormat: (v) => v.toFixed(1),
    pointLabel: (s, i, v) => `${s.name}, B${bits[i]}: ${v.toFixed(3)} b/px`,
    series,
  });
  el.entropyLegend.replaceChildren(
    ...(multi
      ? [...data.channels.map((ch) => legendItem(ch.label, CHANNEL_COLORS[ch.name], false, true)),
        legendItem("H | ctx", "var(--muted)", false), legendItem("H i.i.d.", "var(--muted)", true)]
      : [legendItem("H | ctx", CHANNEL_COLORS.L, false), legendItem("H i.i.d.", CHANNEL_COLORS.L, true)]),
  );

  const groupHead = multi
    ? `<tr><th></th>${data.channels.map((ch) => `<th colspan="4" class="group-start"><span class="ch-name"><span class="channel-dot" style="background:${CHANNEL_COLORS[ch.name]}"></span>${ch.label}</span></th>`).join("")}</tr>`
    : "";
  const colHead = `<tr><th>Plane</th>${data.channels.map(() => `<th class="group-start">ones</th><th>H</th><th>H | ctx</th><th>ρ↔</th>`).join("")}</tr>`;
  const rows = bits.map((b, i) =>
    `<tr><td>B${b}</td>${byChannel.map(({ planes }) => {
      const p = planes[i];
      return `<td class="group-start">${pct(p.ones_ratio)}</td><td>${p.entropy.toFixed(3)}</td>` +
        `<td>${p.context_entropy.toFixed(3)}</td><td>${fmt(p.spatial_correlation)}</td>`;
    }).join("")}</tr>`);
  const totals = `<tr><td><strong>Σ</strong></td>${byChannel.map(({ planes }) =>
    `<td class="group-start"></td><td><strong>${sum(planes.map((p) => p.entropy)).toFixed(2)}</strong></td>` +
    `<td><strong>${sum(planes.map((p) => p.context_entropy)).toFixed(2)}</strong></td><td></td>`).join("")}</tr>`;
  el.entropyTable.innerHTML = `<thead>${groupHead}${colHead}</thead><tbody>${rows.join("")}${totals}</tbody>`;
}

const sum = (xs) => xs.reduce((a, b) => a + b, 0);

function renderCorrelation(data) {
  const { labels, matrix } = data.correlation;
  const size = labels.length;
  const n = data.bit_depth;
  const multi = data.channels.length > 1;
  const showTicks = size <= 24;
  const margin = (multi ? 16 : 0) + (showTicks ? 26 : 8);
  const cell = 18;
  const dim = margin + size * cell;

  const svg = svgEl("svg", { viewBox: `0 0 ${dim} ${dim}`, role: "img", "aria-label": "Plane correlation heatmap" });
  svg.style.maxWidth = `${Math.max(dim * 1.5, 280)}px`; // don't blow an 8×8 matrix up to full width
  if (multi) {
    data.channels.forEach((ch, c) => {
      const start = margin + c * n * cell;
      const mid = start + (n * cell) / 2;
      const colour = CHANNEL_COLORS[ch.name];
      svgEl("rect", { x: start + 1, y: 2, width: n * cell - 2, height: 4, rx: 2, style: `fill:${colour}` }, svg);
      svgEl("rect", { x: 2, y: start + 1, width: 4, height: n * cell - 2, rx: 2, style: `fill:${colour}` }, svg);
      svgEl("text", { x: mid, y: 16, "text-anchor": "middle", class: "group tick", style: `fill:${colour}` }, svg).textContent = ch.name;
      svgEl("text", { x: 14, y: mid, "text-anchor": "middle", "dominant-baseline": "middle", class: "group tick", style: `fill:${colour}` }, svg).textContent = ch.name;
    });
  }
  if (showTicks) {
    labels.forEach((_, i) => {
      const bit = n - 1 - (i % n);
      const pos = margin + i * cell + cell / 2;
      svgEl("text", { x: pos, y: margin - 6, "text-anchor": "middle", class: "tick" }, svg).textContent = bit;
      svgEl("text", { x: margin - 6, y: pos, "text-anchor": "end", "dominant-baseline": "middle", class: "tick" }, svg).textContent = bit;
    });
  }

  svgEl("rect", { x: margin, y: margin, width: size * cell, height: size * cell, class: "cell-bg" }, svg);
  const cells = svgEl("g", {}, svg);
  matrix.forEach((row, i) => row.forEach((v, j) => {
    const attrs = { x: margin + j * cell + 0.5, y: margin + i * cell + 0.5, width: cell - 1, height: cell - 1, "data-i": i, "data-j": j };
    if (v != null) {
      attrs.class = `cell ${v >= 0 ? "pos" : "neg"}`;
      attrs["fill-opacity"] = Math.abs(v).toFixed(3);
    } else {
      attrs.class = "cell";
      attrs["fill-opacity"] = 0;
    }
    const rect = svgEl("rect", attrs, cells);
    svgEl("title", {}, rect).textContent = corrText(labels, matrix, i, j);
  }));

  let active = null;
  const select = (target) => {
    if (!(target instanceof SVGRectElement) || target.dataset.i == null) return;
    active?.classList.remove("active");
    active = target;
    active.classList.add("active");
    el.corrReadout.textContent = corrText(labels, matrix, +target.dataset.i, +target.dataset.j);
  };
  cells.addEventListener("pointerover", (e) => select(e.target));
  cells.addEventListener("click", (e) => select(e.target));

  el.corrChart.replaceChildren(svg);
  el.corrReadout.innerHTML = "&nbsp;";
}

function corrText(labels, matrix, i, j) {
  const v = matrix[i][j];
  return `${labels[i]} × ${labels[j]}: ${v == null ? "undefined (constant plane)" : `ρ = ${v.toFixed(3)}`}`;
}

// ---------- SVG charts ----------
const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent?.append(node);
  return node;
}

function niceCeil(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((c) => c >= v);
}

function legendItem(name, color, dashed, dot = false) {
  const item = document.createElement("span");
  const svg = svgEl("svg", { width: dot ? 10 : 22, height: 10, "aria-hidden": "true" });
  if (dot) svgEl("circle", { cx: 5, cy: 5, r: 4.5, style: `fill:${color}` }, svg);
  else svgEl("line", { x1: 1, x2: 21, y1: 5, y2: 5, class: "series", style: `stroke:${color};stroke-width:2;${dashed ? "stroke-dasharray:5 3" : ""}` }, svg);
  item.append(svg, name);
  return item;
}

// Charts are drawn at the container's pixel width so text stays readable, and redrawn on resize.
const chartSpecs = new Map();
const chartObserver = new ResizeObserver((entries) => {
  for (const { target } of entries) {
    const spec = chartSpecs.get(target);
    if (spec && target.clientWidth !== spec.drawnWidth) drawLineChart(target, spec);
  }
});

function mountChart(container, spec) {
  chartSpecs.set(container, spec);
  chartObserver.observe(container);
  drawLineChart(container, spec);
}

// spec: { xs, xLabel, xTitle, yTitle, yMin, yMax, yFormat, pointLabel, series: [{ name, color, dashed, values }] }
function drawLineChart(container, spec) {
  const width = container.clientWidth;
  spec.drawnWidth = width;
  if (!width) return; // hidden; the observer redraws once it has a size
  const height = 230;
  const m = { top: 10, right: 14, bottom: 42, left: 48 };
  const w = width - m.left - m.right;
  const h = height - m.top - m.bottom;
  const count = spec.xs.length;
  const x = (i) => m.left + (count === 1 ? w / 2 : (i / (count - 1)) * w);
  const y = (v) => m.top + h - ((v - spec.yMin) / (spec.yMax - spec.yMin)) * h;

  const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": spec.yTitle });
  for (let t = 0; t <= 5; t++) {
    const v = spec.yMin + ((spec.yMax - spec.yMin) * t) / 5;
    svgEl("line", { x1: m.left, x2: m.left + w, y1: y(v), y2: y(v), class: t ? "grid" : "axis" }, svg);
    svgEl("text", { x: m.left - 7, y: y(v), "text-anchor": "end", "dominant-baseline": "middle", class: "tick" }, svg)
      .textContent = spec.yFormat(v);
  }
  const step = Math.max(1, Math.ceil(count / Math.max(1, Math.floor(w / 30))));
  spec.xs.forEach((xv, i) => {
    if (i % step && i !== count - 1) return;
    svgEl("text", { x: x(i), y: m.top + h + 16, "text-anchor": "middle", class: "tick" }, svg).textContent = spec.xLabel(xv);
  });
  svgEl("text", { x: m.left + w / 2, y: height - 6, "text-anchor": "middle", class: "title" }, svg).textContent = spec.xTitle;
  svgEl("text", { x: 12, y: m.top + h / 2, "text-anchor": "middle", class: "title", transform: `rotate(-90 12 ${m.top + h / 2})` }, svg)
    .textContent = spec.yTitle;

  const clamp = (v) => Math.min(spec.yMax, Math.max(spec.yMin, v));
  for (const s of spec.series) {
    let d = "";
    let pen = false;
    s.values.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(clamp(v)).toFixed(1)}`;
      pen = true;
    });
    svgEl("path", { d, class: `series${s.dashed ? " dashed" : ""}`, style: `stroke:${s.color}` }, svg);
    s.values.forEach((v, i) => {
      if (v == null) return;
      const dot = svgEl("circle", { cx: x(i), cy: y(clamp(v)), r: s.dashed ? 2.5 : 3.5, class: "dot", style: `fill:${s.color}` }, svg);
      svgEl("title", {}, dot).textContent = spec.pointLabel(s, i, v);
    });
  }
  container.replaceChildren(svg);
}

// ---------- Boot ----------
initTheme();
initDropzone();
initLightbox();
initRecMetric();
initModelsView();
initModeControls();
el.sliceBtn.addEventListener("click", slice);
el.clearBtn.addEventListener("click", clearAll);
