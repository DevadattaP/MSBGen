// MSBGen front-end: pick an image, send it to /api/bitplanes and /api/analysis, render the results.

const API_BITPLANES = "/api/bitplanes";
const API_ANALYSIS = "/api/analysis";
const API_MODELS = "/api/models";
const API_RECONSTRUCT = "/api/reconstruct";
const CHANNEL_COLORS = { R: "var(--ch-r)", G: "var(--ch-g)", B: "var(--ch-b)", L: "var(--ch-gray)" };
const FILL_SERIES = [
  { fill: "mid", name: "Midpoint fill", color: "var(--series-mid)", dashed: false },
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
let analysisData = null;
let recMetric = "psnr";
let availableModels = null; // from /api/models, fetched once at start-up
let modelTiles = []; // { tile, img, image, error } for the reconstruction / error toggle

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
  el.sliceBtn.querySelector(".label").textContent = on ? "Slicing…" : "Slice & analyse";
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
async function postImage(url, file) {
  const body = new FormData();
  body.append("file", file, file.name || "image.png");
  const res = await fetch(url, { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

async function slice() {
  if (!currentFile) return;
  const id = ++requestId;
  const file = currentFile;
  showError("");
  hideResults();
  setLoading(true);

  // All requests start together; the planes usually arrive first and render straight away.
  const analysis = postImage(API_ANALYSIS, file);
  analysis.catch(() => {}); // handled below; avoids an unhandled-rejection warning if slicing fails
  const models = await modelsPromise;
  const reconstruction = models.length ? postImage(API_RECONSTRUCT, file) : null;
  reconstruction?.catch(() => {});
  try {
    const data = await postImage(API_BITPLANES, file);
    if (id !== requestId) return;
    renderResults(data);
  } catch (err) {
    if (id === requestId) showError(err.message || String(err));
    return;
  } finally {
    if (id === requestId) setLoading(false);
  }

  showAnalysisStatus("Analysing bit-planes…", true);
  if (reconstruction) showModelsStatus("Running models…", true);
  else showNoModels();
  await Promise.all([
    analysis.then(
      (data) => id === requestId && renderAnalysis(data),
      (err) => id === requestId && showAnalysisStatus(`Analysis failed: ${err.message || err}`, false),
    ),
    reconstruction?.then(
      (data) => id === requestId && renderModels(data),
      (err) => id === requestId && showModelsStatus(`Model reconstruction unavailable: ${err.message || err}`, false),
    ),
  ]);
}

// ---------- Models ----------
const modelsPromise = fetch(API_MODELS)
  .then((res) => (res.ok ? res.json() : { models: [] }))
  .then((data) => (availableModels = data).models)
  .catch(() => []);

function showModelsStatus(message, busy) {
  el.models.hidden = false;
  el.modelsBody.hidden = true;
  el.modelsSummary.textContent = "";
  el.modelsStatus.hidden = false;
  el.modelsStatus.classList.toggle("error-text", !busy);
  el.modelsStatus.querySelector(".spinner").hidden = !busy;
  el.modelsStatus.querySelector(".status-text").textContent = message;
}

function showNoModels() {
  showModelsStatus("", false);
  el.modelsStatus.classList.remove("error-text");
  const reason = availableModels?.torch_available === false
    ? "PyTorch is not installed, so trained models can't run. "
    : `No trained models found in ${availableModels?.models_dir ?? "runs"}/. `;
  el.modelsStatus.querySelector(".status-text").innerHTML =
    `${escapeHtml(reason)}Train one with <code>python -m msbgen.cli.train --preset unet-res</code>.`;
}

function renderModels(data) {
  el.modelsStatus.hidden = true;
  el.modelsBody.hidden = false;
  const ratio = `${data.width} / ${data.height}`;
  const stem = fileStem(data.filename);
  const n = 8;
  const bits = data.k === 1 ? `B${n - 1}` : `B${n - 1}…B${n - data.k}`;
  const models = data.results.filter((r) => r.kind === "model");
  const scored = data.results.filter((r) => r.psnr != null);
  const best = scored.reduce((a, b) => (b.psnr > a.psnr ? b : a), scored[0]);
  const top = models.reduce((a, b) => (b.psnr > (a?.psnr ?? -Infinity) ? b : a), null);
  el.modelsSummary.textContent = top
    ? `Best model: ${top.label} (${top.name}) · PSNR ${top.psnr.toFixed(2)} dB, ${gainText(top.psnr_gain)} vs midpoint fill.`
    : "";
  modelTiles = [];
  el.modelsView.querySelector('[data-view="image"]').click(); // start on the reconstructions
  const original = { name: "original", label: "Original", image: data.original, error_map: null };
  const tiles = [original, ...data.results].map((r) => {
    const isOriginal = r === original;
    const stats = isOriginal ? `grayscale, ${data.width} × ${data.height}` :
      `PSNR ${fmt(r.psnr, 2)} dB${r.kind === "model" ? ` (${gainText(r.psnr_gain)})` : ""} · SSIM ${fmt(r.ssim)}`;
    const tile = {
      image: r.image,
      label: r.label,
      side: isOriginal ? "reference" : r.kind === "model" ? r.name : `from ${bits}`,
      tag: r === best ? "best" : "",
      highlight: false,
      stats,
      title: isOriginal ? "Original (grayscale)" : `${r.label} — from ${bits} only`,
      caption: isOriginal ? "The image every method is compared against." :
        `${stats} · MAE ${r.mae.toFixed(2)} · MSB consistency ${pct(r.msb_consistency)}.`,
      download: `${stem}_${r.name}.png`,
    };
    const node = renderTile(tile, ratio);
    if (r === best) node.classList.add("best");
    modelTiles.push({ tile, img: node.querySelector("img"), image: r.image, error: r.error_map });
    return node;
  });
  el.modelsGrid.replaceChildren(...tiles);

  const rows = data.results.map((r) => {
    const gain = r.kind === "model" || r.name === "zero" ? gainText(r.psnr_gain, true) : "—";
    return `<tr><td>${escapeHtml(r.label)}${r.kind === "model" ? ` <span class="hint">${escapeHtml(r.name)}</span>` : ""}</td>` +
      `<td>${r.kind === "model" ? "model" : "baseline"}</td><td>${fmt(r.psnr, 2)}</td><td>${gain}</td>` +
      `<td>${fmt(r.ssim, 4)}</td><td>${r.mae.toFixed(2)}</td><td>${pct(r.msb_consistency)}</td></tr>`;
  });
  el.modelsTable.innerHTML = `<thead><tr><th>Method</th><th>Kind</th><th>PSNR (dB)</th><th>Δ vs midpoint</th>` +
    `<th>SSIM</th><th>MAE</th><th>MSB consistency</th></tr></thead><tbody>${rows.join("")}</tbody>`;
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
  el.results.scrollIntoView({ behavior: "smooth", block: "start" });
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
  const head = `<thead><tr><th>k</th><th>Planes</th><th>Raw bpp${perChannel}</th><th>≈ Coded bpp</th>` +
    `<th class="group-start">PSNR mid</th><th>PSNR zero</th><th class="group-start">SSIM mid</th>` +
    `<th>SSIM zero</th><th class="group-start">MAE mid</th><th>MAE zero</th></tr></thead>`;
  const rows = data.reconstructions.map((r) => {
    const psnr = (v) => (v == null ? "∞" : v.toFixed(2));
    const bits = r.k === 1 ? `B${r.bits[0]}` : `B${r.bits[0]}–B${r.bits.at(-1)}`;
    return `<tr><td>${r.k}</td><td>${bits}</td><td>${r.bits_per_pixel}</td><td>${r.coded_bpp.toFixed(3)}</td>` +
      `<td class="group-start">${psnr(r.psnr.mid)}</td><td>${psnr(r.psnr.zero)}</td>` +
      `<td class="group-start">${fmt(r.ssim.mid, 4)}</td><td>${fmt(r.ssim.zero, 4)}</td>` +
      `<td class="group-start">${r.mae.mid.toFixed(2)}</td><td>${r.mae.zero.toFixed(2)}</td></tr>`;
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
el.sliceBtn.addEventListener("click", slice);
el.clearBtn.addEventListener("click", clearAll);
