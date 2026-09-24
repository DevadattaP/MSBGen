// MSBGen front-end: pick an image, send it to /api/bitplanes, render the planes.

const API_BITPLANES = "/api/bitplanes";
const CHANNEL_COLORS = { R: "var(--ch-r)", G: "var(--ch-g)", B: "var(--ch-b)", L: "var(--ch-gray)" };

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
  lightbox: $("#lightbox"),
  lbTitle: $("#lightbox-title"),
  lbImg: $("#lightbox-img"),
  lbCaption: $("#lightbox-caption"),
  lbDownload: $("#lightbox-download"),
  themeToggle: $("#theme-toggle"),
};

let currentFile = null;
let previewUrl = null;

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
  el.sliceBtn.querySelector(".label").textContent = on ? "Slicing…" : "Slice bit-planes";
}

// ---------- File selection ----------
function selectFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    showError(`“${file.name}” is not an image.`);
    return;
  }
  currentFile = file;
  showError("");
  showNotes([]);
  el.results.hidden = true;

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

  el.sliceBtn.disabled = false;
  el.clearBtn.disabled = false;
}

function clearAll() {
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
  el.results.hidden = true;
  el.channels.replaceChildren();
  el.sliceBtn.disabled = true;
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
async function slice() {
  if (!currentFile) return;
  showError("");
  setLoading(true);
  try {
    const body = new FormData();
    body.append("file", currentFile, currentFile.name || "image.png");
    const res = await fetch(API_BITPLANES, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    renderResults(data);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setLoading(false);
  }
}

// ---------- Rendering ----------
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

  const ctx = { ratio: `${data.width} / ${data.height}`, bitDepth: data.bit_depth };
  const stem = (data.filename || "image").replace(/\.[^.]+$/, "");
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
        image: p.image, bit: p.bit, weight: p.weight, label, stats,
        caption: `${stats}. White = 1, black = 0.`,
        download: `${stem}_${p.channel}_B${p.bit}.png`,
      }, ctx));
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
        image: c.image, bit: c.bit, weight: c.weight, label: `RGB · B${c.bit}`, stats,
        caption: `${stats}. Each channel is 255 where its bit ${c.bit} is 1, else 0 — ` +
          "e.g. yellow = R and G set, white = all three, black = none.",
        download: `${stem}_RGB_B${c.bit}.png`,
      }, ctx));
    }
    sections.push(section);
  }

  el.channels.replaceChildren(...sections);
  el.results.hidden = false;
  el.results.scrollIntoView({ behavior: "smooth", block: "start" });
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;

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

// tile: { image, bit, weight, label, stats, caption, download }
function renderTile(tile, { ratio, bitDepth }) {
  const node = el.planeTpl.content.firstElementChild.cloneNode(true);
  const msb = bitDepth - 1;
  const tag = bitDepth > 1 ? (tile.bit === msb ? "MSB" : tile.bit === 0 ? "LSB" : "") : "";
  if (tile.bit === msb) node.classList.add("msb");

  const thumb = node.querySelector(".plane-thumb");
  thumb.style.setProperty("--ratio", ratio);
  const img = thumb.querySelector("img");
  img.src = tile.image;
  img.alt = `Bit-plane ${tile.label}`;

  const name = node.querySelector(".plane-name");
  name.textContent = tile.label;
  if (tag) {
    const b = document.createElement("span");
    b.className = "badge";
    b.textContent = tag;
    name.append(b);
  }
  node.querySelector(".plane-weight").textContent = `×${tile.weight.toLocaleString()}`;
  node.querySelector(".plane-stats").textContent = tile.stats;

  thumb.addEventListener("click", () => openLightbox(tile));
  return node;
}

function openLightbox(tile) {
  el.lbTitle.textContent = `${tile.label} — weight 2^${tile.bit} = ${tile.weight.toLocaleString()}`;
  el.lbImg.src = tile.image;
  el.lbImg.alt = `Bit-plane ${tile.label}`;
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

// ---------- Boot ----------
initTheme();
initDropzone();
initLightbox();
el.sliceBtn.addEventListener("click", slice);
el.clearBtn.addEventListener("click", clearAll);
