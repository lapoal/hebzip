// The page: wiring between the text box, the engine (worker), the built-in comparison and the recorded table.
import { renderBench } from "./bench.js";
import { builtIn } from "./compare.js";
import { Engine } from "./engine.js";
import { issueUrl, shareText } from "./share.js";
import { choosePack, decodeUtf8, utf8, words } from "./text.js";

const $ = (id) => document.getElementById(id);
const MB = (n) => (n / 1e6).toFixed(1);
const PACK_NAME = { fast: "Hebrew pack (mode fast)", en_fast: "English pack (mode en_fast)" };
const ERRORS = {
  CorruptPackage: "This is not a hebzip package, or it is damaged.",
  DamagedStream: "This package is damaged.",
  ChecksumMismatch: "This package is damaged (its checksum does not match).",
  UnsupportedFormat: "This package was made by a newer version of hebzip.",
  TableMismatch: "This package was made with other tables or settings (another version of hebzip), so it cannot be opened here.",
  InputError: "The text is not valid UTF-8.",
};
const state = { file: null, settings: null, manifest: null, engine: null, ready: new Set(), rate: {}, last: null, busy: false };

function notice(text, kind = "warn") {
  const box = $("notice");
  box.className = `flash ${kind}`;
  box.textContent = text;
  box.hidden = !text;
}

// The tables' own size (the prior knowledge a pack brings) and what the browser downloads (some are gzipped).
function packBytes(mode) {
  return state.manifest.packs[mode].reduce((n, f) => n + (state.manifest.files[f].raw_bytes || state.manifest.files[f].bytes), 0);
}

function downloadBytes(mode) {
  return state.manifest.packs[mode].reduce((n, f) => n + state.manifest.files[f].bytes, 0);
}

function showProgress(what, loaded, total) {
  const bar = $(`bar-${what}`);
  if (!bar) return;
  bar.max = total;
  bar.value = loaded;
  $(`txt-${what}`).textContent = `${MB(loaded)} / ${MB(total)} MB`;
  if (state.waitingFor === what) $("txt-work").textContent = `Downloading the English pack: ${MB(loaded)} / ${MB(total)} MB`;
}

function metered() {
  const c = navigator.connection;
  return Boolean(c && (c.saveData || c.type === "cellular"));
}

function onEngine(msg) {
  if (msg.type === "progress") showProgress(msg.what, msg.loaded, msg.total);
  else if (msg.type === "ready") onReady(msg);
  else if (msg.type === "pack-ready") onPackReady(msg);
  else if (msg.type === "error") {
    notice(`The engine could not start (${msg.name}). Reload the page; if it happens again, please report it.`, "error");
    $(`txt-${msg.what === "prefetch" ? "en_fast" : "fast"}`).textContent = "failed";
  }
}

function onReady(msg) {
  state.ready.add("fast");
  $("bar-fast").value = $("bar-fast").max;
  $("txt-fast").textContent = `ready (${msg.seconds.toFixed(1)} s, ${Math.round(msg.heap_mb)} MB memory)`;
  $("go").disabled = false;
  if (metered()) {
    $("txt-en_fast").textContent = `${MB(downloadBytes("en_fast"))} MB download - only when you need it (your connection asks to save data)`;
  } else {
    $("txt-en_fast").textContent = "downloading in the background";
    state.engine.prefetch("en_fast");
  }
}

function onPackReady(msg) {
  const mode = msg.mode;
  state.ready.add(mode);
  const bar = $(`bar-${mode}`);
  if (bar) { bar.value = bar.max; $(`txt-${mode}`).textContent = `ready (${Math.round(msg.heap_mb)} MB memory in all)`; }
}

// A .txt file is compressed as its exact bytes (a text box would turn CR LF into LF); typing drops the file.
function describeInput() {
  const text = $("text").value;
  const bytes = state.file ? state.file.length : utf8(text).length;
  const cap = state.settings.max_input_bytes;
  $("counter").textContent = `${bytes.toLocaleString("en-US")} / ${cap.toLocaleString("en-US")} bytes, ${words(text)} words`;
  $("text").classList.toggle("over", bytes > cap);
  const pick = choosePack(text);
  $("pack").textContent = text.trim() ? PACK_NAME[pick.mode] : "";
  if (text.trim() && pick.mixed) {
    notice("This text mixes Hebrew and Latin letters. It is compressed with the pack of the majority script; the other script is still kept exactly, but compresses less well.");
  } else if (!state.busy) notice("");
  return { text, bytes, pick };
}

function estimate(mode, bytes) {
  const rate = state.rate[mode] || state.settings.default_seconds_per_kb;
  return (rate * bytes) / 1000;
}

function startWork(label, seconds) {
  $("work").hidden = false;
  const t0 = performance.now();
  $("txt-work").textContent = label;
  const timer = setInterval(() => {
    const spent = (performance.now() - t0) / 1000;
    $("bar-work").value = seconds > 0 ? Math.min(0.95, spent / seconds) : 0;
  }, 200);
  return () => { clearInterval(timer); $("bar-work").value = 1; $("work").hidden = true; };
}

async function compress() {
  const { text, bytes, pick } = describeInput();
  if (!text) return;
  if (bytes > state.settings.max_input_bytes) {
    notice(`The website takes up to ${state.settings.max_input_bytes.toLocaleString("en-US")} bytes (it runs as pure Python in your browser). For longer texts use the downloadable tool.`, "error");
    return;
  }
  setBusy(true);
  const raw = state.file || utf8(text);
  const seconds = 2 * estimate(pick.mode, raw.length);
  if (!state.ready.has(pick.mode)) state.waitingFor = pick.mode;
  const stop = startWork(state.waitingFor ? "Waiting for the English pack..." : `Compressing and checking - about ${Math.max(1, Math.round(seconds))} s (estimate for this browser)`, seconds);
  try {
    const packed = await state.engine.run("compress", raw, pick.mode);
    state.waitingFor = null;
    $("txt-work").textContent = "Decompressing to check...";
    const back = await state.engine.run("decompress", packed.data);
    const verified = back.data.length === raw.length && back.data.every((b, i) => b === raw[i]);
    state.rate[pick.mode] = (1000 * packed.seconds) / raw.length;
    const others = await builtIn(raw);
    showResult({ mode: pick.mode, raw, text, packed, back, verified, others });
  } catch (error) {
    notice(ERRORS[error.name] || `Something went wrong (${error.name || "error"}). Please report it with your browser name.`, "error");
  } finally {
    state.waitingFor = null;
    stop();
    setBusy(false);
  }
}

function setBusy(busy) {
  state.busy = busy;
  for (const id of ["go", "sample-he", "sample-en", "clear"]) $(id).disabled = busy || (id === "go" && !state.ready.has("fast"));
}

function row(cells, cls = "") {
  const tr = document.createElement("tr");
  tr.className = cls;
  cells.forEach((c, i) => {
    const cell = document.createElement(i === 0 ? "th" : "td");
    if (i === 0) cell.scope = "row"; else cell.className = "n";
    cell.textContent = c;
    tr.append(cell);
  });
  return tr;
}

function showResult(r) {
  const n = r.raw.length;
  const saved = (b) => `${(100 * (1 - b / n)).toFixed(1)}%`;
  const rows = [row(["original text (UTF-8)", n.toLocaleString("en-US"), "", ""], "muted-row"),
    row([`hebzip ${r.mode} - ${MB(packBytes(r.mode))} MB of tables shipped`, r.packed.data.length.toLocaleString("en-US"), saved(r.packed.data.length),
      `${r.packed.seconds.toFixed(2)} s + ${r.back.seconds.toFixed(2)} s back`], "ours"),
    ...r.others.map((o) => row([`${o.label} - no tables`, o.bytes.toLocaleString("en-US"), saved(o.bytes), `${o.seconds.toFixed(3)} s`]))];
  $("rows").replaceChildren(...rows);
  $("verdict").replaceChildren();
  const ok = document.createElement("span");
  ok.className = r.verified ? "ok" : "bad";
  ok.textContent = r.verified ? "Verified: the package was decompressed here and is byte for byte your text. " : "NOT identical after decompressing - please report this. ";
  $("verdict").append(ok, "The time is pure Python in WebAssembly; the native tool is far faster, and zstd-class compressors faster still (see the recorded table).");
  const url = URL.createObjectURL(new Blob([r.packed.data], { type: "application/octet-stream" }));
  if (state.last && state.last.url) URL.revokeObjectURL(state.last.url);
  $("download").href = url;
  state.last = { url, share: { mode: r.mode, priorBytes: packBytes(r.mode), rawBytes: n, words: words(r.text),
    packageBytes: r.packed.data.length, savedPct: 100 * (1 - r.packed.data.length / n), builtIn: r.others,
    compressS: r.packed.seconds, decompressS: r.back.seconds, verified: r.verified } };
  $("result").hidden = false;
}

async function openPackage(bytes) {
  setBusy(true);
  const stop = startWork("Opening the package...", estimate("fast", bytes.length * 6));
  try {
    const back = await state.engine.run("decompress", bytes);
    if (back.error === "OtherMode") {
      notice(`This package was made with mode ${back.mode}, which runs only in the downloadable tool.`, "warn");
      return;
    }
    $("opened-facts").textContent = `${PACK_NAME[back.mode]}: ${bytes.length.toLocaleString("en-US")} bytes -> ${back.data.length.toLocaleString("en-US")} bytes of text, ${back.seconds.toFixed(2)} s`;
    $("opened-text").textContent = new TextDecoder("utf-8").decode(back.data);
    if (state.savedUrl) URL.revokeObjectURL(state.savedUrl);
    state.savedUrl = URL.createObjectURL(new Blob([back.data], { type: "text/plain;charset=utf-8" }));
    $("save-text").href = state.savedUrl;
    $("opened").hidden = false;
  } catch (error) {
    notice(ERRORS[error.name] || `This file could not be opened (${error.name || "error"}).`, "error");
  } finally {
    stop();
    setBusy(false);
  }
}

async function openFile(file) {
  if (!file) return;
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (/\.hz$/i.test(file.name)) { await openPackage(bytes); return; }
  try {
    $("text").value = decodeUtf8(bytes);
    state.file = bytes;
    describeInput();
  } catch {
    notice("This file is not UTF-8 text. Save it as UTF-8 (.txt) and try again.", "error");
  }
}

async function sample(lang) {
  const response = await fetch(`samples/${lang}.txt`);
  state.file = new Uint8Array(await response.arrayBuffer());
  $("text").value = new TextDecoder("utf-8").decode(state.file);
  describeInput();
}

function share() {
  if (!state.last) return;
  const text = shareText(state.last.share, { userAgent: navigator.userAgent, build: state.manifest.files["py/hebzip.zip"].sha256.slice(0, 12) });
  $("share-preview").textContent = `Title: ${text.title}\n\n${text.body}`;
  $("share-open").href = issueUrl(state.settings.repo, text);
  $("share-dialog").showModal();
}

function wire() {
  $("text").addEventListener("input", () => { state.file = null; describeInput(); });
  $("go").addEventListener("click", compress);
  $("sample-he").addEventListener("click", () => sample("he"));
  $("sample-en").addEventListener("click", () => sample("en"));
  $("clear").addEventListener("click", () => { $("text").value = ""; state.file = null; describeInput(); $("result").hidden = true; $("opened").hidden = true; });
  $("file").addEventListener("change", (e) => openFile(e.target.files[0]));
  $("share").addEventListener("click", share);
  $("share-cancel").addEventListener("click", () => $("share-dialog").close());
  $("share-open").addEventListener("click", () => $("share-dialog").close());
  const box = $("text");
  box.addEventListener("dragover", (e) => { e.preventDefault(); box.classList.add("drop-over"); });
  box.addEventListener("dragleave", () => box.classList.remove("drop-over"));
  box.addEventListener("drop", (e) => { e.preventDefault(); box.classList.remove("drop-over"); openFile(e.dataTransfer.files[0]); });
}

async function main() {
  wire();
  const [settings, manifest] = await Promise.all([fetch("settings.json").then((r) => r.json()), fetch("build.json").then((r) => r.json())]);
  Object.assign(state, { settings, manifest });
  const repo = `https://github.com/${settings.repo}`;
  $("link-repo").href = repo;
  $("link-method").href = `${repo}#how-it-works`;
  $("mb-he").textContent = MB(packBytes("fast"));
  $("mb-en").textContent = MB(packBytes("en_fast"));
  $("build-line").textContent = `Site build ${manifest.files["py/hebzip.zip"].sha256.slice(0, 12)} - Pyodide ${manifest.pyodide} - first visit downloads ${MB(manifest.total_bytes)} MB in all (engine + both packs).`;
  const core = Object.keys(manifest.files).filter((f) => f.startsWith("pyodide/") || f.startsWith("py/"));
  const first = core.reduce((n, f) => n + manifest.files[f].bytes, 0) + downloadBytes("fast");
  $("txt-fast").textContent = `starting (${MB(first)} MB download)`;
  $("txt-en_fast").textContent = `after the Hebrew pack (${MB(downloadBytes("en_fast"))} MB download)`;
  describeInput();
  state.engine = new Engine(onEngine);
  state.engine.boot(manifest);
  fetch("data/bench.json").then((r) => r.json()).then((b) => renderBench($("bench"), b));
}

main().catch(() => notice("The page could not load its own files. Reload the page.", "error"));
