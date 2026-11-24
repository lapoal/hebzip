// The engine: Pyodide + hebzip in a Web Worker, so the page never freezes while Python works.
// Everything is fetched from this site's own folder (relative URLs only). Each file we fetch ourselves is
// checked against its sha256 in build.json (made by tools/site/build_site.py); Pyodide checks its numpy wheel
// against its own lock file. Messages carry numbers, packages and errors - never anything to log.
import { loadPyodide } from "../pyodide/pyodide.mjs";

const SITE = new URL("../", import.meta.url);
const APP = "/app";
let manifest = null;
let py = null;
let entry = null;
const packs = new Map(); // mode -> Promise (loading or loaded)

const post = (msg, transfer = []) => self.postMessage(msg, transfer);
const sizeOf = (names) => names.reduce((n, f) => n + manifest.files[f].bytes, 0);

async function readCounted(response, onBytes) {
  const reader = response.body.getReader();
  const parts = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    onBytes(value.length);
  }
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

async function hex(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

// A table shipped gzipped (build.json says "gzip") is un-gzipped here; its sha256 is that of the pinned table.
async function gunzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function fetchChecked(name, onBytes) {
  const row = manifest.files[name];
  const response = await fetch(new URL(row.file || name, SITE));
  if (!response.ok) throw new Error(`download failed: ${name} (HTTP ${response.status})`);
  const got = await readCounted(response, onBytes);
  const bytes = row.gzip ? await gunzip(got) : got;
  if ((await hex(bytes)) !== manifest.files[name].sha256) {
    throw new Error(`file changed on the server: ${name} - reload the page`);
  }
  return bytes;
}

function writeFile(name, bytes) {
  const path = `${APP}/${name.replace(/^py\//, "")}`;
  py.FS.mkdirTree(path.substring(0, path.lastIndexOf("/")));
  py.FS.writeFile(path, bytes);
}

// Pyodide fetches its own files; wrapping fetch lets the progress bar count those bytes too.
function countingFetch(onBytes) {
  const real = self.fetch.bind(self);
  return async (input, init) => {
    const response = await real(input, init);
    if (!response.ok || !response.body) return response;
    const [mine, theirs] = response.body.tee();
    (async () => { const r = mine.getReader(); for (;;) { const { done, value } = await r.read(); if (done) break; onBytes(value.length); } })();
    return new Response(theirs, { status: response.status, statusText: response.statusText, headers: response.headers });
  };
}

async function bootEngine() {
  const core = Object.keys(manifest.files).filter((f) => f.startsWith("pyodide/") || f.startsWith("py/"));
  const total = sizeOf(core) + sizeOf(manifest.packs.fast);
  let loaded = 0;
  const step = (n) => { loaded += n; post({ type: "progress", what: "fast", loaded, total }); };
  const t0 = performance.now();
  const realFetch = self.fetch;
  self.fetch = countingFetch(step);
  try {
    py = await loadPyodide({ indexURL: new URL("pyodide/", SITE).href, stdout: () => {}, stderr: () => {} });
    await py.loadPackage("numpy", { messageCallback: () => {}, errorCallback: () => {} });
  } finally {
    self.fetch = realFetch;
  }
  for (const name of ["py/hebzip.zip", "py/browser.toml", "py/entry.py"]) {
    const bytes = await fetchChecked(name, step);
    if (name.endsWith(".zip")) py.unpackArchive(bytes, "zip", { extractDir: APP });
    else writeFile(name, bytes);
  }
  py.FS.mkdirTree(`${APP}/golden`); // the config's material folder must exist (it stays empty)
  py.runPython(`import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, "${APP}")`);
  entry = py.pyimport("entry");
  entry.boot(`${APP}/browser.toml`);
  await loadTables("fast", step);
  loaded = total;
  return { seconds: (performance.now() - t0) / 1000, heap_mb: py._module.HEAP8.buffer.byteLength / 2 ** 20 };
}

async function loadTables(mode, step) {
  for (const name of manifest.packs[mode]) writeFile(name, await fetchChecked(name, step));
  const t = performance.now();
  entry.prepare(mode);
  return (performance.now() - t) / 1000;
}

function ensurePack(mode) {
  if (!packs.has(mode)) {
    const total = sizeOf(manifest.packs[mode]);
    let loaded = 0;
    const step = (n) => { loaded += n; post({ type: "progress", what: mode, loaded, total }); };
    const job = loadTables(mode, step).then((prepare_s) => {
      post({ type: "pack-ready", mode, prepare_s, heap_mb: py._module.HEAP8.buffer.byteLength / 2 ** 20 });
    });
    job.catch(() => packs.delete(mode)); // a failed download may be tried again
    packs.set(mode, job);
  }
  return packs.get(mode);
}

function call(fn, ...args) {
  const proxy = fn(...args);
  const out = proxy instanceof Uint8Array ? proxy : proxy.toJs();
  if (proxy && proxy.destroy) proxy.destroy();
  return out;
}

async function run(msg) {
  const t0 = performance.now();
  if (msg.op === "compress") {
    await ensurePack(msg.mode);
    const t = performance.now();
    const data = call(entry.compress, msg.bytes, msg.mode);
    return { data, mode: msg.mode, seconds: (performance.now() - t) / 1000, waited_s: (t - t0) / 1000 };
  }
  const named = entry.mode_of(msg.bytes);
  const mode = named.startsWith("id:") ? (manifest.all_modes[named.slice(3)] || "unknown") : named;
  if (!manifest.packs[mode]) return { error: "OtherMode", mode };
  await ensurePack(mode);
  const t = performance.now();
  const data = call(entry.decompress, msg.bytes);
  return { data, mode, seconds: (performance.now() - t) / 1000, waited_s: (t - t0) / 1000 };
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === "boot") {
      manifest = msg.manifest;
      const info = await bootEngine();
      packs.set("fast", Promise.resolve());
      post({ type: "ready", ...info });
    } else if (msg.type === "prefetch") {
      await ensurePack(msg.mode);
    } else if (msg.type === "run") {
      const out = await run(msg);
      post({ type: "result", id: msg.id, ...out }, out.data ? [out.data.buffer] : []);
    }
  } catch (error) {
    const name = error && error.type ? error.type : "Error"; // a Python exception carries its class in .type
    post({ type: "error", id: msg.id, what: msg.type, name: String(name), message: String(error && error.message || error).slice(-300) });
  }
};
