// What the browser itself has built in (CompressionStream), measured live on the same bytes.
// Formats are feature-detected: only those this browser supports are shown.
const FORMATS = [
  { id: "gzip", label: "gzip (built into the browser)" },
  { id: "deflate", label: "deflate / zlib (built into the browser)" },
  { id: "deflate-raw", label: "raw deflate (built into the browser)" },
  { id: "brotli", label: "brotli (built into the browser)" },
  { id: "zstd", label: "zstd (built into the browser)" },
];

function supported(format) {
  try { new CompressionStream(format); return true; } catch { return false; }
}

async function size(format, bytes) {
  const t0 = performance.now();
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream(format));
  const out = new Uint8Array(await new Response(stream).arrayBuffer());
  return { bytes: out.length, seconds: (performance.now() - t0) / 1000 };
}

export async function builtIn(bytes) {
  if (typeof CompressionStream === "undefined") return [];
  const rows = [];
  for (const f of FORMATS.filter((x) => supported(x.id))) rows.push({ id: f.id, label: f.label, ...(await size(f.id, bytes)) });
  return rows;
}
