// Brotli for the bench, through Node.js's built-in zlib (no install: Node >= 11.7 has brotli).
//
//   node brotli.js c IN OUT QUALITY     compress  (quality 0..11; the bench uses 11, like `brotli -q 11`)
//   node brotli.js d IN OUT             decompress
//
// Why Node and not a Python module: the Python `brotli` package is not a dependency of hebzip, and
// Node ships the reference brotli library. Default window (lgwin 22) and the generic mode, as the
// `brotli` command line uses. The stream has no container: no header beyond brotli's own window bits.
// The time a bench row shows includes starting Node (a new process per call).
"use strict";
const fs = require("fs");
const zlib = require("zlib");

const [op, src, dst, quality] = process.argv.slice(2);
const data = fs.readFileSync(src);
if (op === "c") {
  const q = Number(quality);
  if (!Number.isInteger(q) || q < 0 || q > 11) {
    process.stderr.write("quality must be 0..11\n");
    process.exit(2);
  }
  const params = { [zlib.constants.BROTLI_PARAM_QUALITY]: q,
                   [zlib.constants.BROTLI_PARAM_SIZE_HINT]: data.length };
  fs.writeFileSync(dst, zlib.brotliCompressSync(data, { params }));
} else if (op === "d") {
  fs.writeFileSync(dst, zlib.brotliDecompressSync(data));
} else {
  process.stderr.write("usage: node brotli.js c|d IN OUT [QUALITY]\n");
  process.exit(2);
}
