// The RECORDED benchmark (data/bench.json, made from `python -m atar.bench run`): shown as recorded, never
// re-measured here. Every row says what the tool ships with before it sees the text.
const GROUPS = {
  he_wiki: "Wikipedia openings (5)",
  he_fresh: "Wikipedia, new articles (5)",
  he_own: "Project texts (8)",
  he_everyday_omilab: "Real comments, OMILab (10)",
  "he: all clean texts": "All Hebrew",
  en_fresh: "Wikipedia, new articles (5)",
  en_short: "Wikipedia leads (12)",
  en_long_fresh: "Long, new article (1)",
  en_long: "Long articles (4)",
  "en: all clean texts": "All English",
};

const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...kids.filter((k) => k !== null && k !== undefined));
  return node;
};
const num = (n) => (n === null || n === undefined ? "-" : n.toLocaleString("en-US"));
const pct = (p) => (p === null || p === undefined ? "" : ` (${p > 0 ? "+" : ""}${p.toFixed(1)}%)`);
const secs = (s) => (s === null || s === undefined ? "-" : `${s < 0.01 ? s.toFixed(3) : s.toFixed(2)} s`);

function prior(row) {
  if (row.ours && row.prior_bytes) return `${(row.prior_bytes / 1e6).toFixed(1)} MB of tables (${row.prior})`;
  return row.prior;
}

function timeCell(row) {
  const loaded = `${secs(row.compress_s)} / ${secs(row.decompress_s)}`;
  if (!row.ours || row.cold_compress_s === null) return loaded;
  return `${loaded} (loaded); ${secs(row.cold_compress_s)} / ${secs(row.cold_decompress_s)} (new process)`;
}

// A row measured in a later run than the table's date says its own date (its times are from that run).
const version = (r, recorded) => (r.recorded && r.recorded !== recorded ? `${r.version}; measured ${r.recorded}` : r.version);

function table(lang, data, recorded) {
  const head = el("tr", {}, el("th", { scope: "col", textContent: "Tool (version, flags)" }),
    el("th", { scope: "col", textContent: "Prior knowledge shipped with the tool" }),
    ...data.groups.map((g) => el("th", { scope: "col", className: "n", textContent: GROUPS[g.id] || g.id })),
    el("th", { scope: "col", textContent: "Time per short text: compress / decompress (median)" }));
  const raw = el("tr", { className: "muted-row" }, el("th", { scope: "row", textContent: "original text" }), el("td"),
    ...data.groups.map((g) => el("td", { className: "n", textContent: num(g.raw) })), el("td"));
  const rows = data.rows.map((r) => el("tr", { className: r.ours ? "ours" : "" },
    el("th", { scope: "row" }, el("span", { textContent: r.label }), el("br"), el("small", { className: "muted", textContent: version(r, recorded) })),
    el("td", { className: "prior", textContent: prior(r) }),
    ...data.groups.map((g) => el("td", { className: "n", textContent: num(r.sizes[g.id]) + pct(r.pct[g.id]) })),
    el("td", { className: "time", textContent: timeCell(r) })));
  const caption = el("caption", { textContent: `${lang === "he" ? "Hebrew" : "English"}: package bytes per set (in brackets: % against our best mode on that set)` });
  return el("div", { className: "table-wrap" }, el("table", { className: "table bench-table" }, caption, el("thead", {}, head), el("tbody", {}, raw, ...rows)));
}

export function renderBench(host, bench) {
  host.replaceChildren();
  host.append(el("p", { className: "muted" },
    `Recorded on ${bench.recorded}, on one laptop (${bench.machine}) - not measured in your browser. `,
    "Each text compressed alone; every package was decompressed and compared byte for byte. ",
    "Our modes stayed loaded (", el("code", { textContent: "hebzip serve" }), "); the second time is a new process per text."));
  for (const [lang, data] of Object.entries(bench.languages)) host.append(table(lang, data, bench.recorded));
}
