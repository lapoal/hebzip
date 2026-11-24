// The comparison page: upload one file, then poll /api/status and draw rows as tools finish.
// Every value from the server goes in through textContent (never innerHTML): the file name
// is the user's, and the page must not run anything it contains.
"use strict";

const $ = (id) => document.getElementById(id);
let pollMs = 700;
let maxUpload = 20 * 1024 * 1024;
let timer = null;

const fmtInt = (n) => n.toLocaleString("he-IL");
const fmtSec = (s) => (s === null || s === undefined ? "—" : s.toFixed(2) + " שנ׳");

function fmtSaved(p) {
  if (p === null || p === undefined) return "—";
  return p >= 0 ? "חוסך " + p.toFixed(1) + "%" : "גדל ב-" + (-p).toFixed(1) + "%";
}

function fmtVsZip(p, id) {
  if (id === "zip") return "בסיס";
  if (p === null || p === undefined) return "—";
  if (p === 0) return "שווה";
  return p > 0 ? "קטן ב-" + p.toFixed(1) + "%" : "גדול ב-" + (-p).toFixed(1) + "%";
}

function cell(tr, text, cls) {
  const td = document.createElement("td");
  td.textContent = text;
  if (cls) td.className = cls;
  tr.appendChild(td);
  return td;
}

function markCell(tr, row) {
  const td = cell(tr, "");
  const span = document.createElement("span");
  if (row.status === "ok") {
    span.className = "ok";
    span.textContent = "✓ נפתח זהה למקור";
  } else if (row.status === "mismatch") {
    span.className = "bad-mark";
    span.textContent = row.decompress_s === null ? "✗ לא נפתח!" : "✗ נפתח שונה מהמקור!";
  } else {
    span.className = "fail";
    span.textContent = row.status;
  }
  td.appendChild(span);
}

function nameCell(tr, row) {
  const td = cell(tr, row.label);
  if (row.ours) {
    const tag = document.createElement("span");
    tag.className = "Label Label-ours";
    tag.textContent = "שלנו";
    td.appendChild(tag);
  }
}

function drawRow(body, row) {
  const tr = document.createElement("tr");
  tr.className = (row.ours ? "ours " : "") + (row.status === "mismatch" ? "bad" : "");
  nameCell(tr, row);
  cell(tr, row.size === null ? "—" : fmtInt(row.size), "n");
  cell(tr, fmtSaved(row.saved_pct), "n");
  cell(tr, fmtVsZip(row.vs_zip_pct, row.id), "n");
  cell(tr, fmtSec(row.compress_s), "n");
  cell(tr, fmtSec(row.decompress_s), "n");
  markCell(tr, row);
  body.appendChild(tr);
}

function drawPending(body, label, running) {
  const tr = document.createElement("tr");
  tr.className = "pending";
  cell(tr, label);
  for (let i = 0; i < 5; i++) cell(tr, "", "n");
  cell(tr, running ? "רץ עכשיו..." : "ממתין");
  body.appendChild(tr);
}

function drawJob(job) {
  if (job.state === "idle") {
    setBusy(false);
    return;
  }
  $("result").hidden = false;
  $("f-name").textContent = job.file_name;
  $("f-size").textContent = fmtInt(job.original) + " בתים";
  $("f-words").textContent = fmtInt(job.words);
  const sum = $("summary");
  sum.textContent = job.state === "error" ? job.error : job.summary;
  sum.className = "Box-row summary" +
    (job.state === "error" || job.summary.startsWith("אזהרה") ? " warn" : "");
  const running = job.state === "running";
  $("running").hidden = !running;
  $("running-name").textContent = job.running || "";
  const later = (job.pending || []).filter((l) => l !== job.running);
  $("pending").textContent = later.length ? "· בתור: " + later.join(", ") : "";
  const body = $("rows");
  body.replaceChildren();
  job.rows.forEach((r) => drawRow(body, r));
  (job.pending || []).forEach((l) => drawPending(body, l, l === job.running));
  setBusy(running);
}

function drawVersions(versions) {
  const box = $("versions");
  box.replaceChildren();
  versions.forEach((v) => {
    const li = document.createElement("li");
    const name = document.createElement("bdi");
    name.textContent = v.label;
    const ver = document.createElement("bdi");
    ver.dir = "ltr";
    ver.textContent = v.version;
    li.append(name, ": ", ver);
    box.appendChild(li);
  });
}

// Our tool stays loaded: one line under the upload box with its state, and one note under
// the table that says how the times were measured. Words come from here, numbers from the server.
const STATE_WORDS = {
  loading: "טוען את הכלי שלנו… אפשר כבר להעלות קובץ — הכלי שלנו ימתין עד שיהיה מוכן.",
  stopped: "הכלי שלנו ייטען מחדש בהשוואה הבאה.",
};

function modesOf(s) {
  return s.modes && s.modes.length ? " (" + s.modes.join(", ") + ")" : "";
}

function stateText(s) {
  if (s.state === "ready") {
    return "✓ הכלי שלנו" + modesOf(s) + " מוכן — נטען פעם אחת כשהדף נפתח, ב-" + fmtSec(s.load_s) + ".";
  }
  if (s.state === "failed") return "✗ הכלי שלנו לא נטען: " + (s.error || "סיבה לא ידועה") + " ננסה שוב בהשוואה הבאה.";
  return STATE_WORDS[s.state] || STATE_WORDS.loading;
}

function drawServices(services) {
  const box = $("tool-state");
  const note = $("timing-note");
  box.hidden = note.hidden = services.length === 0;
  box.replaceChildren();
  services.forEach((s) => {
    const line = document.createElement("div");
    line.className = "tool-line " + s.state;
    if (s.state === "loading") {
      const spin = document.createElement("span");
      spin.className = "spinner";
      spin.setAttribute("aria-hidden", "true");
      line.appendChild(spin);
    }
    line.append(stateText(s));
    box.appendChild(line);
  });
  const ready = services.find((s) => s.state === "ready");
  note.textContent = "איך נמדדו הזמנים: הכלי שלנו נטען פעם אחת כשהדף נפתח" +
    (ready ? " (הטעינה לקחה " + fmtSec(ready.load_s) + ")" : "") +
    ", ולכן הזמנים בשורות שלו הם של כלי טעון: זמן הדחיסה כולל גם את בדיקת-הפתיחה הפנימית שלו. " +
    "ZIP, ‏bzip2, ‏xz ו-PPMd רצים בתוך השרת; paq8px מופעל מחדש לכל קובץ — כך משתמשים בו.";
  return services.some((s) => s.state === "loading");
}

function setBusy(busy) {
  $("drop").classList.toggle("disabled", busy);
  $("file").disabled = busy;
}

function flash(text, info) {
  const box = $("flash");
  box.textContent = text;
  box.className = "flash" + (info ? " info" : "");
  box.hidden = !text;
}

async function poll() {
  clearTimeout(timer);
  try {
    const reply = await fetch("/api/status", { cache: "no-store" });
    const data = await reply.json();
    pollMs = data.poll_ms || pollMs;
    maxUpload = data.max_upload_bytes || maxUpload;
    $("limit").textContent = Math.round((data.max_text_bytes || 204800) / 1024);
    drawVersions(data.versions || []);
    const loading = drawServices(data.services || []);
    drawJob(data.job);
    if (data.job.state === "running" || loading) timer = setTimeout(poll, pollMs);
  } catch (e) {
    flash("השרת לא עונה. האם החלון השחור של הכלי עדיין פתוח?");
    timer = setTimeout(poll, 3000);
  }
}

async function upload(file) {
  flash("");
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (ext !== "txt" && ext !== "docx") {
    flash("אפשר להעלות רק קובץ ‎.txt‎ (UTF-8) או ‎.docx‎");
    return;
  }
  if (file.size > maxUpload) {
    flash("הקובץ גדול מדי (מעל " + Math.round(maxUpload / 1048576) + " MB).");
    return;
  }
  setBusy(true);
  try {
    const reply = await fetch("/api/jobs?name=" + encodeURIComponent(file.name),
      { method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" } });
    if (!reply.ok) {
      const data = await reply.json().catch(() => ({}));
      flash(data.error || "ההעלאה נכשלה", reply.status === 409);
    }
  } catch (e) {
    flash("ההעלאה נכשלה: השרת לא עונה.");
  }
  poll();
}

function wireDrop() {
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    if ($("file").disabled) return;
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });
  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (f) upload(f);
  });
}

wireDrop();
poll();
