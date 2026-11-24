// Facts about the input text - counts only, nothing is kept or sent.
const HEBREW = /[א-ת]/gu;
const LATIN = /[A-Za-z]/g;

// Which pack compresses a text: the script most of its letters are written in (fast = Hebrew, en_fast = English).
// A text with both scripts is compressed by the majority; the page says so.
export function choosePack(text) {
  const hebrew = (text.match(HEBREW) || []).length;
  const latin = (text.match(LATIN) || []).length;
  const mode = latin > hebrew ? "en_fast" : "fast";
  const minority = Math.min(hebrew, latin);
  const share = hebrew + latin ? minority / (hebrew + latin) : 0;
  return { mode, hebrew, latin, mixed: share >= 0.1 };
}

export function words(text) {
  const found = text.trim().match(/\S+/gu);
  return found ? found.length : 0;
}

export const utf8 = (text) => new TextEncoder().encode(text);

// A dropped file as text: UTF-8 only (a BOM is kept as part of the bytes, exactly like the CLI).
export function decodeUtf8(bytes) {
  return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
}
