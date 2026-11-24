// "Share your result": a GitHub new-issue link that carries ONLY numbers - never the text.
// The page shows exactly this title and body before the visitor leaves; nothing is sent by the page itself.
const mb = (n) => (n / 1e6).toFixed(1);

export function shareText(result, env) {
  const lines = [
    "Result from the hebzip website (numbers only - the text itself is not included).",
    "",
    "| | value |",
    "|---|---|",
    `| language pack | ${result.mode} (${mb(result.priorBytes)} MB of tables) |`,
    `| text | ${result.rawBytes} bytes, ${result.words} words |`,
    `| hebzip ${result.mode} | ${result.packageBytes} bytes (${result.savedPct.toFixed(1)}% saved) |`,
    ...result.builtIn.map((r) => `| ${r.id} (browser built-in) | ${r.bytes} bytes |`),
    `| compress time | ${result.compressS.toFixed(2)} s |`,
    `| decompress time | ${result.decompressS.toFixed(2)} s |`,
    `| round trip | ${result.verified ? "identical" : "NOT identical"} |`,
    `| browser | ${env.userAgent} |`,
    `| site build | ${env.build} |`,
  ];
  return { title: `Website result: ${result.mode}, ${result.rawBytes} bytes -> ${result.packageBytes} bytes`, body: lines.join("\n") };
}

export function issueUrl(repo, share) {
  const q = new URLSearchParams({ title: share.title, body: share.body });
  return `https://github.com/${repo}/issues/new?${q.toString()}`;
}
