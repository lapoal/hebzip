# hebzip

Lossless compression for **short Hebrew and English texts** - messages, letters, notices, article
openings. It gets them much smaller than general-purpose compressors because it ships with prior
knowledge: tables of words, word pairs and the bytes between words, built from Wikipedia and the fastText
Hebrew word list (25.3 MB for the Hebrew mode `fast`, up to 191 MB for the biggest mode). That knowledge is
the whole trick, so every comparison below says how much each tool ships with.

It is a research tool, not a replacement for zstd: it is **slow** (tens of milliseconds per short text when
kept loaded, seconds for a new process; zstd needs about a millisecond), it needs large tables, and its
gain is largest on short texts.

Try it in your browser: the [website](site/) (once published with GitHub Pages:
`https://lapoal.github.io/hebzip/`) runs mode `fast` for Hebrew, `en_fast` for English; your text never leaves
the browser. · [Hebrew README / קרא בעברית](README.he.md)

## Results

Recorded on 2026-11-22 **on one laptop** (Intel i3-1215U, 11.7 GB RAM, Windows 11, Python 3.12.10) with
`python -m atar.bench run` (see [Reproduce](#reproduce-the-benchmark)). Each text is compressed alone; every
package was decompressed and compared byte for byte. Numbers are package bytes, in brackets the difference
against our best mode on that set. Texts that may have been in our training corpus are never in these
groups (an overlap check found none).

The sets: *Wikipedia openings* (the five golden Wikipedia texts), *new Wikipedia articles* (created in
November 2026, after our corpus was taken), *project texts* (letters, notices, a story, a diary, an essay -
**written by an AI model for this project**, invented people and places), *real comments* (10 public
Facebook comments from the OMILab Hebrew sentiment benchmark; not shipped, rebuilt from the dataset).

**Hebrew** (28 texts, 67,872 bytes)

| tool | prior knowledge | Wikipedia openings (5) | new Wikipedia articles (5) | project texts (8) | real comments (10) | **all (28)** |
|---|---|---:|---:|---:|---:|---:|
| original text | | 10,430 | 8,765 | 41,846 | 6,831 | 67,872 |
| gzip -9 | none | 4,190 (+139.0%) | 3,341 (+158.6%) | 14,359 (+93.9%) | 3,280 (+134.8%) | 25,170 (+112.4%) |
| bzip2 -9 | none | 3,592 (+104.9%) | 2,820 (+118.3%) | 11,377 (+53.6%) | 2,825 (+102.2%) | 20,614 (+74.0%) |
| xz -9e | none | 4,356 (+148.5%) | 3,592 (+178.0%) | 14,060 (+89.8%) | 3,936 (+181.7%) | 25,944 (+119.0%) |
| zstd -19 | none | 4,132 (+135.7%) | 3,334 (+158.0%) | 13,790 (+86.2%) | 3,291 (+135.6%) | 24,547 (+107.2%) |
| brotli -q 11 | built-in dictionary, 0.12 MB | 3,641 (+107.7%) | 2,876 (+122.6%) | 12,561 (+69.6%) | 2,646 (+89.4%) | 21,724 (+83.3%) |
| PPMd (H, order 8) | none | 3,180 (+81.4%) | 2,534 (+96.1%) | 10,616 (+43.3%) | 2,531 (+81.2%) | 18,861 (+59.2%) |
| paq8px v217 -8L | none | 2,885 (+64.6%) | 2,305 (+78.4%) | 9,749 (+31.6%) | 2,274 (+62.8%) | 17,213 (+45.3%) |
| **hebzip fast** | 25.3 MB tables | 1,861 (+6.2%) | 1,381 (+6.9%) | 7,762 (+4.8%) | 1,426 (+2.1%) | 12,430 (+4.9%) |
| **hebzip max** | 82.7 MB tables | 1,805 (+3.0%) | 1,330 (+2.9%) | 7,534 (+1.7%) | 1,413 (+1.1%) | 12,082 (+2.0%) |
| **hebzip max_plus** | 160.6 MB tables | 1,753 (+0.0%) | 1,292 (+0.0%) | 7,407 (+0.0%) | 1,397 (+0.0%) | 11,849 (+0.0%) |

**English** (22 Wikipedia texts, 72,000 bytes)

| tool | prior knowledge | new Wikipedia articles (5) | Wikipedia leads (12) | long, new (1) | long (4) | **all (22)** |
|---|---|---:|---:|---:|---:|---:|
| original text | | 4,878 | 8,229 | 11,217 | 47,676 | 72,000 |
| gzip -9 | none | 2,725 (+156.1%) | 4,677 (+135.9%) | 4,655 (+59.1%) | 20,381 (+82.7%) | 32,438 (+89.4%) |
| bzip2 -9 | none | 2,927 (+175.1%) | 4,971 (+150.7%) | 4,332 (+48.1%) | 18,977 (+70.1%) | 31,207 (+82.2%) |
| xz -9e | none | 3,160 (+197.0%) | 5,832 (+194.1%) | 4,560 (+55.9%) | 20,016 (+79.4%) | 33,568 (+96.0%) |
| zstd -19 | none | 2,725 (+156.1%) | 4,660 (+135.0%) | 4,519 (+54.5%) | 19,671 (+76.3%) | 31,575 (+84.3%) |
| brotli -q 11 | built-in dictionary, 0.12 MB | 1,886 (+77.3%) | 3,359 (+69.4%) | 3,814 (+30.4%) | 15,677 (+40.5%) | 24,736 (+44.4%) |
| PPMd (H, order 8) | none | 2,394 (+125.0%) | 4,190 (+111.3%) | 3,791 (+29.6%) | 16,706 (+49.7%) | 27,081 (+58.1%) |
| paq8px v217 -8L | none | 2,232 (+109.8%) | 3,934 (+98.4%) | 3,390 (+15.9%) | 15,176 (+36.0%) | 24,732 (+44.4%) |
| paq8px v217 -8LT | English dictionary + embeddings, 0.55 MB | 1,553 (+46.0%) | 2,685 (+35.4%) | 2,955 (+1.0%) | 12,193 (+9.3%) | 19,386 (+13.2%) |
| **hebzip en_fast** | 45.7 MB tables | 1,089 (+2.3%) | 2,043 (+3.0%) | 3,001 (+2.6%) | 11,520 (+3.3%) | 17,653 (+3.1%) |
| **hebzip en_max** | 191.4 MB tables | 1,064 (+0.0%) | 1,983 (+0.0%) | 2,925 (+0.0%) | 11,157 (+0.0%) | 17,129 (+0.0%) |

How to read this honestly:

- The claim is narrow: **short Hebrew (and English) messages, with shipped tables of 25-191 MB.** The other
  tools (except brotli's small dictionary and paq8px -8LT's English files) start empty and learn only from
  the text itself, which is why short texts are hard for them. This is not "better than PAQ" in general;
  on long texts the gap shrinks (English, one long new article: en_max 2,925 vs paq8px -8LT 2,955 bytes).
- Versions and flags: zlib 1.3.1; Python's bz2 / lzma; zstandard 0.25.0 (libzstd 1.5.7) without a
  dictionary; Node.js zlib brotli quality 11; pyppmd 1.3.1 PPMd7 order 8, 64 MB; paq8px v217. The table
  without each tool's fixed header bytes is in [bench.md](tools/compare_page/bench/results/bench.md).
- One laptop, one run. The sizes are exact and reproducible; the times are not portable.

**Time per short text** (median, same laptop):

| tool | compress | decompress |
|---|---:|---:|
| gzip -9 / zstd -19 / PPMd | 0.001 s | 0.009-0.015 s |
| brotli -q 11 (includes starting Node.js) | 0.067 s | 0.059 s |
| hebzip fast / en_fast, kept loaded (`hebzip serve`) | 0.054 s / 0.070 s | 0.021 s / 0.028 s |
| hebzip max / max_plus, kept loaded | 0.278 s / 0.340 s | 0.103 s / 0.167 s |
| hebzip fast, a new process per text (CLI) | 2.03 s | 1.62 s |
| paq8px -8L / -8LT | 0.92 s / 38.3 s | 0.86 s / 37.5 s |

## Quick start

Python 3.12 or newer.

    git clone https://github.com/lapoal/hebzip
    cd hebzip
    pip install numpy numba zstandard pyppmd

Run it from the project folder (`python -m hebzip ...`); the tables are found relative to it.

Get the tables (they are not in the repository - see [tables/README.md](tables/README.md)): download the
tables archive from the Releases page and unzip it into the project folder, then

    python -m hebzip.ops.convert_tables       # the memory-mapped copy the default config reads

Use it:

    python -m hebzip compress letter.txt letter.hz                # default mode: fast
    python -m hebzip compress --mode max_plus letter.txt letter.hz
    python -m hebzip compress --mode en_fast note.txt note.hz     # English
    python -m hebzip decompress letter.hz letter.txt
    python -m hebzip info letter.hz                               # mode, sizes, table signature

The package records its mode; `decompress` needs no options. A package can only be opened with the same
tables and rows it was made with (the header carries a signature byte; anything else is refused, exit 3).
Input must be UTF-8; the original bytes come back exactly (BOM and line endings included). Exit codes:
2 config, 3 tables, 4 damaged package, 6 drawer unavailable, 7 input.

To keep the tool loaded (a new process spends about 2 s starting; the coding itself is far faster):

    python -m hebzip serve --modes fast,max

reads one JSON request per line on stdin (`{"id": 1, "op": "compress", "input": "a.txt", "output": "a.hz",
"mode": "fast"}`) and answers one JSON line each; only file paths cross the pipe. Details: the module text
of `hebzip/interface/serve.py`.

Modes: `fast`, `max`, `max_plus` (Hebrew), `en_fast`, `en_max` (English), `generic` (the smallest of 15
general-purpose codecs - also the automatic fallback), `latin_fast` / `latin_max` (a toy language that shows
a new language is rows + tables only).

## Get or build the tables

[tables/README.md](tables/README.md): the download, the tables of `fast` and `en_fast` that the website
folder also holds (`python tools/site/build_site.py --unpack-tables` writes them into `tables/`), and the
recipe that rebuilds every table from a corpus:

    python -m hebzip.ops.build_langpack --language he --corpus hewiki_random.jsonl --corpus-format jsonl

A fresh clone without tables stops with a message that says where to get them.

## Reproduce the benchmark

    cd tools/compare_page
    python -m atar.bench run                                   # every shipped text, every built-in tool
    python -m atar.bench run --omilab DIR --paq8px paq8px.exe --paq8px-dict DIR

Node.js on the PATH gives the brotli row. `--omilab` rebuilds the real-comments set from the public dataset
(each rebuilt text is checked against its sha256 first); paq8px is GPL and not shipped - build v217 yourself.
Whatever is not given is skipped with a line saying so. Results: `tools/compare_page/bench/results/`
(`bench.json` every pair, `summary.json`, `bench.md`). Details: [bench/README.txt](tools/compare_page/bench/README.txt).
The training-overlap check: `python -m atar.bench overlap --corpus he=... --corpus en=...`.

## How it works

The method in short (each part is a "drawer" chosen by a word in `config/hebzip.toml`, and every number is
a row there):

1. **Words and the bytes between them.** The text is split into words of the language's letters and a
   *template* - everything else (spaces, punctuation, digits, unknown words, Hebrew vowel marks as a layer).
2. **Dictionary in frequency order.** Each known word is its line number in a word list sorted by frequency
   (for Hebrew: the fastText vocabulary). Common words get small numbers.
3. **Predictors mixed by a trust mixer.** Several channels each give a probability for the next word: word
   pairs counted on Wikipedia (`corpus_pairs`), and a *text notebook* - pairs and counts of what this text
   has said so far, which learns while it reads (`text_pairs`, `text_counts`). A mixer weighs the channels
   by how well each has predicted in this text (trust learned per text), over a floor that gives every word
   some probability (unigram + Zipf).
4. **Template letter model.** The template bytes go through a small context-mixing model (bit by bit,
   contexts of several orders plus the neighbouring words, counts from Wikipedia plus adaptive ones, a
   logistic mixer) - the PAQ idea at small scale.
5. **Range coding.** A range coder turns the probabilities into bits. All maths is integer or exactly
   ordered floating point, so the decoder (which runs the same models) gets the same probabilities and the
   same text back; the same kernels run as plain Python or compiled with numba, with identical bytes.

Modes: **fast** codes each word in one step. **max** also spells every word letter by letter over an
alphabetical tree, mixing the word model with letter models (corpus letter chains and the text's own
letters), and uses the kind of gap before a word (sentence start, after a comma...). **max_plus** adds word
classes (k-means of fastText vectors: the class of the previous word predicts the next one), trust vectors
selected per situation, and a topic channel (the nearest words of the text's recent content words).
**Language packs:** a language is a `[languages.*]` section (letters, marks, punctuation classes) plus
tables; `en_fast` / `en_max` and the toy `latin_*` modes use the same code as Hebrew.

## Limitations

- Slow compared with zstd-class compressors; the website runs pure Python in WebAssembly (in Edge on the
  test laptop about 0.4 s per KB of Hebrew and 0.7-0.8 s per KB of English, each way) and accepts up to 16 KB.
- Needs the tables: 25-191 MB, which is the prior knowledge behind the results.
- Short texts are the point. Long texts still compress well, but the advantage shrinks.
- A package is tied to the exact tables and config rows (by design); a new table version means new packages.
- Measured on one laptop; Hebrew test sets are small (28 texts).
- Mixed-language text is compressed by one pack; the other script is kept exactly but compresses worse.

## Website

`site/` is a static website (for GitHub Pages) that runs `fast` and `en_fast` in the visitor's browser with
Pyodide in a Web Worker. Everything is self-hosted: no CDN, no analytics, no cookies, and the page fetches only
its own files. Its generated parts are assembled from this repository by `python tools/site/build_site.py`
(see [tools/site/README.md](tools/site/README.md)); `--check` proves the site matches the code.

What was tested (22/11/2026, the same laptop): headless Edge 158 and Chrome 158 on Windows 11 - not Firefox,
not Safari, not a phone. In Edge, 41 public texts (17 Hebrew, 24 English, up to 12 KB) compressed in the page
gave packages byte-identical to the CLI's, and all 41 CLI packages opened in the page to the exact original
bytes; the page made requests to its own origin only. The first visit downloads 36.1 MB (engine 16.7 MB,
Hebrew pack 12.5 MB, English pack 6.9 MB - some tables are shipped gzipped), and the engine uses about
221 MB of memory.

## Credits

- Ideas from the PAQ family of compressors and the context-mixing literature - Matt Mahoney's work (lpaq1,
  paq8, the *Data Compression Explained* book) and paq8px's contributors. No code was copied.
- Tables: Hebrew and English Wikipedia contributors (CC BY-SA 4.0); fastText word vectors - E. Grave,
  P. Bojanowski, P. Gupta, A. Joulin, T. Mikolov, *Learning Word Vectors for 157 Languages*, LREC 2018.
  See [tables/CREDITS.md](tables/CREDITS.md).
- Real-comments test set: OMILab Hebrew sentiment benchmark (MIT).
- Libraries: numpy, numba, zstandard, pyppmd; the website: Pyodide (MPL-2.0). See [NOTICE](NOTICE).

<!-- HOW-THIS-WAS-BUILT -->
**How this was built:** the design direction, the experiments and the decisions are the author's; the
code was written largely with an AI coding assistant.
<!-- /HOW-THIS-WAS-BUILT -->

## Licence

Code: MIT ([LICENSE](LICENSE)). Tables: CC BY-SA 4.0 ([tables/LICENSE](tables/LICENSE)). Details per folder:
[NOTICE](NOTICE). Reporting results and bugs: [CONTRIBUTING.md](CONTRIBUTING.md).
