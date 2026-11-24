hebzip - the public measurement (bench)
=======================================

What it is
----------
One command that compresses every public test text with hebzip and with the well-known compressors,
decompresses every package and compares it byte for byte, and writes the tables:

    results/bench.json    every (tool, text) pair: sizes, sha256 of the package, exact yes/no, times
    results/summary.json  the totals per set, % vs our best mode, with and without fixed headers
    results/bench.md      the same as Markdown tables, with the prior knowledge of each tool and the timings
    overlap.json          which test texts were (or were not) in the corpus our tables were built from

Each text is compressed ALONE (one package per text - the use hebzip is made for: short messages).

How to run (from this folder's parent, tools/compare_page, with a Python that has hebzip's dependencies:
numpy, numba, zstandard, pyppmd; Node.js on the PATH for the brotli row)
------------------------------------------------------------------------------------------------------
    python -m atar.bench run
    python -m atar.bench run --omilab DIR --paq8px paq8px.exe --paq8px-dict DIR

Without the optional inputs the run measures every shipped text and says which sets / rows it skipped.
An interrupted run continues where it stopped (results are saved after every text); --fresh starts over.
    --omilab DIR        token_train.tsv + token_test.tsv of
                        https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew (data/)
    --paq8px EXE        paq8px v217 (GPL, not shipped; build it from https://github.com/hxim/paq8px)
    --paq8px-dict DIR   the folder with its english.dic, english.exp, english.emb (needed for -8LT)
    --rows a,b          only these rows (ids in config/bench.toml); --languages he   only Hebrew
Rebuilding the everyday (OMILab) texts follows atar/bench/datasets.py; every rebuilt text is checked against its
sha256 in manifest.json before it is measured.

The overlap check needs the two training corpora (one JSON article per line):
    python -m atar.bench overlap --corpus he=hewiki_random.jsonl --corpus en=enwiki_random.jsonl
A text whose title is a training article, or that shares >= 5% of its 8-word runs with one, is reported
in a separate "seen in training" group and never in the headline numbers.

The "zstd -19 + trained dictionary" rows use the dictionaries in dicts/, trained from the same two corpora
(the same every-40th-line held out) with zstd's own --train defaults; to rebuild them:
    python -m atar.bench dicts --corpus he=hewiki_random.jsonl --corpus en=enwiki_random.jsonl
The same corpus file gives the same dictionary bytes; dicts/dicts.json records each file's sha256, the
corpus file's sha256, the training parameters and the zstandard / libzstd version.

Rules of the numbers
--------------------
- exact bytes of every package; a total only when every text of the group came back exact;
- "% vs ours" = against our best mode on that group; a second table subtracts each tool's fixed
  per-package header (the header_bytes row of config/bench.toml says what it is for each tool);
- prior knowledge: our tables (MB per mode, computed by hebzip from its config), brotli's built-in
  dictionary, zstd's trained dictionaries (their size), paq8px's English files - written next to the tables;
- times: median per short text, on the machine named in bench.md. "loaded" = hebzip kept in memory
  (python -m hebzip serve, as the comparison page uses it); "cold CLI" = a new process per call.
  In-process tools (gzip, bzip2, xz, zstd with or without its dictionary, PPMd) are timed inside Python; brotli and paq8px start a
  new process per call (included in their time).

Where things are
----------------
    manifest.json        the text sets: every text with its source (title, URL, dates) and sha256
    texts/               the shipped Wikipedia texts (CC BY-SA 4.0, see texts/SOURCES.txt)
    brotli.js            the brotli row (Node.js's built-in zlib)
    dicts/               the zstd dictionaries (CC BY-SA 4.0, derived from Wikipedia; see dicts/LICENSE.txt)
    ../config/bench.toml the rows (tools, options, headers, prior knowledge)
    ../atar/bench/       the code
