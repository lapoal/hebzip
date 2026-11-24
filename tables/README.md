# Tables

hebzip's modes read pinned tables (word lists and counts made from Wikipedia and the fastText Hebrew
vectors). They are **not part of the source repository** (they are large and under a different licence,
CC BY-SA 4.0 - see `LICENSE` and `CREDITS.md`). Without them every mode that needs one stops with
`hebzip: tables not found: ...` (exit 3) and points here.

## Get them

1. **Download** the tables archive (`hebzip-tables-<version>.zip`) from the project's Releases page and
   unzip it into the project folder, so that `tables/he_words.txt`, `tables/en/words.txt`, ... exist.
2. **Only modes `fast` and `en_fast`:** their tables are also in the website folder (some gzipped). Write
   them into `tables/`, each checked against its sha256:

       python tools/site/build_site.py --unpack-tables
3. The default config reads a memory-mapped copy of the tables. Make it once after (1) or (2):

       python -m hebzip.ops.convert_tables                       # after (1): every table
       python -m hebzip.ops.convert_tables --modes fast,en_fast  # after (2)

   (or set `table_store = "eager"` in `config/hebzip.toml` to read the files directly).

Every file is checked against its sha256 in `config/hebzip.toml` before use, wherever it came from.

| mode | tables | size |
|---|---|---|
| fast | he_words, bigram_mc1, template_m128 (+ weights) | 25.3 MB |
| max | fast + corpus_gap_starts, prefix_classes, prefix_prior, trie_he, letters_v15, letter_start_sets | 82.7 MB |
| max_plus | max + classes/, plus/, topic/ | 160.6 MB |
| en_fast | en/words, en/bigram, en/template (+ weights) | 45.7 MB |
| en_max | every file in en/ | 191.4 MB |

## Build them yourself

Every table with a `[tables.<name>.recipe]` row in `config/hebzip.toml` is rebuilt from a corpus by

    python -m hebzip.ops.build_langpack --language he --corpus hewiki_random.jsonl --corpus-format jsonl
    python -m hebzip.ops.build_langpack --language en --corpus enwiki_random.jsonl --corpus-format jsonl
    python -m hebzip.ops.build_langpack --language he --corpus hewiki_random.jsonl --corpus-format jsonl \
        --tables word_classes,class_pairs_wiki,topic_neighbours --vectors he_vectors.npy

(add `--out <folder>` to build next to the pinned files and compare). The tool prints the new
`sha256` / `bytes` rows to paste into the config; then run `python -m hebzip.ops.convert_tables`.

What you need, and honest limits:

- **The corpus:** one JSON object per line, `{"title": ..., "text": ...}`. The article titles we used are in
  `wiki_titles_he.txt` (6,202) and `wiki_titles_en.txt` (12,497). Wikipedia articles change, so a corpus
  fetched today gives tables that work the same way but are **not byte-identical** to the pinned ones
  (and packages made with them get another table signature).
- **Given, not rebuilt:** `he_words.txt` (the Hebrew-letter-only entries among the first 2,000,000 rows of
  fastText `cc.he.300.vec`, in file order), `letter_start_sets.json` and `plus/trust_sets_wiki.json`
  (tuned start weights) have no recipe; they come with the archive.
- **Vectors:** `--vectors` wants a `.npy` file with one row per word number of `he_words.txt`, made from
  `cc.he.300.vec`; the script that made ours is not part of this repository yet.
