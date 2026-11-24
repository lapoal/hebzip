# Credits and sources of the tables

Licence: CC BY-SA 4.0 (see `LICENSE` in this folder). The tables were **modified** from their sources:
texts and vectors were turned into counts, lists and classes; no source text is stored.

## 1. Hebrew Wikipedia

- **What:** the text of 6,202 randomly chosen Hebrew Wikipedia articles (about 40 MB), obtained in
  November 2026 from the Hebrew Wikipedia API and from one shard of a Hebrew Wikipedia CirrusSearch dump
  (the exact dump date was not recorded). The five articles used as test texts were left out.
- **Attribution:** © Hebrew Wikipedia contributors, licensed under CC BY-SA 4.0
  (https://he.wikipedia.org). The article titles are listed in `wiki_titles_he.txt`; each article and its
  list of authors is at `https://he.wikipedia.org/wiki/<title>`.
- **Used for:** `bigram_mc1.npz` (word pairs), `template_m128.npz` + `template_m128_weights.json`
  (the bytes between words), `corpus_gap_starts.npz`, `prefix_prior.npy`, `letters_v15.npz`,
  `letter_start_sets.json`, `classes/class_pairs_wiki.npz`, `plus/trust_sets_wiki.json`,
  `topic/trust_sets.json`. Every 40th article was held out of the counts and used only to tune.

## 2. fastText Hebrew word vectors (cc.he.300)

- **What:** the official Common Crawl + Wikipedia word vectors for Hebrew, https://fasttext.cc/docs/en/crawl-vectors.html,
  distributed under the Creative Commons Attribution-Share-Alike License 3.0 (adaptations may be shared
  under CC BY-SA 4.0).
- **Citation:** E. Grave, P. Bojanowski, P. Gupta, A. Joulin, T. Mikolov,
  *Learning Word Vectors for 157 Languages*, Proceedings of LREC 2018.
- **Used for:** `he_words.txt` (the vocabulary: the entries made only of Hebrew letters among the first
  2,000,000 rows, in the file's frequency order - the word number is the line number), `trie_he.npz`
  (the same words in alphabetical order), `prefix_classes.npy`, `classes/word_classes_512.npy`
  (k-means classes of the vectors) and `topic/neighbours.npz` (the nearest words of each word).

## 3. English Wikipedia (the English language pack, `en/`)

- **What:** 12,497 random articles (about 41 MB) of the Hugging Face dataset `wikimedia/wikipedia`,
  configuration `20231101.en` (https://huggingface.co/datasets/wikimedia/wikipedia).
- **Attribution:** © English Wikipedia contributors, licensed under CC BY-SA (3.0 / 4.0) and the GFDL
  (https://en.wikipedia.org). Titles: `wiki_titles_en.txt`.
- **Used for:** every table in `en/`.

## Files

| file | MB | source |
|---|---|---|
| he_words.txt | 18.2 | fastText cc.he.300 |
| bigram_mc1.npz | 5.5 | Hebrew Wikipedia + he_words |
| template_m128.npz, template_m128_weights.json | 1.6 | Hebrew Wikipedia |
| corpus_gap_starts.npz, prefix_prior.npy, letter_start_sets.json | 0.1 | Hebrew Wikipedia |
| prefix_classes.npy | 1.2 | he_words |
| trie_he.npz | 37.6 | he_words |
| letters_v15.npz | 18.5 | Hebrew Wikipedia |
| classes/word_classes_512.npy | 9.4 | fastText cc.he.300 |
| classes/class_pairs_wiki.npz | 11.7 | Hebrew Wikipedia + word classes |
| plus/trust_sets_wiki.json, topic/trust_sets.json | 0.01 | Hebrew Wikipedia (tuned) |
| topic/neighbours.npz | 56.7 | fastText cc.he.300 |
| en/* | 191.4 | English Wikipedia |

Mode `fast` reads the first three rows (25.3 MB); `max` 82.7 MB; `max_plus` 160.6 MB; `en_fast` 45.7 MB;
`en_max` 191.4 MB. The exact files are pinned by sha256 in `config/hebzip.toml`.
