# hebzip - public measurement

Machine: 12th Gen Intel(R) Core(TM) i3-1215U, 11.7 GB RAM, Windows 11, Python 3.12.10, measured 2026-11-22T19:28:54+02:00

Rows measured in a later run:

- zstd -19 + trained dictionary (112,640 B) (he): 12th Gen Intel(R) Core(TM) i3-1215U, 11.7 GB RAM, Windows 11, Python 3.12.10, measured 2026-11-24T12:45:57+02:00
- zstd -19 + trained dictionary (1 MiB) (he): 12th Gen Intel(R) Core(TM) i3-1215U, 11.7 GB RAM, Windows 11, Python 3.12.10, measured 2026-11-24T12:45:57+02:00
- zstd -19 + trained dictionary (112,640 B) (en): 12th Gen Intel(R) Core(TM) i3-1215U, 11.7 GB RAM, Windows 11, Python 3.12.10, measured 2026-11-24T12:45:57+02:00
- zstd -19 + trained dictionary (1 MiB) (en): 12th Gen Intel(R) Core(TM) i3-1215U, 11.7 GB RAM, Windows 11, Python 3.12.10, measured 2026-11-24T12:45:57+02:00

## he: package bytes as written (in brackets: % vs our best mode on that group)

| tool | he_wiki (5) | he_fresh (5) | he_own (8) | he_everyday_omilab (10) | he: all clean texts (28) |
|---|---|---|---|---|---|
| original text | 10,430 | 8,765 | 41,846 | 6,831 | 67,872 |
| gzip -9 | 4,190 (+139.0%) | 3,341 (+158.6%) | 14,359 (+93.9%) | 3,280 (+134.8%) | 25,170 (+112.4%) |
| bzip2 -9 | 3,592 (+104.9%) | 2,820 (+118.3%) | 11,377 (+53.6%) | 2,825 (+102.2%) | 20,614 (+74.0%) |
| xz -9e | 4,356 (+148.5%) | 3,592 (+178.0%) | 14,060 (+89.8%) | 3,936 (+181.7%) | 25,944 (+119.0%) |
| zstd -19 | 4,132 (+135.7%) | 3,334 (+158.0%) | 13,790 (+86.2%) | 3,291 (+135.6%) | 24,547 (+107.2%) |
| zstd -19 + trained dictionary (112,640 B) | 3,034 (+73.1%) | 2,328 (+80.2%) | 11,523 (+55.6%) | 2,313 (+65.6%) | 19,198 (+62.0%) |
| zstd -19 + trained dictionary (1 MiB) | 2,850 (+62.6%) | 2,206 (+70.7%) | 11,194 (+51.1%) | 2,219 (+58.8%) | 18,469 (+55.9%) |
| brotli -q 11 | 3,641 (+107.7%) | 2,876 (+122.6%) | 12,561 (+69.6%) | 2,646 (+89.4%) | 21,724 (+83.3%) |
| PPMd (H, order 8) | 3,180 (+81.4%) | 2,534 (+96.1%) | 10,616 (+43.3%) | 2,531 (+81.2%) | 18,861 (+59.2%) |
| hebzip fast (loaded) | 1,861 (+6.2%) | 1,381 (+6.9%) | 7,762 (+4.8%) | 1,426 (+2.1%) | 12,430 (+4.9%) |
| hebzip max (loaded) | 1,805 (+3.0%) | 1,330 (+2.9%) | 7,534 (+1.7%) | 1,413 (+1.1%) | 12,082 (+2.0%) |
| hebzip max_plus (loaded) | 1,753 (+0.0%) | 1,292 (+0.0%) | 7,407 (+0.0%) | 1,397 (+0.0%) | 11,849 (+0.0%) |
| hebzip fast (cold CLI) | 1,861 (+6.2%) | 1,381 (+6.9%) | 7,762 (+4.8%) | 1,426 (+2.1%) | 12,430 (+4.9%) |
| hebzip max (cold CLI) | 1,805 (+3.0%) | 1,330 (+2.9%) | 7,534 (+1.7%) | 1,413 (+1.1%) | 12,082 (+2.0%) |
| hebzip max_plus (cold CLI) | 1,753 (+0.0%) | 1,292 (+0.0%) | 7,407 (+0.0%) | 1,397 (+0.0%) | 11,849 (+0.0%) |
| paq8px v217 -8L | 2,885 (+64.6%) | 2,305 (+78.4%) | 9,749 (+31.6%) | 2,274 (+62.8%) | 17,213 (+45.3%) |

## he: package bytes minus each tool's fixed per-package header (in brackets: % vs our best mode on that group)

| tool | he_wiki (5) | he_fresh (5) | he_own (8) | he_everyday_omilab (10) | he: all clean texts (28) |
|---|---|---|---|---|---|
| original text | 10,430 | 8,765 | 41,846 | 6,831 | 67,872 |
| gzip -9 | 4,100 (+135.2%) | 3,251 (+153.6%) | 14,215 (+92.3%) | 3,100 (+125.1%) | 24,666 (+109.2%) |
| bzip2 -9 | 3,457 (+98.3%) | 2,685 (+109.4%) | 11,161 (+51.0%) | 2,555 (+85.5%) | 19,858 (+68.4%) |
| xz -9e | 4,071 (+133.6%) | 3,307 (+158.0%) | 13,604 (+84.1%) | 3,366 (+144.4%) | 24,348 (+106.5%) |
| zstd -19 | 4,082 (+134.2%) | 3,284 (+156.2%) | 13,710 (+85.5%) | 3,191 (+131.7%) | 24,267 (+105.8%) |
| zstd -19 + trained dictionary (112,640 B) | 2,964 (+70.1%) | 2,258 (+76.1%) | 11,411 (+54.4%) | 2,173 (+57.8%) | 18,806 (+59.5%) |
| zstd -19 + trained dictionary (1 MiB) | 2,780 (+59.5%) | 2,136 (+66.6%) | 11,082 (+49.9%) | 2,079 (+51.0%) | 18,077 (+53.3%) |
| brotli -q 11 | 3,641 (+108.9%) | 2,876 (+124.3%) | 12,561 (+69.9%) | 2,646 (+92.2%) | 21,724 (+84.2%) |
| PPMd (H, order 8) | 3,160 (+81.3%) | 2,514 (+96.1%) | 10,584 (+43.2%) | 2,491 (+80.9%) | 18,749 (+59.0%) |
| hebzip fast (loaded) | 1,851 (+6.2%) | 1,371 (+6.9%) | 7,746 (+4.8%) | 1,406 (+2.1%) | 12,374 (+4.9%) |
| hebzip max (loaded) | 1,795 (+3.0%) | 1,320 (+3.0%) | 7,518 (+1.7%) | 1,393 (+1.2%) | 12,026 (+2.0%) |
| hebzip max_plus (loaded) | 1,743 (+0.0%) | 1,282 (+0.0%) | 7,391 (+0.0%) | 1,377 (+0.0%) | 11,793 (+0.0%) |
| hebzip fast (cold CLI) | 1,851 (+6.2%) | 1,371 (+6.9%) | 7,746 (+4.8%) | 1,406 (+2.1%) | 12,374 (+4.9%) |
| hebzip max (cold CLI) | 1,795 (+3.0%) | 1,320 (+3.0%) | 7,518 (+1.7%) | 1,393 (+1.2%) | 12,026 (+2.0%) |
| hebzip max_plus (cold CLI) | 1,743 (+0.0%) | 1,282 (+0.0%) | 7,391 (+0.0%) | 1,377 (+0.0%) | 11,793 (+0.0%) |
| paq8px v217 -8L | 2,840 (+62.9%) | 2,260 (+76.3%) | 9,677 (+30.9%) | 2,184 (+58.6%) | 16,961 (+43.8%) |

## en: package bytes as written (in brackets: % vs our best mode on that group)

| tool | en_fresh (5) | en_short (12) | en_long_fresh (1) | en_long (4) | en: all clean texts (22) |
|---|---|---|---|---|---|
| original text | 4,878 | 8,229 | 11,217 | 47,676 | 72,000 |
| gzip -9 | 2,725 (+156.1%) | 4,677 (+135.9%) | 4,655 (+59.1%) | 20,381 (+82.7%) | 32,438 (+89.4%) |
| bzip2 -9 | 2,927 (+175.1%) | 4,971 (+150.7%) | 4,332 (+48.1%) | 18,977 (+70.1%) | 31,207 (+82.2%) |
| xz -9e | 3,160 (+197.0%) | 5,832 (+194.1%) | 4,560 (+55.9%) | 20,016 (+79.4%) | 33,568 (+96.0%) |
| zstd -19 | 2,725 (+156.1%) | 4,660 (+135.0%) | 4,519 (+54.5%) | 19,671 (+76.3%) | 31,575 (+84.3%) |
| zstd -19 + trained dictionary (112,640 B) | 1,792 (+68.4%) | 3,307 (+66.8%) | 4,045 (+38.3%) | 16,308 (+46.2%) | 25,452 (+48.6%) |
| zstd -19 + trained dictionary (1 MiB) | 1,661 (+56.1%) | 3,101 (+56.4%) | 3,796 (+29.8%) | 15,371 (+37.8%) | 23,929 (+39.7%) |
| brotli -q 11 | 1,886 (+77.3%) | 3,359 (+69.4%) | 3,814 (+30.4%) | 15,677 (+40.5%) | 24,736 (+44.4%) |
| PPMd (H, order 8) | 2,394 (+125.0%) | 4,190 (+111.3%) | 3,791 (+29.6%) | 16,706 (+49.7%) | 27,081 (+58.1%) |
| hebzip en_fast (loaded) | 1,089 (+2.3%) | 2,043 (+3.0%) | 3,001 (+2.6%) | 11,520 (+3.3%) | 17,653 (+3.1%) |
| hebzip en_max (loaded) | 1,064 (+0.0%) | 1,983 (+0.0%) | 2,925 (+0.0%) | 11,157 (+0.0%) | 17,129 (+0.0%) |
| hebzip en_fast (cold CLI) | 1,089 (+2.3%) | 2,043 (+3.0%) | 3,001 (+2.6%) | 11,520 (+3.3%) | 17,653 (+3.1%) |
| hebzip en_max (cold CLI) | 1,064 (+0.0%) | 1,983 (+0.0%) | 2,925 (+0.0%) | 11,157 (+0.0%) | 17,129 (+0.0%) |
| paq8px v217 -8L | 2,232 (+109.8%) | 3,934 (+98.4%) | 3,390 (+15.9%) | 15,176 (+36.0%) | 24,732 (+44.4%) |
| paq8px v217 -8LT | 1,553 (+46.0%) | 2,685 (+35.4%) | 2,955 (+1.0%) | 12,193 (+9.3%) | 19,386 (+13.2%) |

## en: package bytes minus each tool's fixed per-package header (in brackets: % vs our best mode on that group)

| tool | en_fresh (5) | en_short (12) | en_long_fresh (1) | en_long (4) | en: all clean texts (22) |
|---|---|---|---|---|---|
| original text | 4,878 | 8,229 | 11,217 | 47,676 | 72,000 |
| gzip -9 | 2,635 (+150.0%) | 4,461 (+127.7%) | 4,637 (+58.6%) | 20,309 (+82.2%) | 32,042 (+87.5%) |
| bzip2 -9 | 2,792 (+164.9%) | 4,647 (+137.2%) | 4,305 (+47.3%) | 18,869 (+69.2%) | 30,613 (+79.2%) |
| xz -9e | 2,875 (+172.8%) | 5,148 (+162.8%) | 4,503 (+54.1%) | 19,788 (+77.5%) | 32,314 (+89.1%) |
| zstd -19 | 2,675 (+153.8%) | 4,540 (+131.8%) | 4,509 (+54.3%) | 19,631 (+76.1%) | 31,355 (+83.5%) |
| zstd -19 + trained dictionary (112,640 B) | 1,722 (+63.4%) | 3,139 (+60.2%) | 4,031 (+37.9%) | 16,252 (+45.8%) | 25,144 (+47.2%) |
| zstd -19 + trained dictionary (1 MiB) | 1,591 (+50.9%) | 2,933 (+49.7%) | 3,782 (+29.4%) | 15,315 (+37.4%) | 23,621 (+38.3%) |
| brotli -q 11 | 1,886 (+78.9%) | 3,359 (+71.5%) | 3,814 (+30.5%) | 15,677 (+40.6%) | 24,736 (+44.8%) |
| PPMd (H, order 8) | 2,374 (+125.2%) | 4,142 (+111.4%) | 3,787 (+29.6%) | 16,690 (+49.7%) | 26,993 (+58.0%) |
| hebzip en_fast (loaded) | 1,079 (+2.4%) | 2,019 (+3.1%) | 2,999 (+2.6%) | 11,512 (+3.3%) | 17,609 (+3.1%) |
| hebzip en_max (loaded) | 1,054 (+0.0%) | 1,959 (+0.0%) | 2,923 (+0.0%) | 11,149 (+0.0%) | 17,085 (+0.0%) |
| hebzip en_fast (cold CLI) | 1,079 (+2.4%) | 2,019 (+3.1%) | 2,999 (+2.6%) | 11,512 (+3.3%) | 17,609 (+3.1%) |
| hebzip en_max (cold CLI) | 1,054 (+0.0%) | 1,959 (+0.0%) | 2,923 (+0.0%) | 11,149 (+0.0%) | 17,085 (+0.0%) |
| paq8px v217 -8L | 2,187 (+107.5%) | 3,826 (+95.3%) | 3,381 (+15.7%) | 15,140 (+35.8%) | 24,534 (+43.6%) |
| paq8px v217 -8LT | 1,508 (+43.1%) | 2,577 (+31.5%) | 2,946 (+0.8%) | 12,157 (+9.0%) | 19,188 (+12.3%) |

## Prior knowledge each tool brings

| tool | version | fixed header per package | prior knowledge shipped with the tool |
|---|---|---|---|
| gzip -9 | zlib 1.3.1, level 9 | 18 B | none |
| bzip2 -9 | Python 3.12.10 bz2, level 9 | 27 B | none |
| xz -9e | Python 3.12.10 lzma, -9e | 57 B | none |
| zstd -19 | zstandard 0.25.0 (libzstd 1.5.7), level 19 | 10 B | none (no dictionary used) |
| zstd -19 + trained dictionary (112,640 B) | zstandard 0.25.0 (libzstd 1.5.7), level 19, trained dictionary 112,640 B (sha256 141cad015fbf) | 14 B | zstd dictionary trained on the same Hebrew Wikipedia sample as our tables, 112,640 bytes |
| zstd -19 + trained dictionary (1 MiB) | zstandard 0.25.0 (libzstd 1.5.7), level 19, trained dictionary 1,048,576 B (sha256 144a29516a83) | 14 B | zstd dictionary trained on the same Hebrew Wikipedia sample as our tables, 1,048,576 bytes |
| zstd -19 + trained dictionary (112,640 B) | zstandard 0.25.0 (libzstd 1.5.7), level 19, trained dictionary 112,640 B (sha256 49db06bea853) | 14 B | zstd dictionary trained on the same English Wikipedia sample as our tables, 112,640 bytes |
| zstd -19 + trained dictionary (1 MiB) | zstandard 0.25.0 (libzstd 1.5.7), level 19, trained dictionary 1,048,576 B (sha256 77e41c159349) | 14 B | zstd dictionary trained on the same English Wikipedia sample as our tables, 1,048,576 bytes |
| brotli -q 11 | Node.js zlib brotli, quality 11 (time includes starting Node) | 0 B | built-in static dictionary, 122,784 bytes (RFC 7932 Appendix A; mostly English and other web text) |
| PPMd (H, order 8) | pyppmd 1.3.1 PPMd7, order 8, 64 MB | 4 B | none |
| hebzip fast (loaded) | hebzip | 2 B | 25.3 MB of tables (Hebrew Wikipedia sample + fastText Hebrew word list) |
| hebzip max (loaded) | hebzip | 2 B | 82.7 MB of tables (Hebrew Wikipedia sample + fastText Hebrew word list) |
| hebzip max_plus (loaded) | hebzip | 2 B | 160.6 MB of tables (Hebrew Wikipedia sample + fastText Hebrew word list and vectors) |
| hebzip en_fast (loaded) | hebzip | 2 B | 45.7 MB of tables (English Wikipedia sample (dump 2023-11-01)) |
| hebzip en_max (loaded) | hebzip | 2 B | 191.4 MB of tables (English Wikipedia sample (dump 2023-11-01)) |
| hebzip fast (cold CLI) | hebzip | 2 B | 25.3 MB of tables (as fast) |
| hebzip max (cold CLI) | hebzip | 2 B | 82.7 MB of tables (as max) |
| hebzip max_plus (cold CLI) | hebzip | 2 B | 160.6 MB of tables (as max_plus) |
| hebzip en_fast (cold CLI) | hebzip | 2 B | 45.7 MB of tables (as en_fast) |
| hebzip en_max (cold CLI) | hebzip | 2 B | 191.4 MB of tables (as en_max) |
| paq8px v217 -8L | paq8px v217, -8L | 9 B | none (its models, LSTM included, start empty and learn from the text) |
| paq8px v217 -8LT | paq8px v217, -8LT | 9 B | English dictionary + word embeddings for its text transform (english.dic / .exp / .emb - sizes in the summary) |

## Time per short text (this laptop; see the machine line)

| tool | texts | compress (median) | decompress (median) |
|---|---|---|---|
| gzip -9 | 37 | 0.001 s | 0.009 s |
| bzip2 -9 | 37 | 0.001 s | 0.015 s |
| xz -9e | 37 | 0.018 s | 0.013 s |
| zstd -19 | 37 | 0.001 s | 0.013 s |
| zstd -19 + trained dictionary (112,640 B) | 20 | 0.016 s | 0.012 s |
| zstd -19 + trained dictionary (1 MiB) | 20 | 0.244 s | 0.020 s |
| zstd -19 + trained dictionary (112,640 B) | 17 | 0.009 s | 0.015 s |
| zstd -19 + trained dictionary (1 MiB) | 17 | 0.285 s | 0.021 s |
| brotli -q 11 | 37 | 0.067 s | 0.059 s |
| PPMd (H, order 8) | 37 | 0.001 s | 0.015 s |
| hebzip fast (loaded) | 20 | 0.054 s | 0.021 s |
| hebzip max (loaded) | 20 | 0.278 s | 0.103 s |
| hebzip max_plus (loaded) | 20 | 0.340 s | 0.167 s |
| hebzip en_fast (loaded) | 17 | 0.070 s | 0.028 s |
| hebzip en_max (loaded) | 17 | 0.231 s | 0.088 s |
| hebzip fast (cold CLI) | 20 | 2.026 s | 1.618 s |
| hebzip max (cold CLI) | 20 | 2.917 s | 2.449 s |
| hebzip max_plus (cold CLI) | 20 | 3.275 s | 2.673 s |
| hebzip en_fast (cold CLI) | 17 | 1.763 s | 1.708 s |
| hebzip en_max (cold CLI) | 17 | 2.059 s | 1.906 s |
| paq8px v217 -8L | 37 | 0.921 s | 0.855 s |
| paq8px v217 -8LT | 17 | 38.306 s | 37.482 s |

Seen in training (reported apart, never in the headline groups): none

Loaded service and cold CLI wrote the same package for every text: yes
