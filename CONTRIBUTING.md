# Reporting a result or a bug

Thank you for trying hebzip. Two kinds of reports help most.

## A result

- **From the website:** press *Share your result*. It opens a new GitHub issue filled with numbers only (text
  size, package sizes, times, your browser) - you see the exact text before anything is sent, and your text
  itself is never included.
- **From the tool or the benchmark:** open an issue with the command you ran, the numbers (sizes, times),
  your machine (CPU, RAM, OS, Python version) and `python -m hebzip info <package>` for one package. If you
  re-ran `python -m atar.bench run`, attach `tools/compare_page/bench/results/summary.json` (numbers only).

Please do not paste private text into an issue. If a text matters for the report, say what kind it is
(a short message, a news article...) and its size.

## A bug

Include:

1. what you ran (the full command, or what you did on the website and which browser);
2. what you expected and what happened - the exit code and the message;
3. `python --version`, the versions of numpy / numba, and whether the tables came from the release archive,
   from `site/tables` or were built by you;
4. if a package does not open: its size and `python -m hebzip info` of it - not the text.

A package that decompresses to something other than its text is the most serious bug there is - please
report it even if you cannot share the text.

## Changing the code

Run the checks before proposing a change:

    python tests/run_tests.py                  # every test file in its own process (needs the tables)
    python tools/compare_page/run_tests.py     # the comparison page and the benchmark
    python tools/site/build_site.py --check    # the website matches the code

Rules the tests enforce: files up to 400 lines, functions up to 50, mypy strict, ruff, every number a row of
`config/hebzip.toml`, and the golden packages (`golden/`) unchanged unless the format is changed on purpose
(`python -m tests.golden.make_golden`).
