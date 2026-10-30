# Shipped data

Only data whose licence allows redistribution is stored here. Everything else
is streamed from a pinned source at build time (see `brahmai_tokenizer_eval/registry.py`).

| Folder | Source | Licence | Notes |
|---|---|---|---|
| `dakshina/` | Dakshina dataset v1.0, Google Research (Roark et al., 2020), `https://github.com/google-research-datasets/dakshina` | CC BY-SA 4.0 | First 1,000 lines of each language's `romanized/<lang>.romanized.rejoined.test.{native,roman}.txt`, unchanged. Source tarball SHA-256 `83530bec211893d9df429298f42e63c350c25ad56f9f5172f85edb0575ee5b03`. |
| `arxiv/abstracts.*.jsonl` | arXiv API, submissions of 13–15 October 2026 | CC0 1.0 (arXiv metadata) | Abstract text with whitespace collapsed; `arxiv_id` per line. |
| `arxiv/latex.*.jsonl` | arXiv e-prints of the same days | per paper: CC BY 4.0, CC BY-SA 4.0 or CC0 | Main `.tex` file (the one containing `\documentclass`), whole-line comments removed, truncated to 50,000 characters, e-mail addresses replaced by `author@example.org`. Each line records `arxiv_id` and `licence`; `arxiv/MANIFEST.json` lists every paper checked and its licence as reported by arXiv's OAI-PMH record. Papers under arXiv's non-exclusive licence were excluded. |

### Attribution and changes (CC BY, CC BY-SA)

`arxiv/MANIFEST.json`, key `attribution`, lists for every paper whose source is shipped its
title, all authors, arXiv link and licence, as CC BY 4.0 and CC BY-SA 4.0 require.

The shipped LaTeX files are **modified copies** of the authors' sources: whole-line comments
removed, truncated to 50,000 characters, e-mail addresses replaced by `author@example.org`.
Nothing else is changed. The authors and arXiv do not endorse this dataset or its use.

Each paper keeps its own licence. Papers under CC BY-SA 4.0 and anything adapted from them
remain under CC BY-SA 4.0; this is separate from the Apache-2.0 licence of the code.

When you publish results on these files, cite the original works: Dakshina
(Roark et al., 2020) and the individual arXiv papers listed in `MANIFEST.json`.
