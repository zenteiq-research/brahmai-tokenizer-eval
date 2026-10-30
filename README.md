<p align="center"><img src="assets/x.png" alt="X mark" width="110"></p>

<h1 align="center">BrahmAI Tokenizer Eval</h1>

<p align="center">by ZenteiQ AiTech Innovations Pvt. Ltd.</p>

<p align="center">
<a href="https://github.com/zenteiq-research/brahmai-tokenizer-eval/releases">Releases</a> ·
<a href="https://github.com/zenteiq-research/brahmai-tokenizer-eval/issues">Issues</a> ·
<a href="https://github.com/zenteiq-research/brahmai-tokenizer-eval/discussions">Discussions</a> ·
<a href="CITATION.cff">Cite</a> ·
<a href="#acknowledgements">Acknowledgements</a>
</p>

Compare tokenizers fairly, with deep coverage of Indian languages.

Given any tokenizer, this tool counts how many tokens it needs for the same text on fixed public
corpora, reports every number with a confidence interval, and checks the tokenizer for silent
failures: deleted characters, unreachable vocabulary entries, odd number handling, special-token
injection. Runs can be pinned and repeated, so a published number can be reproduced exactly.

It was built to evaluate Dhīr, ZenteiQ's tokenizer (Dhir-131k), which is not yet released. Results
for it will be added here with the release.

*Developed under the BrahmAI project, supported by the IndiaAI Mission (see
[Acknowledgements](#acknowledgements)).*

**Contents:**
[Quick start](#quick-start) ·
[What it measures](#what-it-measures) ·
[Corpora](#corpora) ·
[Offline use](#offline-use-data-archive) ·
[Output](#output) ·
[Reproducible runs](#reproducible-runs) ·
[Reading results](#reading-results) ·
[Extending](#adding-a-corpus) ·
[Development](#development) ·
[Licence and citation](#licence-and-citation) ·
[Acknowledgements](#acknowledgements)

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/zenteiq-research/brahmai-tokenizer-eval.git
cd brahmai-tokenizer-eval
pip install -e ".[tiktoken,transformers]"
export HF_TOKEN=...            # only for gated tokenizers (Llama, Gemma)

# 1. count tokens on the quick suite (7 corpora)
bte run -t sarvamai/sarvam-1 \
        -t meta-llama/Llama-3.2-1B \
        -t google/gemma-3-1b-pt \
        -t tiktoken:o200k_base \
        --suite quick --out results/quick

# 2. audit a tokenizer for silent failures; here OpenAI's general-purpose GPT-4o encoding (o200k_base)
bte probes -t tiktoken:o200k_base --out results/quick
```

`bte` is the short name of `brahmai-tokenizer-eval`. A run shows progress bars and summary tables;
the full tables are written to `results/quick/tables/`. The first tokenizer is the reference for
ratio tables (change it with `--reference`).

Other commands: `bte list` (corpora), `bte corpora` (download and verify), `bte reproduce`
(pinned run), `bte compare` (diff two runs), `bte report` (rebuild tables), `bte data` (offline
archive). Add `--help` to any of them.

### Naming a tokenizer

| You pass | Loaded as |
|---|---|
| `org/repo` or `org/repo@revision` | `tokenizer.json` at the repository root, else the first one in a subfolder, else `transformers.AutoTokenizer` (SentencePiece or vocab/merges files are converted when possible) |
| `path/to/tokenizer.json` or a directory | a local file |
| `tiktoken:<encoding>` | an OpenAI tiktoken encoding, e.g. `tiktoken:o200k_base` |
| `NAME=SPEC` | any of the above, labelled `NAME` in tables |

Tokenizers implemented as custom code in their repository (for example Kimi K2) load only with
`--trust-remote-code`. Pin `@revision` when you publish numbers.

## What it measures

Every metric comes from per-item counts (an item is a sentence, document or source file). That is
why every number carries a bootstrap interval and two tokenizers are compared on exactly the same
items.

| Metric | Definition | Use |
|---|---|---|
| **Tokens per 1,000 bytes** | 1000 · T / B | Primary metric. Lower means fewer tokens, which is better. |
| Bytes per token | B / T | Reciprocal, for comparison with other papers. |
| Characters per token | C / T | Reported, not ranked. |
| Fertility | T / W | Tokens per whitespace word; only for scripts that use spaces. |
| p95 tokens per item | 95th percentile of T per item | Long-tail cost. |
| Paired ratio | T_a / T_b on the same items | "Tokenizer a needs x% more tokens than b." |
| Token premium | T_lang / T_English on parallel text | Cost of a language relative to English [6]. |
| Efficiency index | geometric mean over declared domains of T_ref / T_tok | Optional single summary against one reference. |

T = tokens, B = UTF-8 bytes, C = code points, W = whitespace words. Bytes are the normaliser because
they are the only length unit that is identical across tokenizers and scripts. Special tokens are
never added. Intervals are 95% percentile bootstrap intervals over items (1,000 resamples, fixed
seed); paired ratios and premiums resample both sides with the same indices.

**When fertility is not reported.** Fertility is tokens per whitespace-separated word, so it only makes
sense where words are separated by spaces. It is left out (shown as `–` in tables) for 32 corpora:

* source code (the `code/*` corpora and HumanEval): `x=foo(a,b)` is one whitespace word but several real
  units, so "tokens per word" would be misleading;
* LaTeX source and Wikipedia formulas, for the same reason;
* languages written without spaces between words: Chinese (`flores/zho_Hans`), Japanese
  (`flores/jpn_Jpan`) and Burmese (`flores/mya_Mymr`).

For these corpora compare tokenizers by tokens per 1,000 bytes or bytes per token. A corpus opts out
with `whitespace_words=False` in the registry (see [Adding a corpus](#adding-a-corpus)).

### Probes (`bte probes`)

| Probe | What it catches |
|---|---|
| Unicode coverage | Round trip of every assigned code point (private use included); losses by UTF-8 lead byte and script. |
| Missing byte symbols | Byte values a byte-level vocabulary cannot produce. |
| Reachability | Entries that do not encode back to themselves; they are rarely emitted and their embedding rows go untrained. |
| Numbers | Integers 0–9,999, decimals, separators, years, nine Indian digit sets, GSM8K running text. |
| Normalisation | Token change under NFC/NFD and when ZWJ/ZWNJ are removed. |
| Special-token collision | Special-token strings in plain user text that become control tokens. |
| UTF-8 boundaries | Token boundaries inside a multi-byte character. |
| Throughput | Single-thread bytes per second and scaling with input length. |
| Composition | Vocabulary entries by script; entries spanning whitespace. |

## Corpora

`bte list` prints every corpus with its language, parallel group and licence. Suites:

| Suite | Content |
|---|---|
| `quick` | 7 corpora, a fast smoke test |
| `core` | general and technical text |
| `indic-eval` | benchmark translations in Indian languages |
| `indic` | all Indian-language corpora |
| `full` | all 184 corpora |

| Group | Source | Content | Licence | Obtained |
|---|---|---|---|---|
| `in22/*` | IN22-Gen (`mteb/IN22-Gen`, AI4Bharat) | 1,024 sentences parallel across English and all 22 scheduled languages | CC BY 4.0 | streamed, pinned |
| `flores/*` | FLORES-200 devtest (Meta) | 1,012 parallel sentences; 19 scheduled, 6 foreign, 4 neighbouring languages | CC BY-SA 4.0 | official tarball, SHA-256 checked |
| `indiceval-*/*` | Indic-LLM-Eval-Suite (AI4Bharat, NVIDIA) | GSM8K, MATH-500, ARC-Challenge, MMLU-Pro (2,000 rows), HumanEval, row-aligned translations into 14 languages | Apache-2.0, MIT, CC BY-SA | streamed, pinned |
| `milu/*` | MILU-1.5 | natively authored exam questions, 11 languages | CC BY 4.0 | streamed, pinned |
| `dakshina/*` | Dakshina v1.0 (Google) | 1,000 sentences per language, native script and romanized, 12 languages | CC BY-SA 4.0 | shipped |
| `code/*` | The Heap | 500 files per language with no exact or near duplicate in The Stack v1/v2 (Python, C++, C, Java, JavaScript, Rust, Go, COBOL) | copyleft, per repository | streamed, pinned |
| `code/Fortran` | `Bin12345/fortran-python` | 500 Fortran programs | see card | streamed, pinned |
| `math/*` | MATH, GSM8K test sets | 5,000 and 1,319 items | MIT | streamed, pinned |
| `science/PubMed-abstracts` | `ccdv/pubmed-summarization` | 2,000 test abstracts | see card | streamed, pinned |
| `science/arXiv-*-abstracts` | arXiv | 600 abstracts submitted 13–15 October 2026 | CC0 (metadata) | shipped |
| `latex/arXiv-*-source` | arXiv | main `.tex` of 60 papers from the same days (64 items, see note below) | CC BY, CC BY-SA or CC0, per paper | shipped |
| `latex/Wikipedia-formulas` | Wikipedia (Oct 2025) | 5,000 formulas | CC BY-SA 4.0 | streamed, pinned |

Nothing needs to be downloaded by hand and no PDF is converted. "Shipped" data is plain text inside
the package (`brahmai_tokenizer_eval/data/shipped/`); everything else is fetched on first use and
cached in `$BTE_CACHE` (default `~/.cache/brahmai-tokenizer-eval`).

**Cross-listed papers.** The four `latex/arXiv-*-source` corpora hold 64 items from 60 papers: four
cross-listed papers appear in two field corpora (`2610.22326`, `2610.22238`, `2610.22252`,
`2610.22242`). Each field corpus is complete on its own; pooling all four counts those papers twice.

The arXiv corpora are dated after most current tokenizers were trained, so they cannot be in their
training data. For a tokenizer trained after October 2026, refresh them with
`tools/build_arxiv_heldout.py` and `tools/arxiv_licences.py` (maintainers only).

### Data integrity

Every corpus is checked against its SHA-256 in `brahmai_tokenizer_eval/data/corpora.lock.json`.

* **Strict by default.** A corpus that differs from the lock stops `run`, `probes` and `reproduce`.
  `MISMATCH` means the upstream data changed and results are not comparable with published ones.
  `--allow-data-mismatch` continues for an exploratory run.
* **Check without running:** `bte corpora --suite full` builds every corpus and compares it with the
  lock; `bte data check --suite full` reports ok / missing / MISMATCH for what is already cached.

## Offline use (data archive)

**You usually do not need the archive.** Without it, each corpus is downloaded from its pinned source
the first time it is used and cached in `$BTE_CACHE`. The archive is for machines without internet
access, or when you want every corpus from one verified file.

All redistributable corpora are published as one archive, `bte-corpora-<id>.tar.gz`, with a `.sha256`
file next to it, attached to the
[GitHub Releases](https://github.com/zenteiq-research/brahmai-tokenizer-eval/releases) of this repository.
`<id>` is the first 12 characters of the SHA-256 of `corpora.lock.json`, so it tells you which corpus
versions the archive holds. You do not unpack it yourself: `bte data install` verifies it and
writes the corpora into the cache.

```bash
# 1. download the archive and its checksum (any method works; this one uses the GitHub CLI)
gh release download --repo zenteiq-research/brahmai-tokenizer-eval --pattern 'bte-corpora-*'

# 2. install: checks the SHA-256, then every corpus against the lock file, then fills the cache
bte data install bte-corpora-<id>.tar.gz

# 3. confirm, then run without any download
bte data check --suite full                  # ok / missing / MISMATCH per corpus
bte --offline run -t my/tokenizer --suite core --out results/core
```

Tokenizers are not part of the archive. `--offline` only stops corpus downloads; load tokenizers from
local files or the Hugging Face cache (see [Reproducible runs](#reproducible-runs)).

`install` refuses members with unsafe paths (absolute, `..`, links) and writes nothing to the cache
if any check fails. The archive is deterministic: the same corpora always give a byte-identical file.
Its `LICENCES.md` lists the source, creators and licence of every corpus.

Ten corpora are not in the archive because their licences do not allow redistribution as is: the
eight The Heap code sets (copyleft files that would need their licences and notices), `code/Fortran`
and `science/PubMed-abstracts` (no licence stated). They are built from their pinned source when you
are online, and the lock file checks the result is byte-identical to ours. Maintainers build archives
with `bte data pack --suite full --out dist/`.

## Output

`bte run` writes into `--out`:

| Path | Content |
|---|---|
| `counts/` | per-item counts |
| `run.json` | tokenizer revisions, vocabulary fingerprints, corpus checksums, vocabulary utilisation, software versions |
| `tables/per_corpus` | tokens per 1,000 bytes with 95% interval, bytes per token, fertility, p95, round-trip failures, for every tokenizer and corpus |
| `tables/ratio_vs_<reference>` | paired ratios against the reference tokenizer, pooled by corpus group |
| `tables/premium_<group>` | token premium over English for every parallel group |
| `tables/groups_vs_<reference>` | per named group of corpora: pooled ratio and geometric mean of per-corpus ratios with 95% intervals, plus the median, minimum and maximum per-corpus ratio with the corpus named |

Tables are written as `.csv` and `.md`.

### Example

An excerpt from a real run (`--suite quick`, three public tokenizers, not pinned, run on 29 October 2026):

```bash
bte run -t Sarvam-1=sarvamai/sarvam-1 -t Qwen2.5=Qwen/Qwen2.5-0.5B -t o200k=tiktoken:o200k_base \
        --suite quick --out results/example
```

`tables/per_corpus.md` (four of the seven corpora, a few of the columns):

| Corpus | Tokenizer | Tokens per 1,000 bytes [95% CI] | Bytes per token | Fertility | Round-trip failures |
|---|---|---|---|---|---|
| `in22/eng_Latn` | Sarvam-1 | 253.7 [250.3, 257.2] | 3.94 | 1.60 | 0 |
| `in22/eng_Latn` | Qwen2.5 | 219.4 [216.5, 222.4] | 4.56 | 1.38 | 0 |
| `in22/eng_Latn` | o200k | 210.2 [208.0, 212.5] | 4.76 | 1.32 | 0 |
| `in22/hin_Deva` | Sarvam-1 | 108.2 [107.0, 109.5] | 9.24 | 1.51 | 0 |
| `in22/hin_Deva` | Qwen2.5 | 359.0 [358.2, 359.9] | 2.79 | 5.00 | 1 |
| `in22/hin_Deva` | o200k | 126.1 [124.9, 127.3] | 7.93 | 1.76 | 0 |
| `in22/tam_Taml` | Sarvam-1 | 95.2 [94.2, 96.4] | 10.50 | 2.42 | 0 |
| `in22/tam_Taml` | Qwen2.5 | 400.5 [399.9, 401.3] | 2.50 | 10.17 | 0 |
| `in22/tam_Taml` | o200k | 133.3 [132.2, 134.4] | 7.50 | 3.39 | 0 |
| `code/Python` | Sarvam-1 | 346.4 [340.4, 352.8] | 2.89 | – | 1 |
| `code/Python` | Qwen2.5 | 234.3 [229.5, 239.8] | 4.27 | – | 0 |
| `code/Python` | o200k | 231.2 [227.1, 235.7] | 4.33 | – | 0 |

`tables/ratio_vs_Sarvam-1.md`. A ratio above 1 means the tokenizer needs more tokens than the reference:

| Group | Tokenizer | Ratio vs Sarvam-1 [95% CI] |
|---|---|---|
| natural-language:in22 | Qwen2.5 | 2.97 [2.92, 3.01] |
| natural-language:in22 | o200k | 1.13 [1.12, 1.14] |
| code:code | Qwen2.5 | 0.68 [0.67, 0.68] |
| code:code | o200k | 0.67 [0.66, 0.67] |
| math:math | Qwen2.5 | 0.87 [0.87, 0.87] |
| math:math | o200k | 0.84 [0.84, 0.84] |
| science:science | Qwen2.5 | 0.87 [0.87, 0.87] |
| science:science | o200k | 0.83 [0.83, 0.83] |

In the table, `–` in the Fertility column means fertility is not defined for that corpus (code, see
[When fertility is not reported](#what-it-measures)); the other columns are measured as usual. A
non-zero round-trip failure count is a real result: `code/Python` shows 1 for Sarvam-1, one item that
did not decode back to the original text, so that tokenizer's count there is slightly low.
See [Round-trip failures](#reading-results) for the classes.

Read it like this: on the IN22 group (English plus Hindi, Tamil and Bengali) Qwen2.5 needs about three
times as many tokens as Sarvam-1, driven by the Indian languages (Hindi 359 vs 108 tokens per 1,000
bytes), yet it needs fewer on code, maths and science. That is why results are reported per corpus
and per group and never as one number. The values are only an illustration and change with tokenizer
revisions; pin tokenizers with a manifest to publish a result (see
[Reproducible runs](#reproducible-runs)).

## Reproducible runs

A published number should come from `bte reproduce` with a manifest, not from a hand-typed `bte run`:

```bash
bte reproduce examples/quick.toml --out results/quick
bte reproduce examples/quick.toml --out results/quick-rerun
bte compare results/quick results/quick-rerun     # identical per-item counts, or exit 1
```

A manifest (TOML, documented in `brahmai_tokenizer_eval/manifest.py`, example in
`examples/quick.toml`) lists the corpora (`suite` and/or `corpora`), the reference tokenizer, which
tokenizers to probe, optional named `[groups]`, and every tokenizer with a pin:

| Tokenizer | Pin |
|---|---|
| Hugging Face | `hub`, full 40-character `revision`, and `sha256` of the `tokenizer.json` loaded |
| local file | `path` (relative to the manifest) and `sha256` |
| tiktoken | `tiktoken` encoding and the vocabulary `fingerprint` |

`reproduce` loads every tokenizer, checks it against its pin and stops at the first mismatch, before
anything is counted. It refuses to run from a checkout with uncommitted changes (`--allow-dirty`
overrides, for development).

**Provenance.** Every `run.json` and `probes.run.json` records the suite's git commit and whether the
tree was dirty, the versions of the packages that can change a count (`tokenizers`, `tiktoken`,
`transformers`, `sentencepiece`, `huggingface_hub`, `numpy`, `pandas`, `pyarrow`, `regex`), Python and
platform, the command line, and the manifest's SHA-256. `requirements.lock` is the exact environment
used for published results:

```bash
pip install -r requirements.lock && pip install --no-deps -e .
```

**What is deterministic.** Corpora (pinned sources, seeded sampling, checksums), token counts,
intervals (fixed bootstrap seed) and every probe except throughput are bit-identical between runs on
the same pins and package versions. Throughput is wall-clock time and varies with the machine.

**Offline tokenizers.** When the Hub cannot be reached (`HF_HUB_OFFLINE=1`, no network), a Hub
tokenizer is loaded from the local Hugging Face cache: at the pinned commit if one is given,
otherwise the only cached snapshot. A gated repository can withdraw access; the pinned SHA-256 makes
any copy of the right file equivalent, so placing it in the cache lets `reproduce` accept it.

## Reading results

**Pooled vs geometric mean.** "Tokenizer A needs x% more tokens than B on twelve languages" can mean
two different numbers:

* **pooled**: total tokens of A over total tokens of B across all items. Each language weighs by its
  token count, so one language where A is very poor can dominate.
* **geometric mean** of the twelve per-language ratios. Each language weighs equally.

They can differ a lot: when a tokenizer does not cover a language (Urdu, for example), that language
can dominate the pooled ratio but is one of twelve in the geometric mean. `groups_vs_<reference>`
reports both, with the median and the extreme language, so a text can name its method.

**Token premium.** A premium compares a language with English *within one tokenizer*. A tokenizer
that is very efficient on English shows a higher premium for the same Indian-language cost, so read
premiums next to absolute tokens per 1,000 bytes.

**Round-trip failures.** A tokenizer whose round trip changes text looks efficient because it drops
characters, so check the round-trip column first. Each failure is classified:

* `nfc`: the decoded text is the NFC form of the input; the count measures the NFC text (legitimate
  for Qwen3, GPT-NeoX).
* `leading_space`: the decoder adds one space; nothing is lost.
* `lossy`: characters were dropped or changed, so the count is too low.

The counts files keep the kind and the lost bytes of every failing item.

## Adding a corpus

Add a `Corpus(...)` entry in `brahmai_tokenizer_eval/registry.py` with an existing builder
(`hf_parquet_column`, `hf_wide_translation`, `url_tar_lines`, `shipped`, `heap`, `hf_parquet_concat`,
`hf_parquet_sample`, `hf_csv_sample`, `milu_native`) or a new `b_<name>` function in `corpora.py`.
State the licence, set `parallel` only for sentence-aligned translations and
`whitespace_words=False` where fertility is meaningless, then run
`bte corpora --corpus <id> --lock`.

## Related work

Other tools evaluate tokenizers; this suite differs in what it fixes and what it checks.

* **Tokka-Bench** [1] measures bytes per token, unique-token coverage, fertility and word-split
  rate on FineWeb-2 and StarCoder samples across 100 natural and 20 programming languages.
* **TokLens** [2] reports six intrinsic metrics (fertility, characters per token, compression
  ratio, normalised sequence length, single-token retention, cross-lingual parity) for 24 tokenizers
  and relates them to downstream results.
* **TokEval** [3] adds structure-sensitive metrics and validates them against pretraining.
* **TokSuite** [4] releases fourteen language models that differ only in their tokenizer, to study
  the effect of the tokenizer on model behaviour.

This suite adds pinned and dated held-out corpora for the 22 scheduled Indian languages, paired
bootstrap intervals on identical items, and probes for silent tokenizer artefacts (lossy round
trips, unreachable entries, special-token injection). It does not train models.

The metric definitions build on fertility [5] and token premium [6]. Whether efficiency matters for
downstream quality is debated [7]; see [Limitations](#limitations).

## References

1. B. Gubler. *Tokka-Bench: Evaluating Tokenizers Across 100 Natural and 20 Programming Languages.*
   arXiv:2610.08794. Code: https://github.com/bgub/tokka-bench
2. G.-M. Chiu. *TokLens: A Multilingual Lens on Tokenizer Quality for LLMs.* ACL 2026 Student
   Research Workshop, pp. 188–205. https://aclanthology.org/2026.acl-srw.18/
3. C. Meister. *TokEval: A Tokenizer Evaluation Suite.* COLM 2026. arXiv:2608.18062.
4. G. S. Altıntaş et al. *TokSuite: Measuring the Impact of Tokenizer Choice on Language Model
   Behavior.* ICML 2026. arXiv:2512.20757.
5. P. Rust, J. Pfeiffer, I. Vulić, S. Ruder, I. Gurevych. *How Good is Your Tokenizer? On the
   Monolingual Performance of Multilingual Language Models.* ACL 2021.
6. A. Petrov, E. La Malfa, P. H. S. Torr, A. Bibi. *Language Model Tokenizers Introduce Unfairness
   Between Languages.* NeurIPS 2023. arXiv:2305.15425.
7. C. W. Schmidt et al. *Tokenization Is More Than Compression.* EMNLP 2024. arXiv:2402.18376.

Cite the datasets you report results on as listed in
`brahmai_tokenizer_eval/data/shipped/README.md` and the archive's `LICENCES.md`.

## Limitations

* Efficiency is necessary, not sufficient: fewer tokens do not imply a better model [7].
* Only the arXiv corpora (submitted October 2026) are guaranteed unseen by tokenizers trained earlier. The Heap code is deduplicated against The Stack only. Public benchmarks and web text may appear in a tokenizer's training data, so read results on them as comparisons between tokenizers.
* Bodo, Dogri and Konkani have parallel text only in IN22-Gen.
* Romanized text is Wikipedia-style (Dakshina), not conversational.

## Development

```bash
pip install -e ".[test]"
pytest -q
```

The tests run offline on a tokenizer trained inside the test. They cover metric definitions,
intervals, manifests and pin verification, provenance, round-trip classification, group summaries,
`compare`, the data archive, and edge cases (empty, control-character, combining, emoji and very long
text; missing, damaged or unknown tokenizer specs; empty corpora; damaged archives; duplicate
tokenizer names; command-line errors).

Errors are one line (`error: ...`) with exit code 1; set `BTE_DEBUG=1` for the full traceback.

## Licence and citation

| | Licence |
|---|---|
| **Code** | Apache-2.0 (`LICENSE`, `NOTICE`) |
| **Data** | Not covered by the code licence. Each dataset keeps its own licence and attribution; see `brahmai_tokenizer_eval/data/shipped/README.md` and the archive's `LICENCES.md`. |

**Data notes**

* Dakshina and some of the arXiv papers are CC BY-SA 4.0, and their copies stay under that licence.
* Authors, titles, links and licences of the shipped arXiv papers are listed under `attribution` in
  `brahmai_tokenizer_eval/data/shipped/arxiv/MANIFEST.json`.
* The shipped LaTeX files are modified copies: comments removed, text truncated, and e-mail addresses
  replaced by `author@example.org`.
* When you report results on a corpus, cite its original dataset.

**Citing this software.** Use `CITATION.cff` (GitHub's "Cite this repository" button reads it).

**Questions and contributions**

* Issues are welcome for bug reports and corpus or metric suggestions. Use Discussions for questions
  and ideas.
* We do not accept pull requests at this time.
* Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
* Security problems: see [SECURITY.md](SECURITY.md).
* Contact: research@zenteiq.ai.

## Acknowledgements

### Funding

This tool was developed as part of the BrahmAI project, supported by the IndiaAI Mission.
ZenteiQ acknowledges the support of the IndiaAI Mission, Ministry of Electronics and Information
Technology (MeitY), under the Foundation Model initiative.

### Datasets

Every number this tool reports comes from data that other people created. We thank the creators and
maintainers of these datasets: AI4Bharat, Meta, Google Research, NVIDIA, the authors of MATH and
GSM8K, the authors of The Heap, the arXiv authors, and the Wikipedia contributors.

**If you publish results, cite the datasets you used, as well as this tool.** The BibTeX for every
dataset below is in [`brahmai_tokenizer_eval/data/CITATIONS.bib`](brahmai_tokenizer_eval/data/CITATIONS.bib),
copied from each dataset's own page. The tool's own citation is in [`CITATION.cff`](CITATION.cff).

| Corpora | Created by | Cite these keys from `CITATIONS.bib` |
|---|---|---|
| `in22/*` | AI4Bharat (IndicTrans2) | `gala2023indictrans` |
| `milu/*` | AI4Bharat | `verma-etal-2025-milu` |
| `indiceval-*/*` | AI4Bharat and NVIDIA (Indic-LLM-Eval-Suite), built on GSM8K, MATH, ARC, MMLU-Pro and HumanEval | `indicllmevalsuite2026blog`, plus the papers of the original benchmarks |
| `flores/*` | Meta (FLORES-200, NLLB Team) | `nllb2022`, `goyal2021flores101`, `guzman2019flores` |
| `dakshina/*` | Google Research | `roark-etal-2020-processing` |
| `math/MATH-test` | Hendrycks et al. | `hendrycksmath2021` |
| `math/GSM8K-test` | Cobbe et al., OpenAI | `cobbe2021gsm8k` |
| `code/*` (The Heap) | TU Delft; each file keeps its original code licence | `11052803` |
| `latex/Wikipedia-formulas` | Horn and Keuper; Wikipedia contributors | `horn2025benchmarking` |
| `science/arXiv-*`, `latex/arXiv-*` | the authors of each paper | no BibTeX key; the `attribution` list in [`MANIFEST.json`](brahmai_tokenizer_eval/data/shipped/arxiv/MANIFEST.json) names every paper, its authors and its licence |
| `science/PubMed-abstracts`, `code/Fortran` | no citation supplied | none |

The licence of each dataset, and whom to credit, is also in
[`brahmai_tokenizer_eval/data/shipped/README.md`](brahmai_tokenizer_eval/data/shipped/README.md) and in
`LICENCES.md` inside the data archive.
