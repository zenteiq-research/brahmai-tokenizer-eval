# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Registry of evaluation corpora.

Every corpus is public and pinned. A corpus is either streamed from a fixed
source (a Hugging Face dataset revision or a URL with a known SHA-256) or
shipped in `brahmai_tokenizer_eval/data/shipped/` when its licence allows redistribution. Builders
materialise each corpus to `<cache>/<corpus_id>.jsonl`; `brahmai_tokenizer_eval/data/corpora.lock.json`
records the SHA-256 of every materialised file so a rebuild can be verified.

Corpus ids are "<group>/<subset>". Corpora that share a `parallel` group hold
translations of the same sentences in the same order, which is what token
premium (cost relative to English) requires.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEED = 20261018


@dataclass(frozen=True)
class Corpus:
    id: str
    domain: str                 # natural-language | code | math | science | latex | romanized
    source: str                 # builder name, see corpora.py
    args: dict
    licence: str
    lang: str | None = None     # ISO 639-3 + script, e.g. hin_Deva
    parallel: str | None = None # parallel group name
    whitespace_words: bool = True  # fertility (tokens per word) is meaningful
    note: str = ""


# ── language metadata ─────────────────────────────────────────────────────────

SCHEDULED = {  # the 22 languages of the Eighth Schedule of the Constitution of India
    "asm": "Assamese", "ben": "Bengali", "brx": "Bodo", "doi": "Dogri", "guj": "Gujarati",
    "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri", "gom": "Konkani", "mai": "Maithili",
    "mal": "Malayalam", "mni": "Manipuri", "mar": "Marathi", "npi": "Nepali", "ory": "Odia",
    "pan": "Punjabi", "san": "Sanskrit", "sat": "Santali", "snd": "Sindhi", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}
OTHER_NAMES = {"eng": "English", "deu": "German", "fra": "French", "rus": "Russian",
               "zho": "Chinese (Simplified)", "jpn": "Japanese", "kor": "Korean",
               "pbt": "Pashto", "prs": "Dari", "mya": "Burmese", "sin": "Sinhala"}
NO_SPACE_SCRIPTS = {"Hans", "Hant", "Jpan", "Mymr", "Thai", "Khmr", "Laoo"}


def lang_name(code: str) -> str:
    iso = code.split("_")[0]
    return SCHEDULED.get(iso) or OTHER_NAMES.get(iso) or code


def _ws(code: str) -> bool:
    return code.split("_")[-1] not in NO_SPACE_SCRIPTS


# ── IN22-Gen: 1,024 sentences, English + all 22 scheduled languages ──────────

IN22_REPO, IN22_REV = "mteb/IN22-Gen", "78dfb78e623ef9afbc8db9189a9914db0f0a8403"
IN22_LANGS = ["eng_Latn", "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
              "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Mtei",
              "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck", "snd_Deva", "tam_Taml",
              "tel_Telu", "urd_Arab"]

# ── FLORES-200 devtest: 1,012 sentences ──────────────────────────────────────

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
FLORES_LANGS = ["eng_Latn", "asm_Beng", "ben_Beng", "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab",
                "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Beng", "npi_Deva", "ory_Orya", "pan_Guru",
                "san_Deva", "sat_Olck", "snd_Arab", "tam_Taml", "tel_Telu", "urd_Arab",
                "deu_Latn", "fra_Latn", "rus_Cyrl", "zho_Hans", "jpn_Jpan", "kor_Hang",
                "pbt_Arab", "prs_Arab", "mya_Mymr", "sin_Sinh"]

# ── Dakshina: parallel native-script / romanized Wikipedia sentences ─────────

DAKSHINA = {"bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "ml": "mal_Mlym",
            "mr": "mar_Deva", "pa": "pan_Guru", "sd": "snd_Arab", "si": "sin_Sinh", "ta": "tam_Taml",
            "te": "tel_Telu", "ur": "urd_Arab"}

# ── The Heap: code with no exact/near duplicate in The Stack v1 or v2 ────────

HEAP_REPO, HEAP_REV = "AISE-TUDelft/the-heap", "fdb3fa670069a1d0399d0fea0bf596ffe0dae319"
HEAP_LANGS = ["Python", "C++", "C", "Java", "JavaScript", "Rust", "Go", "COBOL"]


# ── Indic-LLM-Eval-Suite (AI4Bharat & NVIDIA, 2026): English originals with ──
# row-aligned translations into 14 languages, plus natively authored MILU-1.5

INDIC_EVAL_LANGS = {"Assamese": "asm_Beng", "Bengali": "ben_Beng", "Gujarati": "guj_Gujr", "Hindi": "hin_Deva",
                    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Marathi": "mar_Deva", "Nepali": "npi_Deva",
                    "Odia": "ory_Orya", "Punjabi": "pan_Guru", "Sanskrit": "san_Deva", "Tamil": "tam_Taml",
                    "Telugu": "tel_Telu", "Urdu": "urd_Arab"}
INDIC_EVAL = {  # name: (repo, revision, file, [(english field, translated field template)], domain, licence, sample)
    "gsm8k": ("ai4bharat/indic-gsm8k", "8578916cab75ed71d6e89ca9ea7a187d27b2346c", "test.parquet",
              [("question", "question_{L}_translation"), ("answer", "answer_{L}_translation")], "math", "Apache-2.0", None),
    "math500": ("ai4bharat/indic-math-500", "dcb39428f4aeae76562830ba2365ca3b11bdd1d5", "test.parquet",
                [("problem", "problem_{L}_translation"), ("solution", "solution_{L}_translation")], "math", "Apache-2.0", None),
    "arc": ("ai4bharat/indic-arc-challenge", "8a936d8148759ec51a28515d2379b8eae13ce65e", "test.parquet",
            [("question", "question_{L}_translation"), ("choices", "choices_text_{L}_translation")], "science", "CC-BY-SA-4.0", None),
    "mmlupro": ("ai4bharat/indic-mmlu-pro", "7502b81cc6b3ecf4b9556471aafbc18bd25b61e9", "test.parquet",
                [("question", "question_{L}_translation"), ("options", "options_{L}_translation")], "science", "MIT", 2000),
    "humaneval": ("ai4bharat/indic-humaneval", "6e2f8a52b33eab1ce6cb59ec736d630c47299fb5", "test.parquet",
                  [("prompt", "prompt_{L}_translation")], "code", "Apache-2.0", None),
}
MILU_REV = "d8a3cc2b4223989399adef5c2685915099e371a5"
MILU_LANGS = ["English", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Odia", "Punjabi",
              "Tamil", "Telugu"]


def _corpora() -> list[Corpus]:
    out: list[Corpus] = []
    for l in IN22_LANGS:
        out.append(Corpus(f"in22/{l}", "natural-language", "hf_parquet_column",
                          {"repo": IN22_REPO, "revision": IN22_REV, "file": "test.parquet", "column": l},
                          "CC-BY-4.0", lang=l, parallel="in22", whitespace_words=_ws(l)))
    for l in FLORES_LANGS:
        out.append(Corpus(f"flores/{l}", "natural-language", "url_tar_lines",
                          {"url": FLORES_URL, "sha256": FLORES_SHA,
                           "member": f"flores200_dataset/devtest/{l}.devtest"},
                          "CC-BY-SA-4.0", lang=l, parallel="flores", whitespace_words=_ws(l)))
    for code, l in DAKSHINA.items():
        for form in ("native", "roman"):
            out.append(Corpus(f"dakshina/{l}.{form}", "natural-language" if form == "native" else "romanized",
                              "shipped", {"file": f"dakshina/{code}.{form}.txt"}, "CC-BY-SA-4.0",
                              lang=l if form == "native" else f"{l.split('_')[0]}_Latn",
                              parallel=f"dakshina-{code}", note="first 1,000 lines of the rejoined test split"))
    for lang in HEAP_LANGS:
        out.append(Corpus(f"code/{lang}", "code", "heap",
                          {"repo": HEAP_REPO, "revision": HEAP_REV, "language": lang, "n": 500, "min_chars": 200, "max_chars": 50_000},
                          "per-file copyleft licences of the source repositories (The Heap: GPL, LGPL, AGPL, MPL, EPL and others)", whitespace_words=False))
    out.append(Corpus("code/Fortran", "code", "hf_csv_sample",
                      {"repo": "Bin12345/fortran-python", "revision": "186897eb4e3e69b9e302a5da7c7b5c11ebdeed8d", "file": "data.csv", "column": "Fortran",
                       "n": 500, "min_chars": 200, "max_chars": 50_000},
                      "see dataset card", whitespace_words=False,
                      note="Fortran is not in The Heap; may overlap The Stack"))
    out.append(Corpus("math/MATH-test", "math", "hf_parquet_concat",
                      {"repo": "EleutherAI/hendrycks_math", "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3", "suffix": "test-00000-of-00001.parquet",
                       "template": "{problem}\n\n{solution}"}, "MIT"))
    out.append(Corpus("math/GSM8K-test", "math", "hf_parquet_concat",
                      {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866", "files": ["main/test-00000-of-00001.parquet"],
                       "template": "{question}\n{answer}"}, "MIT"))
    out.append(Corpus("science/PubMed-abstracts", "science", "hf_parquet_sample",
                      {"repo": "ccdv/pubmed-summarization", "revision": "6b30a2cae59b11ed77cb19959bffccbbd18e1106", "file": "document/test-00000-of-00001.parquet",
                       "column": "abstract", "n": 2000, "min_chars": 200}, "see dataset card"))
    out.append(Corpus("latex/Wikipedia-formulas", "latex", "hf_parquet_sample",
                      {"repo": "piushorn/wikipedia-latex-formulas-319k",
                       "revision": "7f2feb6ce75e66d6075ad8167d31d9a0c536614f", "file": "data.parquet",
                       "column": "formula", "n": 5000, "min_chars": 0}, "CC-BY-SA-4.0",
                      whitespace_words=False, note="formulas from English Wikipedia (Oct 2025 snapshot)"))
    for g in ("math", "physics", "cs", "qbio-stat"):
        out.append(Corpus(f"science/arXiv-{g}-abstracts", "science", "shipped",
                          {"file": f"arxiv/abstracts.{g}.jsonl"}, "CC0-1.0 (arXiv metadata)",
                          note="submitted 2026-10-13..2026-10-15"))
        out.append(Corpus(f"latex/arXiv-{g}-source", "latex", "shipped",
                          {"file": f"arxiv/latex.{g}.jsonl"}, "CC BY / CC BY-SA / CC0 (per paper)",
                          whitespace_words=False, note="main .tex of CC BY / CC BY-SA / CC0 papers, submitted 2026-10-13..2026-10-15"))
    for name, (repo, rev, f, fields, dom, lic, n) in INDIC_EVAL.items():
        for L, code in [("English", "eng_Latn")] + list(INDIC_EVAL_LANGS.items()):
            out.append(Corpus(f"indiceval-{name}/{code}", dom, "hf_wide_translation",
                              {"repo": repo, "revision": rev, "file": f, "fields": fields, "language": L, "sample": n},
                              lic, lang=code, parallel=f"indiceval-{name}",
                              whitespace_words=dom != "code",
                              note="translated by the benchmark authors" if L != "English" else "English original"))
    for L in MILU_LANGS:
        code = "eng_Latn" if L == "English" else INDIC_EVAL_LANGS[L]
        out.append(Corpus(f"milu/{code}", "knowledge", "milu_native",
                          {"repo": "ai4bharat/MILU-1.5", "revision": MILU_REV, "language": L},
                          "CC-BY-4.0", lang=code, note="natively authored items only (is_translated == False)"))
    return out


CORPORA: dict[str, Corpus] = {c.id: c for c in _corpora()}

SUITES = {
    "core": [c for c in CORPORA if not c.startswith(("dakshina/", "indiceval-", "milu/"))],
    "indic-eval": [c for c in CORPORA if c.startswith(("indiceval-", "milu/"))],
    "indic": [c for c in CORPORA if c.startswith(("in22/", "dakshina/", "indiceval-", "milu/"))
              or (c.startswith("flores/") and CORPORA[c].lang.split("_")[0] in SCHEDULED | {"eng": 1})],
    "full": list(CORPORA),
    "quick": [f"in22/{l}" for l in ("eng_Latn", "hin_Deva", "tam_Taml", "ben_Beng")]
             + ["code/Python", "math/MATH-test", "science/PubMed-abstracts"],
}
