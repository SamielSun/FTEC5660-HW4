"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup  # noqa: E402  (imported here on purpose: it is part of our code)

# Every component of the pipeline can be switched off, one at a time, so that the ablation study of
# Task 5 is a real measurement and not a story. The defaults are the pipeline we submit; ablation.py
# flips the flags and re-runs the whole thing.
CONFIG = {
    "table_to_text": True,    # tables as row/column text instead of a stream of numbers
    "section_split": True,    # chunk along the h2/h3 sections
    "chunk_header": True,      # prepend the metadata + section header to every chunk
    "metadata_filter": True,   # filter candidates by company / fiscal year / document type
    "company_split": True,     # retrieve separately per company for comparison questions
    "abstain": True,           # return [] when the corpus cannot answer
    "stopwords": True,         # financial tokeniser: stop words, 2,710.4 also as 2710.4
}

# =============================================================================
# Task 1.1: tables to text
# =============================================================================


def _cell_text(cell) -> str:
    """The text of one table cell, printed as in the document (whitespace only collapsed)."""
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()


def _table_grid(table) -> tuple[list[dict[int, str]], list[bool]]:
    """The table as rows of {column index: text}, plus a flag per row saying whether it is all <th>.

    rowspan and colspan are resolved here so that every value knows its real column, which is what
    links it to its column header.
    """
    rows: list[dict[int, str]] = []
    all_th: list[bool] = []
    carried: dict[int, list] = {}          # column -> [rows still to fill, text]

    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        if not cells:
            continue
        row: dict[int, str] = {}
        column = 0
        for cell in cells:
            while column in carried:                       # fill cells carried down from a rowspan
                row[column] = carried[column][1]
                carried[column][0] -= 1
                if carried[column][0] <= 0:
                    del carried[column]
                column += 1
            rowspan = int(cell.get("rowspan", 1) or 1)
            colspan = int(cell.get("colspan", 1) or 1)
            text = _cell_text(cell)
            for offset in range(colspan):
                row[column + offset] = text
                if rowspan > 1:
                    carried[column + offset] = [rowspan - 1, text]
            column += colspan
        while column in carried:                           # rowspan cells at the end of the row
            row[column] = carried[column][1]
            carried[column][0] -= 1
            if carried[column][0] <= 0:
                del carried[column]
            column += 1
        rows.append(row)
        all_th.append(tr.find("td") is None)               # a header row holds only <th>

    return rows, all_th


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if table is None:
        return ""

    rows, all_th = _table_grid(table)
    if not rows:
        return ""

    lines: list[str] = []

    caption = table.find("caption")                        # the caption holds the unit: never drop it
    if caption is not None:
        text = _cell_text(caption)
        if text:
            lines.append(text)

    header_index = next((i for i, flag in enumerate(all_th) if flag), 0)
    header = rows[header_index]

    for index, row in enumerate(rows):
        if index == header_index or not row:
            continue
        columns = sorted(row)
        label = row[columns[0]]
        if len(columns) == 1:                              # a section row spanning the whole table
            if label:
                lines.append(label)
            continue
        parts = [label]
        for column in columns[1:]:
            value = row[column]
            if not value:
                continue
            column_header = header.get(column, "")
            parts.append(f"{column_header}: {value}" if column_header else value)
        line = " | ".join(part for part in parts if part)
        if line:
            lines.append(line)

    return "\n".join(lines)


# =============================================================================
# Task 1.2: chunking with metadata
# =============================================================================

_ANNUAL_RE = re.compile(r"annual report\s+(\d{4})", re.I)
_CALL_RE = re.compile(r"(?:q4 and )?full year\s+(\d{4})", re.I)
_REVIEW_RE = re.compile(r"fy\s?(\d{4})", re.I)
_YEAR_RE = re.compile(r"(20\d{2})")


def _clean_company(h1_text: str) -> str:
    """'Orchid Bay Hotels Ltd.' -> 'Orchid Bay Hotels Ltd.'; 'Brightwater Energy Corp. (TSX: BWE)' -> the name.

    A legal suffix may stay (company_key drops it); the ticker in brackets must go.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", h1_text or "").strip()


def _document_identity(soup) -> tuple[str, int]:
    """The document type and its fiscal year, from the <title> and the first lines of the cover."""
    parts = []
    if soup.title is not None:
        parts.append(soup.title.get_text(" ", strip=True))
    h1 = soup.find("h1")
    if h1 is not None:
        parts.append(h1.get_text(" ", strip=True))
    parts.extend(p.get_text(" ", strip=True) for p in soup.find_all("p")[:6])
    head = " ".join(part for part in parts if part)
    low = head.lower()

    if "results review" in low or "equity research" in low:
        doc_type, pattern = "research_report", _REVIEW_RE
    elif "earnings call" in low:
        doc_type, pattern = "earnings_call", _CALL_RE
    else:
        doc_type, pattern = "annual_report", _ANNUAL_RE

    match = pattern.search(head) or _YEAR_RE.search(head)
    year = int(match.group(1)) if match else 2025
    return doc_type, year


def _chunk_header(meta: dict, section: str, subsection: str) -> str:
    """A short header with the metadata and the section title, prepended to every chunk."""
    if not CONFIG["chunk_header"]:
        return ""
    where = " > ".join(part for part in (section, subsection) if part)
    bits = [meta["company"], f"FY{meta['fiscal_year']}", meta["doc_type"].replace("_", " ")]
    if where:
        bits.append(where)
    return " | ".join(bits)


def _make_chunks(meta: dict, header: str, text: str) -> list[Document]:
    """Pack the lines of one section into Documents of at most MAX_CHUNK_CHARS characters."""
    prefix = f"{header}\n" if header else ""
    room = max(MAX_CHUNK_CHARS - len(prefix), 200)
    bodies: list[str] = []
    current: list[str] = []
    size = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) > room:                       # one oversized line: split it as a last resort
            if current:
                bodies.append("\n".join(current))
                current, size = [], 0
            bodies.extend(line[start:start + room] for start in range(0, len(line), room))
            continue
        if size + len(line) + 1 > room:
            bodies.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1

    if current:
        bodies.append("\n".join(current))

    return [Document(page_content=f"{prefix}{body}", metadata=dict(meta)) for body in bodies]


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["nav", "footer", "script", "style"]):
        tag.decompose()

    h1 = soup.find("h1")
    doc_type, fiscal_year = _document_identity(soup)
    meta = {
        "source": file_name,
        "company": _clean_company(h1.get_text(" ", strip=True) if h1 is not None else ""),
        "fiscal_year": fiscal_year,
        "doc_type": doc_type,
    }

    documents: list[Document] = []
    section = subsection = ""
    lines: list[str] = []

    def flush() -> None:
        nonlocal lines
        if lines:
            documents.extend(_make_chunks(meta, _chunk_header(meta, section, subsection), "\n".join(lines)))
            lines = []

    body = soup.body if soup.body is not None else soup
    for block in body.find_all(["h2", "h3", "p", "ol", "ul", "table"]):
        if block.name == "table":
            flush()
            if not CONFIG["table_to_text"]:       # baseline: a table becomes a stream of numbers
                lines.append(re.sub(r"\s+", " ", block.get_text(" ", strip=True)).strip())
                continue
            table_text = table_to_text(str(block))
            if table_text:
                documents.extend(_make_chunks(meta, _chunk_header(meta, section, subsection), table_text))
            continue
        if block.name in ("h2", "h3"):
            flush()
            title = re.sub(r"\s+", " ", block.get_text(" ", strip=True)).strip()
            if not CONFIG["section_split"]:        # baseline: headings are just more text
                lines.append(title)
                continue
            if block.name == "h2":
                section, subsection = title, ""
            else:
                subsection = title
            continue
        if section.lower() == "contents":          # the table of contents is navigation, not content
            continue
        text = block.get_text(" ", strip=True)
        if text:
            lines.append(text)
    flush()

    # Remember what this corpus contains: parse_query may use it, because the runner calls
    # chunk_document for every document before it parses the first question.
    if meta["company"]:
        _KNOWN_COMPANIES.add(meta["company"])
    for table in soup.find_all("table"):
        rows, flags = _table_grid(table)
        header_index = next((i for i, flag in enumerate(flags) if flag), 0)
        for index, row in enumerate(rows):
            if index == header_index or not row:
                continue
            label = row[min(row)]
            if label:
                _NOT_A_COMPANY.add(_plain(label))

    return documents


# =============================================================================
# Task 2.2: parsing the question
# =============================================================================

_KNOWN_COMPANIES: set[str] = set()   # the companies of the corpus, filled in by chunk_document
_NOT_A_COMPANY: set[str] = set()     # row labels of the corpus: regions, segments, line items

_QUESTION_WORDS = {
    "how", "what", "which", "who", "when", "where", "why", "in", "on", "at", "for", "the",
    "did", "do", "does", "is", "are", "was", "were", "its", "it", "if", "by", "and", "or",
    "usd", "eur", "gbp", "cny", "hkd", "percent", "million", "millions", "billion", "fy", "q", "f",
}
_ROLE_WORDS = {
    "chief", "executive", "officer", "financial", "director", "president", "vice", "senior",
    "head", "chair", "chairman", "board", "management", "auditor", "audit",
}
_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december",
}
_BROKER_TAILS = {"research", "securities", "capital", "markets", "advisors", "advisers",
                 "partners", "bank", "brokers", "investments"}
_LEGAL_WORDS = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co", "sa", "ag", "nv"}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*|\d[\d,.]*")
_POSSESSIVE_RE = re.compile(r"['\u2019]s?$")
_FY_RE = re.compile(r"\bFY\s?(\d{4})\b", re.I)
_MONTH_YEAR_RE = re.compile(r"\b(?:" + "|".join(sorted(_MONTHS)) + r")\s+(\d{4})\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_EARNINGS_RE = re.compile(r"earnings\s+call|conference\s+call|transcript", re.I)
_RESEARCH_RE = re.compile(r"results\s+review|target\s+price|price\s+target|\brating\b|forecast|"
                          r"estimate|broker|upgrade|downgrade|analyst", re.I)


def _plain(text: str) -> str:
    """Lower-case, punctuation-free, for vocabulary lookups."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def _strip_possessive(word: str) -> str:
    return _POSSESSIVE_RE.sub("", word)


def _find_year(question: str) -> int | None:
    """The fiscal year the question names ('FY2024', '31 December 2024'); None if it names none."""
    for pattern in (_FY_RE, _MONTH_YEAR_RE, _YEAR_RE):
        match = pattern.search(question)
        if match:
            return int(match.group(1))
    return None


def _find_doc_type(question: str) -> str:
    """'earnings_call' / 'research_report' / 'annual_report' (the default)."""
    if _EARNINGS_RE.search(question):
        return "earnings_call"
    if _RESEARCH_RE.search(question):
        return "research_report"
    return "annual_report"


def _capitalised_runs(question: str) -> list[str]:
    """The runs of consecutive capitalised words, e.g. 'Halcyon Air Cargo' -> ['Halcyon Air Cargo']."""
    runs: list[list[str]] = []
    current: list[str] = []

    def close() -> None:
        nonlocal current
        if current:
            runs.append(current)
            current = []

    for match in _WORD_RE.finditer(question):
        token = match.group()
        if not token[0].isalpha():        # a number ends a run: "FY2024: Kestrel" is two runs
            close()
            continue
        word = _strip_possessive(token)
        if word and word[0].isupper():
            current.append(word)
            if word != token:             # a possessive ends the name: "Software's Chief ..." 
                close()
        else:
            close()
    close()
    return [" ".join(run) for run in runs if run]


def _is_company_name(run: str) -> bool:
    """Whether a capitalised run can name a company that the question asks about."""
    words = run.split()
    if not words or len(run) < 3:
        return False
    lowered = [word.lower() for word in words]
    if all(word in _QUESTION_WORDS for word in lowered):
        return False
    if any(word in _ROLE_WORDS or word in _MONTHS for word in lowered):
        return False
    if lowered[-1] in _BROKER_TAILS:                  # BuyFin Capital, Meridian Equity Research
        return False
    if len(words) == 1 and lowered[-1] in _LEGAL_WORDS:
        return False
    plain = _plain(run)
    known = {company_key(name) for name in _KNOWN_COMPANIES}
    for term in _NOT_A_COMPANY:
        if company_key(term) in known:                # a company name, not a region or a segment
            continue
        if plain in term or term in plain:            # Asia Pacific, Midstream, Europe Middle East
            return False
    return True


def _company_variants(name: str) -> list[str]:
    """The name, then the name with its legal suffix dropped ('Brightwater Energy Corp.' -> ...)."""
    variants = [name]
    words = name.split()
    while len(words) > 1 and words[-1].lower().strip(".") in _LEGAL_WORDS:
        words = words[:-1]
        variants.append(" ".join(words))
    return variants


def _find_companies(question: str) -> list[str]:
    """Every company the question names, in the wording of the question.

    The corpus companies are matched by name; any other capitalised name is kept too, because a
    company that is not in the corpus must still be listed.
    """
    found: list[str] = []

    def add(name: str) -> None:
        name = name.strip(" ,.'\"")
        if name and all(company_key(name) != company_key(other) for other in found):
            found.append(name)

    for name in sorted(_KNOWN_COMPANIES, key=len, reverse=True):
        for variant in _company_variants(name):
            if re.search(rf"(?<![\w]){re.escape(variant)}(?![\w])", question, re.I):
                add(variant)
                break

    for run in _capitalised_runs(question):
        if _is_company_name(run) and all(company_key(run) != company_key(other) for other in found):
            add(run)

    return found


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    return {
        "companies": _find_companies(question),
        "fiscal_year": _find_year(question),
        "doc_type": _find_doc_type(question),
    }


# =============================================================================
# Tasks 2.1, 2.3 and 3: the index, retrieval and abstention
# =============================================================================

from rank_bm25 import BM25Okapi  # noqa: E402

_BM25_TOKEN_RE = re.compile(r"[a-z]+|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "in", "is", "it", "its", "many", "more", "much", "of", "on", "or",
    "our", "that", "the", "their", "there", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "with", "would",
    # words that appear in nearly every chunk of a financial document and only add noise
    "usd", "million", "millions", "billion", "value", "percent", "december", "date", "year",
}

_BM25_K1 = 1.2        # term-frequency saturation
_BM25_B = 0.5         # length normalisation: the table chunks are long, so keep it low
_TABLE_BOOST = 1.0    # a chunk that is a table is where the numbers live
_TABLE_LINE_MARK = " | "


def _tokenize(text: str) -> list[str]:
    """Lower-case word and number tokens; '2,710.4' is also indexed as '2710.4'."""
    if not CONFIG["stopwords"]:                  # baseline: lower-cased whitespace tokens
        return str(text).lower().split()
    tokens: list[str] = []
    for token in _BM25_TOKEN_RE.findall(str(text).lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        if "," in token:
            tokens.append(token.replace(",", ""))
    return tokens


def _is_table_chunk(text: str) -> bool:
    """Whether a chunk is mostly table rows: the shape table_to_text produces."""
    return text.count(_TABLE_LINE_MARK) >= 3


def _rank(index: dict, tokens: list[str]):
    """BM25 scores for the tokens, with the table chunks weighted."""
    scores = index["bm25"].get_scores(tokens)
    if _TABLE_BOOST != 1.0:
        for position, document in enumerate(index["documents"]):
            if _is_table_chunk(document.page_content):
                scores[position] *= _TABLE_BOOST
    return scores


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    documents = [chunk for chunk in chunks if isinstance(chunk, Document)]
    tokens = [_tokenize(chunk.page_content) for chunk in documents]
    metas = [chunk.metadata or {} for chunk in documents]

    years_by_type: dict[str, set[int]] = {}
    for meta in metas:
        year = as_year(meta.get("fiscal_year"))
        if year is not None:
            years_by_type.setdefault(meta.get("doc_type"), set()).add(year)

    return {
        "documents": documents,
        "metas": metas,
        "bm25": BM25Okapi(tokens, k1=_BM25_K1, b=_BM25_B) if tokens else None,
        "years_by_type": years_by_type,
        "corpus_companies": {company_key(meta.get("company")) for meta in metas if meta.get("company")},
    }


def _allowed_years(doc_type: str, year: int | None, years_by_type: dict) -> set[int]:
    """The document years that can state a fact of the year the question names.

    The FY2025 annual report also shows FY2024, so a question about year Y is answered by the
    annual report of Y or of Y + 1. An earnings call or a research report reviews the year before
    the one it forecasts.
    """
    available = set(years_by_type.get(doc_type, ()))
    if year is None:
        return available
    if doc_type == "annual_report":
        return {candidate for candidate in available if candidate in (year, year + 1)}
    return {candidate for candidate in available if candidate in (year - 1, year)}


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    if not index or index.get("bm25") is None:
        return []

    documents = index["documents"]
    metas = index["metas"]
    parsed = parse_query(question)

    asked = [company_key(name) for name in parsed["companies"]]
    in_corpus = index["corpus_companies"]
    known = [key for key in asked if key in in_corpus]
    if CONFIG["abstain"] and asked and not known:   # the company is not in the corpus at all
        return []

    doc_type = parsed["doc_type"]
    years = _allowed_years(doc_type, parsed["fiscal_year"], index["years_by_type"])
    if CONFIG["abstain"] and not years:             # the corpus does not cover that fiscal year
        return []

    def usable(position: int) -> bool:
        if not CONFIG["metadata_filter"]:           # baseline: the whole corpus is a candidate
            return True
        meta = metas[position]
        if meta.get("doc_type") != doc_type:
            return False
        if as_year(meta.get("fiscal_year")) not in years:
            return False
        return not known or company_key(meta.get("company")) in known

    pool = [position for position in range(len(documents)) if usable(position)]
    if not pool:
        return []

    scores = _rank(index, _tokenize(question))

    if len(known) < 2 or not CONFIG["company_split"]:   # one company: plain BM25 over the pool
        pool.sort(key=lambda position: scores[position], reverse=True)
        return [documents[position] for position in pool[:MAX_RETRIEVED]]

    # A comparison needs evidence on both sides: retrieve for each company separately.
    names = [name for name in parsed["companies"] if company_key(name) in known]
    picked: list[int] = []
    for name in names:
        key = company_key(name)
        dropped: set[str] = set()
        for other in names:
            if company_key(other) != key:
                dropped.update(_tokenize(other))
        query = [token for token in _tokenize(question) if token not in dropped]
        company_scores = _rank(index, query)
        own = [position for position in pool if company_key(metas[position].get("company")) == key]
        own.sort(key=lambda position: company_scores[position], reverse=True)
        picked.extend(own[:2])

    picked = sorted(set(picked), key=lambda position: scores[position], reverse=True)
    for position in sorted(pool, key=lambda position: scores[position], reverse=True):
        if len(picked) >= MAX_RETRIEVED:
            break
        if position not in picked:
            picked.append(position)

    return [documents[position] for position in picked[:MAX_RETRIEVED]]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
