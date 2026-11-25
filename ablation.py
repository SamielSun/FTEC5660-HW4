"""Task 5: the ablation study.

Runs the whole pipeline once per variant on public_test and prints the points every variant earns.
Task 4 calls the generator (needs DEEPSEEK_API_KEY in .env); pass --no-generator to skip it.

    python ablation.py                  # everything, including Task 4
    python ablation.py --no-generator   # fast, no API calls
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hw4

DATA = Path("public_test")

# Every variant starts from the naive baseline and switches exactly one more component on.
BASE = {key: False for key in hw4.CONFIG}

# Every variant adds one component on top of the variant above it.
STEPS: list[tuple[str, str, dict]] = [
    ("B0", "naive RAG: get_text, fixed-size chunks, BM25 over everything, never []", {}),
    ("V1", "+ table_to_text: every number keeps its row and its column", {"table_to_text": True}),
    ("V2", "+ chunk along sections with a metadata header", {"section_split": True, "chunk_header": True}),
    ("V3", "+ financial tokeniser and stop words", {"stopwords": True}),
    ("V4", "+ parse_query used as a metadata filter", {"metadata_filter": True}),
    ("V5", "+ one retrieval per company for comparisons", {"company_split": True}),
    ("V6", "+ abstention: [] when the corpus cannot answer", {"abstain": True}),
]


def build_variants() -> list[tuple[str, str, dict]]:
    """Cumulative configurations: each row is the row above plus one change."""
    config = dict(BASE)
    variants = []
    for name, note, changes in STEPS:
        config = {**config, **changes}
        variants.append((name, note, dict(config)))
    return variants


def run(config: dict, corpus: dict, questions: dict, gt: dict | None, docs_meta: dict | None,
        with_generator: bool) -> dict:
    hw4.CONFIG.update({**BASE, **config})
    hw4._KNOWN_COMPANIES.clear()
    hw4._NOT_A_COMPANY.clear()

    chunks_by_file, errors = {}, {}
    for name, html in corpus.items():
        try:
            chunks_by_file[name] = list(hw4.chunk_document(name, html) or [])
        except Exception as exc:  # noqa: BLE001
            chunks_by_file[name], errors[name] = [], repr(exc)
    all_chunks = [chunk for name in sorted(chunks_by_file) for chunk in chunks_by_file[name]
                  if isinstance(chunk, hw4.Document) and len(chunk.page_content) <= hw4.MAX_CHUNK_CHARS]

    index = hw4.build_index(all_chunks)
    parsed = {qid: hw4.parse_query(question) for qid, question in questions.items()}
    retrieved = {qid: hw4.retrieve(index, question) for qid, question in questions.items()}

    if with_generator:
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = dict(zip(questions, pool.map(lambda q: hw4.generate(questions[q], retrieved[q]),
                                                   questions)))
    else:
        answers = {qid: "" for qid in questions}

    out = {"chunks": len(all_chunks)}
    if docs_meta:
        problems = {name: hw4.document_problems(chunks_by_file.get(name), errors.get(name), meta, name)
                    for name, meta in docs_meta.items()}
        out["1.2"] = hw4.POINTS["1.2"] * sum(not p for p in problems.values()) / len(problems)

    if not gt:
        return out

    answerable = [qid for qid in questions if gt[qid]["evidence"]]
    unanswerable = [qid for qid in questions if not gt[qid]["evidence"]]
    recall = sum(hw4.retrieval_hit(retrieved[q], gt[q]) for q in answerable) / max(len(answerable), 1)
    kept = sum(bool(retrieved[q]) for q in answerable) / max(len(answerable), 1)
    rejected = sum(not retrieved[q] for q in unanswerable) / max(len(unanswerable), 1)

    out["2.1"] = hw4.POINTS["2.1"] * sum(bool(retrieved[q]) for q in answerable) / max(len(answerable), 1)
    out["2.2"] = hw4.POINTS["2.2"] * sum(hw4.parse_score(parsed[q], gt[q]) for q in questions) / len(questions)
    out["2.3"] = hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO) / (hw4.RECALL_FULL - hw4.RECALL_ZERO))
    out["3"] = hw4.POINTS["3"] * hw4.clip(kept + rejected - 1)
    out["recall"] = recall
    out["kept"] = kept
    out["rejected"] = rejected
    if with_generator:
        out["4"] = hw4.POINTS["4"] * sum(hw4.answer_correct(answers[q], gt[q]) for q in questions) / len(questions)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--no-generator", action="store_true", help="skip Task 4 (no API calls)")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    gt = hw4.load_json(folder / "ground_truth.json")
    docs_meta = hw4.load_json(folder / "docs.json")
    with_generator = not args.no_generator and gt is not None

    rows = []
    for name, note, config in build_variants():
        result = run(config, corpus, questions, gt, docs_meta, with_generator)
        rows.append({"variant": name, "change": note, **result})
        print(f"{name}  " + "  ".join(f"{key}={value:.2f}" for key, value in result.items()
                                      if isinstance(value, float) and "." in key))

    columns = ["variant", "change", "1.2", "2.1", "2.3", "3", "4", "recall", "kept", "rejected", "chunks"]
    with open("ablation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\n| variant | change | 1.2 | 2.1 | recall@5 | 2.3 | 3 | 4 | total |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        total = sum(row.get(key, 0.0) for key in ("1.2", "2.1", "2.3", "3", "4"))
        cells = " | ".join(f"{row[key]:.2f}" if key in row else "-"
                           for key in ("1.2", "2.1"))
        recall = f"{row['recall']:.1%}" if "recall" in row else "-"
        cells += f" | {recall} | "
        cells += " | ".join(f"{row[key]:.2f}" if key in row else "-" for key in ("2.3", "3", "4"))
        print(f"| {row['variant']} | {row['change']} | {cells} | **{total:.2f}** |")
    print("\nTask 1.1 (8.00) and Task 2.2 (10.00) do not depend on these switches: table_to_text and "
          "parse_query are graded on their own.")
    print("Wrote ablation.csv")


if __name__ == "__main__":
    main()
