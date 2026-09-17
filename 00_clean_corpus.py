from __future__ import annotations

import csv
import re
from pathlib import Path

import config
from common import read_text

SEPARATOR = re.compile(r"^_{10,}\s*$", re.MULTILINE)
FULL_TEXT = re.compile(r"^Full text:\s*", re.MULTILINE)
META = re.compile(
    r"^(Subject|Business indexing term|Company / organization|People|URL|Title|"
    r"Publication title|Pages|Publication year|Publication date|Section|Publisher|"
    r"Place of publication|Country of publication|ISSN|Source type|Language of publication|"
    r"Document type|ProQuest document ID|Document URL|Copyright|Last updated|Database)\s*:",
    re.MULTILINE | re.IGNORECASE,
)
REFERENCES = re.compile(r"^\s*(references|bibliography|acknowledg(e)?ments)\s*$",
                        re.MULTILINE | re.IGNORECASE)


def write_doc(arena: str, doc_id: str, text: str) -> bool:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return False
    path = config.CLEAN_ROOT / arena / f"{doc_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return True


def clean_arxiv() -> list[dict]:
    rows = []
    for path in sorted((config.SOURCE_ROOT / "arxiv").rglob("*.txt")):
        text = read_text(path)
        match = REFERENCES.search(text)
        if match:
            text = text[:match.start()]
        # Remove obvious author/contact lines but retain titles and abstracts.
        text = re.sub(r"^.*(?:e-?mail|@).*?$", "", text, flags=re.I | re.M)
        doc_id = f"arxiv_{path.parent.name}_{path.stem}"
        ok = write_doc("arxiv", doc_id, text)
        rows.append({"arena": "arxiv", "arena_label": config.ARENA_LABELS["arxiv"],
                     "source_type": "arXiv paper", "source_file": str(path.relative_to(config.PROJECT_ROOT)),
                     "document_id": doc_id, "included": int(ok)})
    return rows


def clean_proquest(arena: str) -> list[dict]:
    rows = []
    count = 0
    for path in sorted((config.SOURCE_ROOT / arena).glob("*.txt")):
        for block_no, block in enumerate(SEPARATOR.split(read_text(path)), 1):
            start = FULL_TEXT.search(block)
            if not start:
                continue
            body = block[start.end():]
            end = META.search(body)
            body = body[:end.start()] if end else body
            body = re.sub(r"This article appeared in print on page.*", "", body,
                          flags=re.I | re.S)
            count += 1
            doc_id = f"{arena}_{count:05d}"
            ok = write_doc(arena, doc_id, body)
            rows.append({"arena": arena, "arena_label": config.ARENA_LABELS[arena],
                         "source_type": "ProQuest newspaper export",
                         "source_file": str(path.relative_to(config.PROJECT_ROOT)),
                         "source_block": block_no, "document_id": doc_id, "included": int(ok)})
    return rows


def main():
    rows = clean_arxiv() + clean_proquest("NYT") + clean_proquest("WSJ")
    out = config.AUDIT_ROOT / "corpus_manifest.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {out} ({len(rows)} records). Academic source: arXiv papers.")


if __name__ == "__main__":
    main()
