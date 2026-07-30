"""Stream-parse a MediaWiki XML dump into articles.jsonl.

Reads the bz2 dump without decompressing to disk, keeps only main-namespace
articles (skips redirects), and writes one JSON object per line:
    {"title": ..., "wikitext": ...}

Run:
    python -m pipelines.wikivoyage.parse
"""

import bz2
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DUMP_PATH = DATA_DIR / "dewikivoyage-latest-pages-articles.xml.bz2"
ARTICLES_PATH = DATA_DIR / "articles.jsonl"


def _local(tag: str) -> str:
    """Strip the XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]


def iter_articles(dump_path: Path):
    """Yield (title, wikitext) for every main-namespace, non-redirect page."""
    with bz2.open(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if _local(elem.tag) != "page":
                continue
            ns = title = text = None
            redirect = False
            for child in elem:
                tag = _local(child.tag)
                if tag == "ns":
                    ns = child.text
                elif tag == "title":
                    title = child.text
                elif tag == "redirect":
                    redirect = True
                elif tag == "revision":
                    for rev_child in child:
                        if _local(rev_child.tag) == "text":
                            text = rev_child.text
            if ns == "0" and not redirect and title and text:
                yield title, text
            elem.clear()


def run() -> None:
    if not DUMP_PATH.exists():
        print(f"ERROR: dump not found at {DUMP_PATH}", file=sys.stderr)
        sys.exit(1)

    count = 0
    with open(ARTICLES_PATH, "w", encoding="utf-8") as out:
        for title, wikitext in iter_articles(DUMP_PATH):
            out.write(json.dumps({"title": title, "wikitext": wikitext}, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  parsed {count} articles", flush=True)
    print(f"Done: {count} articles -> {ARTICLES_PATH}")


if __name__ == "__main__":
    run()
