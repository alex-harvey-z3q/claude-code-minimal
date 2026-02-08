import os
from datetime import datetime, timezone
from pathlib import Path

from ingest.wikipedia import fetch_page
from ingest.parser import split_sections
from ingest.models import NormalizedDocument
from ingest.s3 import put_json

RAW_BUCKET = os.environ["RAW_BUCKET"]
PARSED_BUCKET = os.environ["PARSED_BUCKET"]

CONVENTIONS_DIR = Path(os.environ.get("CONVENTIONS_DIR", "/app/conventions_wiki"))


PAGES = [
    "Minesweeper (video game)",
    "Microsoft Minesweeper",
]


def process_page(title: str) -> None:
    """Normalise a single Wikipedia page and persist its contents to S3."""

    page = fetch_page(title)
    page_id = page["pageid"]
    content = page["revisions"][0]["slots"]["main"]["*"]
    fetched_at = datetime.now(timezone.utc).isoformat()

    put_json(RAW_BUCKET, f"pages/{page_id}.json", page)

    for section, text in split_sections(content):
        process_section(page_id, title, section, text, fetched_at)


def process_section(
    page_id: int,
    title: str,
    section: str,
    text: str,
    fetched_at: str,
) -> None:
    """Normalise a single Wikipedia section and persist it to S3."""

    if not text.strip():
        return

    doc = NormalizedDocument(
        doc_id=f"wiki:{page_id}:{section}",
        source="wikipedia",
        page_id=page_id,
        title=title,
        section=section,
        text=text,
        metadata={
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "fetched_at": fetched_at,
        },
    )

    put_json(
        PARSED_BUCKET,
        f"docs/{page_id}/{section}.json",
        doc.to_dict(),
    )


def process_conventions_page(path: Path) -> None:
    """Normalise a single local conventions page and persist it to S3."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    title = path.stem.replace("_", " ")
    page_id = f"conventions-{path.stem}"

    put_json(
        RAW_BUCKET,
        f"conventions/{path.name}",
        {
            "title": title,
            "content": text,
            "source": "conventions",
        },
    )

    # Store parsed conventions doc
    doc = NormalizedDocument(
        doc_id=f"conventions:{path.stem}",
        source="conventions",
        page_id=page_id,
        title=title,
        section="Conventions",
        text=text,
        metadata={
            "url": "",
            "fetched_at": fetched_at,
        },
    )

    put_json(
        PARSED_BUCKET,
        f"docs/conventions/{path.stem}.json",
        doc.to_dict(),
    )


def process_conventions() -> None:
    """Process all local conventions files."""

    if not CONVENTIONS_DIR.exists():
        return

    for path in sorted(CONVENTIONS_DIR.glob("*.md")):
        process_conventions_page(path)


def main() -> None:
    for title in PAGES:
        process_page(title)

    process_conventions()


if __name__ == "__main__":
    main()
