import re
from typing import List, Tuple

# Match any Wikipedia heading level (=, ==, ===, ...)
SECTION_RE = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$", re.MULTILINE)

# Match template-only headings like {{No more links}}
TEMPLATE_ONLY_RE = re.compile(r"^\{\{.*\}\}$")


def sanitise_section_name(section: str) -> str:
    """
    Convert a Wikipedia section title into a safe, stable filename slug.
    """
    slug = section.strip().lower()

    # Drop template-only sections entirely
    if TEMPLATE_ONLY_RE.match(slug):
        return ""

    # Replace common separators
    slug = slug.replace("&", "and")
    slug = slug.replace("/", " ")

    # Remove non-alphanumeric characters (keep spaces)
    slug = re.sub(r"[^a-z0-9 ]+", "", slug)

    # Collapse whitespace to underscores
    slug = re.sub(r"\s+", "_", slug)

    return slug


def split_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split raw Wikipedia wikitext into (section_title, section_text) tuples.
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("Introduction", text)]

    sections: List[Tuple[str, str]] = []

    for i, match in enumerate(matches):
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        sections.append((section_title, section_text))

    return sections
