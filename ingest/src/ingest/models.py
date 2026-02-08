from dataclasses import dataclass
from typing import Dict

@dataclass
class NormalizedDocument:
    doc_id: str
    source: str
    page_id: int
    title: str
    section: str
    text: str
    metadata: Dict

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "page_id": self.page_id,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "metadata": self.metadata,
        }
