from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "user")
os.environ.setdefault("DB_PASSWORD", "password")
os.environ.setdefault("EMBED_DIM", "1536")

from api import retrieval


class RetrievalTest(unittest.TestCase):
    def test_metadata_value_uses_first_present_key_or_default(self) -> None:
        metadata = {"page": "Page", "url": None}

        self.assertEqual(retrieval._metadata_value(metadata, "title", "page"), "Page")
        self.assertEqual(retrieval._metadata_value(metadata, "url", default="missing"), "missing")

    def test_build_retrieval_query_adds_domain_terms_for_minesweeper_tasks(self) -> None:
        query = retrieval._build_retrieval_query("Build a Minesweeper CLI game with tests")

        self.assertIn("Build a Minesweeper CLI game with tests", query)
        self.assertIn("adjacent mine counts", query)
        self.assertIn("win/loss conditions", query)

    def test_retrieve_maps_nodes_to_evidence_items(self) -> None:
        class FakeNode:
            metadata = {
                "page": "Page",
                "section": "Section",
                "url": "https://example.test",
                "revision_id": 123,
                "source_type": "wiki",
            }

            def get_content(self) -> str:
                return "Excerpt text"

        class FakeNodeWithScore:
            node = FakeNode()

        class FakeRetriever:
            def retrieve(self, query: str):
                self.query = query
                return [FakeNodeWithScore()]

        with patch.object(retrieval, "get_retriever", return_value=FakeRetriever()):
            evidence = retrieval.retrieve("question")

        self.assertEqual(
            evidence,
            [
                {
                    "page": "Page",
                    "section": "Section",
                    "url": "https://example.test",
                    "revision_id": 123,
                    "excerpt": "Excerpt text",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
