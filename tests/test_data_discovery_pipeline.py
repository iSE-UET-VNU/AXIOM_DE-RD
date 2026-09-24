from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.data_discovery.pipeline import PageEvidence, PageIndex


class PpocrArtifactTests(unittest.TestCase):
    def test_latest_resume_row_wins_for_duplicate_page(self) -> None:
        page = PageEvidence(
            page_id="doc-1#page=1",
            file_path="/tmp/doc-1.pdf",
            source_uri="doc-1",
            page_index=0,
            page_number=1,
            text="native text",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "light_ocr_ppocrv5.jsonl"
            rows = [
                {"unit": page.page_id, "text": "old OCR", "error": None},
                {"unit": page.page_id, "text": "latest OCR", "error": None},
            ]
            artifact.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            index = PageIndex.build([page])
            stats = index.apply_ppocrv5_jsonl(artifact, strict=True)

        self.assertEqual(stats["duplicate_rows"], 1)
        self.assertEqual(stats["duplicate_policy"], "latest_row_wins")
        self.assertEqual(index.pages[0].ocr_text, "latest OCR")


if __name__ == "__main__":
    unittest.main()
