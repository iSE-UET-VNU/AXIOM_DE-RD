from __future__ import annotations

import base64
import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path

from src import cleaning, enrichment, indexing_cataloging
from src.ingestion.parsing.backends import TableParser
from src.ingestion.runner import run as run_ingestion
from src.models import DataObject, ParseStatus


# Synthetic BIFF8/OLE2 workbook generated once with xlwt 1.3.0. xlwt is not a
# runtime dependency: the committed bytes make the legacy-reader test fully
# self-contained and exercise xlrd against a genuine Excel 97-2003 file.
_XLS_FIXTURE_GZIP_B64 = """
H4sIAAAAAAACCu1YTUhUURQ+9/3MX/7MpAYaTCJkZUpEmzY6WePPxsFapEVQT+dh4sybaRgJW5Rls4ogaFW0EWwRhNUmi1oY
bgoCwxZBEGgtgyAoaKG+zj3vvnGcFBxIYeqd4Z17z7n3vPPde879mfduNjA//qRmAfKoBWRYNr3gytExfLy24AdsN01etUsP
PqZDRUVeDwbSpcKL0rduHkMe7wWQ4LHyCjnAZ3zOQBIiCUOv3UI6Shg0xjE0MxfmnoQZeQ95GVQTsu3E+4lXEH9EvV8SP0Ka
m8Sbse88Ow2zoUjDYZHJvVIdtZUhZzBFNh9JcxCq4DXP5Cu3mNVXhdbUoBYrsoagUgITgEHt0A09xeVyuA8+gFNITV1dTeHw
PFRiwCfgp1kL8MNe2dO1jn5r9QxQ/2u13r2O3oNBzNfflhSAUTDPMZ7MGSiBiwpvUSCspbUMJvkBN88MF3QORqO6kYFSmOG7
NeZKWzyZHlmEQVQBGnIjiGhxHYue87qBdm2G1hfTo7gztMYTw0YaVe2JVHw4pinA/GwqaMAAX0UXBh5GVEy3lNavY+fORDKp
p/A1xxJRHfUntdiwjina6+PHCG07/lXbTiktxRLkUSineoAWpB/HvPjg+1xXX3foLGlG6aixDqTdfOxgwlVugcZlYI2cE391
A1nsJ36N3rqT6jXEK3EGsKzvrhKV9jHqc51a69HPIaL3oT059b1Yz3w9/iyY+RLah/XJjoVLlZMfQuNQhwdkFO35bwwaWSO7
e4fT85BdMrFxfSJe/ccm5pH8ArspTt1yYJJ1DAdgiS9gmhlOliStkmSUWFZSxBxakoqSnJVcKCnCI8vzeFnykS5AkeH9GXlV
s5Ik2qyeMo7q6YwEquRBSSEMVgv3eaOCe9tFGWgh4BcFy5kXwthpOoi2HIi8BhCZgPiEa5mAuLMSB+IRQGQC8u2NBUQmIHYL
BxIEC4i8NhBZAGmRcDR82vAYWaGtyFspy4s1b6Wc4Fl5xoPnzWYdE6G0sy6/v5V127KZxUTYKCaweTGBdWOyQsUWk/XnzCGH
HHLIof+cmLhuyOLSrIoD1C2+6yzhs+x8Jvln6QQk8JfGP5RtYGCZgpGC8mcHqMx+F9ugjf29kFMPek/BEPQRjqGC8xcvQCx3
PBs29P+9JVSo/+VCcG6y/9+5SItqABYAAA==
"""


@unittest.skipUnless(importlib.util.find_spec("xlrd"), "xlrd is not installed")
class LegacyXlsFixtureV1Tests(unittest.TestCase):
    def test_real_biff8_workbook_preserves_types_hidden_sheet_and_lineage(self) -> None:
        workbook_bytes = gzip.decompress(
            base64.b64decode("".join(_XLS_FIXTURE_GZIP_B64.split()))
        )
        self.assertEqual(workbook_bytes[:8], bytes.fromhex("D0CF11E0A1B11AE1"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.xls"
            path.write_bytes(workbook_bytes)
            data_object = DataObject(
                object_id="legacy-xls-object",
                uri=path.as_posix(),
                content_type="application/vnd.ms-excel",
                metadata={"format": "xls"},
            )

            first = TableParser().parse(path, data_object)
            second = TableParser().parse(path, data_object)

        self.assertEqual(first.status, ParseStatus.SUCCESS)
        self.assertEqual(first.route, "table")
        self.assertEqual([table.name for table in first.parsed_data.tables], ["Data", "Hidden"])

        data, hidden = first.parsed_data.tables
        self.assertEqual(
            data.headers,
            ["column_1", "Name", "Name_2", "When", "Enabled", "Amount", "Formula"],
        )
        self.assertEqual(
            data.rows,
            [
                ["1", "Đặng", "東京", "2024-01-02", "true", "12.5", ""],
                ["", "", "", "", "", "", ""],
                ["2", "Grace", "Hopper", "2025-02-03", "false", "7", ""],
            ],
        )
        self.assertEqual(data.metadata["formula_mode"], "cached_value")
        self.assertTrue(hidden.metadata["hidden"])
        self.assertEqual(hidden.metadata["sheet_state"], "hidden")
        self.assertEqual(hidden.rows, [["X", "7"]])

        self.assertEqual(first.parsed_data.rows[0]["__axiom_row_number"], 2)
        self.assertEqual(first.parsed_data.rows[2]["__axiom_row_number"], 4)
        self.assertEqual(first.parsed_data.rows[3]["__axiom_sheet_name"], "Hidden")
        self.assertEqual(
            [table.source_ref for table in first.parsed_data.tables],
            [table.source_ref for table in second.parsed_data.tables],
        )
        self.assertEqual(first.parsed_data.rows, second.parsed_data.rows)

    @unittest.skip("Legacy normalized-table integration is not wired into the main runner.")
    def test_legacy_xls_runs_to_canonical_tables_and_index_records(self) -> None:
        workbook_bytes = gzip.decompress(
            base64.b64decode("".join(_XLS_FIXTURE_GZIP_B64.split()))
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            (raw / "legacy.xls").write_bytes(workbook_bytes)

            ingested = run_ingestion(raw, project_root=root)
            cleaned = cleaning.run(ingested.parsed_data, ingested.initial_schemas)
            enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
            indexed = indexing_cataloging.run(
                enriched.enriched_data,
                enriched.enriched_schemas,
                indexing_config={"embeddings": {"enabled": False}},
                normalized_tables=ingested.normalized_tables,
                normalized_documents=ingested.documents,
            )

        self.assertEqual(len(ingested.normalized_tables), 2)
        self.assertEqual([table["caption"] for table in ingested.normalized_tables], ["Data", "Hidden"])
        self.assertTrue(ingested.documents[0]["quality"]["has_content"])
        self.assertFalse(ingested.documents[0]["quality"]["has_text"])
        index_types = [record.index_type for record in indexed.index_records]
        self.assertEqual(index_types.count("document"), 1)
        self.assertEqual(index_types.count("table"), 2)
        self.assertEqual(index_types.count("catalog"), 1)
        self.assertEqual(indexed.embedding_report["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
