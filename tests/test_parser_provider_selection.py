from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.parsing.parser import parse_raw_file
from src.models import DataObject, ParsedData


class ParserProviderSelectionTests(unittest.TestCase):
    def test_keeps_main_lift_api_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            data_object = _data_object(path)
            expected = _parsed_data(data_object, parser="lift-api")

            with patch(
                "src.ingestion.parsing.parser.LiftAPIParserClient"
            ) as client_class:
                client_class.return_value.parse_file.return_value = expected
                actual = parse_raw_file(
                    path,
                    data_object,
                    {
                        "provider": "lift_api",
                        "lift_api": {"fallback_to_local": False},
                    },
                )

        self.assertIs(actual, expected)
        client_class.return_value.parse_file.assert_called_once_with(path, data_object)

    def test_routes_chandra2_through_main_parser_entry_point(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            data_object = _data_object(path)
            expected = _parsed_data(data_object, parser="chandra2")

            with patch(
                "src.ingestion.parsing.parser.Chandra2Provider"
            ) as provider_class:
                provider_class.return_value.parse_file.return_value = expected
                actual = parse_raw_file(
                    path,
                    data_object,
                    {
                        "provider": "chandra2",
                        "chandra2": {
                            "method": "hf",
                            "batch_size": 1,
                            "max_workers": 1,
                        },
                    },
                )

        self.assertIs(actual, expected)
        provider_class.return_value.parse_file.assert_called_once_with(path, data_object)
        config = provider_class.call_args.args[0]
        self.assertEqual(config.method, "hf")
        self.assertEqual(config.batch_size, 1)


def _data_object(path: Path) -> DataObject:
    return DataObject(
        object_id="document-1",
        uri=path.as_posix(),
        content_type="application/pdf",
        metadata={"format": "pdf"},
    )


def _parsed_data(data_object: DataObject, *, parser: str) -> ParsedData:
    return ParsedData(
        object_id=data_object.object_id,
        source_uri=data_object.uri,
        source_format="pdf",
        text="Parsed document",
        metadata={"parser": parser},
    )


if __name__ == "__main__":
    unittest.main()
