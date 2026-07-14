from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.storage.local import LocalArtifactStore


class LocalArtifactStoreTests(unittest.TestCase):
    def test_json_can_preserve_declared_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalArtifactStore(Path(temp_dir))

            path = store.write_json(
                "schema.json",
                {"$schema": "draft", "$id": "example", "properties": {}},
                sort_keys=False,
            )

            text = path.read_text(encoding="utf-8")
            self.assertLess(text.index('"$schema"'), text.index('"$id"'))
            self.assertLess(text.index('"$id"'), text.index('"properties"'))

    def test_jsonl_is_deterministic_unicode_portable_and_atomically_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalArtifactStore(root / "processed", project_root=root)
            path = store.write_jsonl(
                "records.jsonl",
                [
                    {
                        "z": 1,
                        "path": root / "data" / "work" / "ảnh.png",
                        "a": "xin chào",
                    }
                ],
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"a":"xin chào","path":"data/work/ảnh.png","z":1}\n',
            )
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())

            store.write_jsonl("records.jsonl", [{"b": 2, "a": 1}])

            self.assertEqual(path.read_text(encoding="utf-8"), '{"a":1,"b":2}\n')
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())


if __name__ == "__main__":
    unittest.main()
