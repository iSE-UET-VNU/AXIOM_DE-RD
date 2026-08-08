"""Data-lake file index and evidence-reference resolution.

Evidence references in the question sheet are written by hand: some are paths
relative to the lake root, some are bare file names, some differ in case, and a
few are directories or globs. Resolution is therefore staged from exact to
loose, and anything still unmatched is reported rather than silently dropped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import os

TEXT_EXTS = frozenset({".pdf", ".docx", ".doc", ".md", ".txt", ".epub", ".pptx"})
TABLE_EXTS = frozenset({".csv", ".xlsx", ".xls", ".xlsm"})
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
AUDIO_EXTS = frozenset({".mp3", ".m4a", ".wav"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv"})
MODEL_EXTS = frozenset({".glb", ".gltf", ".obj"})

_MODALITY = (
    ("text", TEXT_EXTS),
    ("table", TABLE_EXTS),
    ("image", IMAGE_EXTS),
    ("audio", AUDIO_EXTS),
    ("video", VIDEO_EXTS),
    ("model3d", MODEL_EXTS),
)


def modality_of(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    for name, extensions in _MODALITY:
        if suffix in extensions:
            return name
    return "other"


@dataclass(frozen=True)
class Lake:
    root: Path
    by_rel: dict[str, Path]
    by_rel_lower: dict[str, Path]
    by_name: dict[str, list[Path]]

    @classmethod
    def index(cls, root: Path | str) -> "Lake":
        root = Path(root)
        by_rel: dict[str, Path] = {}
        by_name: dict[str, list[Path]] = defaultdict(list)
        for path in root.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            by_rel[str(path.relative_to(root))] = path
            by_name[path.name.lower()].append(path)
        return cls(
            root=root,
            by_rel=by_rel,
            by_rel_lower={key.lower(): value for key, value in by_rel.items()},
            by_name=dict(by_name),
        )

    def documents(self, modalities: frozenset[str] | None = None) -> list[Path]:
        selected = sorted(self.by_rel.values())
        if modalities is None:
            return selected
        return [path for path in selected if modality_of(path) in modalities]

    def resolve(self, reference: str) -> Path | None:
        """Resolve one hand-written evidence reference to a single lake file."""
        paths, _ = self.resolve_all(reference)
        return paths[0] if len(paths) == 1 else None

    def resolve_all(self, reference: str) -> tuple[list[Path], str]:
        """Resolve a reference to every file it names, and say how.

        Returns ``(paths, strategy)``. A reference naming a directory or a glob
        legitimately denotes many files, so this cannot return a single path;
        ``strategy`` is recorded per reference in the evalset so a later reader
        can see which labels were literal and which were inferred.

        Strategies are ordered strictest first and the first hit wins. Every
        loose strategy below ``basename`` was added against an observed failure
        in the question sheet, listed with its case, so none of them is
        speculative pattern-matching over ground truth.
        """
        cleaned = _clean(reference)
        if not cleaned or cleaned == "*":
            return [], ""

        for candidate in (cleaned, cleaned.replace("\\", "/")):
            if candidate in self.by_rel:
                return [self.by_rel[candidate]], "exact"
            if candidate.lower() in self.by_rel_lower:
                return [self.by_rel_lower[candidate.lower()]], "case"

        if "*" in cleaned:
            hits = [path for key, path in self.by_rel.items() if fnmatch(key, cleaned)]
            return sorted(hits), "glob" if hits else ""

        stripped = cleaned.rstrip("/")

        # "ise_collection/", "Light_novel/Lord of Mysteries" -- a folder of evidence.
        under = [
            path for key, path in self.by_rel.items() if key.startswith(stripped + "/")
        ]
        if under:
            return sorted(under), "directory"

        matches = self.by_name.get(os.path.basename(stripped).lower(), [])
        if len(matches) == 1:
            return [matches[0]], "basename"

        # "sale/DEAL_LIST_07_07_HA_LINH_OFFICIAL" -- extension omitted.
        stems = [
            path
            for key, path in self.by_rel.items()
            if os.path.splitext(key)[0].lower() == stripped.lower()
        ]
        if len(stems) == 1:
            return stems, "stem"

        # "kinh dịch/Chu dịch diễn giải (1).Phan Bội Châu.Phan Bội Châu" -- the
        # sheet truncated a name that repeats its author suffix.
        prefixed = [
            path
            for key, path in self.by_rel.items()
            if key.lower().startswith(stripped.lower())
        ]
        if len(prefixed) == 1:
            return prefixed, "prefix"

        # "md.صدام حسين" -- right-to-left text renders the extension leading, so
        # the sheet recorded it before the stem.
        if "." in stripped:
            head, _, tail = stripped.partition(".")
            swapped = self.by_name.get(f"{tail}.{head}".lower(), [])
            if len(swapped) == 1:
                return [swapped[0]], "rtl_extension"

        # "bao_cao_thuong_nien_fptBCTN FPT 2023 VN.pdf" -- directory and file
        # name concatenated without a separator.
        joined = [
            path
            for key, path in self.by_rel.items()
            if key.replace("/", "").lower() == stripped.replace("/", "").lower()
        ]
        if len(joined) == 1:
            return joined, "joined_path"

        return [], ""

    def doc_id(self, path: Path) -> str:
        return str(path.relative_to(self.root))


def _clean(reference: str) -> str:
    value = str(reference).strip().strip('"').strip("'")
    while value.endswith(('"', "]", ",")):
        value = value[:-1].strip()
    return value.lstrip("./").strip()
