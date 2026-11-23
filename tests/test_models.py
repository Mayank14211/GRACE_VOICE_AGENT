"""grace/models.py manifest and scripts/fetch_models.py (no network: downloads are faked)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from grace import models

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_models  # noqa: E402


def test_manifest_is_well_formed() -> None:
    paths = [f.path for f in models.MODEL_FILES]
    assert len(paths) == len(set(paths))
    for f in models.MODEL_FILES:
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256), f.path
        assert f.url.startswith("https://"), f.path
        assert not Path(f.path).is_absolute()
    for group in models.REQUIRED_GROUPS:
        assert models.files_in(group), group
    for arch, (rel, _) in models.MOONSHINE_ARCHS.items():
        assert models.moonshine_dir(arch) == models.MODELS_DIR / rel


def test_missing_lists_required_files(tmp_path: Path) -> None:
    assert set(models.missing(models_dir=tmp_path)) == {
        f.path for f in models.MODEL_FILES if f.group in models.REQUIRED_GROUPS
    }
    assert models.missing(groups=(), models_dir=tmp_path) == []


def test_resolve_is_relative_to_the_repo_root() -> None:
    assert models.resolve("models/x.onnx") == ROOT / "models" / "x.onnx"
    absolute = (ROOT / "y.onnx").resolve()
    assert models.resolve(absolute) == absolute


def test_fetch_keeps_a_present_file_and_refuses_a_bad_one(tmp_path: Path) -> None:
    f = models.ModelFile("vad/x.onnx", "https://example.invalid/x", "0" * 64, "vad", "MIT")
    dest = tmp_path / f.path
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"not the model")
    assert fetch_models.fetch(f, tmp_path).startswith("HASH MISMATCH")
    assert dest.read_bytes() == b"not the model"  # never replaced silently

    good = models.ModelFile("vad/x.onnx", f.url, models.sha256(dest), "vad", "MIT")
    assert fetch_models.fetch(good, tmp_path) == "present, hash ok"
