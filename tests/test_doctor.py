from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import doctor  # noqa: E402

# Assembled at runtime so tests/test_no_secrets.py does not flag this file.
SECRET = "sk" + "-THIS-MUST-NEVER-BE-PRINTED-1234567890"


def test_python_check() -> None:
    assert doctor.check_python((3, 11, 9)).status in (
        doctor.PASS,
        doctor.FAIL,
    )  # FAIL only if 32-bit
    assert doctor.check_python((3, 10, 10)).status == doctor.FAIL
    assert doctor.check_python((3, 12, 0)).status == doctor.FAIL


def test_env_missing_is_not_configured(tmp_path: Path) -> None:
    [result] = doctor.check_env(tmp_path / ".env")
    assert result.status == doctor.NOT_CONFIGURED


def test_env_reports_names_never_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = tmp_path / ".env"
    env.write_text(
        f"# comment\nGROQ_API_KEY={SECRET}\nGEMINI_API_KEY=\nexport OPENROUTER_API_KEY='{SECRET}'\n"
    )

    results = doctor.check_env(env)

    by_name = {r.name: r for r in results}
    assert by_name[".env GROQ_API_KEY"].status == doctor.PASS
    assert by_name[".env GEMINI_API_KEY"].status == doctor.NOT_CONFIGURED
    assert by_name[".env OPENROUTER_API_KEY"].status == doctor.PASS
    print(doctor.format_results(results))
    assert SECRET not in capsys.readouterr().out
    assert all(SECRET not in r.detail for r in results)


def test_models_not_configured_when_none_required(tmp_path: Path) -> None:
    [result] = doctor.check_models((), tmp_path)
    assert result.status == doctor.NOT_CONFIGURED


def test_models_hash_verification(tmp_path: Path) -> None:
    (tmp_path / "good.onnx").write_bytes(b"abc")
    (tmp_path / "bad.onnx").write_bytes(b"xyz")
    good = hashlib.sha256(b"abc").hexdigest()

    results = doctor.check_models(
        [("good.onnx", good), ("bad.onnx", good), ("missing.onnx", good)], tmp_path
    )

    assert [r.status for r in results] == [doctor.PASS, doctor.FAIL, doctor.FAIL]
    assert [r.detail for r in results] == ["hash ok", "hash mismatch", "missing"]


def test_failed_import_is_reported_not_raised() -> None:
    [result] = doctor.check_imports([("definitely_not_a_module_xyz", "nope", False)])
    assert result.status == doctor.FAIL


def test_crashing_check_becomes_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> doctor.Result:
        raise RuntimeError("boom")

    monkeypatch.setattr(doctor, "check_tesseract", boom)
    results = doctor.run_checks(hardware=False)
    assert any(r.status == doctor.FAIL and "boom" in r.detail for r in results)


def test_unbuilt_features_are_not_configured() -> None:
    assert all(r.status == doctor.NOT_CONFIGURED for r in doctor.check_not_built())


def test_doctor_no_hardware_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert doctor.main(["--no-hardware"]) == 0, capsys.readouterr().out


@pytest.mark.hardware
@pytest.mark.windows
def test_microphone_opens() -> None:
    result = doctor.check_microphone()
    assert result.status == doctor.PASS, result.detail


@pytest.mark.hardware
@pytest.mark.windows
def test_camera_opens() -> None:
    result = doctor.check_camera()
    assert result.status == doctor.PASS, result.detail


def test_env_keys_match_config() -> None:
    from grace.config import ENV_KEYS

    assert set(doctor.ENV_KEYS) == set(ENV_KEYS.values())


def test_settings_missing_is_not_configured(tmp_path: Path) -> None:
    results = doctor.check_settings(tmp_path / "settings.yaml", tmp_path / ".env")
    assert results[0].status == doctor.NOT_CONFIGURED
    assert {r.name for r in results[1:]} == {
        "provider groq",
        "provider gemini",
        "provider openrouter",
        "provider local",
    }
    assert all(r.status != doctor.FAIL for r in results)


def test_settings_valid_and_invalid(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("inference: {threads: 6}\n")
    assert doctor.check_settings(path, tmp_path / ".env")[0].status == doctor.PASS
    path.write_text("inference: {threads: 12}\n")
    [result] = doctor.check_settings(path, tmp_path / ".env")
    assert result.status == doctor.FAIL
    assert "inference.threads" in result.detail


def test_grammar_check_passes() -> None:
    result = doctor.check_grammar()
    assert result.status == doctor.PASS, result.detail
    assert "system.shutdown" in result.detail


def test_doctor_imports_new_layers() -> None:
    names = {module for module, _, _ in doctor.IMPORTS}
    assert {"grace.actions", "grace.intent.grammar"} <= names
