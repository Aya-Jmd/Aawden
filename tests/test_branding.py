from pathlib import Path
import pytest
from security_wsp.artifacts import atomic_json, edit_manifest_path
from security_wsp.cli import build_parser
from security_wsp.interventions import load_toy_checkpoint, write_toy_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_new_brand_is_used_by_cli_and_dashboard():
    parser = build_parser()
    assert parser.prog == "aawden"
    assert parser.parse_args(["serve"]).run == Path("runs/aawden-demo")
    page = (ROOT / "src/security_wsp/web/index.html").read_text(encoding="utf-8")
    assert "<title>Aawden" in page and "AAWDEN<small>" in page
    assert "python aawden.py" in page


def test_toy_checkpoint_uses_project_format_and_checks_tensor_integrity(tmp_path):
    path = tmp_path / "toy.json"
    record = write_toy_checkpoint(path, [1., 2.], [1., 3.], {})
    assert record["format"] == "aawden-constructed-toy-v1"
    assert load_toy_checkpoint(path) == [1., 3.]
    record["weights"][0] = 99
    atomic_json(path, record)
    with pytest.raises(ValueError, match="integrity"):
        load_toy_checkpoint(path)


def test_edit_manifest_uses_the_project_filename(tmp_path):
    current = tmp_path / "aawden_edit.json"
    atomic_json(current, {"hash": "fixture"})
    assert edit_manifest_path(tmp_path) == current
