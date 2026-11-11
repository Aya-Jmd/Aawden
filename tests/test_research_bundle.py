import json
import zipfile
from pathlib import Path
import pytest
from security_wsp import submission
from security_wsp.artifacts import RunStore, atomic_json, file_hash


def fixture_project(tmp_path, monkeypatch):
    (tmp_path / 'src/security_wsp').mkdir(parents=True)
    monkeypatch.setattr(submission, '__file__', str(tmp_path / 'src/security_wsp/submission.py'))
    run = tmp_path / 'runs/demo'
    store = RunStore(run, {'fixture': True}, stage='educational_demo', evidence_kind='constructed_toy')
    atomic_json(run / 'summary.json', {'evidence_kind': 'constructed_toy'})
    store.finish([], ['summary.json'])
    (tmp_path / 'README.md').write_text('Aawden fixture')
    return run


def test_bundle_inventory_is_verifiable_and_does_not_publish(tmp_path, monkeypatch):
    run = fixture_project(tmp_path, monkeypatch)
    output = tmp_path / 'dist/research.zip'
    result = submission.package_submission(run, output)
    assert result['uploaded'] is False and result['submission_ready'] is False
    with zipfile.ZipFile(output) as archive:
        inventory = json.loads(archive.read('BUNDLE_MANIFEST.json'))
        import hashlib
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == value for name, value in inventory['files'].items())
        assert inventory['model_weights_included'] is False


def test_credential_in_source_prevents_publishing_final_bundle(tmp_path, monkeypatch):
    run = fixture_project(tmp_path, monkeypatch)
    token = 'controlled_secret_for_unit_test'
    (tmp_path / '.env').write_text('PRIVATE_TOKEN=' + token)
    (tmp_path / 'README.md').write_text('Accidental key ' + token)
    output = tmp_path / 'dist/research.zip'
    with pytest.raises(ValueError, match='private credential'):
        submission.package_submission(run, output)
    assert not output.exists()
