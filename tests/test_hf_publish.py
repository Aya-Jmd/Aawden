"""Offline checks for Phase 6 publishing: plan, stage, secret scan, weight refusal and the publish gate.

No network. HfApi is replaced by a recording stub, so publish is exercised without touching Hugging Face.
"""
import json
import struct
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_publish as H  # noqa: E402

SECRET_VALUE = "beamsecretvalue" + "0123456789ab"
FAKE_HF = "hf_" + "Zz9" * 12


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def safetensors_bytes(keys: list[str]) -> bytes:
    header = json.dumps({k: {"dtype": "F32", "shape": [2], "data_offsets": [i * 8, i * 8 + 8]}
                         for i, k in enumerate(keys)}).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * (8 * len(keys))


def fixture(tmp_path: Path, *, weight_bytes: int | None = None, extra_files: dict | None = None) -> tuple[Path, dict]:
    """A miniature project: datasets, a run, a card and a config with the four safety gates."""
    root = tmp_path
    write(root / ".env", f"HF_PUBLISH_TOKEN={'h' * 34}\nBEAM_TOKEN={SECRET_VALUE}\n")
    write(root / "datasets/bench.json", json.dumps({"cases": [
        {"case_id": "c1", "split": "wsp_validation", "target_category": "sql_injection",
         "code": "def f(x):\n    return x", "threat_model": "unauthenticated caller", "oracle_verdict": "vulnerable"},
        {"case_id": "c2", "split": "wsp_final", "target_category": "sql_injection",
         "code": "def g(x):\n    return x", "threat_model": "sealed case", "oracle_verdict": "not_vulnerable_for_target"},
    ]}))
    rows = [
        {"case_id": "c1", "split": "wsp_validation", "category": "sql_injection", "method": "A_pf",
         "condition": "incorrect_suggestion", "raw_output": "{}", "predicted_verdict": "vulnerable",
         "oracle_verdict": "vulnerable"},
        {"case_id": "c1", "split": "wsp_validation", "category": "sql_injection", "method": "B_trace_pf",
         "condition": "incorrect_suggestion", "raw_output": "{}", "predicted_verdict": "vulnerable",
         "oracle_verdict": "vulnerable"},
        {"case_id": "c2", "split": "wsp_final", "category": "sql_injection", "method": "A_pf",
         "condition": "incorrect_suggestion", "raw_output": "{}", "predicted_verdict": "vulnerable",
         "oracle_verdict": "not_vulnerable_for_target"},
    ]
    write(root / "runs/run-1/results.json", json.dumps(rows))
    summary = {"label": "original", "summary": {"by_arm": {"A_pf": {
        "incorrect_suggestion": {"n": 1, "balanced_accuracy": 0.5}}}}}
    write(root / "runs/run-1/summary.json", json.dumps(summary))
    write(root / "runs/run-1/manifest.json", json.dumps({"stage": "wsp-v3-baseline"}))
    write(root / "hf/patch/README.md",
          "# patch\n\n{{RESULTS_TABLE}}\n\n{{SUMMARY_FILES}}\n\n{{FILES_TABLE}}\n")
    write(root / "hf/space/app.py", "print('app')\n")
    write(root / "hf/space/requirements.txt", "gradio==6.31.0\n")
    write(root / "hf/space/README.md", "---\nsdk: gradio\napp_file: app.py\n---\n\ndemo\n")
    for name, text in (extra_files or {}).items():
        write(root / name, text)
    size = weight_bytes if weight_bytes is not None else 64
    patch = root / "checkpoints/patch.safetensors"
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_bytes(safetensors_bytes(["layer::values"]) + b"\0" * max(0, size - 300))
    config = {
        "namespace": "FixtureUser",
        "token_env": "HF_PUBLISH_TOKEN",
        "exclude_patterns": ["runs/**/recovered-*.json"],
        "base_model": {"id": "CohereLabs/aya-expanse-8b", "license": "cc-by-nc-4.0"},
        "limits": {"max_weight_bytes": 1 << 20, "weight_suffixes": [".safetensors", ".bin"],
                   "base_shard_name_patterns": ["^model-\\d{5}-of-\\d{5}\\.safetensors$"],
                   "allowed_suffixes": [".json", ".md", ".txt", ".py", ".safetensors"]},
        "collections": {"models": {"title": "Fixture — models", "description": "d", "item_type": "model"},
                        "datasets": {"title": "Fixture — datasets", "description": "d", "item_type": "dataset"}},
        "repos": [
            {"id": "patch-repo", "type": "model", "collection": "models", "license": "cc-by-nc-4.0",
             "card": "hf/patch/README.md", "forbid_base_slices": True,
             "summary_globs": ["runs/run-1/summary.json"],
             "sources": [{"path": "checkpoints/patch.safetensors", "as": "patch.safetensors"}]},
            {"id": "bench-repo", "type": "dataset", "collection": "datasets", "license": "cc-by-nc-4.0",
             "card": "hf/patch/README.md", "summary_globs": ["runs/run-1/summary.json"],
             "sources": [{"path": "datasets/bench.json"},
                         {"path": "datasets/final.json", "sealed": True, "required": False}]},
            {"id": "demo", "type": "space", "collection": None, "license": "cc-by-nc-4.0",
             "sources": [{"path": "hf/space/app.py", "as": "app.py"},
                         {"path": "hf/space/requirements.txt", "as": "requirements.txt"},
                         {"path": "hf/space/README.md", "as": "README.md"}]},
        ],
    }
    return root, config


class RecordingAPI:
    def __init__(self, name="FixtureUser"):
        self.name = name
        self.calls = []

    def whoami(self, token=None):
        self.calls.append(("whoami", token))
        return {"name": self.name}

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))
        return kwargs["repo_id"]

    def update_repo_settings(self, **kwargs):
        self.calls.append(("update_repo_settings", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))

    def list_collections(self, **kwargs):
        self.calls.append(("list_collections", kwargs))
        return []

    def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))

        class Collection:
            slug = "FixtureUser/fixture-slug"
        return Collection()

    def add_collection_item(self, **kwargs):
        self.calls.append(("add_collection_item", kwargs))


class RefusingAPI(RecordingAPI):
    def __getattr__(self, name):
        raise AssertionError(f"network API must not be touched: {name}")


def test_plan_reports_availability_and_sizes(tmp_path):
    root, config = fixture(tmp_path)
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert report["errors"] == []
    by_id = {r["id"]: r for r in report["repos"]}
    assert by_id["patch-repo"]["available"] and by_id["bench-repo"]["available"] and by_id["demo"]["available"]
    assert by_id["bench-repo"]["excluded"] == ["sealed: datasets/final.json"]
    assert by_id["patch-repo"]["total_bytes"] > 0
    assert report["secret_values_loaded"] == 2
    text = H.format_plan(report)
    assert "patch-repo" in text and "RESULT: clean" in text
    assert SECRET_VALUE not in text


def test_plan_marks_missing_sources_unavailable(tmp_path):
    root, config = fixture(tmp_path)
    config["repos"][0]["sources"] = [{"path": "checkpoints/absent.safetensors"}]
    report = H.build_plan(config, root=root, env_path=root / ".env")
    item = next(r for r in report["repos"] if r["id"] == "patch-repo")
    assert item["available"] is False
    assert item["missing"] == ["checkpoints/absent.safetensors"]


def test_stage_copies_files_and_renders_cards(tmp_path):
    root, config = fixture(tmp_path)
    out = tmp_path / "out"
    manifest = H.stage(config, out, root=root, env_path=root / ".env")
    staged = {r["id"]: r for r in manifest["repos"]}
    assert staged["patch-repo"]["staged"] and staged["bench-repo"]["staged"] and staged["demo"]["staged"]
    card = (out / "patch-repo/README.md").read_text()
    assert "{{" not in card
    assert "layers" not in card and "0.500" in card          # results table rendered from summary.json
    assert "patch.safetensors" in card                        # files table
    assert (out / "demo/app.py").is_file()
    assert json.loads((out / "stage-manifest.json").read_text())["include_final"] is False


def test_stage_refuses_secret_in_source(tmp_path):
    root, config = fixture(tmp_path, extra_files={"datasets/leak.json": f'{{"key": "{SECRET_VALUE}"}}'})
    config["repos"][1]["sources"].append({"path": "datasets/leak.json"})
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert any("secret scan matched" in e for e in report["errors"])
    with pytest.raises(H.PublishRefused):
        H.stage(config, tmp_path / "out", root=root, env_path=root / ".env")


def test_scan_bytes_catches_patterns_without_echoing_them():
    secrets = [("BEAM_TOKEN", SECRET_VALUE.encode())]
    assert "env value of BEAM_TOKEN" in H.scan_bytes(f"x={SECRET_VALUE}".encode(), secrets)
    assert "hf_ token pattern" in H.scan_bytes(f'token = "{FAKE_HF}"'.encode(), [])
    beam_like = "Ab3" * 30
    assert any("Beam" in f for f in H.scan_bytes(beam_like.encode(), []))
    findings = H.scan_bytes(f"x={SECRET_VALUE}".encode(), secrets)
    assert all(SECRET_VALUE not in f for f in findings)
    assert H.scan_bytes(b"harmless text", secrets) == []


def test_publish_refuses_without_owner_flag(tmp_path):
    root, config = fixture(tmp_path)
    api = RefusingAPI()
    with pytest.raises(H.PublishRefused, match="confirm-owner-approved"):
        H.publish(config, tmp_path / "out", False, root=root, api=api, env_path=root / ".env")
    assert api.calls == []


def test_publish_refuses_wrong_account(tmp_path):
    root, config = fixture(tmp_path)
    api = RecordingAPI(name="SomeoneElse")
    with pytest.raises(H.PublishRefused, match="expected namespace"):
        H.publish(config, tmp_path / "out", True, root=root, api=api, env_path=root / ".env")
    assert not [c for c in api.calls if c[0] in ("create_repo", "upload_folder")]


def test_publish_refuses_aya_token_for_writes(tmp_path):
    root, config = fixture(tmp_path)
    write(root / ".env", f"HF_PUBLISH_TOKEN={'h' * 34}\nAAWDEN_AYA_HF_TOKEN={'h' * 34}\n")
    with pytest.raises(H.PublishRefused, match="Aya read token"):
        H.publish(config, tmp_path / "out", True, root=root, api=RefusingAPI(), env_path=root / ".env")


def test_publish_creates_public_repos_and_collections(tmp_path):
    root, config = fixture(tmp_path)
    api = RecordingAPI()
    result = H.publish(config, tmp_path / "out", True, root=root, api=api, env_path=root / ".env")
    created = [c[1] for c in api.calls if c[0] == "create_repo"]
    assert {c["repo_id"] for c in created} == {"FixtureUser/patch-repo", "FixtureUser/bench-repo", "FixtureUser/demo"}
    assert all(c["private"] is False for c in created)
    settings = [c[1] for c in api.calls if c[0] == "update_repo_settings"]
    assert settings and all(s["private"] is False and s["gated"] is False for s in settings)
    uploads = {c[1]["repo_id"] for c in api.calls if c[0] == "upload_folder"}
    assert uploads == {"FixtureUser/patch-repo", "FixtureUser/bench-repo", "FixtureUser/demo"}
    assert set(result["collections"]) == {"Fixture — models", "Fixture — datasets"}
    items = [c[1]["item_id"] for c in api.calls if c[0] == "add_collection_item"]
    assert items == ["FixtureUser/patch-repo", "FixtureUser/bench-repo"]


def test_sealed_split_excluded_unless_requested(tmp_path):
    root, config = fixture(tmp_path)
    write(root / "datasets/final.json", json.dumps({"cases": []}))
    default = H.build_space_examples([root / "runs/run-1"], None, tmp_path / "a.json", root=root)
    assert {e["case_id"] for e in default["examples"]} == {"c1"}
    assert default["excluded_splits"] == ["wsp_final"]
    included = H.build_space_examples([root / "runs/run-1"], None, tmp_path / "b.json", include_final=True,
                                      root=root, cases_paths=["datasets/bench.json"])
    assert {e["case_id"] for e in included["examples"]} == {"c1", "c2"}
    row = included["examples"][0]
    assert set(("case_id", "category", "code", "threat_model", "condition", "method", "raw_output",
                "predicted_verdict", "oracle_verdict")) <= set(row)
    assert row["code"] and row["threat_model"]


def test_space_examples_carry_patch_summary(tmp_path):
    root, config = fixture(tmp_path)
    write(root / "patch.json", json.dumps({"format": "aawden-patch-v1", "changed_scalars": 7,
                                           "total_parameters": 100, "percent_changed": 7.0,
                                           "tensors": [{"name": "l.0::values"}]}))
    payload = H.build_space_examples([root / "runs/run-1"], root / "patch.json", tmp_path / "e.json", root=root)
    assert payload["patch_summary"]["changed_scalars"] == 7
    assert payload["patch_summary"]["changed_tensors"] == ["l.0::values"]


def test_large_weight_file_refused(tmp_path):
    root, config = fixture(tmp_path, weight_bytes=(1 << 20) + 4096)
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert any("over 1048576 bytes refused" in e for e in report["errors"])
    assert any("over 1048576 bytes refused" in e for e in report["errors"])


def test_base_shard_name_refused(tmp_path):
    root, config = fixture(tmp_path)
    shard = root / "checkpoints/model-00001-of-00004.safetensors"
    shard.write_bytes(safetensors_bytes(["w"]))
    config["repos"][0]["sources"] = [{"path": "checkpoints/model-00001-of-00004.safetensors"}]
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert any("base model shard name refused" in e for e in report["errors"])


def test_weight_glob_and_unlisted_suffix_refused(tmp_path):
    root, config = fixture(tmp_path)
    config["repos"][0]["sources"] = [{"glob": "checkpoints/*.safetensors"}]
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert any("explicit path" in e for e in report["errors"])
    assert any("glob under checkpoints/" in e for e in report["errors"])
    write(root / "runs/run-1/weights.safetensors", "x")
    config["repos"][0]["sources"] = [{"path": "runs/run-1/weights.safetensors"}]
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert any("not a safetensors file" in e for e in report["errors"])   # refuses to guess at a corrupt weight
    (root / "runs/run-1/weights.safetensors").write_bytes(safetensors_bytes(["w::values"]))
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert report["errors"] == []


def test_patch_with_base_slices_refused(tmp_path):
    root, config = fixture(tmp_path)
    (root / "checkpoints/patch.safetensors").write_bytes(
        safetensors_bytes(["layer::values", "layer::original"]))
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert any("base tensor slices" in e for e in report["errors"])


def test_expected_sha256_mismatch_refused(tmp_path):
    root, config = fixture(tmp_path)
    config["repos"][0]["expected_sha256"] = {"checkpoints/patch.safetensors": "0" * 64}
    report = H.build_plan(config, root=root, env_path=root / ".env")
    assert any("sha256 does not match" in e for e in report["errors"])


def test_enable_final_includes_sealed_source(tmp_path):
    root, config = fixture(tmp_path)
    write(root / "datasets/final.json", json.dumps({"cases": []}))
    report = H.build_plan(config, root=root, env_path=root / ".env", include_final=True)
    item = next(r for r in report["repos"] if r["id"] == "bench-repo")
    assert item["excluded"] == []
    assert any(f["dest"] == "datasets/final.json" for f in item["files"])


def test_exclude_patterns_skip_recovered_transcripts(tmp_path):
    root, config = fixture(tmp_path, extra_files={"runs/run-1/recovered-task.json": '{"a": 1}'})
    config["repos"][1]["sources"].append({"glob": "runs/**/recovered-*.json", "required": False})
    report = H.build_plan(config, root=root, env_path=root / ".env")
    item = next(r for r in report["repos"] if r["id"] == "bench-repo")
    assert "pattern-excluded: runs/run-1/recovered-task.json" in item["excluded"]
    assert all(f["dest"] != "runs/run-1/recovered-task.json" for f in item["files"])


@pytest.mark.parametrize("card,kind", [
    ("hf/model-aya8b-adapter/README.md", "model"),
    ("hf/dataset-flip-bench/README.md", "dataset"),
    ("hf/space/README.md", "space"),
])
def test_repo_cards_have_front_matter(card, kind):
    text = (ROOT / card).read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---")[1])
    assert front["license"] == "cc-by-nc-4.0"
    if kind in ("model", "space"):
        assert front.get("base_model") == "CohereLabs/aya-expanse-8b"
    if kind == "space":
        assert front["sdk"] == "static" and front["app_file"] == "index.html"
    if kind == "model":
        assert "{{FILES_TABLE}}" in text
        assert "not included" in text.lower()


def test_publish_config_matches_cards_and_namespace():
    config = json.loads((ROOT / "configs" / "hf-publish.json").read_text(encoding="utf-8"))
    assert config["namespace"] == "RishyanthReddy"
    assert config["token_env"] == "HF_PUBLISH_TOKEN"
    assert config["base_model"]["weights_uploaded"] is False
    assert config["limits"]["max_weight_bytes"] == 1 << 30
    ids = [r["id"] for r in config["repos"]]
    assert ids == ["aawden-aya8b-reviewer-adapter", "aawden-reviewer-flip-bench", "aawden-demo"]
    for entry in config["repos"]:
        assert (ROOT / entry["card"]).is_file()
        assert entry["license"] == "cc-by-nc-4.0"
        for spec in entry["sources"]:
            assert not str(spec.get("glob", "")).startswith("checkpoints")
    assert config["collections"]["models"]["title"] == "Aawden — the reviewer adapter"
    assert config["collections"]["datasets"]["title"] == "Aawden — benchmark and results"


def test_space_is_a_self_contained_static_page():
    """The Space is a static page: no inference, no network, no build step.

    It replays saved verdicts, so the only things worth asserting are that it exists, that it carries
    no external dependencies, and that every verdict it shows came from a saved table.
    """
    page = ROOT / "docs" / "demo.html"
    assert page.is_file()
    text = page.read_text(encoding="utf-8")
    assert "<script src=" not in text and "<link rel=\"stylesheet\"" not in text
    assert "http://" not in text and "https://" not in text
    # Saved verdicts are embedded as JSON; a live model would need a fetch or an API token.
    assert "fetch(" not in text
    assert '"models"' in text
    card = (ROOT / "hf" / "space" / "README.md").read_text(encoding="utf-8")
    assert "sdk: static" in card and "app_file: index.html" in card
