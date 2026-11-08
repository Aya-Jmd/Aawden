import json
from pathlib import Path
import pytest
from security_wsp.adapter_evidence import audit_adapter, prepare_adapter_handoff, score_adapter_explanations
from security_wsp.adapter_review import request_for
from security_wsp.artifacts import RunStore, atomic_json, file_hash, read_json
from security_wsp.data import build_cases, write_dataset
from security_wsp.evaluation import score_generation
from security_wsp.models import GenerationResult
from security_wsp.statistics import metrics


def evidence(tmp_path):
    cases = build_cases()[:2]
    dataset = tmp_path / 'cases.json'
    write_dataset(dataset, cases)
    checkpoint = tmp_path / 'adapter.safetensors'
    checkpoint.write_bytes(b'controlled-test-checkpoint')
    adapter_hash = file_hash(checkpoint)
    config = tmp_path / 'model.json'
    atomic_json(config, {'revision': 'pinned', 'lora_adapter': {'sha256': adapter_hash, 'checkpoint': str(checkpoint)}})
    selection = tmp_path / 'selection.json'
    atomic_json(selection, {'model_config': str(config), 'base_revision': 'pinned', 'adapter_sha256': adapter_hash,
                            'confirmation_file_sha256': file_hash(dataset)})
    run = tmp_path / 'run'
    store = RunStore(run, {'fixture': True}, stage='adapter_competence', evidence_kind='cohere_local_checkpoint')
    rows = []
    for case in cases:
        request = request_for(case, 'trace')
        text = json.dumps({'verdict': case.oracle_verdict, 'category': case.target_category, 'evidence_lines': [2],
                           'explanation': 'Controlled test evidence', 'confidence': .8})
        row = score_generation(case, request, GenerationResult(request.request_id, text, 5, 5, 'stop'), 'adapted')
        row['storage_id'] = request.request_id
        rows.append(row)
    atomic_json(run / 'results.json', rows)
    atomic_json(run / 'summary.json', {'model': {'adapter_sha256': adapter_hash}, 'metrics': metrics(rows)})
    store.finish([], ['results.json', 'summary.json'])
    return run, dataset, selection


def test_offline_audit_recomputes_scores_and_checks_checkpoint_bytes(tmp_path):
    run, dataset, selection = evidence(tmp_path)
    assert audit_adapter(run, dataset, selection)['metrics']['balanced_accuracy'] == 1
    (tmp_path / 'adapter.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='tensor bytes'):
        audit_adapter(run, dataset, selection)


def test_handoff_blinds_labels_and_rejects_unfinished_reviews(tmp_path):
    run, dataset, selection = evidence(tmp_path)
    output = tmp_path / 'handoff'
    result = prepare_adapter_handoff(run, dataset, selection, output)
    assert result['independent_review_completed'] is False
    packet = read_json(output / 'label-packet.json')
    assert all('oracle_verdict' not in row and 'split' not in row for row in packet['cases'])
    with pytest.raises(ValueError, match='independent'):
        score_adapter_explanations(output / 'explanation-packet.json', output / 'explanation-review-template.json')
    review = read_json(output / 'explanation-review-template.json')
    review.update(reviewer_id='test-reviewer', independent_of_edit_selection=True, blinded_to_model_identity=True)
    for row in review['explanations']:
        row.update(supported=True, note='Unit-test fixture judgment')
    atomic_json(output / 'review.json', review)
    result = score_adapter_explanations(output / 'explanation-packet.json', output / 'review.json')
    assert result['support_rate_all_sampled'] == 1
    review['explanations'].append(dict(review['explanations'][0]))
    atomic_json(output / 'review.json', review)
    with pytest.raises(ValueError, match='exactly once'):
        score_adapter_explanations(output / 'explanation-packet.json', output / 'review.json')


def test_audit_rejects_prompt_changes_even_when_resealed(tmp_path):
    run, dataset, selection = evidence(tmp_path)
    rows = read_json(run / 'results.json')
    rows[0]['messages'][1]['content'] += '\nLeaked answer'
    atomic_json(run / 'results.json', rows)
    manifest = read_json(run / 'manifest.json')
    manifest['artifact_hashes']['results.json'] = file_hash(run / 'results.json')
    atomic_json(run / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='prompt'):
        audit_adapter(run, dataset, selection)
