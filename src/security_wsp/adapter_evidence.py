"""CPU-only verification and independent review handoff for saved Aya answers."""
from pathlib import Path

from .artifacts import atomic_json, digest, file_hash, read_json, verify_run
from .adapter_review import request_for
from .data import load_dataset, validate_cases
from .final_review import prepare_review_packet
from .parsing import parse_response
from .statistics import metrics


def measurement_fingerprint(rows):
    fields = ('case_id', 'messages', 'raw_output', 'parsed', 'oracle_verdict', 'predicted_verdict', 'correct')
    return digest([{key: row[key] for key in fields if key in row} for row in rows])


def audit_adapter(run: Path, dataset: Path, selection: Path, *, check_weights=True):
    integrity = verify_run(run)
    selected = read_json(selection)
    if file_hash(dataset) != selected['confirmation_file_sha256']:
        raise ValueError('The committed confirmation dataset changed')
    cases = load_dataset(dataset)
    validate_cases(cases)
    by_id = {case.case_id: case for case in cases}
    rows = read_json(run / 'results.json')
    if len(rows) != len(by_id) or {row['case_id'] for row in rows} != set(by_id):
        raise ValueError('Saved answers must cover every committed case exactly once')
    summary = read_json(run / 'summary.json')
    if summary['model']['adapter_sha256'] != selected['adapter_sha256']:
        raise ValueError('The scored adapter differs from the selection')
    for row in rows:
        case = by_id[row['case_id']]
        if row['messages'] != request_for(case, 'trace').messages or row['condition'] != 'neutral':
            raise ValueError('Saved answers used a different model-visible prompt')
        parsed, error = parse_response(row['raw_output'], category=case.target_category,
                                       line_count=len(case.code.splitlines()))
        if row.get('generation_error'):
            parsed, error = None, row['generation_error']
        verdict = parsed['verdict'] if parsed else None
        if (row['oracle_verdict'] != case.oracle_verdict or row['predicted_verdict'] != verdict
                or row['parsed'] != parsed or row['parse_error'] != error
                or row['correct'] != (verdict == case.oracle_verdict)):
            raise ValueError('Saved decisions disagree with strict scoring of the raw answers')
    measured = metrics(rows)
    stored = summary['metrics']
    if 'balanced_accuracy' not in stored:
        stored = stored[next(iter(stored))]
    for key in ('n', 'balanced_accuracy', 'invalid_count', 'true_positive', 'true_negative'):
        if measured[key] != stored[key]:
            raise ValueError('Stored metrics disagree with recomputed metrics')
    manifest = read_json(run / 'manifest.json')
    receipt = manifest.get('export', {}).get('measurement_payload_sha256')
    fingerprint = measurement_fingerprint(rows)
    if receipt and receipt != fingerprint:
        raise ValueError('The measured generation payload changed')
    config = read_json(selected['model_config'])
    if config['revision'] != selected['base_revision'] or config['lora_adapter']['sha256'] != selected['adapter_sha256']:
        raise ValueError('The inference configuration differs from the selected model')
    if check_weights and file_hash(Path(config['lora_adapter']['checkpoint'])) != selected['adapter_sha256']:
        raise ValueError('The selected adapter tensor bytes changed')
    audit_path = run / 'auditkit.json'
    scorer_receipt = (read_json(audit_path).get('config', {}).get('extra', {}).get('aawden_records_sha256')
                      if audit_path.is_file() else None)
    return {'model': 'CohereLabs/aya-expanse-8b', 'adapter_sha256': selected['adapter_sha256'],
            'dataset_sha256': file_hash(dataset), 'manifest_sha256': file_hash(run / 'manifest.json'),
            'measurement_payload_sha256': fingerprint, 'integrity': integrity, 'metrics': measured,
            'tensor_bytes_verified': check_weights, 'measurement_receipt_verified': bool(receipt),
            'canonical_scored_records_sha256': digest(rows),
            'scorer_input_commitment_at_measurement': scorer_receipt,
            'scorer_commitment_scope': 'Original scoring-call metadata receipt; current exported record identity is reported separately.',
            'by_category': {category: metrics([row for row in rows if row['category'] == category])
                            for category in sorted({row['category'] for row in rows})},
            'scope': 'Re-scoring saved neutral generations. No new inference or independent review.'}


def prepare_adapter_handoff(run: Path, dataset: Path, selection: Path, output: Path):
    audit = audit_adapter(run, dataset, selection)
    output.mkdir(parents=True, exist_ok=False)
    cases = load_dataset(dataset)
    prepare_review_packet(dataset, output / 'label-packet.json', development=True)
    template = {'schema_version': 1, 'dataset_sha256': file_hash(dataset), 'reviewer_id': '',
                'independent_of_author_and_edit_selection': False, 'blinded_to_oracle_labels': False,
                'blinded_to_model_outputs': False,
                'cases': [{'case_id': case.case_id, 'source_sha256': case.source_sha256,
                           'verdict': None, 'passed_test_ids': [], 'note': ''} for case in cases]}
    atomic_json(output / 'label-review-template.json', template)
    by_id = {case.case_id: case for case in cases}
    source_hash = file_hash(run / 'results.json')
    explanations = []
    for row in read_json(run / 'results.json'):
        if row['parsed'] is None:
            continue
        case = by_id[row['case_id']]
        parsed = row['parsed']
        explanations.append({'review_id': digest([source_hash, row['storage_id']])[:24],
                             'code': case.code, 'threat_model': case.threat_model,
                             'category': case.target_category, 'verdict': parsed['verdict'],
                             'evidence_lines': parsed['evidence_lines'], 'explanation': parsed['explanation']})
    explanations.sort(key=lambda row: row['review_id'])
    content = {'schema_version': 1, 'explanations': explanations, 'sampled': len(cases)}
    packet = dict(content, packet_sha256=digest(content))
    atomic_json(output / 'explanation-packet.json', packet)
    atomic_json(output / 'explanation-review-template.json', {
        'schema_version': 1, 'packet_sha256': packet['packet_sha256'], 'reviewer_id': '',
        'independent_of_edit_selection': False, 'blinded_to_model_identity': False,
        'explanations': [{'review_id': row['review_id'], 'supported': None, 'note': ''} for row in explanations]})
    (output / 'README.md').write_text(
        '# Aawden independent review\n\n'
        'Complete the label review before opening the explanation packet. Receive only this packet; '
        'the project datasets and model results disclose the answer key. Independently check the code '
        'against the threat model and record behavioral evidence. Do not claim blinding if already exposed.\n\n'
        'The label template and explanation template are deliberately incomplete. An independent reviewer '
        'must supply decisions, evidence notes and truthful attestations. They are not completed reviews.\n\n'
        'These 20 cases are an exposed feasibility check. This review cannot make them a protected final test.\n',
        encoding='utf-8')
    return {'cases': len(cases), 'explanations': len(explanations), 'dataset_sha256': audit['dataset_sha256'],
            'packet_sha256': packet['packet_sha256'], 'output': str(output), 'independent_review_completed': False}


def score_adapter_explanations(packet: Path, review: Path):
    content, document = read_json(packet), read_json(review)
    expected_hash = digest({key: value for key, value in content.items() if key != 'packet_sha256'})
    if content['packet_sha256'] != expected_hash or document.get('packet_sha256') != expected_hash:
        raise ValueError('Explanation review does not match the packet')
    reviewer = document.get('reviewer_id')
    if (not isinstance(reviewer, str) or not reviewer.strip() or document.get('independent_of_edit_selection') is not True
            or document.get('blinded_to_model_identity') is not True):
        raise ValueError('An independent blinded reviewer must provide truthful attestations')
    records = document.get('explanations')
    ids = {row['review_id'] for row in content['explanations']}
    if (not isinstance(records, list) or not all(isinstance(row, dict) for row in records)
            or len(records) != len(ids) or {row.get('review_id') for row in records} != ids):
        raise ValueError('Review must cover each explanation exactly once')
    for row in records:
        if type(row.get('supported')) is not bool or not isinstance(row.get('note'), str) or not row['note'].strip():
            raise ValueError('Every judgment needs a support decision and evidence note')
    supported = sum(row['supported'] for row in records)
    return {'reviewer_id': document['reviewer_id'], 'packet_sha256': expected_hash,
            'sampled': content['sampled'], 'reviewed': len(records), 'supported': supported,
            'support_rate_all_sampled': supported / content['sampled'],
            'scope': 'Reviewer-attested explanation support; identity and independence require external enforcement.'}
