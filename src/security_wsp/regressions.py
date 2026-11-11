"""Independent coding and explanation checks for the frozen comparison."""
from __future__ import annotations

import random
from pathlib import Path

from .artifacts import atomic_json, canonical, digest, file_hash, read_json, verify_run
from .parsing import strict_json, tolerant_json
from .prompts import CONDITIONS, GenerationRequest


def load_coding_cases(dataset: Path):
    document = strict_json(dataset.read_text(encoding="utf-8"))
    cases = document.get("cases") if isinstance(document, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("The coding dataset must contain at least one case")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "code", "question", "expected"}:
            raise ValueError("Each coding case needs case_id, code, question, and expected")
        if any(not isinstance(case[key], str) or not case[key].strip() for key in ("case_id", "code", "question")):
            raise ValueError("Coding case IDs, code, and questions must be nonempty strings")
        if case["case_id"] in ids:
            raise ValueError("Duplicate coding case ID")
        ids.add(case["case_id"])
        canonical(case["expected"])
    return cases


def evaluate_coding(cases, adapter, arm, store, config):
    rows = []
    for case in cases:
        request = GenerationRequest(case["case_id"], "coding_regression", [
            {"role": "system", "content": "Answer the Python 3.11 code question. Return exactly one JSON object "
             "with one key, answer. Use a JSON value for the requested result."},
            {"role": "user", "content": f"<source>\n{case['code']}\n</source>\n{case['question']}"},
        ], config["max_new_tokens"], config["seed"])
        storage_id = digest(["coding_regression", arm, request.request_id])[:24]
        row = store.load_prediction(storage_id)
        if row is None:
            outputs = adapter.generate([request])
            if len(outputs) != 1 or outputs[0].request_id != request.request_id:
                raise ValueError("Misaligned coding regression generation")
            response = outputs[0]
            error = response.error
            correct = False
            if error is None:
                try:
                    # Tolerant by design: this set asks whether the edit preserved coding competence,
                    # not whether the model can emit bare JSON. Aya wraps a correct answer in a
                    # ```json fence and an explanation, and strict_json scored 13 correct answers
                    # as failures (runs/wsp-v3/regressions2, 2026-11-10). The answer still has to
                    # be exactly one object with exactly one key, so the check keeps its teeth.
                    parsed = tolerant_json(response.text)
                    if not isinstance(parsed, dict) or set(parsed) != {"answer"}:
                        raise ValueError("Expected one answer field")
                    correct = canonical(parsed["answer"]) == canonical(case["expected"])
                except (ValueError, TypeError) as exc:
                    error = str(exc)
            row = {"case_id": case["case_id"], "arm": arm, "messages": request.messages,
                   "request_id": request.request_id, "storage_id": storage_id, "raw_output": response.text,
                   "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                   "latency_seconds": response.latency_seconds, "finish_reason": response.finish_reason,
                   "correct": correct, "error": error}
            store.save_prediction(storage_id, row)
        rows.append(row)
    return rows


def coding_summary(rows, dataset_sha256):
    arms = {}
    case_ids = set()
    for arm in ("original", "edited"):
        predictions = [row for row in rows if row["arm"] == arm]
        ids = {row["case_id"] for row in predictions}
        if not ids or len(ids) != len(predictions) or (case_ids and case_ids != ids):
            raise ValueError("Coding regression comparisons require identical, unique cases")
        case_ids = ids
        correct = sum(row["correct"] for row in predictions)
        arms[arm] = {"cases": len(ids), "correct": correct, "accuracy": correct / len(ids),
                     "invalid": sum(row["error"] is not None for row in predictions)}
    return {"dataset_sha256": dataset_sha256, "arms": arms,
            "accuracy_difference": arms["edited"]["accuracy"] - arms["original"]["accuracy"],
            "scope": "Exact answers to ordinary Python code questions. No model-generated code is executed."}


def grounding_packet(run: Path):
    verify_run(run)
    manifest = read_json(run / "manifest.json")
    if manifest["stage"] != "final_evaluation" or manifest["evidence_kind"] != "cohere_local_checkpoint":
        raise ValueError("Explanation review requires a completed frozen model evaluation")
    cases = {row["case_id"]: row for row in read_json(run / "dataset.json")["cases"]}
    protocol = manifest["identity"]["protocol"]
    limit = protocol.get("grounding_case_limit", 20)
    if type(limit) is not int or limit < 1:
        raise ValueError("Explanation review requires a positive case limit fixed before evaluation")
    rng = random.Random(protocol["seed"])
    sample = set(rng.sample(sorted(cases), min(len(cases), limit)))
    results_hash = file_hash(run / "results.json")
    public = []
    arm_by_id = {}
    denominators = dict.fromkeys(("B", "D"), 0)
    coverage = set()
    for row in read_json(run / "results.json"):
        arm, case_id = row["method"], row["case_id"]
        if arm not in denominators or case_id not in sample:
            continue
        coverage.add((case_id, arm, row["condition"]))
        denominators[arm] += 1
        parsed = row["parsed"]
        if not parsed or not parsed["evidence_lines"]:
            continue
        review_id = digest([results_hash, row["storage_id"]])[:24]
        case = cases[case_id]
        arm_by_id[review_id] = arm
        public.append({"review_id": review_id, "code": case["code"], "threat_model": case["threat_model"],
                       "category": case["target_category"], "verdict": parsed["verdict"],
                       "evidence_lines": parsed["evidence_lines"], "explanation": parsed["explanation"]})
    expected = {(case_id, arm, condition) for case_id in sample for arm in denominators for condition in CONDITIONS}
    if coverage != expected or sum(denominators.values()) != len(expected) or len(arm_by_id) != len(public):
        raise ValueError("The explanation sample must cover both arms on the same cases and conditions")
    public.sort(key=lambda row: row["review_id"])
    content = {"schema_version": 1, "sampled_cases": len(sample), "explanations": public}
    return dict(content, packet_sha256=digest(content)), arm_by_id, denominators


def write_grounding_packet(run: Path, output: Path):
    packet, _, _ = grounding_packet(run)
    atomic_json(output, packet)
    return {"packet_sha256": packet["packet_sha256"], "explanations": len(packet["explanations"]), "output": str(output)}


def score_grounding(run: Path, review: Path, output: Path | None = None):
    packet, arms, totals = grounding_packet(run)
    document = read_json(review)
    if document.get("schema_version") != 1 or document.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("The review does not match the blinded explanation packet")
    reviewer = document.get("reviewer_id")
    if (not isinstance(reviewer, str) or not reviewer.strip() or document.get("blinded_to_model_arm") is not True
            or document.get("independent_of_edit_selection") is not True):
        raise ValueError("An independent reviewer must attest that the model arms were blinded")
    records = document.get("explanations")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Expected a list of reviewed explanations")
    by_id = {record.get("review_id"): record for record in records}
    if set(by_id) != set(arms) or len(by_id) != len(records):
        raise ValueError("The review must cover every explanation once")
    counts = {arm: {"sampled": totals[arm], "reviewed": 0, "supported": 0} for arm in totals}
    for review_id, record in by_id.items():
        if type(record.get("supported")) is not bool or not isinstance(record.get("note"), str) or not record["note"].strip():
            raise ValueError("Every explanation needs a support decision and an evidence note")
        group = counts[arms[review_id]]
        group["reviewed"] += 1
        group["supported"] += record["supported"]
    for group in counts.values():
        group["support_rate_all_sampled"] = group["supported"] / group["sampled"]
    summary = {"packet_sha256": packet["packet_sha256"], "review_sha256": file_hash(review),
               "reviewer_id": reviewer, "arms": counts,
               "support_rate_difference": counts["D"]["support_rate_all_sampled"] - counts["B"]["support_rate_all_sampled"],
               "scope": "Reviewer judgments on the frozen sample. Invalid answers and missing citations remain in the denominator."}
    if output is not None:
        atomic_json(output, summary)
    return summary
