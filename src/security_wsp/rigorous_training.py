"""Bounded continuation training on exposed examples and concrete code explanations."""
import json
import random

from .artifacts import atomic_json, file_hash
from .data import CLEAN
from .rigorous_review import checked_dataset, rigorous_request


# These are supervised targets for previously scored confirmation cases.
DETAILS = {
    "trusted_alias": ("The reassigned fragment inserts the name before execute.",
                      "The alias is fixed and the name remains a separate bound value."),
    "conditional_normalization": ("A name containing a quote takes the raw interpolation branch.",
                                   "Alphabetic names cannot close the literal and other names are bound."),
    "dictionary_override": ("A quote triggers the override of the bound query with raw SQL text.",
                            "The override preserves the query placeholder and binds the name."),
    "sort_after_interpolation": ("The raw format operation embeds the name before the order clause is appended.",
                                 "The name stays bound while the fixed order column is appended."),
    "commented_decoy": ("A quote sets index one and executes the raw fallback query.",
                        "The index is fixed at zero so the interpolated fallback is never executed."),
    "escape_then_recover": ("The second replacement undoes quote doubling before building SQL.",
                            "The literal passed into SQLite preserves the doubled quote characters."),
    "named_parameter_and_clause": ("The OR arm interpolates the name even though the other arm binds it.",
                                   "The only use of the name is the named bind argument."),
    "keyed_callable": ("A name containing a quote selects the callable that interpolates it.",
                       "Both modes call the function that binds the name."),
    "tuple_match_guard": ("A same tenant nonowner matches the second alternative and receives the record.",
                          "Only the pattern with a true owner comparison can return the record."),
    "approval_reset": ("The later OR allows a same tenant member even when the owner check failed.",
                       "The later AND retains the owner check for the member path."),
    "early_cache_hit": ("The early cache hit returns the requested record to any authenticated session.",
                        "Both the cache hit and the fallback require the owner check."),
    "negated_denials": ("The all check fails to block an authenticated nonowner.",
                        "A true owner mismatch alone makes any return true and blocks the request."),
    "filter_projection": ("An empty owner-filtered list falls back to the original unfiltered row.",
                          "Only the owner-filtered list can contribute a returned record."),
    "delegated_decision": ("The member OR branch bypasses the owner comparison.",
                          "The delegated decision compares the requested item owner to the session ID."),
    "fallback_principal": ("The assignment overwrites the session principal with the requested record owner.",
                           "The preexisting session ID survives setdefault and the later assignment."),
    "request_key_guard": ("The check uses an owned key but the returned row uses the requested key.",
                          "The same requested key determines both permission and the returned row."),
    "projection_template": ("The format call places the requested name inside SQL quotes.",
                            "The formatted projection is trusted and the name is bound separately."),
    "json_request": ("The request builder replaces the placeholder with raw name text.",
                     "The decoded argument list is passed separately to execute."),
    "late_query_selection": ("A name with a quote selects the interpolated query and no bindings.",
                             "Only the placeholder query is selected, with the name bound."),
    "nested_quote_escaping": ("Replacing a quote with itself leaves it able to close the SQL literal.",
                              "The nested helper doubles each quote before wrapping the name in SQL quotes."),
    "replacement_map": ("The template inserts raw name text and the bind mapping cannot change that text.",
                        "The template inserts a fixed named placeholder, then binds the supplied name."),
    "normalization_then_restore": ("The concatenation uses the original name instead of the escaped value.",
                                   "The concatenation uses the escaped name, preserving its quotes as data."),
    "switch_after_binding": ("The match guard replaces the bound predicate with raw interpolation for a quote.",
                             "The match guard keeps the name as a separate bound value."),
    "identifier_only_branch": ("A quoted name reaches the fallback that interpolates it into SQL.",
                               "Only identifier names can reach literal interpolation and other names are bound."),
    "dictionary_decision_merge": ("A true tenant value satisfies any even when the owner value is false.",
                                  "The return checks the owner decision directly."),
    "identity_set": ("The union adds the caller to the permitted set even for someone else's record.",
                     "The permitted set contains only the requested record owner."),
    "conditioned_early_return": ("A logged-in same-tenant caller can return early before the owner check.",
                                 "Both return paths require ownership of the requested record."),
    "auth_filter_cache": ("The default value returns the unfiltered record when the owner filter misses.",
                          "The requested key must be present in the owner-filtered map to return data."),
    "membership_inversion": ("The denial conjunction leaves a regular nonowner unblocked.",
                             "An owner mismatch alone triggers the denial and blocks the return."),
    "short_circuit_return": ("The authenticated operand makes allowed true for a nonowner.",
                             "The final expression can return the row only when the owner comparison succeeds."),
    "attribute_projection": ("The projected tenant fields match for same-tenant nonowners.",
                             "The projected owner and session identity must match to return the row."),
    "different_record_selector": ("Having any owned key allows a request for someone else's record.",
                                  "The requested key itself must be present in the owned-key list."),
    "different_query_keys": ("The guard uses any owned key while the return uses the requested key.",
                             "The permission list must include the requested key itself."),
    "compound_exception": ("A same tenant logged-in member can take the exception without ownership.",
                           "The exception still requires the same owner comparison as the main path."),
    "set_gate": ("Login removes the ownership issue and lets a foreign record through.",
                 "Discarding the authentication issue leaves the ownership blocker intact."),
    "indexed_criteria": ("An authenticated session selects the tenant comparison instead of ownership.",
                         "The selected criterion is always the requested record owner comparison."),
    "policy_payload": ("The chosen tuple slot compares tenant identity and allows a same-tenant nonowner.",
                       "The chosen slots compare the requested record owner and caller identity."),
    "permission_tuple_truth": ("A nonempty tuple is truthy even if its ownership element is false.",
                               "All requires the sole ownership element to be true before returning data."),
    "multiple_authorized_paths": ("The administrator branch can return another owner's record.",
                                  "Both the early administrator branch and the final return require ownership."),
    "record_permission_projection": ("The helper compares the requested record tenant to the caller tenant.",
                                     "The helper compares the requested record owner to the caller identity."),
    "truth_of_mapping": ("The decision dictionary is truthy even when the read entry is false.",
                         "The read entry controls the return and contains the owner comparison."),
    "conditional_grant_selection": ("Regular members use the office check and can bypass ownership.",
                                    "Both sides of the conditional use the private ownership check."),
    "parallel_gate": ("The nested member check returns data before the owner check.",
                      "The nested early return and final return both require ownership."),
    "permission_cache_key": ("The cache tests an owned key while returning the separately requested row.",
                             "The permission cache is queried using the same key that determines the returned row."),
    "reject_flag_cleanup": ("Login removes the foreign flag and lets the unauthorized request proceed.",
                            "The foreign flag survives and rejects a nonowner."),
    "numeric_gate": ("Login contributes one point, enough to return the record without ownership.",
                     "The threshold requires both the ownership point and the authentication point."),
    "branch_on_owner_mismatch": ("A logged-in nonowner gets the record in the mismatch branch.",
                                 "The mismatch branch returns None and the remaining path requires ownership."),
    "closure_policy_combination": ("The authentication OR branch bypasses the closure's ownership comparison.",
                                   "The closure requires both ownership and authentication before returning data."),
}


def target(case, evidence_first):
    name = case.family_id.removeprefix(f"{case.split}_{case.target_category}_")
    safe = case.oracle_verdict == CLEAN
    if name in DETAILS:
        explanation = DETAILS[name][safe]
    elif case.target_category == "sql_injection":
        explanation = ("The lookup keeps the name literal." if safe else
                       "The lookup can execute SQL syntax originating in the untrusted name.")
    else:
        explanation = ("The requested row is returned only to an actor allowed by the stated policy." if safe else
                       "The code can return the requested row without enforcing the stated access policy.")
    fields = {"category": case.target_category, "explanation": explanation,
              "evidence_lines": list(range(2, len(case.code.splitlines()) + 1)),
              "verdict": case.oracle_verdict, "confidence": .9}
    if evidence_first:
        return json.dumps(fields, separators=(",", ":"))
    verdict = fields.pop("verdict")
    return json.dumps({"verdict": verdict, **fields}, separators=(",", ":"))


def refine(adapter, root, spec, forbidden, output):
    import torch
    from safetensors.torch import save_file
    from .adaptation import LowRankLinear

    cases = checked_dataset(root, spec["exposed"]) + checked_dataset(root, spec["replay"])
    if {case.family_id for case in cases} & {case.family_id for case in forbidden}:
        raise ValueError("Training families overlap with validation or confirmation")
    model = adapter.model
    parameters = {f"{name}.{part}": getattr(module, part) for name, module in model.named_modules()
                  if isinstance(module, LowRankLinear) for part in ("a", "b")}
    if not parameters:
        raise ValueError("Expected a verified LoRA starting point")
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    verdict_weight = spec.get("verdict_weight", 1)
    examples = []
    for case in cases:
        for variant in spec["training_variants"]:
            request = rigorous_request(case, variant)
            prefill = request.messages[-1]["role"] == "assistant"
            messages = request.messages[:-1] if prefill else request.messages
            prefix = adapter.chat_prefix(messages)
            if prefill:
                prefix += request.messages[-1]["content"]
            answer = target(case, prefill)
            forced = request.messages[-1]["content"] if prefill else ""
            if not answer.startswith(forced):
                raise ValueError("Training answer differs from the prescribed response prefix")
            prefix_tokens = adapter._tokens(prefix).cpu()
            tokens = adapter._tokens(prefix + answer[len(forced):] + adapter.tokenizer.eos_token).cpu()
            boundary = (tokens[:, :prefix_tokens.shape[1]] != prefix_tokens).nonzero()
            start = boundary[0, 1].item() if len(boundary) else prefix_tokens.shape[1]
            labels = tokens.clone()
            labels[:, :start] = -100
            verdict_span = None
            if verdict_weight > 1:
                answer_start = len(forced)
                start_char = answer.index('"verdict":') + len('"verdict":')
                end_char = start_char + len(json.dumps(case.oracle_verdict))
                verdict_span = (
                    adapter._tokens(prefix + answer[answer_start:start_char]).shape[1],
                    adapter._tokens(prefix + answer[answer_start:end_char]).shape[1],
                )
            examples.append((tokens, labels, verdict_span))

    randomizer = random.Random(spec["seed"])
    torch.manual_seed(spec["seed"])
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=spec["learning_rate"], weight_decay=0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    history = []
    print("Refining Aya with", len(examples), "labelled review examples per epoch", flush=True)
    try:
        for epoch in range(spec["epochs"]):
            randomizer.shuffle(examples)
            losses = []
            for tokens, labels, verdict_span in examples:
                optimizer.zero_grad(set_to_none=True)
                tokens, labels = tokens.to(adapter.device), labels.to(adapter.device)
                if verdict_span is None:
                    loss = model(input_ids=tokens, labels=labels, use_cache=False).loss
                else:
                    logits = model(input_ids=tokens, use_cache=False).logits
                    per_token = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                                                                  labels[:, 1:], ignore_index=-100, reduction="none")
                    weights = (labels[:, 1:] != -100).to(per_token.dtype)
                    verdict_start, verdict_end = verdict_span
                    weights[:, verdict_start - 1:verdict_end - 1] *= verdict_weight
                    loss = (per_token * weights).sum() / weights.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0)
                optimizer.step()
                losses.append(loss.detach().item())
            history.append({"epoch": epoch + 1, "mean_loss": sum(losses) / len(losses)})
            print("Refinement", history[-1], flush=True)
    finally:
        model.gradient_checkpointing_disable()
        model.disable_input_require_grads()
        model.eval()
        for parameter in parameters.values():
            parameter.requires_grad_(False)
    checkpoint = output / "adapter.safetensors"
    parent = adapter.metadata["adapter_sha256"]
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in parameters.items()}, checkpoint)
    trained = file_hash(checkpoint)
    if trained == parent:
        raise RuntimeError("The training run did not change the adapter")
    adapter.metadata.update(adapter_sha256=trained, parent_adapter_sha256=parent)
    atomic_json(output / "training.json", {"protocol": spec, "parent_adapter_sha256": parent,
                                            "adapter_sha256": trained, "history": history,
                                            "training_case_count": len(cases)})
    return cases
