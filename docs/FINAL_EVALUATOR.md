# External final evaluator handoff

## Ready offline review material

`dist/aawden-label-review.zip` contains a label-blind packet for all 190 exposed development cases and an incomplete response template. It includes no model answers or author verdicts. The separate `dist/aawden-explanation-review.zip` contains 20 saved explanations from the selected Aya result. Complete and commit label decisions before opening explanation material or the labelled repository datasets. A reviewer who has already seen those answers must not attest blinding.

For the development labels, validate the completed review using `python aawden.py verify-label-review --development --dataset datasets/development-review.json --review REVIEW.json`. For the selected explanation packet, use `python aawden.py score-adapter-explanations --packet runs/independent-review-v7/explanation-packet.json --review REVIEW.json --output runs/explanation-review-scores.json`. Empty forms, mismatched commitments, missing decisions and missing independence attestations are rejected.

These packets prepare independent work; they do not constitute completed independent review, and they cannot turn exposed cases into a protected final test. A new final dataset and a separately authorized GPU evaluation remain necessary. No Beam measurement is authorized by these handoff commands.

This stage needs an evaluator and a reviewer independent of the edit search. Keep the final dataset, its private behavioral fixtures, and the completed review outside the editing workspace until freeze. The exposed development datasets in this repository are not eligible final cases.

1. The evaluator prepares the usual Aawden case schema with split set to final_test. Use new source and template families and keep both verdicts represented. Prefer paired vulnerable and fixed examples within each family. Run the behavioral oracle for every case. Assign neutral case identifiers and test identifiers that do not disclose a verdict.
2. Create the reviewer packet in the evaluator workspace. It contains code, threat models, source hashes and required test IDs. It omits author labels and model outputs.
3. Give the packet and the necessary fixture runner to the independent reviewer. The reviewer writes their own verdict and evidence note, runs the specified checks, and fills the review records. Resolve disagreements and regenerate commitments before freezing. Do not disclose outputs or labels to the reviewer while they are assessing cases.
4. Prepare a separate set of ordinary Python questions and independently verify the expected JSON answers. Each case needs `case_id`, `code`, `question`, and `expected`. See the [coding smoke examples](../datasets/coding-smoke.json) for the schema. Commit the new coding set alongside the final security dataset and label review. Send the three SHA-256 commitments and the exposure statement before opening any of these files to the edit team.
5. After freeze, evaluate on the committed files. Keep original/strong as the primary comparator and retain every invalid answer and random control. The coding check scores exact answers without executing model outputs.

~~~bash
# Evaluator workspace; these files remain private until freeze.
python aawden.py review-packet --dataset /evaluator/final.json --output /evaluator/packet.json
python aawden.py verify-final-review --dataset /evaluator/final.json --review /evaluator/review.json
python aawden.py validate-coding-data --dataset /evaluator/coding.json

# Editing workspace; do not load the final files at this step.
python aawden.py freeze --search runs/aya-pilot/candidates \
  --final-dataset-sha256 DATASET_SHA --final-review-sha256 REVIEW_SHA \
  --coding-dataset-sha256 CODING_SHA \
  --exposure-note "Describe actual separation and any prior exposure."

# After freeze, on the approved evaluation host.
python aawden.py final-evaluate --freeze-manifest runs/aya-pilot/candidates/frozen.json \
  --dataset /evaluator/final.json --review /evaluator/review.json \
  --coding-dataset /evaluator/coding.json --output runs/aya-final
~~~

The reviewer document follows this example. Replace every example value with the reviewer's actual evidence and include every case exactly once.

~~~json
{
  "schema_version": 1,
  "dataset_sha256": "hash reported by review-packet",
  "reviewer_id": "reviewer's stable identifier",
  "independent_of_author_and_edit_selection": true,
  "blinded_to_oracle_labels": true,
  "blinded_to_model_outputs": true,
  "cases": [
    {
      "case_id": "case id from packet",
      "source_sha256": "source hash from packet",
      "verdict": "vulnerable",
      "passed_test_ids": ["test identifiers observed passing"],
      "note": "What behavior was observed and why it supports this verdict."
    }
  ]
}
~~~

Aawden checks exact byte commitments, review coverage, matching source hashes, behavioral test IDs, verdict agreement and development overlap, including exact AST templates. These checks cannot establish a reviewer's identity or prove workspace isolation. The evaluator must enforce both and report any exposure.

## Independent explanation review

After the frozen run, give the reviewer only the explanation packet. It contains code, threat models, verdicts, citations and explanations for the sampled answers. The reviewer checks whether the cited lines support the verdict and explanation under the stated threat model. Keep arm assignments, reference labels and score summaries private until the review is complete.

~~~bash
python aawden.py grounding-packet --run runs/aya-final --output /evaluator/grounding-packet.json
python aawden.py score-grounding --run runs/aya-final --review /evaluator/grounding-review.json --output /evaluator/grounding-scores.json
python aawden.py report --run runs/aya-final --grounding-review /evaluator/grounding-review.json > /evaluator/complete-report.md
~~~

The review includes `schema_version` set to `1`, the packet's `packet_sha256`, a `reviewer_id`, and attestations `independent_of_edit_selection` and `blinded_to_model_arm` set to `true`. In `explanations`, provide each packet `review_id` once with a boolean `supported` and a `note` citing the evidence for that decision. Scoring keeps invalid answers and missing citations in the denominator.

To prepare the existing development cases for a separate label review, run `review-packet --development --dataset datasets/screen.json --output runs/development-review-packet.json`. Validate the returned review with `verify-label-review --development` using the same dataset and the review file. The final set stays under the evaluator's control.

