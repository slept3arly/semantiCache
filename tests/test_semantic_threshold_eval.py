import json
from pathlib import Path

import pytest

from evaluation.semantic_threshold_eval import evaluate_thresholds


def test_threshold_report_counts_accepted_false_accept_and_false_reject():
    records = [
        {"pair_id": "positive-close", "distance": 0.5, "expected_reuse": True, "metadata_eligible": True},
        {"pair_id": "positive-boundary", "distance": 0.6, "expected_reuse": True, "metadata_eligible": True},
        {"pair_id": "negative-close", "distance": 0.4, "expected_reuse": False, "metadata_eligible": True},
        {"pair_id": "negative-far", "distance": 0.8, "expected_reuse": False, "metadata_eligible": True},
        {"pair_id": "ineligible", "distance": 0.1, "expected_reuse": True, "metadata_eligible": False},
    ]

    report = evaluate_thresholds(records, [0.6])[0]

    assert report["accepted_pair_ids"] == ["positive-close", "negative-close"]
    assert report["rejected_pair_ids"] == ["positive-boundary", "negative-far", "ineligible"]
    assert report["false_accept_count"] == 1
    assert report["false_accept_pair_ids"] == ["negative-close"]
    assert report["false_reject_count"] == 1
    assert report["false_reject_pair_ids"] == ["positive-boundary"]
    assert report["labeled_pair_count"] == 4
    assert report["ineligible_pair_ids"] == ["ineligible"]


def test_threshold_is_strict_and_candidate_thresholds_are_independent():
    records = [
        {"pair_id": "at-cutoff", "distance": 0.6, "expected_reuse": True, "metadata_eligible": True}
    ]

    below, above = evaluate_thresholds(records, [0.6, 0.7])

    assert below["accepted_count"] == 0
    assert below["false_reject_count"] == 1
    assert above["accepted_pair_ids"] == ["at-cutoff"]
    assert above["false_accept_count"] == 0
    assert above["false_reject_count"] == 0


@pytest.mark.parametrize("thresholds", [[0], [-0.1], [float("inf")]])
def test_thresholds_must_be_finite_and_positive(thresholds):
    with pytest.raises(ValueError):
        evaluate_thresholds([], thresholds)


def test_fixed_dataset_has_independent_labels_and_rationales():
    dataset_path = Path(__file__).parents[1] / "evaluation" / "semantic_pairs.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    assert dataset["dataset_version"] == 1
    assert len(dataset["pairs"]) == 9
    assert {pair["group"] for pair in dataset["pairs"]} == {
        "paraphrase",
        "related_distinct",
        "unrelated",
    }
    assert all(isinstance(pair["expected_reuse"], bool) for pair in dataset["pairs"])
    assert all(pair["label_rationale"] for pair in dataset["pairs"])
