"""The cross-architecture gate must not silently skip or tolerate bad artifacts."""

import copy
import json

import pytest

from oracle_cases import case, raw_feature, id_feature
from oracle_support import assert_output, corpus_digest
from original_runner import evaluate
from test_architecture_parity import CASES, oracle


def test_missing_requested_oracle_is_a_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SQLREC_COMPAT_ORACLE", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        oracle.__wrapped__()


@pytest.mark.parametrize("snapshot", [
    {"schema": 0, "corpus_sha256": corpus_digest(CASES), "outputs": {}},
    {"schema": 1, "corpus_sha256": "stale", "outputs": {}},
    {"schema": 1, "corpus_sha256": corpus_digest(CASES), "outputs": {}},
])
def test_stale_or_partial_oracle_is_a_failure(snapshot, monkeypatch, tmp_path):
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setenv("SQLREC_COMPAT_ORACLE", str(path))
    with pytest.raises(AssertionError):
        oracle.__wrapped__()


def test_sparse_comparison_has_no_numerical_tolerance():
    request = case("check-sparse", id_feature("num_buckets", 2**63 - 1), [2**60], "int64")
    output = evaluate(request)
    reference = copy.deepcopy(output)
    reference["result"]["feature"]["values"][0] += 1
    with pytest.raises(AssertionError):
        assert_output(output, reference, request)


@pytest.mark.parametrize("replacement", [0.0, -1e-9])
def test_dense_comparison_catches_small_absolute_errors_and_signed_zero(replacement):
    request = case("check-dense", raw_feature(), [-0.0], "float32")
    output = evaluate(request)
    reference = copy.deepcopy(output)
    reference["result"]["feature"]["dense_values"][0][0] = replacement
    with pytest.raises(AssertionError):
        assert_output(output, reference, request)
