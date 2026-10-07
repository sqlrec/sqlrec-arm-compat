"""The cross-architecture gate must not silently skip or tolerate bad artifacts."""

import copy
import json

import numpy as np
import pytest

from oracle_cases import all_cases, case, raw_feature, id_feature
from oracle_support import assert_float_buckets, assert_output, corpus_digest
from float_support import assert_float_output
from original_runner import evaluate
from test_architecture_parity import oracle


def test_missing_requested_oracle_is_a_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SQLREC_COMPAT_ORACLE", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        oracle.__wrapped__()


@pytest.mark.parametrize("snapshot", [
    {"schema": 0, "corpus_sha256": corpus_digest(all_cases()), "outputs": {}},
    {"schema": 1, "corpus_sha256": "stale", "outputs": {}},
    {"schema": 1, "corpus_sha256": corpus_digest(all_cases()), "outputs": {}},
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


def test_ordinary_raw_buckets_remain_exact():
    request = case("check-ordinary-buckets", raw_feature(boundaries=[0, 1]), [0.5], "float32")
    output = evaluate(request)
    reference = copy.deepcopy(output)
    reference["result"]["feature"]["values"][0] += 1
    with pytest.raises(AssertionError):
        assert_output(output, reference, request)


def test_log_bucket_tolerance_preserves_sparse_metadata():
    request = case("check-log-metadata", raw_feature(normalizer="method=log10", boundaries=[0, 1]),
                   [3.0], "float32")
    request["bucket_float_max_ulp"] = 4
    output = evaluate(request)
    reference = copy.deepcopy(output)
    reference["result"]["feature"]["lengths"][0] = 0
    with pytest.raises(AssertionError):
        assert_output(output, reference, request)


@pytest.mark.parametrize("replacement", [0.0, -1e-4])
def test_dense_comparison_catches_material_errors_and_signed_zero(replacement):
    request = case("check-dense", raw_feature(), [-0.0], "float32")
    output = evaluate(request)
    reference = copy.deepcopy(output)
    reference["result"]["feature"]["dense_values"][0][0] = replacement
    with pytest.raises(AssertionError):
        assert_output(output, reference, request)


def test_dense_comparison_allows_small_float32_rounding_errors():
    request = case("check-dense-tolerance", raw_feature(), [1.0, -0.0], "float32")
    output = evaluate(request)
    reference = copy.deepcopy(output)
    reference["result"]["feature"]["dense_values"] = [[1.000005], [-1e-9]]
    assert_output(output, reference, request)


@pytest.mark.parametrize("replacement", [1.0001, float("nan"), float("inf")])
def test_float_comparison_rejects_large_errors_and_nonfinite_values(replacement):
    with pytest.raises(AssertionError):
        assert_float_output([1.0], [replacement])


def test_log_bucket_tolerance_accepts_only_nearby_crossed_boundaries():
    baseline = np.float32(0.47712125471966244)
    previous = np.nextafter(baseline, np.float32(-np.inf))
    following = np.nextafter(baseline, np.float32(np.inf))
    bounds = [previous, baseline, following]
    assert_float_buckets([3], [2], [following], [baseline], bounds, 4, "near boundary")
    with pytest.raises(AssertionError):
        assert_float_buckets([4], [0], [following], [baseline], [-10, *bounds, 10], 4, "far boundary")
    with pytest.raises(AssertionError):
        assert_float_buckets([2], [2], [following], [baseline], bounds, 4, "incorrect actual bucket")
    with pytest.raises(AssertionError):
        value = np.float32(baseline + 8 * np.spacing(baseline))
        assert_float_buckets([3], [2], [value], [baseline], bounds, 4, "large normalization error")


def test_missing_oracle_is_an_error_in_full_mode(monkeypatch, tmp_path):
    import oracle_support
    monkeypatch.delenv("SQLREC_COMPAT_ORACLE", raising=False)
    monkeypatch.delenv("SQLREC_ORIGINAL_PYTHON", raising=False)
    monkeypatch.setattr(oracle_support, "DEFAULT_ORACLE", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        oracle.__wrapped__()


def test_export_corpus_includes_all_differential_cases():
    from differential_cases import differential_cases, separator_cases
    ids = [item["id"] for item in all_cases()]
    assert len(ids) == len(set(ids))
    assert {item["id"] for item in differential_cases() + separator_cases()} <= set(ids)
    assert "graphlearn-import-surface" in ids


def test_oracle_digest_changes_with_tzrec_runtime(monkeypatch):
    import oracle_support
    previous = corpus_digest(all_cases())
    monkeypatch.setattr(oracle_support, "RUNTIME", ["numpy==2.4.6", *oracle_support.RUNTIME[1:]])
    assert corpus_digest(all_cases()) != previous


def test_oracle_digest_changes_with_original_wheels(monkeypatch):
    import oracle_support
    previous = corpus_digest(all_cases())
    monkeypatch.setattr(oracle_support, "ORIGINAL_REQUIREMENTS", ["pyfg @ new-original-wheel"])
    assert corpus_digest(all_cases()) != previous


def test_default_oracle_and_explicit_original_environment(monkeypatch, tmp_path):
    import oracle_cases
    import oracle_support
    requests = [{"id": "example"}]
    outputs = {"example": {"pyfg_version": "1.0.5"}}
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({"schema": 1, "corpus_sha256": corpus_digest(requests),
                                "outputs": outputs}), encoding="utf-8")
    monkeypatch.setattr(oracle_cases, "all_cases", lambda: requests)
    monkeypatch.setattr(oracle_support, "DEFAULT_ORACLE", path)
    monkeypatch.delenv("SQLREC_COMPAT_ORACLE", raising=False)
    monkeypatch.delenv("SQLREC_ORIGINAL_PYTHON", raising=False)
    assert oracle_support.load_oracle() == outputs
    calls = []

    def original_outputs(python, cases):
        calls.append((python, cases))
        return {"live-original": {}}

    monkeypatch.setattr(oracle_support, "original_outputs", original_outputs)
    monkeypatch.setenv("SQLREC_ORIGINAL_PYTHON", "original-python")
    assert oracle_support.load_oracle() == {"live-original": {}}
    assert calls == [("original-python", requests)]
    monkeypatch.setenv("SQLREC_COMPAT_ORACLE", str(path))
    assert oracle_support.load_oracle() == outputs
