"""Same input corpus on every architecture; sparse/bucket outputs must be exact."""

import json
import os
import platform
from pathlib import Path

import numpy as np
import pyfg
import pytest

from sqlrec_arm_compat import UnsupportedAPIError
from oracle_cases import supported_cases, rejected_arrow_cases
from oracle_support import assert_output, corpus_digest, original_outputs
from original_runner import evaluate


CASES = supported_cases() + rejected_arrow_cases()


def test_native_runner_architecture():
    expected = os.environ.get("SQLREC_EXPECTED_MACHINE")
    if expected:
        assert platform.machine() == expected
    assert np.dtype(np.int64).itemsize == 8
    assert np.dtype(np.int32).itemsize == 4
    assert np.dtype(np.float32).itemsize == 4


@pytest.fixture(scope="module")
def oracle():
    snapshot_path = os.environ.get("SQLREC_COMPAT_ORACLE")
    if snapshot_path:
        # A missing/stale artifact is a failure, never a skipped parity gate.
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        assert snapshot["schema"] == 1
        assert snapshot["corpus_sha256"] == corpus_digest(CASES), "Stale/mismatched oracle corpus"
        outputs = snapshot["outputs"]
        assert set(outputs) == {case["id"] for case in CASES}
        assert all(output["pyfg_version"] == "1.0.5" for output in outputs.values() if "error" not in output)
        return outputs
    original = os.environ.get("SQLREC_ORIGINAL_PYTHON")
    if original:
        return original_outputs(original, CASES)
    pytest.skip("Set SQLREC_COMPAT_ORACLE to an exported x86 oracle, or SQLREC_ORIGINAL_PYTHON")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_original_corpus_replay(case, oracle):
    if case.get("expect_error"):
        assert "error" in oracle[case["id"]]
        with pytest.raises(UnsupportedAPIError, match=case.get("error_match", "Arrow type")):
            evaluate(case)
        return
    actual = evaluate(case)
    assert Path(actual["pyfg_file"]).resolve() == Path(pyfg.__file__).resolve()
    assert_output(actual, oracle[case["id"]], case)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_corpus_shape_dtype_and_sparse_invariants(case):
    if case.get("expect_error"):
        with pytest.raises(UnsupportedAPIError, match=case.get("error_match", "Arrow type")):
            evaluate(case)
        return
    output = evaluate(case)
    rows = len(next(iter(case["data"].values())))
    for feature in case["config"]["features"]:
        values = output["result"][feature["feature_name"]]
        if "dense_values" in values:
            assert values["values_shape"] == [rows, feature.get("value_dim", 1)]
            assert values["values_dtype"] == "float32"
            assert np.isfinite(values["dense_values"]).all()
        else:
            assert values["values_dtype"] == "int64"
            assert values["lengths_dtype"] == "int32"
            assert len(values["lengths"]) == rows
            assert sum(values["lengths"]) == len(values["values"])
            assert all(length >= 0 for length in values["lengths"])
            bound = len(feature["boundaries"]) + 1 if feature.get("boundaries") else feature.get("num_buckets", feature.get("hash_bucket_size"))
            assert all(0 <= value < bound for value in values["values"])
