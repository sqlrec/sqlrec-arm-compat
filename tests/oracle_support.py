"""Shared oracle export/replay helpers; never infer an oracle from compat output."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from float_support import assert_float_output

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = ROOT / "tests/data/x86-oracle.json"
RUNTIME = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines()
ORIGINAL_REQUIREMENTS = (ROOT / "requirements-oracle.txt").read_text(encoding="utf-8").splitlines()

def corpus_digest(cases):
    return hashlib.sha256(json.dumps({"runtime": RUNTIME, "original": ORIGINAL_REQUIREMENTS, "cases": cases},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_oracle():
    from oracle_cases import all_cases

    cases = all_cases()
    snapshot_path = os.environ.get("SQLREC_COMPAT_ORACLE")
    original = os.environ.get("SQLREC_ORIGINAL_PYTHON")
    if snapshot_path or not original:
        snapshot_path = snapshot_path or DEFAULT_ORACLE
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        assert snapshot["schema"] == 1
        assert snapshot["corpus_sha256"] == corpus_digest(cases), "Stale/mismatched oracle corpus; regenerate it"
        outputs = snapshot["outputs"]
        assert set(outputs) == {case["id"] for case in cases}
        assert all(output["pyfg_version"] == "1.0.5" for output in outputs.values() if "error" not in output)
        return outputs
    return original_outputs(original, cases)


def original_outputs(python, cases):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    runner = Path(__file__).with_name("original_runner.py")
    process = subprocess.run([python, str(runner)], input=json.dumps({"cases": cases}),
                             text=True, capture_output=True, cwd=runner.parent.parent,
                             env=environment, timeout=180)
    prefix = "SQLREC_ORIGINAL_RESULT="
    lines = [line[len(prefix):] for line in process.stdout.splitlines() if line.startswith(prefix)]
    if process.returncode or len(lines) != 1:
        raise AssertionError((process.stdout + process.stderr)[-8000:])
    records = json.loads(lines[0])
    assert [record["id"] for record in records] == [case["id"] for case in cases]
    errors = [record for record, case in zip(records, cases)
              if ("error" in record) != bool(case.get("expect_error"))]
    assert not errors, json.dumps(errors, ensure_ascii=False, indent=2)
    for record in records:
        if "error" in record:
            continue
        output = record["output"]
        assert output["pyfg_version"] == "1.0.5", "Oracle must use original pyfg, not compat"
        assert output["machine"].lower() in {"x86_64", "amd64"}
    return {record["id"]: record.get("output", {"error": record.get("error")}) for record in records}


def assert_output(actual, expected, case):
    dense = {feature["feature_name"] for feature in case["config"]["features"]
             if feature["feature_type"] == "raw_feature" and not feature.get("boundaries")}
    for section in ("result", "direct", "defaults"):
        assert actual[section].keys() == expected[section].keys(), case["id"]
        for name, values in actual[section].items():
            reference = expected[section][name]
            if name in dense:
                if section == "result":
                    assert values["values_dtype"] == reference["values_dtype"] == "float32"
                    assert values["values_shape"] == reference["values_shape"]
                    values, reference = values["dense_values"], reference["dense_values"]
                assert_float_output(values, reference, err_msg=f"{case['id']}: {section}/{name}")
            elif case.get("bucket_float_max_ulp"):
                feature = next(item for item in case["config"]["features"] if item["feature_name"] == name)
                normalized = actual["normalized"][section][name]
                native = expected["normalized"][section][name]
                if section == "result":
                    assert values.keys() == reference.keys()
                    assert {key: value for key, value in values.items() if key != "values"} == {
                        key: value for key, value in reference.items() if key != "values"}
                    assert normalized["values_shape"] == native["values_shape"]
                    assert normalized["values_dtype"] == native["values_dtype"] == "float32"
                    values, reference = values["values"], reference["values"]
                    normalized, native = normalized["dense_values"], native["dense_values"]
                assert_float_output(normalized, native, err_msg=f"{case['id']}: normalized {section}/{name}")
                assert_float_buckets(values, reference, normalized, native, feature["boundaries"],
                                     case["bucket_float_max_ulp"], f"{case['id']}: {section}/{name}")
            else:
                assert values == reference, f"{case['id']}: {section}/{name}: {values} != {reference}"


def assert_float_buckets(actual, expected, normalized, native, boundaries, max_ulp, err_msg):
    """Only tolerate normalized values and crossed boundaries within a few ULPs."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, err_msg
    normalized = np.asarray(normalized, dtype=np.float32).reshape(-1)
    native = np.asarray(native, dtype=np.float32).reshape(-1)
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    assert actual.size == expected.size == normalized.size == native.size, err_msg
    bounds = np.asarray(boundaries, dtype=np.float32)
    np.testing.assert_array_equal(actual, np.searchsorted(bounds, normalized, side="right"), err_msg=err_msg)
    # Native bucket and dense paths can round differently, so validate only
    # disagreements using native dense values and the exact crossed boundaries.
    for bucket, reference, value, baseline in zip(actual, expected, normalized, native):
        assert 0 <= bucket <= bounds.size and 0 <= reference <= bounds.size, err_msg
        if bucket == reference:
            continue
        tolerance = max_ulp * abs(float(np.spacing(baseline)))
        assert abs(float(value) - float(baseline)) <= tolerance, err_msg
        crossed = bounds[min(bucket, reference):max(bucket, reference)]
        assert np.all(np.abs(crossed.astype(np.float64) - float(baseline)) <= tolerance), err_msg
