"""Shared oracle export/replay helpers; never infer an oracle from compat output."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def corpus_digest(cases):
    return hashlib.sha256(json.dumps(cases, sort_keys=True, allow_nan=False).encode()).hexdigest()


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
                assert np.asarray(values).shape == np.asarray(reference).shape
                np.testing.assert_allclose(values, reference, rtol=1e-6, atol=1e-7,
                                           err_msg=f"{case['id']}: {section}/{name}")
                if np.asarray(values).size:
                    np.testing.assert_array_max_ulp(np.asarray(values, dtype=np.float32),
                                                    np.asarray(reference, dtype=np.float32), maxulp=2)
                # Preserve zero's sign; allclose alone would miss +0 versus -0.
                values, reference = np.asarray(values), np.asarray(reference)
                zeros = (values == 0) & (reference == 0)
                np.testing.assert_array_equal(np.signbit(values[zeros]), np.signbit(reference[zeros]))
            else:
                assert values == reference, f"{case['id']}: {section}/{name}: {values} != {reference}"
