"""Compare the supported path against Alibaba's original x86 pyfg wheel."""

import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyfg
import pytest
import numpy as np


ORIGINAL_PYTHON = os.environ.get("SQLREC_ORIGINAL_PYTHON")
RESULT_PREFIX = "SQLREC_ORIGINAL_RESULT="
pytestmark = pytest.mark.skipif(
    not ORIGINAL_PYTHON,
    reason="Set SQLREC_ORIGINAL_PYTHON to an x86 Python with original pyfg installed",
)


def _case(name, values, arrow_type, kind, default=""):
    return {
        "config": {
            "features": [
                {
                    "feature_type": "id_feature",
                    "feature_name": name,
                    "expression": f"item:{name}",
                    "default_value": default,
                    "value_type": "string",
                    "need_prefix": False,
                    "value_dim": 0,
                    kind: 100,
                }
            ]
        },
        "data": {name: values},
        "types": {name: arrow_type},
    }


def _raw_case(name, values, arrow_type="float64", **options):
    feature = {
        "feature_type": "raw_feature",
        "feature_name": name,
        "expression": f"item:{name}",
        "value_type": "float",
        "default_value": "0.1",
    }
    feature.update(options)
    return {"config": {"features": [feature]}, "data": {name: values}, "types": {name: arrow_type}}


def _assert_direct_output(actual, expected, dense_names):
    assert actual.keys() == expected.keys()
    for name, values in actual.items():
        if name in dense_names:
            np.testing.assert_allclose(values, expected[name], rtol=1e-6, atol=1e-7)
        else:
            assert values == expected[name]


@pytest.mark.parametrize(
    "case",
    [
        _case("int_string", ["1\x1d2", "", None, "3"], "string", "num_buckets"),
        _case("int_arrow", [1, 2, None, 3], "int64", "num_buckets", "0"),
        _case("hash_string", ["abc\x1defg", None, "hij"], "string", "hash_bucket_size", "xyz"),
        _case("hash_integer", [1, 2, None, 3], "int64", "hash_bucket_size", "4"),
        _case(
            "hash_list",
            [["Action", "Adventure"], [], None, ["Comedy"]],
            "list<string>",
            "hash_bucket_size",
            "unknown",
        ),
        _raw_case("float_scalar", [0.2,0.3,None]),
        _raw_case("float_strings", ["0.2","",None,"0.3"], "string"),
        _raw_case("zscore", [0.2,0.3,None], normalizer="method=zscore,mean=0.1,standard_deviation=10"),
        _raw_case("minmax", [0.2,0.3,None], normalizer="method=minmax,min=0.1,max=0.6"),
        _raw_case("log10", [0.1,0.01,10,None], normalizer="method=log10,threshold=0.05,default=-5"),
        _raw_case("float_vector", [[0.2,0.5],[],None], "list<float32>", value_dim=2, default_value="0.1\x1d0.4"),
        _raw_case("bucket_boundary", [0.05,0.1,0.2,0.3,None], boundaries=[0.1,0.2,0.3],default_value=""),
        _raw_case("float32_scalar", [0.2, 0.3, None], "float32", value_dim=1),
        _raw_case("scalar_list", [[0.2], [], None], "list<float32>"),
        _raw_case("vector_strings", ["0.2|0.5", "", None], "string", value_dim=2, separator="|", default_value="0.1|0.4"),
        _raw_case("bucket_float32", [0.05, 0.1, 0.2, None], "float32", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_integer", [0, 1, None], "int64", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_negative", [-3, -2, -1, None], boundaries=[-2, -1], default_value=""),
        _raw_case("bucket_zero_boundary", [-1, 0, 1, None], boundaries=[0], default_value=""),
        _raw_case("bucket_strings", ["0.05", "0.1", "", None], "string", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_list", [[0.2], [], None], "list<float32>", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_nonempty_default", [0.05, None], boundaries=[0.1, 0.2], default_value="0.2"),
        _raw_case("zscore_bucket", [0.2, None], boundaries=[0.01, 0.1], normalizer="method=zscore,mean=0.1,standard_deviation=10"),
        _raw_case("zscore_empty_bucket", [0.0, None], boundaries=[-1, 0, 1], default_value="", normalizer="method=zscore,mean=10,standard_deviation=1"),
    ],
    ids=lambda case: next(iter(case["data"])),
)
def test_original_pyfg_output(case, monkeypatch):
    monkeypatch.setenv("USE_FARM_HASH_TO_BUCKETIZE", "true")
    original_env = os.environ.copy()
    original_env.pop("PYTHONPATH", None)
    runner = Path(__file__).with_name("original_runner.py")
    completed = subprocess.run(
        [ORIGINAL_PYTHON, str(runner)],
        input=json.dumps(case),
        text=True,
        capture_output=True,
        check=True,
        cwd=runner.parent.parent,
        env=original_env,
    )
    result_lines = [
        line.removeprefix(RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(result_lines) == 1, completed.stdout
    original = json.loads(result_lines[0])
    assert original["pyfg_version"] == "1.0.5"
    assert Path(original["pyfg_file"]).resolve() != Path(pyfg.__file__).resolve()

    handler = pyfg.FgArrowHandler(case["config"], 1)
    data = {
        name: pa.array(
            values,
            type=(
                pa.list_(pa.type_for_alias(case["types"][name][5:-1]))
                if case["types"][name].startswith("list<")
                else pa.type_for_alias(case["types"][name])
            ),
        )
        for name, values in case["data"].items()
    }
    output, status = handler.process_arrow(data)
    assert status.ok()
    dense_names = {feature["feature_name"] for feature in case["config"]["features"] if feature["feature_type"] == "raw_feature" and not feature.get("boundaries")}
    result = {
        name: {
            "dense_values": value.dense_values.tolist(),
            "values_dtype": str(value.dense_values.dtype),
            "values_shape": list(value.dense_values.shape),
        } if name in dense_names else {
            "values": value.np_values.tolist(),
            "lengths": value.np_lengths.tolist(),
            "values_dtype": str(value.np_values.dtype),
            "lengths_dtype": str(value.np_lengths.dtype),
        }
        for name, value in output.items()
    }
    assert result.keys() == original["result"].keys()
    for name, feature in result.items():
        if name in dense_names:
            assert feature["values_dtype"] == original["result"][name]["values_dtype"]
            assert feature["values_shape"] == original["result"][name]["values_shape"]
            np.testing.assert_allclose(feature["dense_values"], original["result"][name]["dense_values"], rtol=1e-6,atol=1e-7)
        else:
            assert feature == original["result"][name]
    direct, status = handler(case["data"])
    assert status.ok()
    _assert_direct_output(direct, original["direct"], dense_names)
    defaults, status = handler({name: [None] for name in case["data"]})
    assert status.ok()
    _assert_direct_output(defaults, original["defaults"], dense_names)


def test_original_graphlearn_import_surface():
    code = (
        "import json, graphlearn as gl; "
        "from graphlearn.python.data.values import Values; "
        "from importlib.metadata import version; "
        "print(json.dumps({'version': version('graphlearn'), "
        "'names': [gl.Graph.__name__, gl.Nodes.__name__, "
        "gl.Decoder.__name__, Values.__name__]}))"
    )
    original_env = os.environ.copy()
    original_env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [ORIGINAL_PYTHON, "-c", code],
        text=True,
        capture_output=True,
        check=True,
        cwd=Path(__file__).parent.parent,
        env=original_env,
    )
    original = json.loads(completed.stdout)
    assert original == {
        "version": "1.3.8",
        "names": ["Graph", "Nodes", "Decoder", "Values"],
    }
