"""Compare the supported path against Alibaba's original x86 pyfg wheel."""

import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyfg
import pytest


ORIGINAL_PYTHON = os.environ.get("SQLREC_ORIGINAL_PYTHON")
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
    original = json.loads(completed.stdout)
    assert original["pyfg_version"] == "1.0.5"
    assert Path(original["pyfg_file"]).resolve() != Path(pyfg.__file__).resolve()

    handler = pyfg.FgArrowHandler(case["config"], 1)
    data = {
        name: pa.array(
            values,
            type=(
                pa.list_(pa.string())
                if case["types"][name] == "list<string>"
                else pa.type_for_alias(case["types"][name])
            ),
        )
        for name, values in case["data"].items()
    }
    output, status = handler.process_arrow(data)
    assert status.ok()
    result = {
        name: {
            "values": value.np_values.tolist(),
            "lengths": value.np_lengths.tolist(),
            "values_dtype": str(value.np_values.dtype),
            "lengths_dtype": str(value.np_lengths.dtype),
        }
        for name, value in output.items()
    }
    assert result == original["result"]
    defaults, status = handler({name: [None] for name in case["data"]})
    assert status.ok()
    assert defaults == original["defaults"]


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
