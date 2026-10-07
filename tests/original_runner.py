"""Run in a separate x86 Python environment containing the original pyfg."""

import json
import os
import platform
import sys
from importlib.metadata import version

import pyarrow as pa
import numpy as np
import pyfg


RESULT_PREFIX = "SQLREC_ORIGINAL_RESULT="


def arrow_type(name):
    if name.startswith("large_list<"):
        return pa.large_list(pa.type_for_alias(name[11:-1]))
    if name.startswith("list<"):
        return pa.list_(pa.type_for_alias(name[5:-1]))
    return pa.type_for_alias(name)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def evaluate(request):
    os.environ["USE_FARM_HASH_TO_BUCKETIZE"] = "true"
    pyfg.set_env("USE_FARM_HASH_TO_BUCKETIZE", "true")
    handler = pyfg.FgArrowHandler(request["config"], 1)
    data = {
        name: pa.array(
            [np.float16(value) if value is not None else None for value in values]
            if request["types"][name] == "float16" else values,
            type=arrow_type(request["types"][name]),
        )
        for name, values in request["data"].items()
    }
    output, status = handler.process_arrow(data)
    if not status.ok():
        raise RuntimeError(status.message())
    dense_names = {feature["feature_name"] for feature in request["config"]["features"] if feature["feature_type"] == "raw_feature" and not feature.get("boundaries")}
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
    direct, status = handler(request.get("direct_data", request["data"]))
    if not status.ok():
        raise RuntimeError(status.message())
    defaults, status = handler({name: [None] for name in request["data"]})
    if not status.ok():
        raise RuntimeError(status.message())
    return {
        "result": result,
        "direct": _jsonable(direct),
        "defaults": _jsonable(defaults),
        "pyfg_file": pyfg.__file__,
        "pyfg_version": version("pyfg"),
        "machine": platform.machine(),
    }


def main():
    request = json.load(sys.stdin)
    if "cases" in request:
        result = []
        for case in request["cases"]:
            try:
                result.append({"id": case["id"], "output": evaluate(case)})
            except Exception as error:
                result.append({"id": case["id"], "error": f"{type(error).__name__}: {error}"})
    else:
        result = evaluate(request)
    print(RESULT_PREFIX + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
