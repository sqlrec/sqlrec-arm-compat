"""Run in a separate x86 Python environment containing the original pyfg."""

import json
import os
import sys
from importlib.metadata import version

import pyarrow as pa
import pyfg


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main():
    request = json.load(sys.stdin)
    os.environ["USE_FARM_HASH_TO_BUCKETIZE"] = "true"
    pyfg.set_env("USE_FARM_HASH_TO_BUCKETIZE", "true")
    handler = pyfg.FgArrowHandler(request["config"], 1)
    data = {
        name: pa.array(values, type=pa.type_for_alias(request["types"][name]))
        for name, values in request["data"].items()
    }
    output, status = handler.process_arrow(data)
    if not status.ok():
        raise RuntimeError(status.message())
    result = {
        name: {
            "values": value.np_values.tolist(),
            "lengths": value.np_lengths.tolist(),
            "values_dtype": str(value.np_values.dtype),
            "lengths_dtype": str(value.np_lengths.dtype),
        }
        for name, value in output.items()
    }
    defaults, status = handler({name: [None] for name in request["data"]})
    if not status.ok():
        raise RuntimeError(status.message())
    print(
        json.dumps(
            {
                "result": result,
                "defaults": _jsonable(defaults),
                "pyfg_file": pyfg.__file__,
                "pyfg_version": version("pyfg"),
            }
        )
    )


if __name__ == "__main__":
    main()
