"""Export original x86 outputs for replay on native ARM/amd64 runners."""

import argparse
import json
import os
from pathlib import Path

from oracle_cases import supported_cases, rejected_arrow_cases
from oracle_support import corpus_digest, original_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--original-python", default=os.environ.get("SQLREC_ORIGINAL_PYTHON"))
    arguments = parser.parse_args()
    if not arguments.original_python:
        parser.error("Set SQLREC_ORIGINAL_PYTHON or --original-python")
    cases = supported_cases() + rejected_arrow_cases()
    outputs = original_outputs(arguments.original_python, cases)
    snapshot = {"schema": 1, "corpus_sha256": corpus_digest(cases), "outputs": outputs}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(snapshot, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Exported {len(cases)} original x86 cases to {arguments.output}")


if __name__ == "__main__":
    main()
