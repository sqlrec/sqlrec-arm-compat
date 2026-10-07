"""Export original x86 outputs for replay on native ARM/amd64 runners."""

import argparse
import json
import os
import sys
from pathlib import Path

from oracle_cases import all_cases
from oracle_support import DEFAULT_ORACLE, corpus_digest, original_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_ORACLE,
                        help="Reference file to update (default: tests/data/x86-oracle.json)")
    parser.add_argument("--original-python", default=os.environ.get("SQLREC_ORIGINAL_PYTHON", sys.executable))
    arguments = parser.parse_args()
    cases = all_cases()
    outputs = original_outputs(arguments.original_python, cases)
    # Import paths are useful for live diagnostics, but are machine-specific
    # and do not belong in the committed reference data.
    for output in outputs.values():
        output.pop("pyfg_file", None)
        if "normalized" in output:
            output["normalized"].pop("pyfg_file", None)
    snapshot = {"schema": 1, "corpus_sha256": corpus_digest(cases), "outputs": outputs}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(snapshot, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(cases)} original x86 cases to {arguments.output}")


if __name__ == "__main__":
    main()
