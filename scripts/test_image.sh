#!/usr/bin/env bash
set -euo pipefail

root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
image="${1:?Usage: test_image.sh ARM_TZREC_IMAGE REPORT_DIR}"
if [[ ! -f "$root/tests/data/x86-oracle.json" ]]; then
  echo "Missing tests/data/x86-oracle.json; export it in the original x86 environment first." >&2
  exit 1
fi
mkdir -p "${2:?Specify the report directory}"
reports="$(realpath "$2")"

# Test the installed wheels in a disposable container from the deployment image.
docker run --rm --entrypoint sh --workdir /compat \
  --mount "type=bind,source=$root,target=/compat,readonly" \
  --mount "type=bind,source=$reports,target=/reports" \
  --env USE_FARM_HASH_TO_BUCKETIZE=true \
  --env SQLREC_EXPECTED_MACHINE=aarch64 \
  --env SQLREC_TEST_REPORT_DIR=/reports \
  "$image" -eu -c '
    python -m pip install --no-cache-dir "pytest>=8,<10" "coverage>=7,<8"
    python scripts/test.py --require-tzrec
  '
