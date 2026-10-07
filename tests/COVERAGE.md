# Compatibility coverage and limits

The contract is the SQLREC subset documented in the project README, not all of
Alibaba pyfg or graphlearn. Adding a supported feature requires adding an original
oracle case and a fail-fast test for neighboring unsupported configurations.

## Scenario matrix

| Area | Covered scenarios |
| --- | --- |
| ID bucketization | `num_buckets`/FarmHash; counts 1, 100, above int32, int64 maximum; signed limits, negative/out-of-range IDs, padded decimal strings, malformed IDs |
| FarmHash | ASCII, Chinese, emoji, composed/decomposed Unicode, embedded NUL; lengths around 4/8/16/32/64/128 bytes; long strings and fixed-seed random UTF-8 tokens |
| ID defaults | Missing/null/empty string/list, empty/nonempty defaults, default containing separator as one hash token, rejection of multivalue integer defaults |
| Raw inputs | float32/float64/int32/int64/string scalars and ordinary lists of each; scalar/vector output; dimensions 1/2/8 and local maximum-dimension regression |
| Raw defaults | Dense and bucketized outputs, empty/nonempty scalar/vector defaults, defaults bypass normalization; numeric null zero versus string/list missing and all-null direct columns |
| Normalizers | None/zscore/minmax/log10; explicit and omitted log10 parameters; scalar/vector, dense/sparse, numeric versus string precision, values outside minmax range |
| Floating point | Boundary equality and both adjacent float32 values; subnormal/minimum-normal/maximum-finite values, signed zero; log10 threshold and adjacent transcendental-result buckets |
| Batch/output | Empty and all-null batches, multiple features sharing one input, aliases, sparse lengths/flattening, int64 values/int32 lengths/float32 dense values, dense shape and direct scalar/vector distinction |
| Lifecycle/API | Repeated calls, reset, recovery after failure, no mutation of inputs/config/previous outputs; environment set/unset, unsupported methods, weighted output and graphlearn placeholders |
| Fail-fast validation | Missing/duplicate names, missing/mismatched inputs, config types, thread count, bucket bounds/modes, expressions, unsupported Arrow types, malformed/nonfinite/complex/bool numbers, dimensions/defaults, normalizer scales/parameters, rounded duplicate boundaries, normalization overflow |

`oracle_cases.py` uses architecture-independent JSON scalars and a fixed random
seed. `export_oracle.py` runs the original **x86 pyfg 1.0.5** in an isolated process;
compat output is never used to generate expected values. Corpus hashes and exact
case IDs prevent stale/partial reference files from silently passing. Rejected cases
must fail on both sides, although native RuntimeError and compat UnsupportedAPIError
are intentionally not required to have identical exception classes/messages.

The direct original binding has inconsistent NumPy scalar conversion. Such
objects are covered by local regression tests in the pinned runtime rather than falsely
claiming equality with that native binding. Arrow null rows in nested lists remain
in the oracle, but the original direct nested-list path uses empty lists because
its binding does not accept mixed nested-list/None columns.

## Deployment CI gate

SQLREC resolves one compat revision, including the committed original x86
reference file `tests/data/x86-oracle.json`. Ordinary CI runs do not regenerate
or transfer this file. When changing the corpus, runtime pins or original-wheel
requirements, export it again in an original Linux amd64 Python 3.11 environment
with `python tests/export_oracle.py` and commit it with the change.
The existing TZRec image jobs build and run integration checks on amd64 and arm64.
Only the ARM image installs the replacement library, so its final image runs the
complete compat suite once, with mandatory TZRec integration, before publication.
The ARM runner is native; no emulation is used for the deployment gate.

`requirements-runtime.txt` pins NumPy 1.26.4, PyArrow 17.0.0 and pyfarmhash 0.4.0.
Python 3.11 is fixed in the Dockerfiles and package metadata. Local tests install
`requirements-test.txt` and the current checkout's editable packages, then call
`scripts/test.py` in that environment. The same entry point runs against installed
wheels inside the final ARM image through `scripts/test_image.sh`. Test tools live
only in a disposable container. The Dockerfiles pin deployment versions directly;
keep them aligned with the test requirements and package metadata. Image tests
install only pytest/coverage, then verify the existing NumPy/PyArrow/pyfarmhash
versions before running the suite.

The entry point executes all tests, rejects unexpected skips, prints actual
versions, and produces JUnit and coverage reports. It requires 100% measured
Python statement/branch coverage for pyfg, graphlearn and sqlrec_arm_compat.
This does not cover internal native FarmHash code; the original oracle checks
its output. Missing/stale/incomplete oracles fail. `--unit-only` explicitly
selects a smaller suite. Oracle digests include the corpus, runtime pins and
original-wheel requirements. Local tests and plain pytest use the committed
reference by default; explicit oracle/original-interpreter overrides are retained.

Dense outputs use `rtol=1e-5, atol=1e-7` with signed-zero preservation. Sparse
IDs, row lengths, shapes and dtypes are exact. The three adjacent-boundary log10
cases allow a bucket difference only if normalized values and every crossed
boundary are within four float32 ULPs of the original normalized value. Their
native dense normalization is exported with the buckets; wider errors fail.
The business test cases and these tolerances are unchanged by CI simplification.

## Explicit limits

- No finite test suite exhausts every input, dependency patch version, CPU/compiler
  or operating system. The deployment test is a regression gate, not a proof of universal
  numerical identity. Every newly reported production input should become a case.
- Only the pinned TZRec deployment runtime is certified. Linux little-endian
  amd64/arm64 is the deployment target; other Python/NumPy versions and
  architectures are outside the test contract.
- Do not claim native ARM verification from an amd64-only local run. CI must run
  after pushing both compat changes and the parent workflow changes.
- TorchEasyRec integration is optional in the standalone compat environment and
  can be made mandatory with `scripts/test.py --require-tzrec`. It must run
  in the actual TZRec images; no substitute Torch environment is installed
  solely to remove a skip.
- Weighted/user/sequence/combo features, sampling/training through graphlearn,
  multiple FG executor threads, and `bucketize_only` are outside this subset.
- Allocation-limit batches and concurrent process-wide environment mutations are
  not certified; SQLREC uses one FG executor and sets hash mode before processing.
