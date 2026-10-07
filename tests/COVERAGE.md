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
case IDs prevent stale/partial artifacts from silently passing. Rejected cases
must fail on both sides, although native RuntimeError and compat UnsupportedAPIError
are intentionally not required to have identical exception classes/messages.

The direct original binding has inconsistent NumPy scalar conversion. Such
objects are covered by local NumPy 1/2 regression tests rather than falsely
claiming equality with that native binding. Arrow null rows in nested lists remain
in the oracle, but the original direct nested-list path uses empty lists because
its binding does not accept mixed nested-list/None columns.

## Native CI gates

SQLREC's `_build-images.yml` resolves one compat commit and exports one x86 oracle.
The same commit and artifact are tested on **native amd64 and arm64**, without QEMU:

| Profile | Python | NumPy | Arrow |
| --- | --- | --- | --- |
| minimum-py311 | 3.11 | 1.24.4 | 14.0.2 |
| reference-py311 | 3.11 | 1.26.4 | 17.0.0 |
| numpy2-py311 | 3.11 | Latest supported 2.x | Latest supported `<26` |
| numpy2-py312 | 3.12 | Latest supported 2.x | Latest supported `<26` |

Tests install built pyfg/graphlearn wheels; pyfarmhash is compiled for the native
architecture. A machine assertion detects wrong-architecture execution. Dense
outputs allow at most two float32 ULPs with a relative/absolute safety bound and
signed-zero preservation. Sparse IDs and buckets are exact, with no tolerance.
Both TZRec images depend on the complete native matrix; existing image smoke tests
then exercise actual TZRec/DataParser/model/native-server integration.
The Docker test target also requires 100% measured Python statement/branch coverage
for pyfg, graphlearn and sqlrec_arm_compat. This does not certify all input values or
cover the internal native FarmHash implementation; the oracle handles that layer.

## Explicit limits

- No finite test suite exhausts every input, dependency patch version, CPU/compiler
  or operating system. The matrix is a regression gate, not a proof of universal
  numerical identity. Every newly reported production input should become a case.
- Test supported major-version endpoints and representative combinations, not the
  full Cartesian product of every NumPy/Arrow release. Linux little-endian
  amd64/arm64 is the deployment target; other architectures are not certified.
- Do not claim native ARM verification from an amd64-only local run. CI must run
  after pushing both compat changes and the parent workflow changes.
- TorchEasyRec integration is optional in the standalone compat environment and
  must run in the actual TZRec images; no substitute Torch environment is installed
  solely to remove a skip.
- Weighted/user/sequence/combo features, sampling/training through graphlearn,
  multiple FG executor threads, and `bucketize_only` are outside this subset.
- Allocation-limit batches and concurrent process-wide environment mutations are
  not certified; SQLREC uses one FG executor and sets hash mode before processing.
