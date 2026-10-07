# Original x86 reference

`x86-oracle.json` contains outputs from the original Linux amd64 Python 3.11
pyfg/graphlearn wheels, not from this compatibility implementation. The complete
test suite reads it by default, locally and inside the ARM TZRec deployment image.

After changing oracle cases, `requirements-runtime.txt` or
`requirements-oracle.txt`, regenerate and commit the reference file:

```sh
python3.11 -m venv .venv/original
.venv/original/bin/python -m pip install -r requirements-oracle.txt
.venv/original/bin/python tests/export_oracle.py
```

Run these commands on Linux amd64. The exporter checks the original pyfg version
and architecture. Its digest covers the complete corpus and both requirements
files, so stale or partial references fail instead of silently skipping tests.
