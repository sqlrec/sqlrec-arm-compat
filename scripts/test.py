"""Run all tests and coverage using the current Python environment."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = ROOT / "tests/data/x86-oracle.json"


def check_environment(parser):
    if sys.version_info[:2] != (3, 11):
        parser.error("TZRec tests require Python 3.11")
    versions = {"python": sys.version.split()[0], "machine": platform.machine()}
    for requirement in (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines():
        name, expected = requirement.split("==")
        actual = version(name)
        if actual != expected:
            parser.error(f"Expected {requirement}, got {actual}; install requirements-test.txt first")
        versions[name] = actual
    return versions


def unexpected_skips(report, *, unit_only=False, require_tzrec=False):
    skipped = []
    for case in ET.parse(report).iter("testcase"):
        reason = case.find("skipped")
        if reason is None:
            continue
        name = f"{case.get('classname')}.{case.get('name')}"
        message = reason.get("message", "")
        if unit_only and "Original parity explicitly disabled" in message:
            continue
        if not require_tzrec and case.get("classname", "").endswith("test_tzrec_integration"):
            print(f"Optional integration skipped: {name}: {message}", flush=True)
            continue
        skipped.append(f"{name}: {message}")
    return skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--oracle", type=Path, help="Original x86 reference output")
    reference.add_argument("--original-python", help="Separate Linux x86 environment with original wheels")
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--require-tzrec", action="store_true")
    parser.add_argument("--reports", type=Path,
                        default=Path(os.environ.get("SQLREC_TEST_REPORT_DIR", ROOT / ".venv/test-reports")))
    args = parser.parse_args()
    environment = os.environ.copy()
    if args.oracle:
        environment["SQLREC_COMPAT_ORACLE"] = str(args.oracle.resolve())
        environment.pop("SQLREC_ORIGINAL_PYTHON", None)
    elif args.original_python:
        environment["SQLREC_ORIGINAL_PYTHON"] = args.original_python
        environment.pop("SQLREC_COMPAT_ORACLE", None)
    if args.unit_only:
        environment.pop("SQLREC_COMPAT_ORACLE", None)
        environment.pop("SQLREC_ORIGINAL_PYTHON", None)
    elif not (environment.get("SQLREC_COMPAT_ORACLE") or environment.get("SQLREC_ORIGINAL_PYTHON")):
        environment["SQLREC_COMPAT_ORACLE"] = str(DEFAULT_ORACLE)
    if not args.unit_only and environment.get("SQLREC_COMPAT_ORACLE"):
        oracle = Path(environment["SQLREC_COMPAT_ORACLE"])
        if not oracle.is_file():
            parser.error(f"Oracle does not exist: {oracle}; regenerate it with tests/export_oracle.py in the original x86 environment")
    print(json.dumps(check_environment(parser)), flush=True)
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / "pytest.xml"
    report.unlink(missing_ok=True)
    (reports / "coverage.xml").unlink(missing_ok=True)
    # A full run must not inherit pytest filters or write into a read-only checkout.
    environment.pop("PYTEST_ADDOPTS", None)
    environment["COVERAGE_FILE"] = str(reports / ".coverage")
    command = [sys.executable, "-m", "coverage", "run", "--branch",
               "--source=pyfg,graphlearn,sqlrec_arm_compat", "-m", "pytest", "tests",
               "-q", "-ra", f"--junitxml={report}"]
    if args.unit_only:
        command.append("--unit-only")
    if args.require_tzrec:
        command.append("--require-tzrec")
    status = subprocess.run(command, cwd=ROOT, env=environment).returncode
    if not report.exists():
        print("No pytest report was produced; the complete suite did not run.", flush=True)
        return status or 1
    skipped = unexpected_skips(report, unit_only=args.unit_only, require_tzrec=args.require_tzrec)
    if skipped:
        print("Unexpected skipped tests:\n" + "\n".join(skipped), flush=True)
        status = status or 1
    if status:
        return status
    status = subprocess.run([sys.executable, "-m", "coverage", "xml", "-o", str(reports / "coverage.xml")],
                            cwd=ROOT, env=environment).returncode
    if status:
        return status
    return subprocess.run([sys.executable, "-m", "coverage", "report", "--fail-under=100"],
                          cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    sys.exit(main())
