"""Check full-suite execution without creating environments or invoking Docker."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compat_test_runner", ROOT / "scripts/test.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_suite_rejects_unexpected_skips_and_can_require_integration(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text('''<testsuites><testsuite>
        <testcase classname="tests.test_architecture_parity" name="parity">
            <skipped message="oracle unavailable" /></testcase>
        <testcase classname="tests.test_tzrec_integration" name="integration">
            <skipped message="tzrec unavailable" /></testcase>
        </testsuite></testsuites>''', encoding="utf-8")
    assert len(runner.unexpected_skips(report)) == 1
    assert len(runner.unexpected_skips(report, require_tzrec=True)) == 2


def test_suite_allows_only_explicit_unit_mode_skips(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text('''<testsuites><testsuite>
        <testcase classname="tests.test_differential" name="parity">
            <skipped message="Original parity explicitly disabled by --unit-only" /></testcase>
        </testsuite></testsuites>''', encoding="utf-8")
    assert len(runner.unexpected_skips(report)) == 1
    assert not runner.unexpected_skips(report, unit_only=True)


def test_runtime_pins_match_package_metadata():
    import tomllib
    pins = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines()
    metadata = tomllib.loads((ROOT / "packages/pyfg/pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert metadata["dependencies"] == pins
    assert metadata["requires-python"] == ">=3.11,<3.12"


def test_runner_rejects_wrong_python_and_dependency_versions(monkeypatch):
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(runner, "sys", SimpleNamespace(version_info=(3, 12)))
    with pytest.raises(SystemExit) as error:
        runner.check_environment(parser)
    assert error.value.code == 2
    monkeypatch.setattr(runner, "sys", SimpleNamespace(version_info=(3, 11), version="3.11.0"))
    pins = {"numpy": "1.26.4", "pyarrow": "17.0.0", "pyfarmhash": "0.4.0"}
    monkeypatch.setattr(runner, "version", lambda name: pins[name])
    assert runner.check_environment(parser)["numpy"] == "1.26.4"
    for name in pins:
        expected = pins[name]
        pins[name] = "99.0.0"
        with pytest.raises(SystemExit) as error:
            runner.check_environment(parser)
        assert error.value.code == 2
        pins[name] = expected


@pytest.mark.parametrize("pytest_status,coverage_status", [(0, 0), (1, 0), (0, 2)])
def test_runner_executes_all_tests_and_enforces_coverage(monkeypatch, tmp_path, pytest_status, coverage_status):
    oracle = tmp_path / "oracle.json"
    oracle.write_text("{}", encoding="utf-8")
    reports = tmp_path / "reports"
    monkeypatch.setattr(runner.sys, "argv", ["test.py", "--oracle", str(oracle), "--require-tzrec",
                                           "--reports", str(reports)])
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k only_one_test")
    monkeypatch.setenv("SQLREC_ORIGINAL_PYTHON", "old-original")
    monkeypatch.setattr(runner, "check_environment", lambda parser: {})
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "pytest" in command:
            (reports / "pytest.xml").write_text('<testsuites><testsuite /></testsuites>', encoding="utf-8")
            return SimpleNamespace(returncode=pytest_status)
        return SimpleNamespace(returncode=coverage_status if "report" in command else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main() == (pytest_status or coverage_status)
    command, options = calls[0]
    assert command[0] == runner.sys.executable
    assert "tests" in command and "--require-tzrec" in command
    assert "PYTEST_ADDOPTS" not in options["env"]
    assert "SQLREC_ORIGINAL_PYTHON" not in options["env"]
    assert options["env"]["SQLREC_COMPAT_ORACLE"] == str(oracle)
    assert options["env"]["COVERAGE_FILE"] == str(reports / ".coverage")
    if not pytest_status:
        assert calls[-1][0][-1] == "--fail-under=100"
    else:
        assert len(calls) == 1


def test_runner_requires_an_oracle_in_full_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("SQLREC_COMPAT_ORACLE", raising=False)
    monkeypatch.delenv("SQLREC_ORIGINAL_PYTHON", raising=False)
    monkeypatch.setattr(runner.sys, "argv", ["test.py", "--reports", str(tmp_path)])
    monkeypatch.setattr(runner, "DEFAULT_ORACLE", tmp_path / "missing.json")
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_runner_uses_committed_oracle_by_default(monkeypatch, tmp_path):
    oracle = tmp_path / "oracle.json"
    oracle.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "DEFAULT_ORACLE", oracle)
    monkeypatch.delenv("SQLREC_COMPAT_ORACLE", raising=False)
    monkeypatch.delenv("SQLREC_ORIGINAL_PYTHON", raising=False)
    monkeypatch.setattr(runner.sys, "argv", ["test.py", "--reports", str(tmp_path)])
    monkeypatch.setattr(runner, "check_environment", lambda parser: {})
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "pytest" in command:
            (tmp_path / "pytest.xml").write_text('<testsuites><testsuite /></testsuites>', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main() == 0
    assert calls[0][1]["env"]["SQLREC_COMPAT_ORACLE"] == str(oracle)


def test_runner_unit_mode_clears_reference_and_rejects_missing_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.sys, "argv", ["test.py", "--unit-only", "--reports", str(tmp_path)])
    monkeypatch.setenv("SQLREC_COMPAT_ORACLE", "old-oracle")
    monkeypatch.setenv("SQLREC_ORIGINAL_PYTHON", "old-original")
    monkeypatch.setattr(runner, "check_environment", lambda parser: {})
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main() == 1
    assert "--unit-only" in calls[0][0]
    assert "SQLREC_COMPAT_ORACLE" not in calls[0][1]["env"]
    assert "SQLREC_ORIGINAL_PYTHON" not in calls[0][1]["env"]
