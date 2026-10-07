"""Replay original outputs on any architecture, including older differential cases."""

from pathlib import Path

import graphlearn as gl
from graphlearn.python.data.values import Values
import pyfg
import pytest

from sqlrec_arm_compat import UnsupportedAPIError
from differential_cases import differential_cases, separator_cases
from oracle_support import assert_output
from original_runner import evaluate
from test_architecture_parity import oracle


@pytest.mark.parametrize("case", differential_cases(), ids=lambda case: case["id"])
def test_original_pyfg_output(case, oracle):
    actual = evaluate(case)
    assert Path(actual["pyfg_file"]).resolve() == Path(pyfg.__file__).resolve()
    assert_output(actual, oracle[case["id"]], case)


@pytest.mark.parametrize("case", separator_cases(), ids=lambda case: case["id"])
def test_original_rejects_unsupported_separators(case, oracle):
    assert "error" in oracle[case["id"]]
    with pytest.raises(UnsupportedAPIError, match="separator"):
        pyfg.FgArrowHandler(case["config"], 1)


def test_original_graphlearn_import_surface(oracle):
    original = oracle["graphlearn-import-surface"]["graphlearn"]
    assert original == {"version": "1.3.8", "names": ["Graph", "Nodes", "Decoder", "Values"]}
    assert [gl.Graph.__name__, gl.Nodes.__name__, gl.Decoder.__name__, Values.__name__] == original["names"]
