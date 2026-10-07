"""Make full parity the default; smaller local runs must be explicit."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--unit-only", action="store_true",
                     help="Explicitly skip original-wheel oracle comparisons")
    parser.addoption("--require-tzrec", action="store_true",
                     help="Fail instead of skipping when TorchEasyRec integration is unavailable")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--unit-only"):
        marker = pytest.mark.skip(reason="Original parity explicitly disabled by --unit-only")
        for item in items:
            if (Path(str(item.path)).name == "test_differential.py"
                    or getattr(item, "originalname", None) == "test_original_corpus_replay"):
                item.add_marker(marker)
