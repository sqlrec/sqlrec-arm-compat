"""Namespace needed by tzrec.datasets.sampler."""

from sqlrec_arm_compat import UnsupportedAPIError


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    raise UnsupportedAPIError(f"graphlearn.python.{name} is unsupported")
