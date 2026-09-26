"""Small pyfg surface used by SQLREC's FG_NORMAL ID features."""

import os

from sqlrec_arm_compat import UnsupportedAPIError

from ._handler import FgArrowHandler


def set_env(name: str, value: str) -> None:
    if name != "USE_FARM_HASH_TO_BUCKETIZE":
        raise UnsupportedAPIError(f"pyfg.set_env({name!r}) is unsupported")
    os.environ[name] = value


def unset_env(name: str) -> None:
    if name != "USE_FARM_HASH_TO_BUCKETIZE":
        raise UnsupportedAPIError(f"pyfg.unset_env({name!r}) is unsupported")
    os.environ.pop(name, None)


class FeatureFactory:
    @staticmethod
    def create(*args, **kwargs):
        raise UnsupportedAPIError("pyfg.FeatureFactory.create is unsupported")


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    raise UnsupportedAPIError(f"pyfg.{name} is unsupported")


__all__ = ["FgArrowHandler", "FeatureFactory", "set_env", "unset_env"]
