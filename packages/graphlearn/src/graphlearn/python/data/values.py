from sqlrec_arm_compat import UnsupportedAPIError


class Values:
    """Class placeholder that permits tzrec's import-time property patch."""

    def __init__(self, *args, **kwargs):
        raise UnsupportedAPIError("graphlearn.python.data.values.Values is unsupported")


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    raise UnsupportedAPIError(f"graphlearn.python.data.values.{name} is unsupported")
