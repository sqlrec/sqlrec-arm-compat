"""Import-only graphlearn surface for SQLREC configurations without sampling."""

from sqlrec_arm_compat import UnsupportedAPIError


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise UnsupportedAPIError(
            f"graphlearn.{type(self).__name__} is unsupported: SQLREC does not configure a sampler"
        )


class Graph(_Unavailable):
    pass


class Nodes(_Unavailable):
    pass


class Decoder(_Unavailable):
    pass


def set_field_delimiter(*args, **kwargs):
    raise UnsupportedAPIError("graphlearn.set_field_delimiter is unsupported")


def set_use_string_hash_id(*args, **kwargs):
    raise UnsupportedAPIError("graphlearn.set_use_string_hash_id is unsupported")


def set_load_graph_thread_num(*args, **kwargs):
    raise UnsupportedAPIError("graphlearn.set_load_graph_thread_num is unsupported")


def set_tracker_mode(*args, **kwargs):
    raise UnsupportedAPIError("graphlearn.set_tracker_mode is unsupported")


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    raise UnsupportedAPIError(f"graphlearn.{name} is unsupported")


__all__ = ["Graph", "Nodes", "Decoder"]
