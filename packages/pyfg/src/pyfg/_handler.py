"""ID and continuous feature subset of pyfg.FgArrowHandler used by SQLREC."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import farmhash
import numpy as np
import pyarrow as pa

from sqlrec_arm_compat import UnsupportedAPIError
from pyfg._raw import RawFeature, DenseData


@dataclass(frozen=True)
class _Feature:
    name: str
    input_name: str
    bucket_kind: str
    bucket_count: int
    default_value: str
    separator: str


@dataclass(frozen=True)
class _SparseData:
    np_values: np.ndarray
    np_lengths: np.ndarray

    @property
    def np_weights(self):
        raise UnsupportedAPIError("pyfg weighted ID features are unsupported")

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise UnsupportedAPIError(f"pyfg sparse output {name} is unsupported")


class _Status:
    def ok(self) -> bool:
        return True

    def message(self) -> str:
        return ""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise UnsupportedAPIError(f"pyfg status {name} is unsupported")


_OK = _Status()
_ALLOWED_KEYS = {
    "feature_type",
    "feature_name",
    "expression",
    "default_value",
    "value_type",
    "need_prefix",
    "value_dim",
    "num_buckets",
    "hash_bucket_size",
    "separator",
}


class FgArrowHandler:
    """Handle SQLREC's ordinary, unweighted item-side ID features."""

    def __init__(
        self, config: Mapping[str, Any], threads: int, *, bucketize_only: bool = False
    ) -> None:
        if bucketize_only:
            raise UnsupportedAPIError("pyfg bucketize_only mode is unsupported")
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if threads != 1:
            raise UnsupportedAPIError("pyfg.FgArrowHandler supports only threads=1")
        if not isinstance(config, Mapping) or set(config) != {"features"}:
            raise UnsupportedAPIError("pyfg config must contain only 'features'")
        raw_features = config["features"]
        if not isinstance(raw_features, list) or not raw_features:
            raise UnsupportedAPIError("pyfg config needs a nonempty features list")
        self._features = tuple(self._parse_feature(raw) for raw in raw_features)
        if len({feature.name for feature in self._features}) != len(self._features):
            raise ValueError("feature_name must be unique")

    @staticmethod
    def _parse_feature(raw: Any) -> _Feature:
        if not isinstance(raw, Mapping):
            raise UnsupportedAPIError("pyfg feature config must be an object")
        name = raw.get("feature_name", "<unnamed>")
        if raw.get("feature_type") == "raw_feature":
            return RawFeature.parse(raw)
        if raw.get("feature_type") != "id_feature":
            raise UnsupportedAPIError(
                f"pyfg feature {name!r}: only id_feature and raw_feature are supported"
            )
        unsupported = set(raw) - _ALLOWED_KEYS
        if unsupported:
            raise UnsupportedAPIError(
                f"pyfg feature {name!r}: unsupported config keys {sorted(unsupported)}"
            )
        if not isinstance(name, str) or not name:
            raise ValueError("feature_name must be a nonempty string")
        expression = raw.get("expression")
        if not isinstance(expression, str) or not expression.startswith("item:"):
            raise UnsupportedAPIError(
                f"pyfg feature {name!r}: only item:<input> expressions are supported"
            )
        input_name = expression[5:]
        if not input_name or ":" in input_name:
            raise ValueError(f"pyfg feature {name!r}: invalid expression {expression!r}")
        for key, expected in (
            ("value_type", "string"),
            ("need_prefix", False),
            ("value_dim", 0),
        ):
            if raw.get(key, expected) != expected:
                raise UnsupportedAPIError(
                    f"pyfg feature {name!r}: {key}={raw[key]!r} is unsupported"
                )
        kinds = [key for key in ("num_buckets", "hash_bucket_size") if key in raw]
        if len(kinds) != 1:
            raise UnsupportedAPIError(
                f"pyfg feature {name!r}: exactly one of num_buckets or "
                "hash_bucket_size is required"
            )
        kind = kinds[0]
        count = raw[kind]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 2**63 - 1:
            raise ValueError(f"pyfg feature {name!r}: {kind} must be positive")
        default = raw.get("default_value", "")
        if not isinstance(default, str):
            raise UnsupportedAPIError(
                f"pyfg feature {name!r}: default_value must be a string"
            )
        separator = raw.get("separator", "\x1d")
        if not isinstance(separator, str) or not separator:
            raise ValueError("separator must be a nonempty string")
        return _Feature(name, input_name, kind, count, default, separator)

    @staticmethod
    def _bucketize(feature: _Feature, token: Any) -> int:
        if feature.bucket_kind == "num_buckets":
            try:
                if isinstance(token, str) and not re.fullmatch(r"\s*[+-]?[0-9]+\s*", token):
                    raise ValueError("Expected a complete decimal integer")
                value = int(token)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pyfg feature {feature.name!r}: expected integer ID, got {token!r}"
                ) from exc
            if value < 0 or value >= feature.bucket_count:
                raise ValueError(
                    f"pyfg feature {feature.name!r}: ID {value} is outside "
                    f"[0, {feature.bucket_count})"
                )
            return value
        if os.environ.get("USE_FARM_HASH_TO_BUCKETIZE", "").lower() != "true":
            raise UnsupportedAPIError(
                f"pyfg feature {feature.name!r}: hash_bucket_size requires "
                "USE_FARM_HASH_TO_BUCKETIZE=true for cross-architecture parity"
            )
        return farmhash.fingerprint64(str(token)) % feature.bucket_count

    @classmethod
    def _encode_row(cls, feature: _Feature, value: Any) -> list[int]:
        if value is None or value == "" or value == []:
            value = feature.default_value
        if value is None or value == "":
            return []
        if isinstance(value, str):
            tokens = value.split(feature.separator)
        elif isinstance(value, list):
            tokens = value
            if any(not isinstance(token, str) or not token for token in tokens):
                raise UnsupportedAPIError(
                    f"pyfg feature {feature.name!r}: list IDs must be nonempty strings"
                )
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            tokens = [value]
        else:
            raise UnsupportedAPIError(
                f"pyfg feature {feature.name!r}: input type "
                f"{type(value).__name__} is unsupported"
            )
        if any(token == "" for token in tokens):
            raise UnsupportedAPIError(
                f"pyfg feature {feature.name!r}: empty IDs inside multivalue input "
                "are unsupported"
            )
        return [cls._bucketize(feature, token) for token in tokens]

    def _encode(self, input_data: Mapping[str, Any], *, require_arrow: bool):
        if not isinstance(input_data, Mapping):
            raise TypeError("pyfg input_data must be a mapping")
        output: dict[str, list[list[int]]] = {}
        row_count = None
        for feature in self._features:
            if feature.input_name not in input_data:
                raise KeyError(
                    f"pyfg feature {feature.name!r}: missing input "
                    f"{feature.input_name!r}"
                )
            column = input_data[feature.input_name]
            if require_arrow:
                if not isinstance(column, pa.Array):
                    raise UnsupportedAPIError(
                        f"pyfg feature {feature.name!r}: process_arrow needs "
                        "pyarrow.Array input"
                    )
                is_string_list = (
                    (pa.types.is_list(column.type) or pa.types.is_large_list(column.type))
                    and pa.types.is_string(column.type.value_type)
                )
                if not (
                    pa.types.is_string(column.type)
                    or pa.types.is_integer(column.type)
                    or is_string_list
                    or isinstance(feature, RawFeature) and (
                        pa.types.is_floating(column.type)
                        or pa.types.is_null(column.type)
                        or (pa.types.is_list(column.type) and pa.types.is_floating(column.type.value_type))
                    )
                ):
                    raise UnsupportedAPIError(
                        f"pyfg feature {feature.name!r}: Arrow type {column.type} "
                        "is unsupported"
                    )
                values = column.to_pylist()
            else:
                if not isinstance(column, (list, tuple)):
                    raise UnsupportedAPIError(
                        f"pyfg feature {feature.name!r}: direct call needs a list input"
                    )
                values = column
            if row_count is None:
                row_count = len(values)
            elif len(values) != row_count:
                raise ValueError("pyfg input columns must have equal lengths")
            output[feature.name] = [feature.encode(value) if isinstance(feature, RawFeature) else self._encode_row(feature, value) for value in values]
        return output

    def process_arrow(self, input_data: Mapping[str, pa.Array]):
        encoded = self._encode(input_data, require_arrow=True)
        output = {}
        for name, rows in encoded.items():
            feature = next(feature for feature in self._features if feature.name == name)
            if isinstance(feature, RawFeature) and not feature.boundaries:
                output[name] = DenseData(np.asarray(rows, dtype=np.float32).reshape(-1, feature.value_dim))
                continue
            output[name] = _SparseData(
                np_values=np.asarray([value for row in rows for value in row], dtype=np.int64),
                np_lengths=np.asarray([len(row) for row in rows], dtype=np.int32),
            )
        return output, _OK

    def __call__(self, input_data: Mapping[str, list[Any]]):
        output = self._encode(input_data, require_arrow=False)
        for feature in self._features:
            if isinstance(feature, RawFeature) and feature.boundaries:
                output[feature.name] = [row if row else None for row in output[feature.name]]
        return output, _OK

    def reset_executor(self) -> None:
        return None

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise UnsupportedAPIError(f"pyfg.FgArrowHandler.{name} is unsupported")
