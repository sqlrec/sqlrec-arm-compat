"""Continuous RawFeature subset used by SQLRec training and serving."""

from dataclasses import dataclass
import re

import numpy as np

from sqlrec_arm_compat import UnsupportedAPIError


@dataclass(frozen=True)
class DenseData:
    dense_values: np.ndarray


def number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, np.number)):
        raise ValueError("Expected a number or numeric string")
    if isinstance(value, str) and not re.fullmatch(r"\s*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\s*", value):
        raise ValueError("Expected a complete decimal number")
    try:
        parsed = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Number must be finite float32") from error
    if not np.isfinite(parsed) or abs(parsed) > float(np.finfo(np.float32).max):
        raise ValueError("Number must be finite float32")
    return np.float32(parsed)


def normalizer(text):
    if not text:
        return "", {}
    parts = {}
    for part in text.split(","):
        pair = part.strip().split("=")
        if len(pair) != 2 or pair[0] in parts:
            raise ValueError("Invalid normalizer")
        parts[pair[0]] = pair[1]
    method = parts.pop("method", "")
    if method == "log10":
        required = {"threshold", "default"}
        parts.setdefault("threshold", "1e-10")
        parts.setdefault("default", "-10")
    elif method == "zscore":
        required = {"mean", "standard_deviation"}
    elif method == "minmax":
        required = {"min", "max"}
    else:
        raise UnsupportedAPIError("RawFeature normalizer must use zscore, minmax or log10")
    if set(parts) != required:
        raise ValueError("Invalid normalizer parameters")
    params = {key: number(value) for key, value in parts.items()}
    if (method == "zscore" and params["standard_deviation"] <= 0
            or method == "minmax" and params["max"] <= params["min"]
            or method == "log10" and params["threshold"] <= 0):
        raise ValueError("Normalizer scale must be positive")
    return method, params


@dataclass(frozen=True)
class RawFeature:
    name: str
    input_name: str
    value_dim: int
    default_value: str
    separator: str
    boundaries: tuple
    method: str
    params: dict

    @classmethod
    def parse(cls, raw):
        allowed = {"feature_type", "feature_name", "expression", "value_type", "value_dim", "default_value", "separator", "boundaries", "normalizer"}
        if set(raw) - allowed:
            raise UnsupportedAPIError(f"RawFeature unsupported config keys: {sorted(set(raw) - allowed)}")
        name = raw.get("feature_name")
        expression = raw.get("expression", "")
        if not isinstance(name, str) or not name:
            raise ValueError("feature_name must be a nonempty string")
        if not isinstance(expression, str) or not expression.startswith("item:") or not expression[5:] or ":" in expression[5:]:
            raise UnsupportedAPIError("RawFeature expects item:<input> expression")
        if raw.get("value_type", "float") != "float":
            raise UnsupportedAPIError("RawFeature requires value_type=float")
        dim = raw.get("value_dim", 1)
        if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 65536:
            raise ValueError("Invalid RawFeature value_dim")
        separator = raw.get("separator", "\x1d")
        default = raw.get("default_value", "0")
        if not isinstance(separator, str) or not separator or not isinstance(default, str):
            raise ValueError("Invalid RawFeature separator/default_value")
        boundaries = raw.get("boundaries", [])
        if not isinstance(boundaries, list):
            raise ValueError("boundaries must be an array")
        bounds = tuple(number(value) for value in boundaries)
        if any(a >= b for a, b in zip(bounds, bounds[1:])):
            raise ValueError("boundaries must be strictly increasing")
        method, params = normalizer(raw.get("normalizer", ""))
        feature = cls(name, expression[5:], dim, default, separator, bounds, method, params)
        if not default and not bounds:
            raise ValueError("Dense RawFeature requires default_value")
        feature.encode(None)
        return feature

    def encode(self, value):
        missing = value is None or value == "" or value == []
        if missing:
            value = self.default_value
        if isinstance(value, str):
            tokens = value.split(self.separator) if value else []
        elif isinstance(value, list):
            tokens = value
        else:
            tokens = [value]
        if not tokens and self.boundaries:
            return []
        if len(tokens) != self.value_dim:
            raise ValueError(f"RawFeature {self.name!r}: value count must match value_dim")
        values = np.asarray([number(token) for token in tokens], dtype=np.float32)
        # FG defaults are already in the normalized value space.
        if not missing and self.method:
            p = self.params
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if self.method == "zscore":
                    values = (values - p["mean"]) / p["standard_deviation"]
                elif self.method == "minmax":
                    values = (values - p["min"]) / (p["max"] - p["min"])
                else:
                    values = np.where(values > p["threshold"], np.log10(values), p["default"])
            if not np.all(np.isfinite(values)):
                raise ValueError("Normalized values must be finite float32")
        if self.boundaries:
            return np.searchsorted(np.asarray(self.boundaries, dtype=np.float32), values, side="right").tolist()
        return values.tolist()
