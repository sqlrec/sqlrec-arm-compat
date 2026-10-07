"""Deterministic, architecture-independent inputs for the original-wheel oracle.

Use only JSON/Python scalar inputs: original bindings do not consistently accept
NumPy scalar objects. NumPy scalar support is tested separately by unit tests.
"""

import random
import struct


def float32_neighbors(value):
    """Previous/equal/next float32, without platform-dependent NumPy arithmetic."""
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    if value == 0:
        return [-struct.unpack("!f", struct.pack("!I", 1))[0], 0.0,
                struct.unpack("!f", struct.pack("!I", 1))[0]]
    return sorted(struct.unpack("!f", struct.pack("!I", bit))[0]
                  for bit in (bits - 1, bits, bits + 1))


def case(case_id, feature, values, arrow_type):
    direct = [row if row is not None else [] for row in values] if arrow_type.startswith("list<") else values
    return {"id": case_id, "config": {"features": [feature]},
            "data": {"input": values}, "direct_data": {"input": direct},
            "types": {"input": arrow_type}}


def id_feature(kind, count=100, default="", **options):
    return {"feature_type": "id_feature", "feature_name": "feature",
            "expression": "item:input", kind: count, "default_value": default, **options}


def raw_feature(**options):
    return {"feature_type": "raw_feature", "feature_name": "feature",
            "expression": "item:input", "value_type": "float", "default_value": "0.25", **options}


def supported_cases():
    cases = []
    for kind in ("num_buckets", "hash_bucket_size"):
        for count in (1, 100, 2**31 + 11, 2**63 - 1):
            for dtype in ("int32", "int64", "string", "list<string>"):
                maximum = {"int8": 127, "int16": 32767, "int32": 2**31 - 1}.get(dtype, 2**63 - 1)
                values = [-maximum - 1, -1, 0, 1, 99, 100, maximum, None]
                if dtype == "string":
                    values = [str(row) if row is not None else None for row in values] + ["", "+01", "00099", "9223372036854775808", "-9223372036854775809"]
                elif dtype.startswith("list<"):
                    values = [["0", "1"], [], None, ["99", "100"]]
                for default in ("", "1"):
                    cases.append(case(f"id-{kind}-{count}-{dtype}-default={default}",
                                      id_feature(kind, count, default), values, dtype))
    # Exercise all FarmHash short/long-input branches, Unicode, NUL and signed IDs.
    tokens = ["中文", "🙂", "é", "e\u0301", "a\0b", "\0", "-9223372036854775808"]
    tokens += ["x" * size for size in (1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1024)]
    rng = random.Random(20261007)
    tokens += ["".join(rng.choice("abcXYZ019中文🙂") for _ in range(rng.randrange(1, 160))) for _ in range(64)]
    cases.append(case("hash-utf8-lengths-seeded", id_feature("hash_bucket_size", 2**63 - 1), tokens, "string"))
    cases.append(case("hash-int64-extremes", id_feature("hash_bucket_size"),
                      [-2**63, -2**31, -1, 0, 2**31, 2**63 - 1, None], "int64"))
    for separator in ("|", ",", "\t", "\x1d"):
        cases.append(case(f"id-separator-{ord(separator)}", id_feature("num_buckets", separator=separator),
                          [separator.join(["1", "2", "1"]), "", None], "string"))
    for kind in ("num_buckets", "hash_bucket_size"):
        item = case(f"multivalue-default-{kind}", id_feature(kind, default="1|2", separator="|"),
                    ["3|4", "", None], "string")
        if kind == "num_buckets":
            item.update(expect_error=True, error_match="default_value")
        cases.append(item)
    for separator, default in (("1", "1"), ("-", "-1")):
        cases.append(case(f"integer-default-containing-separator-{separator}",
                          id_feature("num_buckets", default=default, separator=separator),
                          ["2", "", None], "string"))

    normalizers = ("", "method=zscore,mean=0.1,standard_deviation=0.3",
                   "method=minmax,min=0.1,max=0.8", "method=log10,threshold=0.05,default=-5")
    for dtype in ("float32", "float64", "int32", "int64", "string", "list<float32>", "list<float64>", "list<int32>", "list<int64>", "list<string>"):
        for index, normalizer in enumerate(normalizers):
            for bucketized in (False, True):
                options = {"normalizer": normalizer}
                if bucketized:
                    options["boundaries"] = [-1, 0, 0.1, 0.5, 1]
                values = [-1, 0, 1, 10, None] if dtype.startswith("int") or dtype.startswith("list<int") else [-1, 0, 0.05, 0.1, 0.5, 1, 10, None]
                if dtype == "string":
                    values = [str(row) if row is not None else None for row in values] + [""]
                elif dtype.startswith("list<"):
                    values = [[str(row) if dtype == "list<string>" else row] if row is not None else None for row in values] + [[]]
                for default in (("", "0.25") if bucketized else ("0.25",)):
                    cases.append(case(f"raw-{dtype}-normalizer={index}-bucket={bucketized}-default={default}",
                                      raw_feature(default_value=default, **options), values, dtype))
    for dim in (2, 8):
        for dtype in ("list<float32>", "list<float64>", "list<int32>", "list<int64>", "list<string>", "string"):
            for index, normalizer in enumerate(normalizers):
                for bucketized in (False, True):
                    options = {"value_dim": dim, "normalizer": normalizer,
                               "default_value": "|".join(["0.25"] * dim), "separator": "|"}
                    if bucketized:
                        options["boundaries"] = [-1, 0, 0.1, 0.5, 1]
                    values = [[(column + 1) / 10 for column in range(dim)], [], None]
                    if dtype.startswith("list<int"):
                        values[0] = list(range(dim))
                    if dtype == "string":
                        values = ["|".join(map(str, values[0])), "", None]
                    elif dtype == "list<string>":
                        values[0] = list(map(str, values[0]))
                    cases.append(case(f"vector-{dim}-{dtype}-normalizer={index}-bucket={bucketized}",
                                      raw_feature(**options), values, dtype))
    # Exactly on either side of boundaries and log10 thresholds in float32.
    for boundary in (-1.0, 0.0, 0.05, 0.1, 1.0, 16777216.0):
        for dtype in ("float32", "float64", "string"):
            values = float32_neighbors(boundary)
            if dtype == "string":
                values = list(map(repr, values))
            cases.append(case(f"float32-ulp-{boundary}-{dtype}", raw_feature(boundaries=[boundary]), values, dtype))
    for index, normalizer in enumerate(normalizers[1:]):
        # Threshold and normalized boundary values: bucket comparisons are exact,
        # unlike dense outputs, where a small floating-point tolerance is allowed.
        values = [value for center in (0.05, 0.1, 0.25, 0.8, 1, 10) for value in float32_neighbors(center)]
        values += [rng.uniform(-2, 20) for _ in range(128)]
        cases.append(case(f"normalized-boundary-seeded-{index}",
                          raw_feature(normalizer=normalizer, boundaries=[-5, -1, 0, 0.1, 0.5, 1, 2]), values, "float32"))
    cases.append(case("raw-finite-extremes", raw_feature(),
                      [-3.4028234663852886e38, -0.0, 0.0, 1.401298464324817e-45,
                       1.1754943508222875e-38, 3.4028234663852886e38], "float32"))
    # Put boundaries one float32 ULP apart around transcendental results. A
    # Tiny rounding differences may cross these boundaries; comparison checks
    # native normalized values and every crossed boundary within four ULPs.
    log_values = (0.3010299956639812, 0.47712125471966244, 0.8450980400142568,
                  1.0, 1.505149978319906, 2.0)
    bounds = sorted({value for center in log_values for value in float32_neighbors(center)})
    for dtype in ("float32", "float64", "string"):
        values = [value for center in (2, 3, 7, 10, 32, 100) for value in float32_neighbors(center)]
        if dtype == "string":
            values = list(map(repr, values))
        cases.append(case(f"log10-one-ulp-buckets-{dtype}",
                          raw_feature(normalizer="method=log10", boundaries=bounds), values, dtype))
        cases[-1]["bucket_float_max_ulp"] = 4
    for feature, dtype in ((id_feature("num_buckets"), "int64"),
                           (id_feature("hash_bucket_size"), "string"),
                           (raw_feature(), "float32"),
                           (raw_feature(boundaries=[0, 1], default_value=""), "float64"),
                           (raw_feature(value_dim=2, default_value="0|0", separator="|"), "list<float32>")):
        for values, suffix in (([], "empty"), ([None, None], "nulls")):
            cases.append(case(f"batch-{feature['feature_type']}-{dtype}-{suffix}", feature, values, dtype))
    # Two features read the same column, with different names and encodings.
    mixed = case("shared-column-mixed", id_feature("num_buckets"), [1, 2, None], "int64")
    mixed["config"]["features"] += [raw_feature(feature_name="dense"),
                                     raw_feature(feature_name="bucket", boundaries=[0, 1], default_value="")]
    cases.append(mixed)
    assert len({item["id"] for item in cases}) == len(cases)
    return cases


def all_cases():
    # Include the older differential cases so ARM/local replay runs them too.
    from differential_cases import differential_cases, separator_cases
    return (supported_cases() + rejected_arrow_cases() + differential_cases() + separator_cases()
            + [{"id": "graphlearn-import-surface", "graphlearn_import": True}])


def rejected_arrow_cases():
    cases = []
    for feature_type in ("id_feature", "raw_feature"):
        for dtype, values in (("int8", [1, None]), ("int16", [1, None]),
                              ("uint8", [1, None]), ("uint16", [1, None]),
                              ("uint32", [1, None]), ("uint64", [2**63, None]),
                              ("bool", [True, None]), ("float16", [0.5, None]),
                              ("null", [None, None]),
                              ("large_string", ["1", None]),
                              ("large_list<string>", [["1"], None])):
            feature = id_feature("num_buckets") if feature_type == "id_feature" else raw_feature()
            item = case(f"rejected-{feature_type}-{dtype}", feature, values, dtype)
            item["expect_error"] = True
            cases.append(item)
    return cases
