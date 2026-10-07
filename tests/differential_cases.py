"""JSON-only differential corpus shared by export and replay."""

def _case(name, values, arrow_type, kind, default=""):
    return {
        "config": {
            "features": [
                {
                    "feature_type": "id_feature",
                    "feature_name": name,
                    "expression": f"item:{name}",
                    "default_value": default,
                    "value_type": "string",
                    "need_prefix": False,
                    "value_dim": 0,
                    kind: 100,
                }
            ]
        },
        "data": {name: values},
        "types": {name: arrow_type},
    }


def _raw_case(name, values, arrow_type="float64", **options):
    feature = {
        "feature_type": "raw_feature",
        "feature_name": name,
        "expression": f"item:{name}",
        "value_type": "float",
        "default_value": "0.1",
    }
    feature.update(options)
    return {"config": {"features": [feature]}, "data": {name: values}, "types": {name: arrow_type}}



def differential_cases():
    cases = [
        _case("int_string", ["1\x1d2", "", None, "3"], "string", "num_buckets"),
        _case("int_arrow", [1, 2, None, 3], "int64", "num_buckets", "0"),
        _case("int_out_of_range", [-1, 0, 99, 100, 101, 2**63 - 1, None], "int64", "num_buckets", "99"),
        _case("int_string_out_of_range", ["-1", "99", "100", "9223372036854775807", None], "string", "num_buckets", "99"),
        _case("hash_string", ["abc\x1defg", None, "hij"], "string", "hash_bucket_size", "xyz"),
        _case("hash_integer", [1, 2, None, 3], "int64", "hash_bucket_size", "4"),
        _case(
            "hash_list",
            [["Action", "Adventure"], [], None, ["Comedy"]],
            "list<string>",
            "hash_bucket_size",
            "unknown",
        ),
        _raw_case("float_scalar", [0.2,0.3,None]),
        _raw_case("float_strings", ["0.2","",None,"0.3"], "string"),
        _raw_case("zscore", [0.2,0.3,None], normalizer="method=zscore,mean=0.1,standard_deviation=10"),
        _raw_case("minmax", [0.2,0.3,None], normalizer="method=minmax,min=0.1,max=0.6"),
        _raw_case("log10", [0.1,0.01,10,None], normalizer="method=log10,threshold=0.05,default=-5"),
        _raw_case("log10_defaults", [0, 0.05, 0.1, 1, 10, None], "float32", normalizer="method=log10"),
        _raw_case("log10_default_fallback", [0, 0.05, 0.1, 1, 10, None], "float32", normalizer="method=log10,threshold=0.05"),
        _raw_case("log10_default_threshold", [0, 0.05, 0.1, 1, 10, None], "float32", normalizer="method=log10,default=-5"),
        _raw_case("float_vector", [[0.2,0.5],[],None], "list<float32>", value_dim=2, default_value="0.1\x1d0.4"),
        _raw_case("bucket_vector_empty", [[0.1,0.2],[],None], "list<float32>", value_dim=2, boundaries=[0.1,0.2], default_value=""),
        _raw_case("bucket_vector_default", [[0.1,0.2],[],None], "list<float32>", value_dim=2, boundaries=[0.1,0.2], default_value="0.1\x1d0.3"),
        _raw_case("bucket_boundary", [0.05,0.1,0.2,0.3,None], boundaries=[0.1,0.2,0.3],default_value=""),
        _raw_case("float32_scalar", [0.2, 0.3, None], "float32", value_dim=1),
        _raw_case("scalar_list", [[0.2], [], None], "list<float32>"),
        _raw_case("vector_strings", ["0.2|0.5", "", None], "string", value_dim=2, separator="|", default_value="0.1|0.4"),
        _raw_case("bucket_float32", [0.05, 0.1, 0.2, None], "float32", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_integer", [0, 1, None], "int64", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_negative", [-3, -2, -1, None], boundaries=[-2, -1], default_value=""),
        _raw_case("bucket_zero_boundary", [-1, 0, 1, None], boundaries=[0], default_value=""),
        _raw_case("bucket_strings", ["0.05", "0.1", "", None], "string", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_list", [[0.2], [], None], "list<float32>", boundaries=[0.1, 0.2], default_value=""),
        _raw_case("bucket_nonempty_default", [0.05, None], boundaries=[0.1, 0.2], default_value="0.2"),
        _raw_case("zscore_bucket", [0.2, None], boundaries=[0.01, 0.1], normalizer="method=zscore,mean=0.1,standard_deviation=10"),
        _raw_case("zscore_empty_bucket", [0.0, None], boundaries=[-1, 0, 1], default_value="", normalizer="method=zscore,mean=10,standard_deviation=1"),
    ]
    for item in cases:
        item['id'] = 'differential-' + item['config']['features'][0]['feature_name']
        item['direct_data'] = {
            name: [row if row is not None else [] for row in values]
            if item['types'][name].startswith('list<') else values
            for name, values in item['data'].items()
        }
    return cases


def separator_cases():
    cases = []
    for feature_type in ("id_feature", "raw_feature"):
        for separator in ("||", "\u4e2d"):
            if feature_type == "id_feature":
                item = _case("separator", [f"1{separator}2"], "string", "num_buckets")
                item["config"]["features"][0]["separator"] = separator
            else:
                item = _raw_case("separator", [f"0.1{separator}0.2"], "string", value_dim=2,
                                 default_value=f"0.1{separator}0.4", separator=separator)
            item.update(id=f"rejected-separator-{feature_type}-{separator}", expect_error=True)
            cases.append(item)
    return cases
