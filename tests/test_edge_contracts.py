"""Fail-fast input/config contracts and handler lifecycle regressions."""

import copy

import farmhash
import numpy as np
import pyarrow as pa
import pyfg
import pytest

from sqlrec_arm_compat import UnsupportedAPIError
from oracle_cases import id_feature, raw_feature


def handler(feature=None, threads=1):
    return pyfg.FgArrowHandler({"features": [feature or id_feature("num_buckets")]}, threads)


@pytest.mark.parametrize("threads", [True, False, 0, -1, 1.0, "1", None])
def test_thread_count_requires_positive_integer(threads):
    with pytest.raises(ValueError, match="threads"):
        handler(threads=threads)


@pytest.mark.parametrize("config", [None, [], {}, {"features": []}, {"features": {}},
                                   {"features": [None]}, {"features": [1]},
                                   {"features": [id_feature("num_buckets")], "extra": True}])
def test_invalid_top_level_config(config):
    with pytest.raises(UnsupportedAPIError):
        pyfg.FgArrowHandler(config, 1)


@pytest.mark.parametrize("kind", ["id", "raw"])
@pytest.mark.parametrize("name", [None, "", 1, False])
def test_feature_name_required_and_string(kind, name):
    feature = id_feature("num_buckets") if kind == "id" else raw_feature()
    feature["feature_name"] = name
    with pytest.raises(ValueError, match="feature_name"):
        handler(feature)
    feature.pop("feature_name")
    with pytest.raises(ValueError, match="feature_name"):
        handler(feature)


@pytest.mark.parametrize("kind", ["num_buckets", "hash_bucket_size"])
@pytest.mark.parametrize("count", [True, False, 0, -1, 2**63, 1.5, "100", None, np.int64(100)])
def test_bucket_count_bounds_and_config_types(kind, count):
    with pytest.raises(ValueError):
        handler(id_feature(kind, count))


@pytest.mark.parametrize("feature", [
    {key: value for key, value in id_feature("num_buckets").items() if key != "num_buckets"},
    id_feature("num_buckets", hash_bucket_size=100),
])
def test_bucket_modes_are_mutually_exclusive_and_required(feature):
    with pytest.raises(UnsupportedAPIError, match="exactly one"):
        handler(feature)


@pytest.mark.parametrize("options", [
    {"value_dim": True}, {"value_dim": -1}, {"value_dim": 65537}, {"value_dim": "2"},
    {"boundaries": (0, 1)}, {"boundaries": [np.inf]}, {"boundaries": [np.nan]},
    {"boundaries": [0.1, 0.1000000001]}, {"boundaries": [True]},
    {"default_value": None}, {"value_dim": 2, "default_value": "0.1"},
    {"value_dim": 2, "default_value": "0.1\x1d0.2\x1d0.3"},
    {"separator": ""}, {"separator": None},
    {"normalizer": None}, {"normalizer": 1}, {"normalizer": []},
    {"normalizer": "method=zscore,mean=0"},
    {"normalizer": "method=zscore,mean=0,standard_deviation=-1"},
    {"normalizer": "method=minmax,min=1,max=1"},
    {"normalizer": "method=minmax,min=2,max=1"},
    {"normalizer": "method=log10,threshold=0"},
    {"normalizer": "method=log10,default=nan"},
    {"normalizer": "method=log10,method=log10"},
    {"normalizer": "method=log10,unknown=1"},
    {"normalizer": "method=log10,broken"},
])
def test_invalid_raw_config_fails_early(options):
    with pytest.raises((ValueError, UnsupportedAPIError)):
        handler(raw_feature(**options))


@pytest.mark.parametrize("value", [np.complex64(1 + 2j), np.complex128(1), np.bool_(True),
                                   b"1", "1_000", "0x10", "1,2", "1e999", 10**400,
                                   [None], [True], [float("nan")]])
def test_invalid_raw_values_never_truncate_or_produce_nan(value):
    with pytest.raises(ValueError):
        handler(raw_feature())({"input": [value]})


@pytest.mark.parametrize("normalizer", ["method=zscore,mean=-3e38,standard_deviation=1",
                                      "method=zscore,mean=0,standard_deviation=1e-30"])
def test_normalization_overflow_is_rejected(normalizer):
    with pytest.raises(ValueError, match="finite float32"):
        handler(raw_feature(normalizer=normalizer))({"input": [3e38]})


def test_string_normalization_cannot_overflow_final_float32_cast():
    with pytest.raises(ValueError, match="finite float32"):
        handler(raw_feature(normalizer="method=zscore,mean=0,standard_deviation=0.1"))({"input": ["3e38"]})


@pytest.mark.parametrize("kind", ["id", "raw"])
@pytest.mark.parametrize("expression", [None, "", "user:input", "item:", "item:a:b", 1])
def test_expression_subset_is_enforced(kind, expression):
    feature = id_feature("num_buckets") if kind == "id" else raw_feature()
    feature["expression"] = expression
    with pytest.raises((ValueError, UnsupportedAPIError)):
        handler(feature)


@pytest.mark.parametrize("change", [{"default_value": None}, {"separator": None},
                                     {"separator": ""}, {"need_prefix": True},
                                     {"value_type": "float"}])
def test_id_unsupported_defaults_and_options(change):
    with pytest.raises((ValueError, UnsupportedAPIError)):
        handler(id_feature("num_buckets", **change))


@pytest.mark.parametrize("row", [True, np.bool_(True), 1.5, np.float32(1), {}, (1,), b"1", "1\x1d", "\x1d1", "1\x1d\x1d2"])
def test_invalid_id_rows_are_not_silently_coerced(row):
    with pytest.raises(UnsupportedAPIError):
        handler()({"input": [row]})


@pytest.mark.parametrize("options", [{"weighted": True}, {"value_type": "string"}])
def test_raw_unsupported_options(options):
    with pytest.raises(UnsupportedAPIError):
        handler(raw_feature(**options))


def test_bucketize_only_and_output_unsupported_api_contracts():
    with pytest.raises(UnsupportedAPIError, match="bucketize_only"):
        pyfg.FgArrowHandler({"features": [id_feature("num_buckets")]}, 1, bucketize_only=True)
    result, status = handler().process_arrow({"input": pa.array([1])})
    assert status.ok() and status.message() == ""
    for item in (status, result["feature"]):
        with pytest.raises(UnsupportedAPIError, match="unknown_api"):
            item.unknown_api
        with pytest.raises(AttributeError):
            item.__unknown_api__
    with pytest.raises(UnsupportedAPIError, match="weighted"):
        result["feature"].np_weights


def test_maximum_supported_raw_dimension():
    dim = 65536
    fg = handler(raw_feature(value_dim=dim, default_value="\x1d".join(["0"] * dim)))
    output, _ = fg.process_arrow({"input": pa.array([None], type=pa.list_(pa.float32()))})
    assert output["feature"].dense_values.shape == (1, dim)
    assert not output["feature"].dense_values.any()


@pytest.mark.parametrize("kind", ["id", "raw"])
@pytest.mark.parametrize("column", [pa.chunked_array([[1], [2]]), pa.table({"input": [1]}),
                                   np.array([1]), [1], (1,)])
def test_arrow_api_requires_an_array(kind, column):
    feature = id_feature("num_buckets") if kind == "id" else raw_feature()
    with pytest.raises(UnsupportedAPIError, match="pyarrow.Array"):
        handler(feature).process_arrow({"input": column})


@pytest.mark.parametrize("column", [pa.array([1]), np.array([1]), "1", 1, None])
def test_direct_api_requires_a_list_or_tuple(column):
    with pytest.raises(UnsupportedAPIError, match="list input"):
        handler()({"input": column})


def test_missing_columns_duplicate_names_and_mismatched_batches():
    fg = handler()
    for method in (fg, fg.process_arrow):
        with pytest.raises(KeyError, match="missing input"):
            method({})
        with pytest.raises(TypeError, match="mapping"):
            method([1, 2])
    first = id_feature("num_buckets")
    with pytest.raises(ValueError, match="unique"):
        pyfg.FgArrowHandler({"features": [first, first]}, 1)
    second = raw_feature(feature_name="raw", expression="item:other")
    fg = pyfg.FgArrowHandler({"features": [first, second]}, 1)
    for data in ({"input": [1], "other": []},
                 {"input": pa.array([1]), "other": pa.array([], type=pa.float32())}):
        method = fg.process_arrow if isinstance(data["input"], pa.Array) else fg
        with pytest.raises(ValueError, match="equal lengths"):
            method(data)


def test_repeated_calls_reset_and_failure_do_not_mutate_inputs_or_outputs():
    feature = id_feature("num_buckets", default="1")
    original = copy.deepcopy(feature)
    fg = handler(feature)
    data = {"input": [["1", "2"], [], None]}
    original_data = copy.deepcopy(data)
    before, _ = fg(data)
    arrow, _ = fg.process_arrow({"input": pa.array(data["input"], type=pa.list_(pa.string()))})
    with pytest.raises(ValueError):
        fg({"input": ["oops"]})
    fg.reset_executor()
    after, _ = fg(data)
    assert before == after == {"feature": [[1, 2], [1], [1]]}
    assert arrow["feature"].np_values.tolist() == [1, 2, 1, 1]
    assert data == original_data and feature == original
    assert fg({"input": (1, 2)})[0] == {"feature": [[1], [2]]}


def test_hash_default_is_one_token_not_split_by_separator(monkeypatch):
    monkeypatch.setenv("USE_FARM_HASH_TO_BUCKETIZE", "true")
    fg = handler(id_feature("hash_bucket_size", default="1|2", separator="|"))
    expected = farmhash.fingerprint64("1|2") % 100
    assert fg({"input": [None, "", []]})[0]["feature"] == [[expected]] * 3
    output, _ = fg.process_arrow({"input": pa.array([None, "", "1|2"], type=pa.string())})
    assert output["feature"].np_lengths.tolist() == [1, 1, 2]
    assert output["feature"].np_values[:2].tolist() == [expected, expected]


@pytest.mark.parametrize("separator,default,expected", [("1", "1", 1), ("-", "-1", 0)])
def test_integer_default_is_not_split_even_if_separator_is_digit_or_sign(separator, default, expected):
    fg = handler(id_feature("num_buckets", default=default, separator=separator))
    assert fg({"input": [None, "", []]})[0]["feature"] == [[expected]] * 3


def test_environment_mode_changes_and_unsupported_api_names(monkeypatch):
    monkeypatch.delenv("USE_FARM_HASH_TO_BUCKETIZE", raising=False)
    fg = handler(id_feature("hash_bucket_size"))
    pyfg.set_env("USE_FARM_HASH_TO_BUCKETIZE", "true")
    before, _ = fg({"input": ["abc"]})
    pyfg.unset_env("USE_FARM_HASH_TO_BUCKETIZE")
    with pytest.raises(UnsupportedAPIError, match="FARM_HASH"):
        fg({"input": ["abc"]})
    pyfg.unset_env("USE_FARM_HASH_TO_BUCKETIZE")
    pyfg.set_env("USE_FARM_HASH_TO_BUCKETIZE", "true")
    assert fg({"input": ["abc"]})[0] == before
    for method in (pyfg.set_env, pyfg.unset_env):
        with pytest.raises(UnsupportedAPIError, match="UNKNOWN_MODE"):
            method("UNKNOWN_MODE", "true") if method is pyfg.set_env else method("UNKNOWN_MODE")
    with pytest.raises(UnsupportedAPIError, match="unknown_api"):
        pyfg.unknown_api
    with pytest.raises(AttributeError):
        fg.__unknown_api__
