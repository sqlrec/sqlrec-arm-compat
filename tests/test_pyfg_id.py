import farmhash
import numpy as np
import pyarrow as pa
import pyfg
import pytest

from sqlrec_arm_compat import UnsupportedAPIError


def handler(kind="num_buckets", default="", name="feature", input_name="value"):
    return pyfg.FgArrowHandler(
        {
            "features": [
                {
                    "feature_type": "id_feature",
                    "feature_name": name,
                    "expression": f"item:{input_name}",
                    "default_value": default,
                    "value_type": "string",
                    "need_prefix": False,
                    "value_dim": 0,
                    kind: 100,
                }
            ]
        },
        1,
    )


def test_integer_id_and_empty_default():
    output, status = handler().process_arrow(
        {"value": pa.array(["1\x1d2", "", None, "3"])}
    )
    assert status.ok()
    assert status.message() == ""
    np.testing.assert_array_equal(output["feature"].np_values, [1, 2, 3])
    np.testing.assert_array_equal(output["feature"].np_lengths, [2, 0, 0, 1])


def test_integer_arrow_and_default():
    fg = handler(default="0")
    output, _ = fg.process_arrow({"value": pa.array([1, 2, None, 3])})
    np.testing.assert_array_equal(output["feature"].np_values, [1, 2, 0, 3])
    np.testing.assert_array_equal(output["feature"].np_lengths, [1, 1, 1, 1])
    default, status = fg({"value": [None]})
    assert status.ok() and default["feature"][0] == [0]
    fg.reset_executor()


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
@pytest.mark.parametrize("kind", ["num_buckets", "hash_bucket_size"])
def test_direct_numpy_integer_ids(dtype, kind, monkeypatch):
    monkeypatch.setenv("USE_FARM_HASH_TO_BUCKETIZE", "true")
    output, status = handler(kind, default="0")({"value": [dtype(2), None]})
    expected = [2, 0] if kind == "num_buckets" else [
        farmhash.fingerprint64(token) % 100 for token in ("2", "0")
    ]
    assert status.ok()
    assert output["feature"] == [[value] for value in expected]


@pytest.mark.parametrize("dtype", [pa.int64(), pa.string()])
def test_out_of_range_integer_ids_fall_back_to_zero(dtype):
    rows = [-1, 0, 99, 100, 101, 2**63 - 1, None]
    if pa.types.is_string(dtype):
        rows = [str(value) if value is not None else None for value in rows]
    fg = handler(default="99")
    expected = [0, 0, 99, 0, 0, 0, 99]
    output, status = fg.process_arrow({"value": pa.array(rows, type=dtype)})
    assert status.ok()
    assert output["feature"].np_values.tolist() == expected
    assert output["feature"].np_lengths.tolist() == [1] * len(rows)
    direct, status = fg({"value": rows})
    assert status.ok()
    assert direct["feature"] == [[value] for value in expected]


@pytest.mark.parametrize("token", ["12oops", "1.5"])
def test_malformed_integer_ids_still_raise(token):
    with pytest.raises(ValueError, match="expected integer ID"):
        handler()({"value": [token]})


@pytest.mark.parametrize("separator", ["||", "::", "\u4e2d"])
def test_unsupported_id_separators_raise(separator):
    config = {
        "features": [{
            "feature_type": "id_feature",
            "feature_name": "feature",
            "expression": "item:value",
            "num_buckets": 100,
            "separator": separator,
        }]
    }
    with pytest.raises(UnsupportedAPIError, match="separator"):
        pyfg.FgArrowHandler(config, 1)


def test_farm_hash_matches_tzrec_reference(monkeypatch):
    monkeypatch.setenv("USE_FARM_HASH_TO_BUCKETIZE", "true")
    fg = handler("hash_bucket_size", default="xyz")
    output, _ = fg.process_arrow(
        {"value": pa.array(["abc\x1defg", None, "hij"])}
    )
    np.testing.assert_array_equal(output["feature"].np_values, [85, 59, 20, 95])
    np.testing.assert_array_equal(output["feature"].np_lengths, [2, 1, 1])
    default, _ = fg({"value": [None]})
    assert default["feature"][0] == [20]


def test_arrow_string_lists_are_multivalue_ids(monkeypatch):
    monkeypatch.setenv("USE_FARM_HASH_TO_BUCKETIZE", "true")
    fg = handler("hash_bucket_size", default="unknown", name="genres")
    rows = [["Action", "Adventure", "Sci-Fi"], [], None, ["Comedy"]]
    output, status = fg.process_arrow(
        {"value": pa.array(rows, type=pa.list_(pa.string()))}
    )
    assert status.ok()
    expected_tokens = ["Action", "Adventure", "Sci-Fi", "unknown", "unknown", "Comedy"]
    np.testing.assert_array_equal(
        output["genres"].np_values,
        [farmhash.fingerprint64(token) % 100 for token in expected_tokens],
    )
    np.testing.assert_array_equal(output["genres"].np_lengths, [3, 1, 1, 1])


def test_arrow_string_lists_without_default_keep_empty_rows(monkeypatch):
    monkeypatch.setenv("USE_FARM_HASH_TO_BUCKETIZE", "true")
    output, _ = handler("hash_bucket_size", name="genres").process_arrow(
        {"value": pa.array([["Action"], [], None], type=pa.list_(pa.string()))}
    )
    np.testing.assert_array_equal(
        output["genres"].np_values, [farmhash.fingerprint64("Action") % 100]
    )
    np.testing.assert_array_equal(output["genres"].np_lengths, [1, 0, 0])


@pytest.mark.parametrize("row", [["Action", None], [""], [1]])
def test_string_lists_reject_invalid_tokens(row):
    fg = handler("hash_bucket_size", name="genres")
    with pytest.raises(UnsupportedAPIError, match="list IDs"):
        fg({"value": [row]})


def test_hash_requires_farm_mode(monkeypatch):
    monkeypatch.delenv("USE_FARM_HASH_TO_BUCKETIZE", raising=False)
    with pytest.raises(UnsupportedAPIError, match="USE_FARM_HASH_TO_BUCKETIZE=true"):
        handler("hash_bucket_size").process_arrow({"value": pa.array(["abc"])})


@pytest.mark.parametrize(
    "change,match",
    [
        ({"feature_type": "combo_feature"}, "only id_feature"),
        ({"weighted": True}, "weighted"),
        ({"expression": "user:id"}, "item:<input>"),
        ({"value_dim": 2}, "value_dim"),
        ({"vocab_list": ["a"]}, "vocab_list"),
    ],
)
def test_unsupported_config_raises(change, match):
    feature = {
        "feature_type": "id_feature",
        "feature_name": "feature",
        "expression": "item:value",
        "num_buckets": 100,
    }
    feature.update(change)
    with pytest.raises(UnsupportedAPIError, match=match):
        pyfg.FgArrowHandler({"features": [feature]}, 1)


def test_unsupported_methods_and_input_raise():
    fg = handler()
    config = {
        "features": [{
            "feature_type": "id_feature",
            "feature_name": "feature",
            "expression": "item:value",
            "num_buckets": 100,
        }]
    }
    with pytest.raises(UnsupportedAPIError, match="threads=1"):
        pyfg.FgArrowHandler(config, 2)
    with pytest.raises(UnsupportedAPIError, match="user_inputs"):
        fg.user_inputs()
    with pytest.raises(UnsupportedAPIError, match="FeatureFactory.create"):
        pyfg.FeatureFactory.create({})
    with pytest.raises(UnsupportedAPIError, match="Arrow type"):
        fg.process_arrow({"value": pa.array([1.2])})
