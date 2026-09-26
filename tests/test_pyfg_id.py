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


def test_hash_requires_farm_mode(monkeypatch):
    monkeypatch.delenv("USE_FARM_HASH_TO_BUCKETIZE", raising=False)
    with pytest.raises(UnsupportedAPIError, match="USE_FARM_HASH_TO_BUCKETIZE=true"):
        handler("hash_bucket_size").process_arrow({"value": pa.array(["abc"])})


@pytest.mark.parametrize(
    "change,match",
    [
        ({"feature_type": "raw_feature"}, "only id_feature"),
        ({"weighted": True}, "weighted"),
        ({"separator": ","}, "separator"),
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
    with pytest.raises(ValueError, match="outside"):
        fg.process_arrow({"value": pa.array([100])})
