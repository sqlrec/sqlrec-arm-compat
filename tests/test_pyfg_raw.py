import numpy as np
import pyarrow as pa
import pyfg
import pytest

from sqlrec_arm_compat import UnsupportedAPIError


def handler(**options):
    feature = {"feature_type": "raw_feature", "feature_name": "price", "expression": "item:input", "value_type": "float", "default_value": "0.1"}
    feature.update(options)
    return pyfg.FgArrowHandler({"features": [feature]}, 1)


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64(), pa.string()])
def test_scalar_float_types_and_null_defaults(dtype):
    values = ["0.2", "0.3", None] if pa.types.is_string(dtype) else [0.2, 0.3, None]
    output, _ = handler().process_arrow({"input": pa.array(values, type=dtype)})
    assert output["price"].dense_values.dtype == np.float32
    np.testing.assert_allclose(output["price"].dense_values, [[0.2], [0.3], [0.1]])


@pytest.mark.parametrize("normalizer,expected", [
    ("method=zscore,mean=0.1,standard_deviation=10.0", [[0.01], [0.02], [0.1]]),
    ("method=minmax,min=0.1,max=0.6", [[0.2], [0.4], [0.1]]),
    ("method=log10,threshold=0.05,default=-5", [[-0.6989700043], [-0.5228787453], [0.1]]),
])
def test_normalizers_leave_defaults_in_encoded_value_space(normalizer, expected):
    output, _ = handler(normalizer=normalizer).process_arrow({"input": pa.array([0.2,0.3,None])})
    np.testing.assert_allclose(output["price"].dense_values, expected, rtol=1e-6)
    defaults, _ = handler(normalizer=normalizer)({"input": [None]})
    np.testing.assert_allclose(defaults["price"], [0.1])


@pytest.mark.parametrize("normalizer,expected", [
    ("method=log10", [0, 0, 0, 0, 1, 0.1]),
    ("method=log10,threshold=0.05", [0, 0, -1, 0, 1, 0.1]),
    ("method=log10,default=-5", [-5, -5, -5, -5, 1, 0.1]),
    ("method=log10,threshold=0.05,default=-5", [-5, -5, -1, 0, 1, 0.1]),
])
def test_log10_effective_defaults_match_original(normalizer, expected):
    fg = handler(normalizer=normalizer)
    values = [0, 0.05, 0.1, 1, 10, None]
    direct, status = fg({"input": values})
    assert status.ok()
    np.testing.assert_allclose(direct["price"], expected, rtol=1e-6)
    output, status = fg.process_arrow({"input": pa.array(values, type=pa.float32())})
    assert status.ok()
    np.testing.assert_allclose(output["price"].dense_values, np.asarray(expected).reshape(-1, 1), rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_direct_numpy_raw_scalars(dtype):
    values = [dtype(1), None]
    direct, status = handler()({"input": values})
    assert status.ok()
    np.testing.assert_allclose(direct["price"], [1, 0.1])
    bucketed, status = handler(boundaries=[0.1, 0.2], default_value="")({"input": values})
    assert status.ok()
    assert bucketed["price"] == [2, 0]


@pytest.mark.parametrize("separator", ["||", "::", "\u4e2d"])
@pytest.mark.parametrize("dim", [1, 2])
def test_unsupported_raw_separators_raise(separator, dim):
    default = separator.join(["0.1"] * dim)
    with pytest.raises(UnsupportedAPIError, match="separator"):
        handler(value_dim=dim, separator=separator, default_value=default)


def test_vector_and_bucket_boundaries():
    output, _ = handler(value_dim=2, default_value="0.1|0.4", separator="|").process_arrow({"input": pa.array([[0.2,0.5], [], None], type=pa.list_(pa.float32()))})
    np.testing.assert_allclose(output["price"].dense_values, [[0.2,0.5],[0.1,0.4],[0.1,0.4]])
    output, _ = handler(boundaries=[0.1,0.2,0.3], default_value="").process_arrow({"input": pa.array([0.05,0.1,0.2,0.3,None])})
    assert output["price"].np_values.tolist() == [0,1,2,3,0]
    assert output["price"].np_lengths.tolist() == [1,1,1,1,1]
    default, _ = handler(boundaries=[0.1], default_value="")({"input": [None]})
    assert default["price"] == [None]


@pytest.mark.parametrize("value", [True,"12oops","NaN","Inf",float("nan"),float("inf"),1e39,[1,2],{}])
def test_invalid_numbers_are_rejected(value):
    with pytest.raises((ValueError,TypeError)):
        handler()({"input": [value]})


@pytest.mark.parametrize("options", [
    {"value_dim":0}, {"boundaries":[2,1]}, {"boundaries":[1,1]},
    {"normalizer":"method=zscore,mean=0,standard_deviation=0"},
    {"normalizer":"method=expression,expr=x"}, {"default_value":""},
])
def test_bad_config_is_rejected(options):
    with pytest.raises((ValueError, RuntimeError)):
        handler(**options)
