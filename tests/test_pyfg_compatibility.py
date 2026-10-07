"""Regression checks for the distinct direct and Arrow output contracts."""

import unittest

import numpy as np
import pyarrow as pa
import pyfg

from float_support import assert_float_output


def raw(name="raw", **options):
    feature = {
        "feature_type": "raw_feature",
        "feature_name": name,
        "expression": f"item:{name}",
        "value_type": "float",
        "default_value": "0.1",
    }
    feature.update(options)
    return feature


def handler(*features):
    return pyfg.FgArrowHandler({"features": list(features)}, 1)


class OutputContractTest(unittest.TestCase):
    def test_scalar_direct_rows_are_flat_and_arrow_rows_are_dense(self):
        fg = handler(raw())
        for rows in ([None], [0.2, 0.3, None], ["0.2", "", None], []):
            with self.subTest(rows=rows):
                expected = [0.1 if value in (None, "") else float(value) for value in rows]
                direct, status = fg({"raw": rows})
                self.assertTrue(status.ok())
                self.assertEqual(np.asarray(direct["raw"]).shape, (len(rows),))
                assert_float_output(direct["raw"], expected)
                dtype = pa.string() if rows and isinstance(rows[0], str) else pa.float64()
                arrow, status = fg.process_arrow({"raw": pa.array(rows, type=dtype)})
                self.assertTrue(status.ok())
                self.assertEqual(arrow["raw"].dense_values.dtype, np.float32)
                self.assertEqual(arrow["raw"].dense_values.shape, (len(rows), 1))
                assert_float_output(arrow["raw"].dense_values, np.asarray(expected).reshape(-1, 1))

    def test_scalar_normalizers_preserve_default_shape_and_values(self):
        cases = [
            ("method=zscore,mean=0.1,standard_deviation=10", 0.01),
            ("method=minmax,min=0.1,max=0.6", 0.2),
            ("method=log10,threshold=0.05,default=-5", np.log10(0.2)),
        ]
        for normalizer, value in cases:
            with self.subTest(normalizer=normalizer):
                direct, _ = handler(raw(normalizer=normalizer))({"raw": [0.2, None, "", []]})
                assert_float_output(direct["raw"], [value, 0.1, 0.1, 0.1])

    def test_vector_direct_rows_keep_their_dimension(self):
        fg = handler(raw(value_dim=2, default_value="0.1|0.4", separator="|"))
        rows = [[0.2, 0.5], [], None, "0.3|0.6"]
        direct, _ = fg({"raw": rows})
        assert_float_output(direct["raw"], [[0.2, 0.5], [0.1, 0.4], [0.1, 0.4], [0.3, 0.6]])
        arrow, _ = fg.process_arrow({"raw": pa.array(rows[:3], type=pa.list_(pa.float32()))})
        assert_float_output(arrow["raw"].dense_values, direct["raw"][:3])
        empty, _ = fg.process_arrow({"raw": pa.array([], type=pa.list_(pa.float32()))})
        self.assertEqual(empty["raw"].dense_values.shape, (0, 2))
        self.assertEqual(fg({"raw": []})[0]["raw"], [])

    def test_numeric_arrow_empty_bucket_defaults_bucketize_zero(self):
        for dtype in (pa.float32(), pa.float64(), pa.int64()):
            for boundaries in ([0.1, 0.2, 0.3], [-2, -1], [0]):
                with self.subTest(dtype=dtype, boundaries=boundaries):
                    fg = handler(raw(boundaries=boundaries, default_value=""))
                    output, _ = fg.process_arrow({"raw": pa.array([None, 1, None], type=dtype)})
                    zero_bucket = sum(boundary <= 0 for boundary in boundaries)
                    self.assertEqual(output["raw"].np_values.tolist(), [zero_bucket, len(boundaries), zero_bucket])
                    self.assertEqual(output["raw"].np_lengths.tolist(), [1, 1, 1])
                    self.assertEqual(output["raw"].np_values.dtype, np.int64)
                    self.assertEqual(output["raw"].np_lengths.dtype, np.int32)
                    # Arrow materialization must not change subsequent direct calls.
                    self.assertEqual(fg({"raw": [None]})[0]["raw"], [None])

    def test_numeric_arrow_zero_fallback_bypasses_normalization(self):
        fg = handler(raw(boundaries=[-1, 0, 1], default_value="", normalizer="method=zscore,mean=10,standard_deviation=1"))
        output, _ = fg.process_arrow({"raw": pa.array([None, 0.0])})
        self.assertEqual(output["raw"].np_values.tolist(), [2, 0])
        self.assertEqual(output["raw"].np_lengths.tolist(), [1, 1])

    def test_string_and_list_arrow_empty_buckets_remain_missing(self):
        fg = handler(raw(boundaries=[0.1, 0.2, 0.3], default_value=""))
        for values, dtype in (
            ([None, "", "0.2"], pa.string()),
            ([None, [], [0.2]], pa.list_(pa.float32())),
        ):
            with self.subTest(dtype=dtype):
                output, _ = fg.process_arrow({"raw": pa.array(values, type=dtype)})
                self.assertEqual(output["raw"].np_values.tolist(), [2])
                self.assertEqual(output["raw"].np_lengths.tolist(), [0, 0, 1])

    def test_numeric_arrow_nonempty_bucket_default_is_not_replaced(self):
        fg = handler(raw(boundaries=[0.1, 0.2, 0.3], default_value="0.2"))
        output, _ = fg.process_arrow({"raw": pa.array([None, 0.1, None])})
        self.assertEqual(output["raw"].np_values.tolist(), [2, 1, 2])
        self.assertEqual(output["raw"].np_lengths.tolist(), [1, 1, 1])
        self.assertEqual(fg({"raw": [None]})[0]["raw"], [2])

    def test_direct_bucket_scalars_match_original_numeric_nulls(self):
        fg = handler(raw(boundaries=[-1, 0, 1], default_value="", normalizer="method=zscore,mean=10,standard_deviation=1"))
        for values in ([0.0, None], [0, None]):
            with self.subTest(values=values):
                output, status = fg({"raw": values})
                self.assertTrue(status.ok())
                self.assertEqual(output["raw"], [0, 2])
        self.assertEqual(fg({"raw": [None]})[0]["raw"], [None])
        self.assertEqual(fg({"raw": []})[0]["raw"], [])

    def test_direct_bucket_strings_and_lists_preserve_missing_rows(self):
        fg = handler(raw(boundaries=[0.1, 0.2], default_value=""))
        for values in (["0.2", "", None], [[0.2], [], None]):
            with self.subTest(values=values):
                output, status = fg({"raw": values})
                self.assertTrue(status.ok())
                self.assertEqual(output["raw"], [2, None, None])

    def test_bucket_vectors_preserve_empty_rows_and_default_shape(self):
        fg = handler(raw(value_dim=2, boundaries=[0.1, 0.2], default_value=""))
        rows = [[0.1, 0.2], [], None]
        direct, status = fg({"raw": rows})
        self.assertTrue(status.ok())
        self.assertEqual(direct["raw"], [[1, 2], [], []])
        self.assertEqual(fg({"raw": [None]})[0]["raw"], [[]])
        self.assertEqual(fg({"raw": []})[0]["raw"], [])
        arrow, status = fg.process_arrow({"raw": pa.array(rows, type=pa.list_(pa.float32()))})
        self.assertTrue(status.ok())
        self.assertEqual(arrow["raw"].np_values.tolist(), [1, 2])
        self.assertEqual(arrow["raw"].np_lengths.tolist(), [2, 0, 0])

    def test_mixed_features_keep_each_output_contract(self):
        fg = handler(
            raw(),
            raw("vector", value_dim=2, default_value="0.1\x1d0.4"),
            raw("bucket", boundaries=[0.1, 0.2], default_value=""),
            {"feature_type": "id_feature", "feature_name": "id", "expression": "item:id", "num_buckets": 10},
        )
        direct, _ = fg({"raw": [None], "vector": [None], "bucket": [None], "id": [None]})
        assert_float_output(direct["raw"], [0.1])
        assert_float_output(direct["vector"], [[0.1, 0.4]])
        self.assertEqual(direct["bucket"], [None])
        self.assertEqual(direct["id"], [[]])


if __name__ == "__main__":
    unittest.main()
