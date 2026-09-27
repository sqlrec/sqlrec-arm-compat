import farmhash
import pyarrow as pa
import pytest


def test_sqlrec_id_feature_path(monkeypatch):
    monkeypatch.setenv("USE_FARM_HASH_TO_BUCKETIZE", "true")
    pytest.importorskip("tzrec")
    from tzrec.datasets.data_parser import DataParser
    from tzrec.features.id_feature import IdFeature
    from tzrec.protos import feature_pb2

    configs = [
        ("age", "num_buckets", 100, [1, None, 3], [1, 3], [1, 0, 1]),
        ("city", "hash_bucket_size", 100, ["abc", None, "hij"], [85, 95], [1, 0, 1]),
        (
            "genres", "hash_bucket_size", 100,
            [["Action", "Adventure"], [], None],
            [farmhash.fingerprint64(value) % 100 for value in ("Action", "Adventure")],
            [2, 0, 0],
        ),
    ]
    features = []
    input_data = {"label": pa.array([1, 0, 1])}
    for name, bucket_kind, count, values, expected_values, expected_lengths in configs:
        kwargs = {bucket_kind: count}
        feature = IdFeature(
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name=name,
                    expression=f"item:{name}",
                    embedding_dim=16,
                    **kwargs,
                )
            ),
            fg_mode=2,
        )
        parsed = feature.parse({name: pa.array(values)})
        assert parsed.values.tolist() == expected_values
        assert parsed.lengths.tolist() == expected_lengths
        features.append(feature)
        input_data[name] = pa.array(values)

    parsed_batch = DataParser(features, labels=["label"]).parse(input_data)
    assert parsed_batch["age.values"].tolist() == [1, 3]
    assert parsed_batch["age.lengths"].tolist() == [1, 0, 1]
    assert parsed_batch["city.values"].tolist() == [85, 95]
    assert parsed_batch["city.lengths"].tolist() == [1, 0, 1]
    assert parsed_batch["genres.values"].tolist() == configs[2][4]
    assert parsed_batch["genres.lengths"].tolist() == [2, 0, 0]
    assert parsed_batch["label"].tolist() == [1, 0, 1]
