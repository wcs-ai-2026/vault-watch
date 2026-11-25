import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from vaultwatch.features.features import make_features


def sample_transactions():
    return pd.DataFrame({
        "step": [23, 24, 25],
        "type": ["TRANSFER", "PAYMENT", "CASH_IN"],
        "amount": [1000.0, 100.0, 500.0],
        "oldbalanceOrg": [1000.0, 500.0, 0.0],
        "newbalanceOrig": [0.0, 400.0, 500.0],
        "oldbalanceDest": [200.0, 0.0, 1000.0],
        "newbalanceDest": [1200.0, 0.0, 500.0],
        "nameOrig": ["C1", "C2", "C3"],
        "nameDest": ["C4", "M1", "C5"],
        "isFraud": [1, 0, 0],
        "isFlaggedFraud": [0, 0, 0]
    })


def test_feature_shape():
    features = make_features(sample_transactions())
    assert features.shape == (3, 22)


def test_balance_drained():
    features = make_features(sample_transactions())
    assert features["balance_drained"].tolist() == [1, 0, 0]


def test_balance_change():
    features = make_features(sample_transactions())

    assert features["orig_balance_change"].tolist() == [
        -1000, -100, 500
    ]


def test_depletion_ratio():
    features = make_features(sample_transactions())

    np.testing.assert_allclose(
        features["orig_balance_depletion_ratio"],
        [1.0, 0.2, 0.0]
    )


def test_amount_to_balance_ratio():
    features = make_features(sample_transactions())

    np.testing.assert_allclose(
        features["amount_to_balance_ratio"],
        [1.0, 0.2, 0.0]
    )


def test_no_missing_or_infinite_values():
    features = make_features(sample_transactions())

    assert not features.isna().any().any()

    numeric = features.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


def test_no_label_leakage():
    features = make_features(sample_transactions())

    excluded = [
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest"
    ]

    for column in excluded:
        assert column not in features.columns


def test_one_hot_encoding():
    features = make_features(sample_transactions())

    assert features["type_TRANSFER"].tolist() == [1, 0, 0]
    assert features["type_PAYMENT"].tolist() == [0, 1, 0]
    assert features["type_CASH_IN"].tolist() == [0, 0, 1]

    type_columns = [
        col for col in features.columns
        if col.startswith("type_")
    ]

    assert (features[type_columns].sum(axis=1) == 1).all()


def test_hour_of_day():
    features = make_features(sample_transactions())

    assert features["hour_of_day"].tolist() == [23, 0, 1]


def test_destination_balance_availability():
    features = make_features(sample_transactions())

    assert features["dest_balance_available"].tolist() == [
        1, 0, 1
    ]


def test_transaction_ids():
    features = make_features(sample_transactions())

    assert features["txn_id"].is_unique

    assert features["txn_id"].tolist() == [
        "seed-000000",
        "seed-000001",
        "seed-000002"
    ]