import numpy as np
import pytest

from vaultwatch.models.xgb import FEATURE_COLUMNS, XGBModel, prepare_features

FAST = {"n_estimators": 20}  # keep tests quick


def test_prepare_features_always_has_the_same_columns(toy_transactions):
    only_payments = toy_transactions[toy_transactions["type"] == "PAYMENT"]
    assert list(prepare_features(only_payments).columns) == list(prepare_features(toy_transactions).columns)


def test_fit_returns_the_model(toy_transactions):
    model = XGBModel(**FAST)
    assert model.fit(toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]) is model


def test_auto_scale_pos_weight_is_normal_to_fraud_ratio(toy_transactions):
    y = toy_transactions["isFraud"]
    model = XGBModel(**FAST).fit(toy_transactions[FEATURE_COLUMNS], y)
    assert model.fitted_scale_pos_weight == pytest.approx((y == 0).sum() / (y == 1).sum())


def test_predict_score_is_one_score_between_0_and_1_per_row(toy_transactions):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    scores = XGBModel(**FAST).fit(X, y).predict_score(X)

    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_fraud_scores_higher_than_normal(toy_transactions):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    scores = XGBModel(**FAST).fit(X, y).predict_score(X)

    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_save_and_load_give_the_same_scores_and_settings(toy_transactions, tmp_path):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    model = XGBModel(max_depth=3, **FAST).fit(X, y)

    path = tmp_path / "xgb.joblib"
    model.save(path)
    reloaded = XGBModel.load(path)

    np.testing.assert_allclose(reloaded.predict_score(X), model.predict_score(X))
    assert reloaded.params == model.params


def test_predict_before_fit_raises(toy_transactions):
    with pytest.raises(RuntimeError):
        XGBModel().predict_score(toy_transactions[FEATURE_COLUMNS])


def test_fit_rejects_bad_labels(toy_transactions):
    X = toy_transactions[FEATURE_COLUMNS]
    with pytest.raises(ValueError):
        XGBModel(**FAST).fit(X, np.full(len(X), 2))
    with pytest.raises(ValueError):
        XGBModel(**FAST).fit(X, np.zeros(len(X), dtype=int))


def test_feature_importance_covers_every_feature(toy_transactions):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    importance = XGBModel(**FAST).fit(X, y).feature_importance()

    assert set(importance.index) == set(prepare_features(X).columns)
    assert importance.is_monotonic_decreasing
