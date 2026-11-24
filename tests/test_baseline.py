import numpy as np
import pytest

from vaultwatch.models.baseline import FEATURE_COLUMNS, LogRegModel


def test_fit_returns_the_model(toy_transactions):
    model = LogRegModel()
    assert model.fit(toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]) is model


def test_predict_score_is_one_score_between_0_and_1_per_row(toy_transactions):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    scores = LogRegModel().fit(X, y).predict_score(X)

    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_fraud_scores_higher_than_normal(toy_transactions):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    scores = LogRegModel().fit(X, y).predict_score(X)

    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_save_and_load_give_the_same_scores(toy_transactions, tmp_path):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    model = LogRegModel().fit(X, y)

    path = tmp_path / "baseline.joblib"
    model.save(path)
    reloaded = LogRegModel.load(path)

    np.testing.assert_allclose(reloaded.predict_score(X), model.predict_score(X))


def test_fit_rejects_labels_that_are_not_0_or_1(toy_transactions):
    X = toy_transactions[FEATURE_COLUMNS]
    with pytest.raises(ValueError):
        LogRegModel().fit(X, np.full(len(X), 2))


def test_fit_rejects_wrong_number_of_labels(toy_transactions):
    X, y = toy_transactions[FEATURE_COLUMNS], toy_transactions["isFraud"]
    with pytest.raises(ValueError):
        LogRegModel().fit(X, y[:-1])


def test_fit_needs_both_classes(toy_transactions):
    X = toy_transactions[FEATURE_COLUMNS]
    with pytest.raises(ValueError):
        LogRegModel().fit(X, np.zeros(len(X), dtype=int))
