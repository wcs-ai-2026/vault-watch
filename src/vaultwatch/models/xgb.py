import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

# VW-29: XGBoost fraud classifier v1 on the PaySim seed sample.
# Same VW-18 chronological split as the baseline: steps 1-323 train, 324-378 validation,
# 379-743 test. Metrics are reported on validation; test is only scored to write
# predictions/xgb_test.csv (the test set is locked, nothing is tuned on it).
#
# Uses the raw PaySim columns for now. When Rund's make_features() lands (VW-21),
# FEATURE_COLUMNS / prepare_features() get swapped for the shared feature file.
#
# Run from the repo root: python src/vaultwatch/models/xgb.py

NUMERIC_FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURE_COLUMNS = NUMERIC_FEATURES + ["type"]
# Left out on purpose (same as the baseline):
#   nameOrig / nameDest -> account IDs
#   isFlaggedFraud      -> PaySim's own rule, would leak label-like info
#   step                -> only used for splitting

# Settings chosen in VW-30 on the full validation split (see DECISIONS.md and docs/tuning/).
# scale_pos_weight=1 means no reweighting: on full PaySim, reweighting did not rank fraud any
# better, it only pushed every score up. The class imbalance is handled by picking the alert
# threshold instead (see docs/xgb_report.md). scale_pos_weight="auto" (normal rows / fraud rows)
# and "sqrt" are still supported for experiments.
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1,
}


def prepare_features(transactions):
    # numeric columns as-is, plus one 0/1 column per transaction type (always the same 5 columns)
    features = transactions[NUMERIC_FEATURES].astype(float).copy()
    for transaction_type in TRANSACTION_TYPES:
        features[f"type_{transaction_type}"] = (transactions["type"] == transaction_type).astype(int)
    return features


# XGBOOST MODEL CLASS (same interface as LogRegModel and IsolationForestModel)
class XGBModel:
    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}
        self.booster = None  # the trained XGBClassifier, set by fit()

    def fit(self, X, y):  # supervised model, so fraud labels are required
        labels = np.asarray(y)
        if labels.shape != (len(X),) or not np.isin(labels, [0, 1]).all():  # error check
            raise ValueError("Provide one binary fraud label per training row.")
        if len(np.unique(labels)) < 2:
            raise ValueError("Training needs both fraud and normal transactions.")

        params = dict(self.params)
        ratio = (labels == 0).sum() / (labels == 1).sum()
        if params["scale_pos_weight"] == "auto":
            params["scale_pos_weight"] = ratio
        elif params["scale_pos_weight"] == "sqrt":
            params["scale_pos_weight"] = np.sqrt(ratio)
        self.fitted_scale_pos_weight = params["scale_pos_weight"]

        self.booster = XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )
        self.booster.fit(prepare_features(X), labels)
        return self

    def predict_score(self, features):  # returns probability of fraud, 0-1 (higher = more suspicious)
        if self.booster is None:
            raise RuntimeError("Call fit() or load() before predict_score().")
        return self.booster.predict_proba(prepare_features(features))[:, 1]

    def feature_importance(self):  # gain-based importance per feature, biggest first
        gains = self.booster.get_booster().get_score(importance_type="gain")
        names = prepare_features(pd.DataFrame(columns=FEATURE_COLUMNS)).columns
        return pd.Series({name: gains.get(name, 0.0) for name in names}).sort_values(ascending=False)

    def save(self, model_path):
        joblib.dump({"params": self.params, "booster": self.booster}, model_path)

    @classmethod
    def load(cls, path):  # loads a model from disk and returns an instance of the class
        saved = joblib.load(path)
        model = cls(**saved["params"])
        model.booster = saved["booster"]
        return model


def write_predictions(rows, output_path):
    # txn_id uses the same seed-row scheme as iforest.py and baseline.py so predictions line up
    predictions = pd.DataFrame({
        "txn_id": [f"seed-{row_number:06d}" for row_number in rows.index],
        "score": rows["score"].to_numpy(),
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions):,} predictions to {output_path}")


def main():  # main driver function

    transactions = pd.read_parquet("data/seed/paysim_sample.parquet")

    # chronological split from VW-18
    training_rows = transactions.loc[transactions["step"] <= 323]
    training_labels = training_rows["isFraud"]
    validation_rows = transactions[transactions["step"].between(324, 378)].copy()
    validation_labels = validation_rows["isFraud"]
    test_rows = transactions[transactions["step"] >= 379].copy()

    # train the model
    model = XGBModel()
    model.fit(training_rows[FEATURE_COLUMNS], training_labels)
    print(f"Trained on {len(training_rows):,} transactions ({training_labels.sum()} fraud), "
          f"scale_pos_weight={model.fitted_scale_pos_weight:.1f}.")

    # score validation rows; flag as fraud when probability >= 0.5
    validation_rows["score"] = model.predict_score(validation_rows[FEATURE_COLUMNS])
    threshold = 0.5
    flagged = (validation_rows["score"] >= threshold).astype(int)

    precision = precision_score(validation_labels, flagged, zero_division=0)
    recall = recall_score(validation_labels, flagged, zero_division=0)
    f1 = f1_score(validation_labels, flagged, zero_division=0)
    pr_auc = average_precision_score(validation_labels, validation_rows["score"])

    print(f"\nValidation: {len(validation_rows):,} rows, {validation_labels.sum()} fraud")
    print(f"Precision at {threshold}: {precision:.4f}")
    print(f"Recall at {threshold}:    {recall:.4f}")
    print(f"F1 at {threshold}:        {f1:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(validation_labels, flagged, labels=[0, 1]))

    print("\nTop features (gain):")
    print(model.feature_importance().round(1).to_string())

    # save predictions in the contract format: predictions/<model>_<split>.csv -> txn_id, score
    test_rows["score"] = model.predict_score(test_rows[FEATURE_COLUMNS])
    print()
    for split_name, rows in [("val", validation_rows), ("test", test_rows)]:
        write_predictions(rows, f"predictions/xgb_{split_name}.csv")

    # save the model, reload it, and check it gives the same scores
    # saved models live in models_saved/ (ignored by git, shared through the Drive folder)
    model_path = Path("models_saved/xgb_seed.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    reloaded_model = XGBModel.load(model_path)
    example_features = validation_rows[FEATURE_COLUMNS].head(5)
    np.testing.assert_allclose(reloaded_model.predict_score(example_features), model.predict_score(example_features))
    print(f"Reloaded {model_path} and verified the same scores on five rows.")


if __name__ == "__main__":
    main()
