import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# VW-28: Logistic regression baseline on the PaySim seed sample.
# Uses the VW-18 chronological split: steps 1-323 train, 324-378 validation, 379-743 test.
# Metrics are reported on validation only. The test split is just scored to write
# predictions/baseline_test.csv; nothing is tuned on it.
#
# Run from the repo root: python src/vaultwatch/models/baseline.py

NUMERIC_FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
CATEGORICAL_FEATURES = ["type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Left out on purpose:
#   nameOrig / nameDest -> account IDs, no signal for a linear baseline
#   isFlaggedFraud      -> PaySim's own rule-based flag, would leak label-like info
#   step                -> only used for splitting


# LOGISTIC REGRESSION MODEL CLASS (same interface as IsolationForestModel)
class LogRegModel:
    def __init__(self):
        preprocess = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),  # scale numeric columns, fit on training data only
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),  # one column per transaction type
        ])
        classifier = LogisticRegression(
            class_weight="balanced",  # fraud is ~0.13% of rows, so weight it up
            max_iter=1000,
            random_state=42,
        )
        self.pipeline = Pipeline([("preprocess", preprocess), ("classifier", classifier)])

    def fit(self, X, y):  # supervised model, so fraud labels are required
        labels = np.asarray(y)
        if labels.shape != (len(X),) or not np.isin(labels, [0, 1]).all():  # error check
            raise ValueError("Provide one binary fraud label per training row.")
        if len(np.unique(labels)) < 2:
            raise ValueError("Training needs both fraud and normal transactions.")

        self.pipeline.fit(X, labels)
        return self

    def predict_score(self, features):  # returns probability of fraud (higher = more suspicious)
        return self.pipeline.predict_proba(features)[:, 1]

    def save(self, model_path):
        joblib.dump(self.pipeline, model_path)

    @classmethod
    def load(cls, path):  # loads a model from disk and returns an instance of the class
        model = cls()
        model.pipeline = joblib.load(path)
        return model


def write_predictions(rows, output_path):
    # txn_id uses the same seed-row scheme as iforest.py so predictions from different models line up
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

    # train the model
    model = LogRegModel()
    model.fit(training_rows[FEATURE_COLUMNS], training_labels)
    print(f"Trained on {len(training_rows):,} transactions ({training_labels.sum()} fraud).")

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

    # save predictions in the contract format: predictions/<model>_<split>.csv -> txn_id, score
    # test rows are only scored here, never used for tuning (the test set is locked)
    test_rows = transactions[transactions["step"] >= 379].copy()
    test_rows["score"] = model.predict_score(test_rows[FEATURE_COLUMNS])
    for split_name, rows in [("val", validation_rows), ("test", test_rows)]:
        write_predictions(rows, f"predictions/baseline_{split_name}.csv")

    # save the model, reload it, and check it gives the same scores
    model_path = Path("data/processed/logreg_seed.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    reloaded_model = LogRegModel.load(model_path)
    example_features = validation_rows[FEATURE_COLUMNS].head(5)
    np.testing.assert_allclose(reloaded_model.predict_score(example_features), model.predict_score(example_features))
    print(f"Reloaded {model_path} and verified the same scores on five rows.")


if __name__ == "__main__":
    main()
