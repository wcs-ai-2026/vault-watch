import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `vaultwatch` imports work when run as a script
from vaultwatch.models.xgb import FEATURE_COLUMNS, XGBModel  # noqa: E402

# VW-30: handle class imbalance and tune XGBoost on the validation split.
#
# Every setting is trained on the training split (steps 1-323) and scored on the
# validation split (steps 324-378). The test split is never loaded here.
#
#   python src/vaultwatch/models/tune_xgb.py --data seed   # 200k seed sample, ~1 min
#   python src/vaultwatch/models/tune_xgb.py --data full   # full PaySim from data/raw/, much slower
#
# The seed validation split only has 11 fraud rows, so seed results are noisy.
# Decisions in DECISIONS.md are based on the full run (564 fraud rows in validation).

RAW_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
SEED_PATH = Path("data/seed/paysim_sample.parquet")
TARGET_RECALL = 0.85  # project goal: catch 85%+ of fraud

# What we try. "auto" = normal/fraud ratio in training (~775 on full data, ~1215 on the seed),
# "sqrt" = its square root, 1 = no reweighting at all.
GRID = {
    "scale_pos_weight": [1, "sqrt", "auto"],
    "max_depth": [4, 6],
    "learning_rate": [0.1, 0.05],
}
N_ESTIMATORS = {0.1: 300, 0.05: 600}  # smaller steps get more trees so total learning is similar


def load_splits(data):
    if data == "seed":
        transactions = pd.read_parquet(SEED_PATH)
    else:
        columns = ["step", "type", "isFraud"] + [c for c in FEATURE_COLUMNS if c != "type"]
        transactions = pd.read_csv(RAW_PATH, usecols=columns)
    train = transactions[transactions["step"] <= 323]
    val = transactions[transactions["step"].between(324, 378)]
    return train, val


def precision_at_recall(labels, scores, target_recall):
    # best precision we can get while still catching at least target_recall of the fraud
    precision, recall, _ = precision_recall_curve(labels, scores)
    reachable = recall >= target_recall
    return precision[reachable].max() if reachable.any() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["seed", "full"], default="seed")
    args = parser.parse_args()

    train, val = load_splits(args.data)
    y_train, y_val = train["isFraud"].to_numpy(), val["isFraud"].to_numpy()
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"[{args.data}] train {len(train):,} rows / {y_train.sum():,} fraud (ratio {ratio:.0f}:1), "
          f"val {len(val):,} rows / {y_val.sum():,} fraud")

    results = []
    for weight, depth, rate in product(*GRID.values()):
        scale_pos_weight = {"sqrt": float(np.sqrt(ratio)), "auto": float(ratio)}.get(weight, weight)
        start = time.time()
        model = XGBModel(scale_pos_weight=scale_pos_weight, max_depth=depth,
                         learning_rate=rate, n_estimators=N_ESTIMATORS[rate])
        model.fit(train[FEATURE_COLUMNS], y_train)
        scores = model.predict_score(val[FEATURE_COLUMNS])
        flagged = scores >= 0.5

        row = {
            "scale_pos_weight": weight,
            "scale_pos_weight_value": round(scale_pos_weight, 1),
            "max_depth": depth,
            "learning_rate": rate,
            "n_estimators": N_ESTIMATORS[rate],
            "pr_auc": float(average_precision_score(y_val, scores)),
            f"precision_at_recall_{TARGET_RECALL}": float(precision_at_recall(y_val, scores, TARGET_RECALL)),
            "precision_at_0.5": float(y_val[flagged].mean()) if flagged.any() else 0.0,
            "recall_at_0.5": float(flagged[y_val == 1].mean()),
            "seconds": round(time.time() - start, 1),
        }
        results.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    table = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    output_path = Path(f"docs/tuning/xgb_tuning_{args.data}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(output_path, index=False)
    print(f"\nBest by validation PR-AUC:\n{table.head(5).round(4).to_string(index=False)}")
    print(f"\nSaved {len(table)} results to {output_path}")


if __name__ == "__main__":
    main()
