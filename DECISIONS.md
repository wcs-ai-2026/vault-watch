# Decisions

One entry per decision and why, so everyone knows what we chose and can find the evidence.

## 2026-11-23: XGBoost class imbalance handling and settings (VW-30, Mousa)

**Decision:** `XGBModel` uses `scale_pos_weight=1` (no reweighting), `max_depth=6`,
`learning_rate=0.1`, `n_estimators=300`, `subsample=0.8`, `colsample_bytree=0.8`.
The 0.13% fraud rate is handled by choosing the alert threshold on validation
(see `docs/xgb_report.md`), not by reweighting the training data.

**Why:**

- Tried 12 settings: `scale_pos_weight` of 1, sqrt(normal/fraud) = 35, and normal/fraud = 1,224,
  crossed with `max_depth` 4/6 and learning rate 0.1 (300 trees) / 0.05 (600 trees).
  Each was trained on the full training split (4,463,587 rows, 3,643 fraud) and scored on the
  full validation split (980,416 rows, 564 fraud). Results: `docs/tuning/xgb_tuning_full.csv`.
- Reweighting did not rank fraud any better. Best validation PR-AUC was 0.9046 with weight 1,
  0.9044 with sqrt and 0.8995 with the full ratio. What reweighting changed was where the scores
  land: at a 0.5 threshold, weight 1 gave precision 0.94 / recall 0.74, the full ratio gave
  precision 0.24 / recall 0.99. That trade-off is the threshold's job, and leaving weights at 1
  keeps the score close to a real probability of fraud, which is easier to combine in the ensemble.
- `max_depth=6` beat 4 in every pairing.
- Among the top settings, weight 1 / depth 6 / learning rate 0.1 had the best precision at 85%
  recall (0.736, the project's recall goal) and trains in about half the time of the 600-tree
  version (56s vs 113s on a laptop), whose PR-AUC is only 0.0002 higher.
- The seed sample could not choose between settings: its validation split has 11 fraud rows and
  all 12 settings scored PR-AUC 0.89-0.91 (`docs/tuning/xgb_tuning_seed.csv`).

**Still open:** precision at 85% recall is 0.74, below the 80% precision goal. These runs use raw
PaySim columns only. Rerun `python src/vaultwatch/models/tune_xgb.py --data full` once
`make_features()` lands (VW-21/VW-22), since account-history features may change the best settings.
