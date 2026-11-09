# VW-18: Proposed Chronological Train/Validation/Test Split

## Objective

Define reproducible, time-based train/validation/test boundaries for the PaySim dataset to support realistic fraud detection evaluation and reduce temporal data leakage.

## Dataset

The full PaySim dataset contains 6,362,620 transactions across 743 simulated hourly steps, with 8,213 fraudulent transactions (approximately 0.1291%).

The `step` column represents simulated time in hours. Multiple transactions can occur within the same step.

## Proposed Split

We propose an approximately 70%/15%/15% split by transaction count, while keeping complete simulated hours together and preserving chronological order.

| Split | Step Range | Transactions | % of Total | Fraud Cases | Fraud Rate |
|---|---|---:|---:|---:|---:|
| Training | 1–323 | 4,463,587 | 70.15% | 3,643 | 0.0816% |
| Validation | 324–378 | 980,416 | 15.41% | 564 | 0.0575% |
| Testing | 379–743 | 918,617 | 14.44% | 4,006 | 0.4361% |

## Methodology

1. Group transactions by `step` and count the transactions occurring during each simulated hour.
2. Calculate cumulative transaction counts in chronological order.
3. Identify the first step where cumulative transaction volume reaches at least 70% of the dataset.
4. Identify the first step where cumulative transaction volume reaches at least 85% of the dataset.
5. Assign complete steps to each split, ensuring no step is shared across splits.

The resulting cut points are steps 323 and 378.

## Rationale

**Chronological integrity:** Training uses earlier transactions, validation uses later transactions, and testing uses the latest transactions. This better represents prospective fraud detection than randomly splitting transactions.

**Transaction-based proportions:** PaySim transaction volume varies considerably across simulated hours. Choosing boundaries using cumulative transaction counts produces splits close to the target proportions without dividing individual hourly steps.

**Reproducibility:** The boundaries are determined using transaction counts rather than random sampling.

## Important Finding: Fraud Distribution Shift

The fraud rate differs considerably across the splits:

- Training: 0.0816%
- Validation: 0.0575%
- Testing: 0.4361%

The testing fraud rate is approximately 5.3 times the training fraud rate.

Exploratory analysis also shows that transaction volume declines substantially during later simulated steps, while fraud counts remain relatively stable across most 100-hour blocks.

This change in fraud prevalence is important when interpreting precision, recall, and PR-AUC. It may reflect characteristics of PaySim's synthetic simulation rather than real-world fraud behaviour.

We recommend retaining these chronological boundaries as the initial proposal rather than adjusting them to artificially equalize fraud prevalence.

## Evaluation Considerations

- Fit preprocessing transformations only on training data.
- Use validation data for model selection and hyperparameter tuning.
- Reserve the test set for final evaluation.
- Ensure engineered historical features do not use future transactions.
- Report fraud prevalence alongside precision, recall, F1, PR-AUC, and other relevant metrics.
- Treat this as a proposed evaluation protocol pending team review.

## Recommendation

Adopt steps **1–323 for training, 324–378 for validation, and 379–743 for testing** as the initial chronological split for VaultWatch.

Review the observed fraud-prevalence shift with the team before finalizing the evaluation protocol.