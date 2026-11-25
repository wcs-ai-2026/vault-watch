# VaultWatch Feature Dictionary

## Version 0 — VW-19

The initial feature pipeline transforms raw PaySim transactions into 21 numerical features and one transaction identifier. Features are generated using `make_features()` in `src/vaultwatch/features/features.py`.

## Transaction Amount Features

| Feature | Description |
|---|---|
| `amount` | Original transaction amount. |
| `log_amount` | Natural logarithm of 1 + amount, reducing the effect of large transaction values. |
| `amount_to_balance_ratio` | Transaction amount divided by the initial origin balance. Defaults to zero when the denominator is zero. |

## Balance Features

| Feature | Description |
|---|---|
| `oldbalanceOrg` | Origin account balance before the transaction. |
| `newbalanceOrig` | Origin account balance after the transaction. |
| `oldbalanceDest` | Destination account balance before the transaction. |
| `newbalanceDest` | Destination account balance after the transaction. |
| `orig_balance_change` | New origin balance minus old origin balance. |
| `dest_balance_change` | New destination balance minus old destination balance. |
| `balance_drained` | Binary indicator equal to 1 when a positive origin balance becomes zero. |
| `orig_balance_depletion_ratio` | Proportion of the initial origin balance removed. Negative values indicate an increase in balance. Defaults to zero when the initial balance is zero. |
| `orig_balance_zero` | Binary indicator for an initial origin balance of zero. |
| `dest_balance_available` | Binary indicator based on whether the destination is a non-merchant account. Merchant balance information is unavailable in PaySim. |

## Temporal Features

| Feature | Description |
|---|---|
| `hour_of_day` | Simulated hour calculated as step modulo 24. |
| `hour_sin` | Sine transformation of the simulated hour. |
| `hour_cos` | Cosine transformation of the simulated hour. |

Cyclical encoding preserves the proximity between hours at the beginning and end of a daily cycle.

## Transaction Type Features

| Feature | Description |
|---|---|
| `type_CASH_IN` | Binary indicator for cash-in transactions. |
| `type_CASH_OUT` | Binary indicator for cash-out transactions. |
| `type_DEBIT` | Binary indicator for debit transactions. |
| `type_PAYMENT` | Binary indicator for payments. |
| `type_TRANSFER` | Binary indicator for transfers. |

All five columns are generated consistently, even when a transaction category is absent from the input.

## Transaction Identifier

`txn_id` is used to align transactions across feature datasets, model predictions and evaluation results. It must not be included as a model predictor.

Existing transaction IDs are preserved. For the original seed dataset, IDs follow the `seed-000000` convention using the original row index. Full datasets and live transactions should supply stable transaction IDs.

## Data Leakage Prevention

- `isFraud` is excluded from the feature dataset and used separately as the target label.
- `isFlaggedFraud` is excluded because it contains information from PaySim's existing fraud detection rules.
- Raw account identifiers are excluded from model inputs.
- Chronological split boundaries established in VW-18 remain unchanged.
- Post-transaction balances are included because VaultWatch monitors completed transactions.
- Balance-derived features require further review for simulator-specific artifacts and potential leakage.

## Generated Dataset

- Input: `data/seed/paysim_sample.parquet`
- Output: `data/seed/paysim_features.parquet`
- Transactions: 200,000
- Numerical features: 21
- Total columns: 22
- Missing values: 0
- Infinite values: 0

## Testing

The feature pipeline is tested using `tests/test_features.py`. Tests cover feature dimensions, balance calculations, safe division, transaction type encoding, temporal features, transaction identifiers and exclusion of prohibited columns.
