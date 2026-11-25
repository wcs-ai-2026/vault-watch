import pandas as pd
import numpy as np


TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]


def make_features(txns: pd.DataFrame) -> pd.DataFrame:

    features = pd.DataFrame(index=txns.index)

    # Transaction ID (metadata, not a model feature)
    if "txn_id" in txns.columns:
        features["txn_id"] = txns["txn_id"]
    else:
        features["txn_id"] = [
            f"seed-{row_number:06d}"
            for row_number in txns.index
        ]

    # Transaction amount features
    features["amount"] = txns["amount"]
    features["log_amount"] = np.log1p(txns["amount"])

    # Original balance features
    features["oldbalanceOrg"] = txns["oldbalanceOrg"]
    features["newbalanceOrig"] = txns["newbalanceOrig"]
    features["oldbalanceDest"] = txns["oldbalanceDest"]
    features["newbalanceDest"] = txns["newbalanceDest"]

    # Balance change features
    features["orig_balance_change"] = (
        txns["newbalanceOrig"] - txns["oldbalanceOrg"]
    )

    features["dest_balance_change"] = (
        txns["newbalanceDest"] - txns["oldbalanceDest"]
    )

    # Detect completely drained accounts
    features["balance_drained"] = (
        (txns["oldbalanceOrg"] > 0) &
        (txns["newbalanceOrig"] == 0)
    ).astype(int)

    # Replace zero denominators to avoid division by zero
    safe_balance = txns["oldbalanceOrg"].replace(0, np.nan)

    # Proportion of original balance removed
    features["orig_balance_depletion_ratio"] = (
        (txns["oldbalanceOrg"] - txns["newbalanceOrig"]) /
        safe_balance
    ).fillna(0)

    # Transaction amount relative to original balance
    features["amount_to_balance_ratio"] = (
        txns["amount"] / safe_balance
    ).fillna(0)

    # Detect accounts with an initial balance of zero
    features["orig_balance_zero"] = (
        txns["oldbalanceOrg"] == 0
    ).astype(int)

    # PaySim does not provide balance information for merchants
    features["dest_balance_available"] = (
        ~txns["nameDest"].str.startswith("M")
    ).astype(int)

    # Temporal features
    features["hour_of_day"] = txns["step"] % 24

    # Cyclical encoding of time
    features["hour_sin"] = np.sin(
        2 * np.pi * features["hour_of_day"] / 24
    )

    features["hour_cos"] = np.cos(
        2 * np.pi * features["hour_of_day"] / 24
    )

    # One-hot encoding of transaction types
    for transaction_type in TRANSACTION_TYPES:
        features[f"type_{transaction_type}"] = (
            txns["type"] == transaction_type
        ).astype(int)

    return features

def main():

    # Load the original seed dataset
    txns = pd.read_parquet("data/seed/paysim_sample.parquet")

    # Generate features
    features = make_features(txns)

    # Inspect the generated features
    print("Original dataset shape:", txns.shape)
    print("Feature dataset shape:", features.shape)

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nFirst five transactions:")
    print(features.head())

    # Check for missing and infinite values
    print("\nMissing values:", features.isna().sum().sum())

    numeric_features = features.select_dtypes(include=[np.number])

    print(
        "Infinite values:",
        np.isinf(numeric_features.to_numpy()).sum()
    )

    # Save the engineered features
    features.to_parquet(
        "data/seed/paysim_features.parquet",
        index=False
    )

    print("\nSeed features saved successfully.")


if __name__ == "__main__":
    main()