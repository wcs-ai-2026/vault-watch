import pandas as pd
from pathlib import Path


RAW_PATH = Path(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = Path("data/processed")

TRAIN_END = 323
VAL_END = 378


def split_transactions(txns: pd.DataFrame):

    # Remove PaySim's existing fraud detection flag
    txns = txns.drop(columns=["isFlaggedFraud"])

    # Assign stable transaction IDs before sorting
    # We use a paysim prefix to avoid collisions wuth the seed sample, which uses the same index values
    txns["txn_id"] = [
        f"paysim-{i:07d}" for i in txns.index
    ]

    # Sort transactions chronologically
    txns = txns.sort_values("step", kind="stable")

    # Create chronological splits
    train = txns.loc[
        txns["step"] <= TRAIN_END
    ]

    val = txns.loc[
        (txns["step"] > TRAIN_END) &
        (txns["step"] <= VAL_END)
    ]

    test = txns.loc[
        txns["step"] > VAL_END
    ]

    return train, val, test


def main():

    print("Loading full PaySim dataset...")
    txns = pd.read_csv(RAW_PATH)

    print("Original dataset shape:", txns.shape)

    train, val, test = split_transactions(txns)

    # Verify that no transactions were lost
    assert len(train) + len(val) + len(test) == len(txns)

    # Verify chronological boundaries
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    for name, split in splits.items():

        # Verify that prohibited column was removed
        assert "isFlaggedFraud" not in split.columns

        # Verify unique transaction IDs
        assert split["txn_id"].is_unique

        fraud_count = split["isFraud"].sum()
        fraud_rate = split["isFraud"].mean() * 100

        print(f"\n{name.upper()} SPLIT")
        print(f"Transactions: {len(split):,}")
        print(f"Fraud transactions: {fraud_count:,}")
        print(f"Fraud rate: {fraud_rate:.4f}%")
        print(
            f"Steps: {split['step'].min()} "
            f"- {split['step'].max()}"
        )

        output_path = OUTPUT_DIR / f"{name}.parquet"

        split.to_parquet(output_path, index=False)

        print(f"Saved to {output_path}")

    print("\nAll splits generated successfully.")


if __name__ == "__main__":
    main()