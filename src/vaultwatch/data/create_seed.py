from pathlib import Path
import pandas as pd

raw_data_path = Path("data/raw/PS_20174392719_1491204439457_log.csv")
seed_path = Path("data/seed/paysim_sample.parquet")
sample_size = 200000

df = pd.read_csv(raw_data_path)

fraud_rate = df.isFraud.mean()
n_fraud = int(sample_size * fraud_rate)
n_non_fraud = sample_size - n_fraud

fraud_df = df[df.isFraud == 1].sample(n=n_fraud, random_state=42)
# selects n_fraud rows from the df

non_fraud_df = df[df.isFraud == 0].sample(n=n_non_fraud, random_state=42)
# selects n_non_fraud rows from the df

seed = pd.concat([fraud_df, non_fraud_df], ignore_index=True)
# concatenates the two dataframes into a single dataframe

seed = seed.sample(frac=1, random_state=42).reset_index(drop=True)
# shuffles the rows of the dataframe and resets the index
# frac specifies the fraction of rows to return in the random sample, so frac=1 means return all rows

seed_path.parent.mkdir(parents=True, exist_ok=True)
seed.to_parquet(seed_path, index=False)

# Verify the result
print("Original dataset:", df.shape)
print("Seed dataset:", seed.shape)
print("\nOriginal fraud rate:", fraud_rate)
print("Seed fraud rate:", seed["isFraud"].mean())
print("\nSeed fraud counts:")
print(seed["isFraud"].value_counts())
print("\nSaved to:", seed_path)
