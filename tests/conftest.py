import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# make `import vaultwatch...` work without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def toy_transactions():
    """Small fake PaySim-shaped table where fraud is easy to spot.

    Fraud rows are TRANSFER/CASH_OUT that drain the sender's balance to zero,
    which is the main fraud pattern in PaySim. Kept tiny so tests run in seconds.
    """
    rng = np.random.default_rng(0)
    n_normal, n_fraud = 400, 40

    normal_amount = rng.uniform(10, 5_000, n_normal)
    normal_old = rng.uniform(5_000, 50_000, n_normal)
    normal = pd.DataFrame({
        "step": rng.integers(1, 700, n_normal),
        "type": rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], n_normal),
        "amount": normal_amount,
        "oldbalanceOrg": normal_old,
        "newbalanceOrig": normal_old - normal_amount,
        "oldbalanceDest": rng.uniform(0, 50_000, n_normal),
        "newbalanceDest": rng.uniform(0, 50_000, n_normal),
        "isFraud": 0,
    })

    fraud_amount = rng.uniform(50_000, 500_000, n_fraud)
    fraud = pd.DataFrame({
        "step": rng.integers(1, 700, n_fraud),
        "type": rng.choice(["TRANSFER", "CASH_OUT"], n_fraud),
        "amount": fraud_amount,
        "oldbalanceOrg": fraud_amount,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 1,
    })

    return pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
