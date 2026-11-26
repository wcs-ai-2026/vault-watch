import streamlit as st

# Display the dashboard heading and identify the data as fake.
st.title("VaultWatch")
st.write("Live transaction feed")
st.caption("Demo data: these transactions and risk scores are fake.")


# Define three sample alerts to simulate incoming transactions.
alerts = [
    {"txn_id": "T001", "type": "PAYMENT", "amount": 50.0, "risk_score": 0.05, "flagged": False},
    {"txn_id": "T002", "type": "TRANSFER", "amount": 181000.0, "risk_score": 0.92, "flagged": True},
    {"txn_id": "T003", "type": "CASH_OUT", "amount": 8000.0, "risk_score": 0.85, "flagged": True},
]


# Preserve the transaction feed between updates and provide a flagged-only filter.
if "feed" not in st.session_state:
    st.session_state.feed = []

show_flagged = st.checkbox("Show only flagged transactions")


# Refresh this section automatically every second while the page is open.
@st.fragment(run_every="1s")
def live_feed():

    # Cycle through the sample alerts, copying each one and assigning a unique ID.
    number = st.session_state.get("txn_number", 0)
    alert = alerts[number % len(alerts)].copy()
    alert["txn_id"] = f"T{number + 1:06d}"
    st.session_state.txn_number = number + 1

    # Add the new alert and keep only the latest 100 transactions.
    st.session_state.feed.append(alert)
    st.session_state.feed = st.session_state.feed[-100:]
    feed = st.session_state.feed

    # Display counts for all transactions and flagged transactions in the retained feed.
    total_col, flagged_col = st.columns(2)
    total_col.metric("Transactions in view", len(feed))
    flagged_col.metric("Flagged in view", sum(item["flagged"] for item in feed))

    # Display newest transactions first, applying the flagged-only filter when checked.
    visible = [item for item in reversed(feed) if not show_flagged or item["flagged"]]
    st.dataframe(visible)


# Display the live feed when the page first loads.
live_feed()