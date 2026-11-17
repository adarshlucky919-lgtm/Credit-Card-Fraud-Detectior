import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detector")
st.title("Credit Card Fraud Detector")
st.caption("Random Forest model trained on the ULB credit card fraud dataset. "
           "Demo project, not a production fraud system.")

@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

model = load_model()
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

mode = st.radio("Input source", ["Demo transactions", "Upload CSV"])

df = None
if mode == "Demo transactions":
    df = pd.read_csv("demo_transactions.csv")
else:
    up = st.file_uploader("CSV with columns Time, V1 to V28, Amount", type="csv")
    if up is not None:
        df = pd.read_csv(up)

if df is not None:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    threshold = st.slider("Fraud probability threshold", 0.05, 0.95, 0.50, 0.05)

    result = df.copy()
    result["fraud_probability"] = model.predict_proba(df[FEATURES])[:, 1]
    result["flagged"] = result["fraud_probability"] >= threshold

    c1, c2 = st.columns(2)
    c1.metric("Transactions", len(result))
    c2.metric("Flagged as fraud", int(result["flagged"].sum()))

    if "Class" in result.columns:
        tp = int((result["flagged"] & (result["Class"] == 1)).sum())
        fp = int((result["flagged"] & (result["Class"] == 0)).sum())
        fn = int((~result["flagged"] & (result["Class"] == 1)).sum())
        st.write(f"Against the true labels: {tp} frauds caught, "
                 f"{fn} frauds missed, {fp} false alarms.")

    cols = ["fraud_probability", "flagged"] + (["Class"] if "Class" in result.columns else [])
    cols += [c for c in result.columns if c not in cols]
    st.dataframe(result[cols].sort_values("fraud_probability", ascending=False))