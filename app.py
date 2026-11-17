import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.main{
    background-color:#0E1117;
}
.block-container{
    padding-top:1rem;
}
.metric-card{
    background:#1E1E1E;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 0 12px rgba(0,0,0,.3);
}
.metric-title{
    color:#BBBBBB;
    font-size:16px;
}
.metric-value{
    color:white;
    font-size:34px;
    font-weight:bold;
}
div[data-testid="stMetric"]{
    background:#1E1E1E;
    padding:15px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection System")
st.caption(
    "Machine Learning Dashboard • Random Forest • Streamlit"
)

@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

model = load_model()

FEATURES = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

with st.sidebar:
    st.header("⚙ Dashboard")

    mode = st.radio(
        "Input Source",
        ["Demo Transactions","Upload CSV"]
    )

    threshold = st.slider(
        "Fraud Threshold",
        0.05,
        0.95,
        0.50,
        0.05
    )

    st.divider()

    st.subheader("📘 Project")

    st.write("""
**Model:** Random Forest

**Dataset:** ULB Credit Card Fraud Dataset

**Framework:** Streamlit

**Language:** Python
""")

    st.divider()

    st.success("Developed by")
    st.write("**Adarsh Kumar Dubey**")
    st.write("B.Tech CSE (Cloud Computing)")
    st.write("Bennett University")

df=None

if mode=="Demo Transactions":
    df=pd.read_csv("demo_transactions.csv")
else:
    uploaded=st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )
    if uploaded:
        df=pd.read_csv(uploaded)

if df is not None:

    missing=[c for c in FEATURES if c not in df.columns]

    if missing:
        st.error(f"Missing columns : {missing}")
        st.stop()

    result=df.copy()

    result["fraud_probability"]=model.predict_proba(
        result[FEATURES]
    )[:,1]

    result["flagged"]=(
        result["fraud_probability"]>=threshold
    )

    total=len(result)

    fraud_actual=0

    legit_actual=0

    if "Class" in result.columns:
        fraud_actual=int(result["Class"].sum())
        legit_actual=total-fraud_actual

    flagged=int(result["flagged"].sum())

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Transactions",total)

    c2.metric("Fraud",fraud_actual)

    c3.metric("Legitimate",legit_actual)

    c4.metric("Flagged",flagged)
    st.divider()

    left,right=st.columns([1,1])

    with left:

        if "Class" in result.columns:

            pie_data=pd.DataFrame({
                "Category":["Legitimate","Fraud"],
                "Count":[legit_actual,fraud_actual]
            })

            fig=px.pie(
                pie_data,
                values="Count",
                names="Category",
                hole=0.5,
                title="Dataset Distribution"
            )

            fig.update_layout(
                template="plotly_dark",
                height=420
            )

            st.plotly_chart(fig,use_container_width=True)

    with right:

        fig2=px.histogram(
            result,
            x="fraud_probability",
            nbins=30,
            title="Fraud Probability Distribution"
        )

        fig2.update_layout(
            template="plotly_dark",
            xaxis_title="Fraud Probability",
            yaxis_title="Transactions",
            height=420
        )

        st.plotly_chart(fig2,use_container_width=True)

    st.divider()

    if "Class" in result.columns:

        tp=int((result["flagged"] & (result["Class"]==1)).sum())
        fp=int((result["flagged"] & (result["Class"]==0)).sum())
        fn=int((~result["flagged"] & (result["Class"]==1)).sum())
        tn=int((~result["flagged"] & (result["Class"]==0)).sum())

        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        accuracy=(tp+tn)/len(result)

        st.subheader("📊 Model Performance")

        m1,m2,m3,m4=st.columns(4)

        m1.metric("Accuracy",f"{accuracy*100:.2f}%")
        m2.metric("Precision",f"{precision*100:.2f}%")
        m3.metric("Recall",f"{recall*100:.2f}%")
        m4.metric("False Alarms",fp)

        perf=pd.DataFrame({
            "Metric":[
                "Frauds Caught",
                "Frauds Missed",
                "False Alarms"
            ],
            "Count":[tp,fn,fp]
        })

        fig3=px.bar(
            perf,
            x="Metric",
            y="Count",
            text="Count",
            title="Prediction Summary"
        )

        fig3.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig3,use_container_width=True)
    st.divider()

    st.subheader("🔍 Filter Transactions")

    option = st.selectbox(
        "Select Records",
        [
            "All Transactions",
            "Fraud Only",
            "Legitimate Only"
        ]
    )

    display = result.copy()

    if option == "Fraud Only":
        display = display[display["flagged"]]

    elif option == "Legitimate Only":
        display = display[~display["flagged"]]

    display = display.sort_values(
        "fraud_probability",
        ascending=False
    )

    display["Prediction"] = display["flagged"].map(
        {
            True: "🔴 Fraud",
            False: "🟢 Legitimate"
        }
    )

    show_cols = [
        "Prediction",
        "fraud_probability"
    ]

    if "Class" in display.columns:
        show_cols.append("Class")

    show_cols += [
        c for c in display.columns
        if c not in show_cols
    ]

    st.dataframe(
        display[show_cols],
        use_container_width=True,
        height=500
    )

    csv = display.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions",
        csv,
        "fraud_predictions.csv",
        "text/csv"
    )

    st.divider()

    st.markdown(
        """
        ### 📖 About this Project

        This application predicts fraudulent credit card transactions using a
        **Random Forest Machine Learning model** trained on the
        **ULB Credit Card Fraud Detection Dataset**.

        #### Technologies Used

        - Python
        - Streamlit
        - Pandas
        - Scikit-Learn
        - Plotly

        #### Features

        ✅ Fraud Probability

        ✅ Interactive Dashboard

        ✅ CSV Upload

        ✅ Adjustable Threshold

        ✅ Prediction Download

        ---
        """
    )

    st.markdown(
        """
        <center>

        <h4>👨‍💻 Developed by</h4>

        <h3>Adarsh Kumar Dubey</h3>

        B.Tech CSE 

        Bennett University

        </center>
        """,
        unsafe_allow_html=True
    )
