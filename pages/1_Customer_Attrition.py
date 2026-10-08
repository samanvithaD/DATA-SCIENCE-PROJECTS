import numpy as np, pandas as pd, streamlit as st

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve)

from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Attrition", layout="wide")

# Reduce white space at the top of the Streamlit page
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Customer Attrition Prediction")


def add_features(d):

    d = d.copy()

    d["avg_call_length"] = np.where(d.DayCalls > 0, d.DayMins / d.DayCalls, 0)

    d["overage_ratio"] = d.OverageFee / d.MonthlyCharge

    d["charge_per_day_min"] = np.where(d.DayMins > 0, d.MonthlyCharge / d.DayMins, 0)

    d["high_cs_calls"] = (d.CustServCalls >= 4).astype(int)

    return d


@st.cache_resource(show_spinner="Training models...")
def train():

    df = pd.read_csv("data/telecom_churn.csv")

    X, y = add_features(df.drop(columns="Churn")), df["Churn"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    ratio = (ytr == 0).sum() / (ytr == 1).sum()

    models = {

        "Logistic Regression": Pipeline([
            ("s", StandardScaler()),
            ("c", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=42
        ),
    }

    rows, probs = {}, {}

    for n, m in models.items():

        m.fit(Xtr, ytr)

        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[n] = p

        rows[n] = {
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred),
            "Recall": recall_score(yte, pred),
            "F1": f1_score(yte, pred),
            "ROC-AUC": roc_auc_score(yte, p),
            "PR-AUC": average_precision_score(yte, p)
        }

    return df, models, pd.DataFrame(rows).T.round(3), probs, yte, list(X.columns)


df, models, results, probs, yte, cols = train()

tab1, tab2, tab3 = st.tabs([
    "Model comparison",
    "Churn drivers",
    "Score a customer"
])


with tab1:

    st.write(
        f"{len(df):,} customers, churn rate {df.Churn.mean():.1%}. "
        f"Predicting 'nobody churns' would score {1 - yte.mean():.1%} accuracy, "
        f"so recall and PR-AUC matter more."
    )

    st.dataframe(results, use_container_width=True)

    # Reduced graph size
    fig, ax = plt.subplots(figsize=(4.5, 3))

    for n, p in probs.items():

        fpr, tpr, _ = roc_curve(yte, p)

        ax.plot(fpr, tpr, label=n)

    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()

    st.pyplot(fig)


with tab2:

    a, b = st.columns(2)

    rate = df.groupby("CustServCalls").Churn.mean() * 100

    a.write("Churn rate (%) by customer-service calls")

    a.bar_chart(rate)

    b.write("Churn rate (%) by contract renewal")

    b.bar_chart(df.groupby("ContractRenewal").Churn.mean() * 100)

    imp = pd.Series(
        models["XGBoost"].feature_importances_,
        index=cols
    ).sort_values()

    st.write("XGBoost feature importance")

    st.bar_chart(imp)


with tab3:

    st.write(
        "Enter customer details. The XGBoost model returns a churn probability and a risk tier."
    )

    c1, c2, c3 = st.columns(3)

    w = c1.number_input("Account weeks", 1, 250, 100)

    ren = c1.selectbox(
        "Contract renewed?",
        [1, 0],
        format_func=lambda v: "Yes" if v else "No"
    )

    plan = c1.selectbox(
        "Data plan?",
        [0, 1],
        format_func=lambda v: "Yes" if v else "No"
    )

    du = c1.number_input("Data usage (GB)", 0.0, 6.0, 0.0)

    cs = c2.number_input("Customer-service calls", 0, 10, 1)

    dm = c2.number_input("Day minutes", 0.0, 400.0, 180.0)

    dc = c2.number_input("Day calls", 0, 170, 100)

    mc = c3.number_input("Monthly charge", 10.0, 120.0, 56.0)

    of = c3.number_input("Overage fee", 0.0, 20.0, 10.0)

    rm = c3.number_input("Roaming minutes", 0.0, 20.0, 10.0)

    if st.button("Predict churn risk"):

        row = pd.DataFrame([{
            "AccountWeeks": w,
            "ContractRenewal": ren,
            "DataPlan": plan,
            "DataUsage": du,
            "CustServCalls": cs,
            "DayMins": dm,
            "DayCalls": dc,
            "MonthlyCharge": mc,
            "OverageFee": of,
            "RoamMins": rm
        }])

        p = float(
            models["XGBoost"]
            .predict_proba(add_features(row)[cols])[0, 1]
        )

        tier = "Low" if p < 0.3 else "Medium" if p < 0.6 else "High"

        st.metric(
            "Churn probability",
            f"{p:.1%}",
            f"{tier} risk",
            delta_color="off"
        )

        if cs >= 3:
            st.warning(
                "3+ service calls: escalate to a retention specialist before the 4th call."
            )

        if ren == 0:
            st.warning(
                "Contract not renewed: offer a renewal incentive."
            )