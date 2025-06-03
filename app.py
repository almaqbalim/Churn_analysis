from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Telecommunication_icon.png/120px-Telecommunication_icon.png", width=60)
st.title("📡 Telecom Customer Churn Dashboard")
st.markdown(
    "An interactive dashboard to analyze, predict, and reduce customer churn.")


# ----------------------------------
# Load Data
# ----------------------------------


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/almaqbalim/Churn_analysis/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.dropna(inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df = df.dropna(subset=["TotalCharges"])
    return df


df = load_data()

# ----------------------------------
# Sidebar Filters
# ----------------------------------
st.sidebar.header("🔎 Filter Data")
contract_filter = st.sidebar.multiselect(
    "Contract Type", df["Contract"].unique(), default=df["Contract"].unique())
internet_filter = st.sidebar.multiselect(
    "Internet Service", df["InternetService"].unique(), default=df["InternetService"].unique())

df_filtered = df[(df["Contract"].isin(contract_filter)) &
                 (df["InternetService"].isin(internet_filter))]

# ----------------------------------
# Dashboard Header
# ----------------------------------
st.title("📊 Telecom Customer Churn Dashboard")
st.markdown("Analyze churn behavior and extract actionable insights.")

# ----------------------------------
# Key Metrics
# ----------------------------------
total_customers = df_filtered.shape[0]
churned_customers = df_filtered[df_filtered["Churn"] == "Yes"].shape[0]
churn_rate = churned_customers / total_customers * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned_customers)
col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")

# ----------------------------------
# Visualizations
# ----------------------------------

st.markdown("### 🔍 Churn Distribution")
fig1 = px.pie(df_filtered, names="Churn",
              title="Overall Churn Distribution", hole=0.4)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### 📉 Churn Rate by Contract Type")
contract_churn = df_filtered.groupby("Contract")["Churn"].value_counts(
    normalize=True).unstack().fillna(0) * 100
fig2 = px.bar(contract_churn, barmode="group",
              title="Churn Rate by Contract Type", labels={"value": "Churn Rate (%)"})
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 📈 Churn Rate by Tenure Group")
df_filtered["TenureGroup"] = pd.cut(df_filtered["tenure"], bins=[
                                    0, 6, 12, 24, 36, 72], labels=["0–6", "6–12", "12–24", "24–36", "36+"])
tenure_churn = df_filtered.groupby("TenureGroup")["Churn"].value_counts(
    normalize=True).unstack().fillna(0) * 100
fig3 = px.bar(tenure_churn, barmode="group",
              title="Churn Rate by Tenure Group", labels={"value": "Churn Rate (%)"})
st.plotly_chart(fig3, use_container_width=True)

st.markdown("### 💰 Monthly Charges vs. Churn")
fig4 = px.histogram(df_filtered, x="MonthlyCharges",
                    color="Churn", barmode="overlay", nbins=40)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("### 🌐 Internet Service vs. Churn")
internet_churn = df_filtered.groupby("InternetService")["Churn"].value_counts(
    normalize=True).unstack().fillna(0) * 100
fig5 = px.bar(internet_churn, barmode="group",
              title="Churn Rate by Internet Service", labels={"value": "Churn Rate (%)"})
st.plotly_chart(fig5, use_container_width=True)

st.markdown("### 🎁 Value-Added Services vs. Churn")
service_cols = ["OnlineBackup", "TechSupport",
                "StreamingTV", "StreamingMovies"]
for col in service_cols:
    st.markdown(f"#### {col} vs. Churn")
    svc_churn = df_filtered.groupby(col)["Churn"].value_counts(
        normalize=True).unstack().fillna(0) * 100
    fig = px.bar(svc_churn, barmode="group", title=f"Churn Rate by {col}", labels={
                 "value": "Churn Rate (%)"})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# Business Insights & Recommendations
# ----------------------------------

st.markdown("### 📌 Business Insights & Recommendations")

with st.expander("🔍 View Insights and Strategic Actions"):
    churn_monthly = contract_churn.loc["Month-to-month",
                                       "Yes"] if "Month-to-month" in contract_churn.index else 0
    churn_two_year = contract_churn.loc["Two year",
                                        "Yes"] if "Two year" in contract_churn.index else 0

    st.markdown("#### 🔎 Key Data-Driven Insights")
    st.markdown(f"""
    - **Month-to-month contracts** show a high churn rate of **{churn_monthly:.1f}%**.
    - **Two-year contracts** have significantly lower churn (**{churn_two_year:.1f}%**).
    - Churn is highest among customers with **tenure < 12 months**.
    - **Fiber optic** users show more churn than DSL or no internet.
    - Higher **monthly charges** correlate with higher churn.
    - Users with **extra services** (e.g. Tech Support, Online Backup) churn less.
    """)

    st.markdown("#### ✅ Strategic Recommendations")
    st.markdown("""
    1. 💡 **Promote Long-Term Contracts**  
       Offer incentives for 1–2 year plans to reduce churn.

    2. 🔁 **Focus on Early Retention**  
       Engage customers in their first 6–12 months with onboarding support and perks.

    3. 📉 **Optimize Pricing Strategy**  
       Create flexible plans or targeted discounts for high-bill customers.

    4. 🛠️ **Investigate Fiber Feedback**  
       Address service issues or expectations with fiber users.

    5. 🎁 **Upsell Value-Added Services**  
       Promote bundles like tech support to improve stickiness.

    6. 🧠 **Implement Predictive Retention Models**  
       Proactively identify at-risk customers for retention campaigns.
    """)

    st.info("📌 These insights adjust based on filters. Try filtering by Internet type or contract to explore different behaviors.")

st.markdown("### 🤖 Predict Customer Churn (Demo)")

with st.expander("🎯 Try the Prediction Tool"):
    demo_data = df_filtered.copy()

    # Encode categorical variables
    df_encoded = demo_data.copy()
    le_dict = {}
    for col in df_encoded.select_dtypes(include="object").columns:
        if col != "customerID":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            le_dict[col] = le

    # Drop non-model columns
    X = df_encoded.drop(
        columns=["customerID", "Churn", "TenureGroup"], errors="ignore")
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Inputs from user
    st.markdown("Enter the following to predict churn:")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
    total = st.slider("Total Charges", 0.0, 10000.0, 1500.0)
    contract_type = st.selectbox("Contract Type", df["Contract"].unique())
    internet_type = st.selectbox(
        "Internet Service", df["InternetService"].unique())

    # Prepare input row
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": le_dict["Contract"].transform([contract_type])[0],
        "InternetService": le_dict["InternetService"].transform([internet_type])[0]
    }

    # Fill missing columns with 0 if any
    model_features = X.columns
    input_df = pd.DataFrame([input_data])
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_features]  # ensure correct order

    prediction = model.predict(input_df)[0]
    prediction_label = "Churn" if prediction == 1 else "No Churn"
    st.success(f"🔮 Prediction: {prediction_label}")
