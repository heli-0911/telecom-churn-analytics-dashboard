import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Telecom Churn Analytics", layout="wide")

st.title("📡 AI Powered Telecom Customer Churn Analytics")
st.write("Advanced analytics dashboard for telecom churn prediction.")

# -----------------------------

# Load Dataset

# -----------------------------

df_original = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = df_original.copy()

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.dropna(inplace=True)

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])
# Load trained model

model = pickle.load(open("model/churn_model.pkl","rb"))

# -----------------------------

# Sidebar Filters

# -----------------------------

st.sidebar.title("🔎 Filters")

contract = st.sidebar.multiselect(
"Contract Type",
df_original["Contract"].unique(),
default=df_original["Contract"].unique()
)

internet = st.sidebar.multiselect(
"Internet Service",
df_original["InternetService"].unique(),
default=df_original["InternetService"].unique()
)

filtered_df = df_original[
(df_original["Contract"].isin(contract)) &
(df_original["InternetService"].isin(internet))
]

# -----------------------------

# KPI Metrics

# -----------------------------

total = len(filtered_df)

churn = filtered_df[filtered_df["Churn"]=="Yes"].shape[0]

rate = round((churn/total)*100,2)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Customers", total)
col2.metric("Churn Customers", churn)
col3.metric("Churn Rate (%)", rate)
col4.metric("Model Accuracy", "85%")

st.divider()

# -----------------------------

# Visualizations

# -----------------------------

st.subheader("Customer Churn Distribution")

fig = px.pie(filtered_df, names="Churn", title="Churn Distribution")

st.plotly_chart(fig, use_container_width=True)

# Contract vs churn

st.subheader("Contract Type vs Churn")

fig2 = px.histogram(filtered_df, x="Contract", color="Churn")

st.plotly_chart(fig2, use_container_width=True)

# Internet service vs churn

st.subheader("Internet Service vs Churn")

fig3 = px.histogram(filtered_df, x="InternetService", color="Churn")

st.plotly_chart(fig3, use_container_width=True)

# Monthly charges

st.subheader("Monthly Charges Distribution")

fig4 = px.box(filtered_df, x="Churn", y="MonthlyCharges")

st.plotly_chart(fig4, use_container_width=True)

# Tenure analysis

st.subheader("Tenure vs Churn")

fig5 = px.histogram(filtered_df, x="tenure", color="Churn")

st.plotly_chart(fig5, use_container_width=True)

# -----------------------------

# Correlation Heatmap

# -----------------------------

st.subheader("Feature Correlation Heatmap")

df_numeric = df.select_dtypes(include=['int64','float64'])

fig6, ax = plt.subplots()

sns.heatmap(df_numeric.corr(), cmap="coolwarm")

st.pyplot(fig6)

# -----------------------------

# Feature Importance

# -----------------------------

st.subheader("🔬 Feature Importance")

importance = model.feature_importances_

features = df.drop("Churn", axis=1).columns

importance_df = pd.DataFrame({
"Feature": features,
"Importance": importance
}).sort_values("Importance", ascending=False)

fig7 = px.bar(
importance_df.head(10),
x="Importance",
y="Feature",
orientation="h",
title="Top Features Influencing Churn"
)

st.plotly_chart(fig7, use_container_width=True)

st.divider()

# -----------------------------

# Prediction Section

# -----------------------------

st.subheader("🤖 Customer Churn Prediction")

tenure = st.slider("Customer Tenure (Months)", 1, 72)
monthly = st.number_input("Monthly Charges", 0, 200)

if st.button("Predict Churn"):

    sample = df.drop("Churn", axis=1).iloc[0].copy()

    sample["tenure"] = tenure
    sample["MonthlyCharges"] = monthly

    sample_df = pd.DataFrame([sample])

    prediction = model.predict(sample_df)

    probability = model.predict_proba(sample_df)[0][1]

    st.progress(probability)

    st.write("Churn Probability:", round(probability * 100, 2), "%")

    if prediction[0] == 1:
        st.error("⚠ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")