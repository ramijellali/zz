import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data and models
@st.cache_data
def load_data():
    data_path_1 = "churn-bigml-20.csv"
    data_path_2 = "churn-bigml-80.csv"
    df1 = pd.read_csv(data_path_1)
    df2 = pd.read_csv(data_path_2)
    data = pd.concat([df1, df2], ignore_index=True)

    return data

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# App Layout
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.markdown(
    "<style>h1, h2, h3 { color: #4B8BBE; text-align: center; }</style>",
    unsafe_allow_html=True
)

st.title("📊 Customer Churn Prediction Dashboard")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose Section", [
        "Overview",
        "Data Visualization",
        "Feature Correlations",
        "Boxplots",
        "PCA Visualization",
        "Model Performance",
        "Model Prediction"
    ]
)

# Load the data
if "data" not in st.session_state:
    st.session_state.data = load_data()
data = st.session_state.data

# Overview Section
if option == "Overview":
    st.header("Business Understanding & Data Overview")

    st.markdown(
        """
        <p style='font-size:16px; text-align:justify;'>
        Customer churn, also known as customer retention, customer turnover, or customer defection, is the loss of clients or customers.

Telephone service companies, Internet service providers, pay TV companies, insurance firms, and alarm monitoring services, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.

Companies usually make a distinction between voluntary churn and involuntary churn. Voluntary churn occurs due to a decision by the customer to switch to another company or service provider, involuntary churn occurs due to circumstances such as a customer's relocation to a long-term care facility, death, or the relocation to a distant location. In most applications, involuntary reasons for churn are excluded from the analytical models. Analysts tend to concentrate on voluntary churn, because it typically occurs due to factors of the company-customer relationship which companies control, such as how billing interactions are handled or how after-sales help is provided.

predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn. Since these models generate a small prioritized list of potential defectors, they are effective at focusing customer retention marketing programs on the subset of the customer base who are most vulnerable to churn.
        </p>
        <h4 style='color: #4B8BBE;'>Key Metrics:</h4>
        <ul style='font-size:16px;'>
        <li>Churn Distribution</li>
        <li>Feature Correlations</li>
        <li>Interactive Visualizations</li>
        <li>Model Predictions</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Dataset Overview")
    st.write("First 10 rows of the dataset:")
    st.dataframe(data.head(10))

    st.subheader("Statistical Summary")
    st.write(data.describe())

    st.subheader("Data Shape")
    st.write(f"Number of rows: **{data.shape[0]}**, Number of columns: **{data.shape[1]}**")

# Data Visualization Section
elif option == "Data Visualization":
    st.header("📊 Data Visualization")

    st.subheader("Distribution of Churn")
    churn_distribution = data["Churn"].value_counts().reset_index()
    churn_distribution.columns = ["Churn", "Count"]
    fig_pie = px.pie(
        churn_distribution,
        names="Churn",
        values="Count",
        title="Churn Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie)
    st.markdown("**Interpretation:** This chart shows the proportion of customers who churned vs. retained.")

    st.subheader("Histogram of Features")
    selected_feature = st.selectbox("Choose a feature to plot:", data.columns)
    if selected_feature:
        fig_hist = px.histogram(
            data,
            x=selected_feature,
            color="Churn",
            barmode="overlay",
            title=f"Histogram of {selected_feature} by Churn",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_hist)
        st.markdown("**Interpretation:** Histogram helps identify the distribution of a specific feature and how it differs between churned and retained customers.")

# Feature Correlations Section
elif option == "Feature Correlations":
    st.header("🔗 Feature Correlations with Churn")

    # Check if 'Churn' column exists and ensure it's numeric
    if 'Churn' not in data.columns:
        st.error("The target variable 'Churn' is missing from the dataset!")
    else:
        data['Churn'] = pd.to_numeric(data['Churn'], errors='coerce')

        # Select only numeric columns for correlation calculation
        numeric_data = data.select_dtypes(include=[np.number])
        correlations = numeric_data.corr()['Churn'].sort_values(ascending=False)
        corr_df = correlations.reset_index()
        corr_df.columns = ['Feature', 'Correlation']

        # Plot Correlations
        fig_corr = px.bar(
            corr_df,
            x="Feature",
            y="Correlation",
            title="Correlation of Features with Churn",
            text="Correlation",
            color="Correlation",
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig_corr.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_corr.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_corr)

        st.markdown("**Interpretation:** Positive or negative correlations indicate how strongly a feature impacts churn.")

# Boxplots Section
elif option == "Boxplots":
    st.header("📦 Boxplots of Features")
    selected_feature = st.selectbox("Select a feature for boxplot:", ['Total day minutes', 'Customer service calls'])

    fig_box = px.box(
        data,
        x="Churn",
        y=selected_feature,
        color="Churn",
        title=f"Distribution of {selected_feature} by Churn",
        color_discrete_sequence=["red", "blue"]
    )
    st.plotly_chart(fig_box)
    st.markdown("**Interpretation:** Boxplots show the spread and outliers of numeric features for churned and retained customers.")

# PCA Visualization Section
elif option == "PCA Visualization":
    st.header("🔍 PCA Visualization")
    pca = PCA(n_components=2)
    X_pca_visual = pca.fit_transform(StandardScaler().fit_transform(data.select_dtypes(include=[np.number])))
    pca_df = pd.DataFrame(X_pca_visual, columns=["PCA Component 1", "PCA Component 2"])
    pca_df["Churn"] = data["Churn"]

    fig_pca = px.scatter(
        pca_df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Churn",
        title="PCA Scatter Plot by Churn",
        color_discrete_sequence=["blue", "red"]
    )
    st.plotly_chart(fig_pca)
    st.markdown("**Interpretation:** PCA reduces features to two components for visualization, showing clusters of churned and retained customers.")

# Model Performance Section
elif option == "Model Performance":
    st.header("🏆 Model Performance Comparison")

    metrics_df = pd.read_csv("metrics.csv")

    st.subheader("Accuracy Comparison")
    fig_acc = px.bar(
        metrics_df,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison",
        labels={"Accuracy": "Accuracy Score"},
        color="Accuracy",
        color_continuous_scale=px.colors.sequential.Teal
    )
    st.plotly_chart(fig_acc)

    st.subheader("ROC AUC Comparison")
    fig_roc = px.line(
        metrics_df,
        x="Model",
        y="ROC AUC",
        title="ROC AUC Comparison",
        markers=True
    )
    st.plotly_chart(fig_roc)
    st.markdown("**Interpretation:** Higher ROC AUC values indicate better model performance.")

# Model Prediction Section
elif option == "Model Prediction":
    st.header("🔮 Make a Prediction")
    models = {
        "Random Forest": "models/Random_Forest.pkl",
        "KNN": "models/KNN.pkl",
        "Decision Tree": "models/Decision_Tree.pkl",
        "SVM (RBF)": "models/SVM_(RBF).pkl",
        "Logistic Regression": "models/Logistic_Regression.pkl",
        "Gradient Boosting": "models/Gradient_Boosting.pkl",
        "XGBoost": "models/XGBoost.pkl"
    }

    selected_model = st.selectbox("Choose a Model", list(models.keys()))
    model_path = models[selected_model]
    model = load_model(model_path)

    st.subheader("Input Customer Features")
    input_data = {feature: st.number_input(f"{feature}", value=0.0) for feature in model.feature_names_in_}
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Results")
    st.write(f"Prediction: **{'Churn' if prediction[0] == 1 else 'Not Churn'}**")
    st.write(f"Probability of Churn: **{prediction_proba[0][1] * 100:.2f}%**")

st.sidebar.markdown("---")
st.sidebar.markdown("**Created for Customer Churn Analysis Project**")
