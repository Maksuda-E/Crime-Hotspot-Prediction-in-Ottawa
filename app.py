import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page configuration
st.set_page_config(page_title="Crime Hotspot Prediction - Ottawa", layout="wide")

# App title
st.title("Crime Hotspot Prediction - Ottawa")
st.write("Machine Learning application to analyze crime patterns and identify hotspots")

# Sidebar
st.sidebar.header("Controls")

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload crime dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Check required columns
    required_columns = ["X", "Y", "Sector", "Division", "Neighbourh", "Weekday"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Data cleaning
    df = df.dropna(subset=["X", "Y"])

    # Clean Sector column
    df["Sector"] = df["Sector"].astype(str)
    df["Sector"] = df["Sector"].str.replace("SECTOR", "", regex=False)
    df["Sector"] = pd.to_numeric(df["Sector"], errors="coerce")
    df = df.dropna(subset=["Sector"])
    df = df[df["Sector"] != 408]

    # Encode categorical columns
    categorical_cols = ["Division", "Neighbourh", "Weekday"]
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    # Create target variable
    df["Hotspot"] = (df.groupby("Neighbourh")["Neighbourh"].transform("count") > 50).astype(int)

    # Feature selection
    features = ["X", "Y", "Sector", "Division", "Neighbourh", "Weekday"]
    X = df[features]
    y = df["Hotspot"]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Map visualization
    st.subheader("Crime Locations Map")
    st.map(df[["Y", "X"]].rename(columns={"Y": "lat", "X": "lon"}))

    # Feature importance
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(importance_df["Feature"], importance_df["Importance"])
    ax2.invert_yaxis()
    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to begin.")
