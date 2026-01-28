import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

    # Required columns check
    required_columns = ["X", "Y", "Sector", "Division", "Neighbourh", "Weekday"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # Drop missing coordinates
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
    df["Hotspot"] = (
        df.groupby("Neighbourh")["Neighbourh"].transform("count") > 50
    ).astype(int)

    # Features and target
    features = ["X", "Y", "Sector", "Division", "Neighbourh", "Weekday"]
    X = df[features]
    y = df["Hotspot"]

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

    # Map visualization
    st.subheader("Crime Locations Map")

    # Make a safe map dataframe
    df_map = df[["Y", "X"]].copy()

    # Convert to numeric
    df_map["Y"] = pd.to_numeric(df_map["Y"], errors="coerce")
    df_map["X"] = pd.to_numeric(df_map["X"], errors="coerce")

    # Drop missing
    df_map = df_map.dropna(subset=["Y", "X"])

    # Rename for streamlit
    df_map = df_map.rename(columns={"Y": "lat", "X": "lon"})

    # Debug info
    st.write("Rows available for map:", len(df_map))
    if len(df_map) > 0:
        st.write(
            "Latitude range:",
            float(df_map["lat"].min()),
            "to",
            float(df_map["lat"].max())
        )
        st.write(
            "Longitude range:",
            float(df_map["lon"].min()),
            "to",
            float(df_map["lon"].max())
        )
        st.dataframe(df_map.head())

    # Validate lat lon
    df_map_valid = df_map[
        (df_map["lat"].between(-90, 90)) &
        (df_map["lon"].between(-180, 180))
    ].copy()

    st.write("Rows with valid lat lon:", len(df_map_valid))

    if len(df_map_valid) == 0:
        st.error(
            "No valid latitude and longitude values found. "
            "X and Y may be projected coordinates, not lat lon."
        )
    else:
        st.map(df_map_valid)

else:
    st.info("Please upload a CSV file to begin.")
