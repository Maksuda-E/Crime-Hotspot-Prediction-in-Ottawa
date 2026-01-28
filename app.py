import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Libraries for interactive maps
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

# Machine learning libraries
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Configure Streamlit page
st.set_page_config(page_title="Crime Hotspot Prediction Ottawa", layout="wide")

# App title and description
st.title("Crime Hotspot Prediction Ottawa")
st.write(
    "This application analyzes crime data, visualizes patterns, "
    "and identifies hotspot areas using interactive maps and a simple machine learning model."
)


# Function to convert Web Mercator coordinates to latitude and longitude
def web_mercator_to_wgs84(x, y):
    """
    Converts X and Y Web Mercator coordinates into latitude and longitude.
    """
    earth_radius = 6378137.0
    lon = (x / earth_radius) * (180.0 / math.pi)
    lat = (2.0 * math.atan(math.exp(y / earth_radius)) - (math.pi / 2.0)) * (180.0 / math.pi)
    return lat, lon


# Load and preprocess the dataset
@st.cache_data
def load_data(uploaded_file):
    """
    Loads the CSV file and prepares data for visualization and modeling.
    """
    df = pd.read_csv(uploaded_file, low_memory=False)

    # Convert coordinate columns to numeric
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    # Convert coordinates to latitude and longitude
    latitudes = []
    longitudes = []

    for x_val, y_val in zip(df["X"], df["Y"]):
        lat, lon = web_mercator_to_wgs84(x_val, y_val)
        latitudes.append(lat)
        longitudes.append(lon)

    df["lat"] = latitudes
    df["lon"] = longitudes

    # Keep only valid latitude and longitude values
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]

    # Extract hour from time column if present
    if "Occur_Time" in df.columns:
        df["Occur_Time"] = df["Occur_Time"].astype(str)
        df["OccurHour"] = pd.to_numeric(df["Occur_Time"].str.slice(0, 2), errors="coerce")

    # Ensure Year is numeric
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    return df


# Sidebar controls
st.sidebar.header("Controls")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload crime CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload the crime dataset to begin.")
    st.stop()

# Load dataset
df = load_data(uploaded_file)

# Check required columns
required_columns = [
    "Year", "Weekday", "OffSummary", "PrimViolat",
    "Neighbourh", "Sector", "Division", "lat", "lon"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# Sidebar filters
years = sorted(df["Year"].dropna().unique().tolist())
divisions = sorted(df["Division"].dropna().astype(str).unique().tolist())
off_summaries = sorted(df["OffSummary"].dropna().astype(str).unique().tolist())

selected_years = st.sidebar.multiselect("Year", years, default=years)
selected_divisions = st.sidebar.multiselect("Division", divisions, default=divisions)
selected_off_summaries = st.sidebar.multiselect("Offence Summary", off_summaries, default=off_summaries)

max_map_points = st.sidebar.slider("Maximum map points", 2000, 50000, 15000, 1000)

# Apply filters
df_filtered = df[
    df["Year"].isin(selected_years)
    & df["Division"].astype(str).isin(selected_divisions)
    & df["OffSummary"].astype(str).isin(selected_off_summaries)
].copy()

# Create tabs
tab_overview, tab_map, tab_trends, tab_model = st.tabs(
    ["Overview", "Interactive Map", "Trends", "Model"]
)


# Overview tab
with tab_overview:
    col1, col2, col3 = st.columns(3)

    col1.metric("Total incidents", f"{len(df_filtered):,}")
    col2.metric("Unique neighborhoods", df_filtered["Neighbourh"].nunique())
    col3.metric("Unique offence types", df_filtered["PrimViolat"].nunique())

    st.subheader("Data preview")
    st.dataframe(
        df_filtered[
            ["Year", "Weekday", "Division", "Neighbourh",
             "OffSummary", "PrimViolat", "lat", "lon"]
        ].head(30),
        use_container_width=True
    )


# Interactive map tab
with tab_map:
    st.subheader("Crime hotspot visualization")

    map_type = st.radio("Select map type", ["Cluster map", "Heatmap"], horizontal=True)

    df_map = df_filtered[["lat", "lon", "Neighbourh", "Division", "OffSummary", "PrimViolat"]].dropna()

    if len(df_map) > max_map_points:
        df_map = df_map.sample(n=max_map_points, random_state=42)

    center_lat = df_map["lat"].mean()
    center_lon = df_map["lon"].mean()

    # Create base map
    crime_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    if map_type == "Cluster map":
        cluster = MarkerCluster().add_to(crime_map)

        for row in df_map.itertuples(index=False):
            popup_text = (
                f"Neighbourhood: {row.Neighbourh}<br>"
                f"Division: {row.Division}<br>"
                f"Offence: {row.OffSummary}<br>"
                f"Violation: {row.PrimViolat}"
            )

            folium.Marker(
                location=[row.lat, row.lon],
                popup=popup_text
            ).add_to(cluster)

    if map_type == "Heatmap":
        HeatMap(df_map[["lat", "lon"]].values.tolist(), radius=10).add_to(crime_map)

    st_folium(crime_map, width=1100, height=650)

    st.subheader("Top hotspot neighborhoods")
    hotspot_table = df_filtered["Neighbourh"].value_counts().reset_index()
    hotspot_table.columns = ["Neighbourhood", "Incidents"]
    st.dataframe(hotspot_table.head(20), use_container_width=True)


# Trends tab
with tab_trends:
    st.subheader("Incidents by division")
    division_counts = df_filtered["Division"].astype(str).value_counts().head(12)

    fig1, ax1 = plt.subplots()
    ax1.bar(division_counts.index, division_counts.values)
    ax1.set_xlabel("Division")
    ax1.set_ylabel("Incidents")
    ax1.tick_params(axis="x", rotation=45)
    st.pyplot(fig1)

    st.subheader("Incidents by weekday")
    weekday_counts = df_filtered["Weekday"].astype(str).value_counts()

    fig2, ax2 = plt.subplots()
    ax2.bar(weekday_counts.index, weekday_counts.values)
    ax2.set_xlabel("Weekday")
    ax2.set_ylabel("Incidents")
    ax2.tick_params(axis="x", rotation=45)
    st.pyplot(fig2)

    if "OccurHour" in df_filtered.columns:
        st.subheader("Incidents by hour")
        hour_counts = df_filtered["OccurHour"].dropna().astype(int).value_counts().sort_index()

        fig3, ax3 = plt.subplots()
        ax3.plot(hour_counts.index, hour_counts.values)
        ax3.set_xlabel("Hour")
        ax3.set_ylabel("Incidents")
        st.pyplot(fig3)


# Model tab
with tab_model:
    st.subheader("Hotspot prediction model")

    st.write(
        "This model predicts whether a crime occurred in a hotspot neighborhood "
        "based on historical incident density."
    )

    percentile = st.slider("Hotspot threshold percentile", 70, 95, 85, 5)

    neighborhood_counts = df_filtered["Neighbourh"].value_counts()
    threshold = neighborhood_counts.quantile(percentile / 100.0)

    df_ml = df_filtered.copy()
    df_ml["Hotspot"] = df_ml["Neighbourh"].map(neighborhood_counts) >= threshold
    df_ml["Hotspot"] = df_ml["Hotspot"].astype(int)

    # Clean and encode features
    df_ml["Sector"] = df_ml["Sector"].astype(str).str.replace("SECTOR", "", regex=False)
    df_ml["Sector_num"] = pd.to_numeric(df_ml["Sector"], errors="coerce")
    df_ml = df_ml.dropna(subset=["Sector_num"])

    for col in ["Weekday", "OffSummary", "PrimViolat", "Neighbourh", "Division"]:
        encoder = LabelEncoder()
        df_ml[col + "_enc"] = encoder.fit_transform(df_ml[col].astype(str))

    feature_columns = [
        "lon", "lat", "Sector_num",
        "Division_enc", "Neighbourh_enc",
        "Weekday_enc", "OffSummary_enc", "PrimViolat_enc"
    ]

    X = df_ml[feature_columns]
    y = df_ml["Hotspot"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if y.nunique() < 2:
        st.error("Not enough class variation to train the model.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.metric("Model accuracy", f"{accuracy:.2f}")

    cm = confusion_matrix(y_test, predictions)

    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig_cm)

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.subheader("Feature importance")
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(importance_df["Feature"], importance_df["Importance"])
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)
