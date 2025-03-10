import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os

# ===============================
# Load the dataset
# ===============================

# Define file path for the cleaned dataset
FILE_PATH = os.path.join(BASE_DIR, "merged_sentiment_data_updated.csv")

# Read CSV file
try:
    df = pd.read_csv(FILE_PATH)

    # Standardize column names
    if "date" in df.columns:
        df.rename(columns={"date": "month"}, inplace=True)

    # Convert "month" column to datetime format
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # Remove rows with invalid dates
    df = df.dropna(subset=["month"])

    # Filter dataset within the time range of 2017-2025
    start_date = pd.Timestamp("2017-01-01")
    end_date = pd.Timestamp("2025-12-31")
    df = df[(df["month"] >= start_date) & (df["month"] <= end_date)]

    # Ensure "Sentiment Category" column is treated as string
    df["Sentiment Category"] = df["Sentiment Category"].astype(str)

    # Ensure "count" column exists
    if "count" not in df.columns:
        df["count"] = 1  # Assign default count of 1

    # Define the cutoff date for COVID-19 impact (December 2019)
    cutoff_date = pd.Timestamp("2019-12-31")

    # Create a new column "Period" to categorize data into Pre-COVID and Post-COVID
    df["Period"] = df["month"].apply(lambda x: "Pre-COVID" if x <= cutoff_date else "Post-COVID")

    print(f"Data loaded successfully! Shape: {df.shape}")

except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()  # Create an empty DataFrame to prevent crashes

# ===============================
# Initialize Dash App
# ===============================

app = dash.Dash(__name__)

# ===============================
# Layout Design
# ===============================

app.layout = html.Div([
    html.H1("Sentiment Analysis Dashboard"),

    # Dropdown for selecting platform
    html.Label("Select Platform:"),
    dcc.Dropdown(
        id="platform-dropdown",
        options=[{"label": src, "value": src} for src in df["source"].unique()] if not df.empty else [],
        value=df["source"].unique()[0] if not df.empty else None
    ),

    # Dropdown for selecting period (Pre-COVID or Post-COVID)
    html.Label("Select Time Period:"),
    dcc.Dropdown(
        id="period-dropdown",
        options=[
            {"label": "Pre-COVID (Before 2020)", "value": "Pre-COVID"},
            {"label": "Post-COVID (After 2020)", "value": "Post-COVID"}
        ],
        value="Pre-COVID"
    ),

    # Date Picker for selecting time range
    html.Label("Select Date Range:"),
    dcc.DatePickerRange(
        id="date-picker",
        start_date=df["month"].min() if not df.empty else None,
        end_date=df["month"].max() if not df.empty else None,
        display_format="YYYY-MM-DD"
    ),

    # Line Chart for Sentiment Trends
    html.H3(id="sentiment-trend-title"),
    dcc.Graph(id="sentiment-trend"),

    # Radar Chart for Sentiment Distribution
    html.H3(id="sentiment-radar-title"),
    dcc.Graph(id="sentiment-radar")
])

# ===============================
# Callbacks: Interactive Data Updates
# ===============================

@app.callback(
    [Output("sentiment-radar-title", "children"),
     Output("sentiment-radar", "figure")],
    [Input("platform-dropdown", "value"),
     Input("period-dropdown", "value")]
)
def update_radar_chart(platform, period):
    """ Updates the radar chart for sentiment distribution. """

    if df.empty or platform is None or period is None:
        return "Sentiment Distribution", px.line_polar(title="No Data Available")

    # Filter dataset by selected platform and time period
    filtered_df = df[(df["source"] == platform) & (df["Period"] == period)]

    # If no valid data is found, return a placeholder chart
    if filtered_df.empty:
        return f"Sentiment Distribution on {platform} ({period})", px.line_polar(
            r=[1], theta=["No Data"], line_close=True,
            title=f"No Sentiment Data for {platform} ({period})"
        )

    # Count occurrences of each sentiment category
    sentiment_counts = filtered_df["Sentiment Category"].value_counts()

    # Create a radar chart
    fig = px.line_polar(
        r=sentiment_counts.values.tolist(),
        theta=sentiment_counts.index.tolist(),
        line_close=True,
        title=f"Sentiment Distribution on {platform} ({period})"
    )

    return f"Sentiment Distribution on {platform} ({period})", fig


@app.callback(
    [Output("sentiment-trend-title", "children"),
     Output("sentiment-trend", "figure")],
    [Input("platform-dropdown", "value"),
     Input("period-dropdown", "value")]
)
def update_trend_chart(platform, period):
    """ Updates the trend chart for sentiment trends over time. """

    if df.empty or platform is None or period is None:
        return "Sentiment Trend Over Time", px.line(title="No Data Available")

    # Filter dataset by selected platform and time period
    filtered_df = df[(df["source"] == platform) & (df["Period"] == period)]

    # If no valid data is found, return a placeholder chart
    if filtered_df.empty:
        return f"Sentiment Trend on {platform} ({period})", px.line(
            title=f"No Sentiment Data for {platform} ({period})"
        )

    # Aggregate sentiment counts by month
    df_grouped = filtered_df.groupby(["month", "Sentiment Category"])["count"].sum().reset_index()

    # Create a line chart to show sentiment trends over time
    fig = px.line(
        df_grouped,
        x="month",
        y="count",
        color="Sentiment Category",
        title=f"Sentiment Trend on {platform} ({period})"
    )

    return f"Sentiment Trend on {platform} ({period})", fig

# ===============================
# Run the App
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  
app.run_server(host="0.0.0.0", port=port, debug=False)
