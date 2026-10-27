
# Prodigy Infotech - Data Science Internship
# Task 05: Traffic Accident Data Analysis

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("output", exist_ok=True)
sns.set_theme(style="whitegrid")

# Dataset path
file_path = "dataset/US_Accidents_March23.csv"

# Load only required columns to reduce memory usage
columns = [
    "Severity",
    "Start_Time",
    "Start_Lat",
    "Start_Lng",
    "Weather_Condition",
    "State",
    "City",
    "Sunrise_Sunset",
    "Traffic_Signal",
    "Junction",
    "Crossing"
]

df = pd.read_csv(file_path, usecols=columns)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Convert time column to datetime
df["Start_Time"] = pd.to_datetime(
    df["Start_Time"], errors="coerce"
)

# Remove rows missing essential analysis fields
df = df.dropna(
    subset=["Start_Time", "Start_Lat", "Start_Lng"]
).copy()

# Extract hour and day
df["Hour"] = df["Start_Time"].dt.hour
df["Day"] = df["Start_Time"].dt.day_name()

# Classify time of day
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["Time_of_Day"] = df["Hour"].apply(get_time_of_day)

# 1. Accidents by hour
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Hour", color="steelblue")
plt.title("Traffic Accidents by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Recorded Accidents")
plt.tight_layout()
plt.savefig("output/accidents_by_hour.png", dpi=300)
plt.show()

# 2. Accidents by time of day
order = ["Morning", "Afternoon", "Evening", "Night"]

plt.figure(figsize=(8, 5))
sns.countplot(
    data=df, x="Time_of_Day", order=order,
    color="cornflowerblue"
)
plt.title("Accidents by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Recorded Accidents")
plt.tight_layout()
plt.savefig("output/accidents_by_time.png", dpi=300)
plt.show()

# 3. Accidents by weather condition
weather_counts = (
    df["Weather_Condition"]
    .fillna("Unknown")
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 6))
weather_counts.sort_values().plot(kind="barh")
plt.title("Top 10 Weather Conditions in Accident Records")
plt.xlabel("Number of Recorded Accidents")
plt.ylabel("Weather Condition")
plt.tight_layout()
plt.savefig("output/accidents_by_weather.png", dpi=300)
plt.show()

# 4. Accident severity
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Severity", color="salmon")
plt.title("Distribution of Accident Severity")
plt.xlabel("Severity (1 = Lowest, 4 = Highest)")
plt.ylabel("Number of Recorded Accidents")
plt.tight_layout()
plt.savefig("output/accident_severity.png", dpi=300)
plt.show()

# 5. Top 10 states by recorded accident count
state_counts = df["State"].value_counts().head(10)

plt.figure(figsize=(10, 6))
state_counts.sort_values().plot(kind="barh")
plt.title("Top 10 States by Recorded Accidents")
plt.xlabel("Number of Recorded Accidents")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("output/top_states.png", dpi=300)
plt.show()

# 6. Geographic hotspot visualization
# Plot a sample of coordinates to avoid overcrowding
hotspots = df.dropna(
    subset=["Start_Lat", "Start_Lng"]
)

hotspots = hotspots.sample(
    n=min(10000, len(hotspots)),
    random_state=42
)

plt.figure(figsize=(10, 7))
plt.scatter(
    hotspots["Start_Lng"],
    hotspots["Start_Lat"],
    s=1,
    alpha=0.3
)
plt.title("Geographical Distribution of Recorded Accidents")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("output/accident_hotspots.png", dpi=300)
plt.show()

# 7. Road features recorded near accidents
road_features = [
    "Traffic_Signal",
    "Junction",
    "Crossing"
]

available_features = [
    col for col in road_features if col in df.columns
]

if available_features:
    feature_counts = {
        col: int(df[col].fillna(False).astype(bool).sum())
        for col in available_features
    }

    plt.figure(figsize=(8, 5))
    pd.Series(feature_counts).plot(
        kind="bar", color="seagreen"
    )
    plt.title("Accident Records with Road Features")
    plt.xlabel("Road Feature")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("output/road_features.png", dpi=300)
    plt.show()

# 8. Save summary tables
df.groupby("Time_of_Day").size().to_csv(
    "output/time_summary.csv"
)

df.groupby("Severity").size().to_csv(
    "output/severity_summary.csv"
)

print("\nTask 05 completed successfully!")
print("Charts and summary files are saved in output/")
