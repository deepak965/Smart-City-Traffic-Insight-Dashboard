import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="Smart City Traffic Dashboard", layout="wide")

# Title
st.title("🚦 Smart City Traffic Insight Dashboard")
st.write("Analyze CSV data, detect congestion & upload videos for YOLO + Optical Flow Traffic analysis.")

# Sidebar navigation
menu = st.sidebar.radio(
    "Choose Section",
    ["📄 CSV Traffic Analytics", "🎥 Video Analysis (YOLO + Optical Flow)"]
)

# ------------------ CSV SECTION ------------------ #
if menu == "📄 CSV Traffic Analytics":
    
    st.subheader("Upload CSV File")
    csv_file = st.file_uploader("Upload Traffic CSV", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)

        st.write("### 📌 Preview Data")
        st.dataframe(df.head())

        # Timestamp conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour

        # Busiest junction
        avg_per_junction = df.groupby("junction_id")["vehicle_count"].mean()
        busiest = avg_per_junction.idxmax()

        st.write(f"### 🚦 Busiest Junction: **{busiest}**")

        # Peak hour
        hourly_avg = df.groupby("hour")["vehicle_count"].mean()
        peak_hour = hourly_avg.idxmax()

        st.write(f"### ⏰ Peak Hour: **{peak_hour}:00**")

        st.write("### 📊 Avg Vehicle Count Per Junction")
        st.bar_chart(avg_per_junction)

        st.write("### ⏰ Hourly Vehicle Distribution")
        st.line_chart(hourly_avg)

        # Speed vs count scatter plot
        st.write("### ⚡ Speed vs Vehicle Count")
        fig_scatter, ax = plt.subplots()
        ax.scatter(df["avg_speed"], df["vehicle_count"])
        ax.set_xlabel("Average Speed km/h")
        ax.set_ylabel("Vehicle Count")
        st.pyplot(fig_scatter)

# ------------------ VIDEO SECTION ------------------ #
elif menu == "🎥 Video Analysis (YOLO + Optical Flow)":
    
    st.subheader("Upload a Traffic Video")

    video_file = st.file_uploader("Upload .mp4 video file", type=["mp4"])

    if video_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        st.video(tfile.name)

        run_button = st.button("Start Video Analysis")

        if run_button:

            st.write("Running YOLO detection... Please wait.")

            model = YOLO("yolov8n.pt")

            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)

            ret, prev = cap.read()
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            points = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 7)
            
            speeds = []
            vehicle_counts = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO
                results = model(frame, verbose=False)
                count = len(results[0].boxes)
                vehicle_counts.append(count)

                # Optical Flow speed estimation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)

                movement = np.sqrt(np.sum((new_points - points)**2, axis=2))
                pixel_speed = np.mean(movement)

                meter_per_pixel = 0.2
                kmh = (pixel_speed * meter_per_pixel * fps) * 3.6  
                speeds.append(kmh)

                prev_gray = gray.copy()
                points = new_points

            cap.release()

            avg_speed = np.mean(speeds)
            avg_count = np.mean(vehicle_counts)

            st.success("Video processed!")

            st.metric(label="Average Speed (km/h)", value=f"{avg_speed:.2f}")
            st.metric(label="Avg Vehicles per Frame", value=f"{avg_count:.0f}")

            if avg_speed < 15 and avg_count > 20:
                st.error("🚨 High Congestion Detected!")
            else:
                st.success("🟢 Traffic Flow Normal")

