import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

st.title("🚦 Smart Traffic Video Analyzer")
st.write("Upload a traffic video to detect vehicles & estimate speed")

uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])

if uploaded_file is not None:
    
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(tfile.name)

    st.write("Processing video... please wait")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(tfile.name)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev = cap.read()

    if not ret:
        st.error("Video could not be read.")
    else:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 7)
        speeds = []
        vehicle_counts = []
        
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            annotated = results[0].plot()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)

            movement = np.sqrt(np.sum((new_points - points)**2, axis=2))
            pixel_speed = np.mean(movement)

            meter_per_pixel = 0.2
            kmh = (pixel_speed * meter_per_pixel * fps) * 3.6
            speeds.append(kmh)

            count = len(results[0].boxes)
            vehicle_counts.append(count)

            prev_gray = gray.copy()
            points = new_points

            frames.append(annotated)

        cap.release()

        avg_speed = np.mean(speeds)
        avg_count = np.mean(vehicle_counts)

        st.subheader("📌 Analysis Results:")
        st.write(f"Average Detected Speed: **{avg_speed:.2f} km/h**")
        st.write(f"Average Vehicle Count: **{avg_count:.0f} vehicles/frame**")

        if avg_speed < 15 and avg_count > 20:
            st.write("🚨 Traffic Congestion Detected!")
        else:
            st.write("🟢 Traffic Normal")

        st.success("Processing Complete!")
