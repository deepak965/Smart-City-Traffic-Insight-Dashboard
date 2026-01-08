 AI Traffic Analysis System (Computer Vision & ML – Decoupled)

1 Overview
The AI Traffic Analysis System is an educational and experimental project demonstrating two independent AI approaches to traffic analysis:
Computer Vision–based vehicle detection using YOLO
Dataset-driven traffic level prediction using Machine Learning
Both models operate independently and are showcased through a Streamlit-based interface.
 The YOLO detection outputs are not directly used as inputs to the machine learning model.

 
2 Problem Statement
Traffic congestion analysis typically requires:
Real-time visual understanding of traffic scenes
Data-driven prediction of congestion levels
This project explores both approaches separately to understand their strengths and limitations.


3 Project Design Philosophy
The project follows a decoupled architecture:
✔ YOLO demonstrates real-time computer vision
✔ ML model demonstrates structured data learning
 No pipeline-level fusion between CV and ML
 
This design allows:
Independent testing and improvement of each model
Clear learning of both AI paradigms


4 System Architecture
        ┌──────────────────────────┐
        │  Traffic Image / Video   │
        └───────────┬──────────────┘
                    ↓
        ┌──────────────────────────┐
        │   YOLO Vehicle Detection │
        │  (Standalone CV Module) │
        └──────────────────────────┘


        ┌──────────────────────────┐
        │  Traffic Dataset (CSV)   │
        └───────────┬──────────────┘
                    ↓
        ┌──────────────────────────┐
        │ Machine Learning Model   │
        │ (Traffic Level Predictor)│
        └──────────────────────────┘


        ┌──────────────────────────┐
        │     Streamlit Interface  │
        │  (Demonstration Layer)   │
        └──────────────────────────┘

        
5 Modules

5.1 Computer Vision Module (YOLO)
Detects vehicles from traffic images/videos
Outputs:
Bounding boxes
Vehicle classes
Purpose:
Demonstrates real-time object detection capability

5.2 Machine Learning Module (Dataset-Based)
Trained using a labeled traffic dataset
Features:
Vehicle counts
Time-based or contextual attributes
Model:
Random Forest Classifier
Output:
Traffic level (Low / Medium / High)

5.3 Streamlit Application
Allows users to:
Run YOLO-based vehicle detection
Run ML-based traffic prediction
Models are executed independently
Results are displayed separately


6 Tech Stack
Language
Python
Computer Vision
YOLO
OpenCV
Machine Learning
Scikit-learn
Pandas
NumPy
Interface
Streamlit
