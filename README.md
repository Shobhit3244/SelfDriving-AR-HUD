# Autonomous Perception Engine: Self-Driving AR HUD

[![Test Me](https://img.shields.io/badge/Test_Me-Live_Demo-0ea5e9?style=for-the-badge)](https://shobhit3244.github.io/SelfDriving-AR-HUD/)

This project is a real-time computer vision web application simulating the "Perception Engine" of a self-driving car. It accesses the user's webcam and uses a pre-trained machine learning model directly in the browser to detect specific objects (cars and stop signs), overlaying an augmented reality (AR) Heads-Up Display (HUD) onto the live video feed.

## Features

- **Live Webcam Integration**: Streams video directly from the device's camera.
- **Real-Time Object Detection**: Uses TensorFlow.js and the COCO-SSD model to continuously scan frames.
- **Targeted Tracking**: Specifically filters for and tracks **cars** and **stop signs**.
- **AR Overlays**: Draws futuristic, dynamic bounding boxes (with crosshairs and confidence labels) directly onto the video feed. Elements are color-coded (Cyan for cars, Red for stop signs).
- **Telemetry Dashboard**: A glassmorphic UI panel providing real-time system metrics:
  - FPS (Frames Per Second)
  - Inference Time (ms)
  - Camera Resolution
  - Active Detection Count
- **Detection Log**: A scrolling, timestamped log of the most confident detections.

## Tech Stack

- **Frontend**: HTML5, CSS3 (Vanilla, Glassmorphism design), JavaScript (ES6)
- **Machine Learning**: TensorFlow.js (`@tensorflow/tfjs`)
- **Model**: COCO-SSD (`@tensorflow-models/coco-ssd`)

## How to Run Locally

Since this project requires webcam access, modern browsers require it to be served over a secure origin (`localhost` or `https`).

1. Clone the repository.
2. Use a local development server to serve the files in the project root. For example:
   - **Python 3:** `python -m http.server 8000`
   - **Node.js:** `npx http-server`
3. Open your browser and navigate to `http://localhost:8000` (or the port specified by your server).
