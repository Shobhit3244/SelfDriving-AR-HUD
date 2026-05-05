# Autonomous Perception Engine: Self-Driving AR HUD

[![Test Me](https://img.shields.io/badge/Test_Me-Live_Demo-0ea5e9?style=for-the-badge)](https://shobhit3244.github.io/SelfDriving-AR-HUD/)

This project is a real-time computer vision web application simulating the "Perception Engine" of a self-driving car. It processes live video (via webcam or screen-sharing tools) using a hybrid approach: **TensorFlow.js** for object detection and **OpenCV.js** for pure Digital Image Processing (DIP). The system autonomously detects drivable lanes, identifies red lights, and tracks braking vehicles to provide Forward Collision Warnings in a dynamic Augmented Reality (AR) HUD.

---

## 🏗 System Architecture & Performance Optimization

To achieve real-time performance inside a web browser, the system decouples computationally expensive tasks from the main rendering thread.

The engine runs on three asynchronous loops:
1. **`drawLoop` (~60 FPS)**: Bound to `requestAnimationFrame`. Handles all UI rendering, canvas drawing, and smooth HUD overlays without blocking.
2. **`mlLoop` (~24 FPS)**: Throttled using `setTimeout` to ~42ms. Runs the COCO-SSD object detection neural network.
3. **`dipLoop` (~24 FPS)**: Throttled using `setTimeout` to ~42ms. Runs the OpenCV.js pipeline for lane detection and geometric calculations.

This decoupled architecture ensures that even if ML inference spikes to 150ms on slower hardware, the HUD UI and camera feed remain silky smooth and responsive.

---

## 🧠 Methodology 1: Digital Image Processing (DIP) Lane Detection

The system uses classical computer vision techniques via OpenCV to detect and highlight the drivable lane path.

### The Pipeline:
1. **Downscaling**: The raw video feed is downscaled to a maximum processing width of `640px` (`MAX_DIP_WIDTH`). This exponentially reduces the number of pixels processed per frame, ensuring fast Canny edge detection.
2. **Grayscale & Blurring**: The image is converted to grayscale (`cv.COLOR_RGBA2GRAY`) and smoothed using a `5x5` Gaussian Blur to eliminate high-frequency noise (like asphalt texture).
3. **Canny Edge Detection**: Calculates the intensity gradients of the image. Thresholds are set to `(50, 150)` to strictly isolate distinct lines.
4. **Region of Interest (ROI) Masking**: A static trapezoidal polygon is defined covering the bottom half of the screen. `cv.fillPoly` creates a binary mask that zeroes out everything above the horizon (sky, trees, oncoming traffic).
5. **Hough Line Transform**: `cv.HoughLinesP` is applied to the masked edges to extract line segments based on polar coordinate voting. 
6. **Geometric Filtering & Averaging**:
   - Lines with near-horizontal or near-vertical slopes (`|m| < 0.2` or `|m| > 10`) are discarded.
   - Lines with negative slopes on the left side of the screen are classified as "Left Lane".
   - Lines with positive slopes on the right side are classified as "Right Lane".
   - The arrays are averaged out to produce two master line equations: $y = mx + b$.

### Rendering the Drivable Path & No-Go Zones
Using the averaged $m$ and $b$, the system calculates the exact intersection coordinates at the bottom of the screen ($y_{bottom}$) and the horizon ($y_{top}$). 

$$x = \frac{y - b}{m}$$

This yields a 4-point polygon representing the **Drivable Lane**.
By mathematically inverting this space, the system defines **No-Go Zones** as the polygons stretching from the left lane boundary to the left edge of the screen, and the right lane boundary to the right edge of the screen.

---

## 🛑 Methodology 2: Forward Collision & Traffic Signal Logic

Instead of relying on heavy Machine Learning models specifically trained for brake lights or traffic light states, this system uses a highly robust, mathematical DIP heuristic applied to the bounding boxes provided by the ML model.

### 1. Object Detection (ML)
The **COCO-SSD** model tracks entities like `car`, `truck`, `person`, `bicycle`, `stop sign`, and `traffic light`. Bounding boxes $[x, y, width, height]$ are scaled to the canvas.

### 2. Traffic Light State Analysis (Strict Chromaticity)
When a `traffic light` or `stop sign` is detected, the system extracts the exact pixel data (`ImageData`) from inside the bounding box using an isolated, hidden canvas.

A strict chromaticity threshold evaluates every pixel:
```javascript
if (R > 180 && R > G * 2.0 && R > B * 2.0) { redPixels++; }
```
If `redPixels / totalPixels > 0.05` (5%), the engine confirms a Red Light/Stop Sign and triggers the `STOPPING` state.

### 3. Forward Collision Warning (Braking Car Detection)
When a `car` or `truck` is detected, a 3-stage heuristic evaluates danger:
1. **Proximity Filter**: Calculates if the vehicle occupies $>20\%$ of the screen width. $width_{bbox} > width_{screen} \times 0.20$.
2. **Trajectory Filter**: Calculates the center point of the bounding box $x_{center} = x + \frac{width}{2}$. It checks if the car falls within the central $40\%$ of the screen ($30\% < x_{center} < 70\%$), ignoring parked cars on the side of the road.
3. **Brake Light Activation**: If the car is directly ahead and too close, the algorithm samples the **bottom 30%** of the car's bounding box (where taillights are located) and runs the same strict Chromaticity algorithm used for red lights. If intense red clusters are found, the vehicle is mathematically proven to be braking.

### State Machine & Debouncing
When danger is detected, the system enters the `STOPPING` state, flashing a red **STOP** warning in the center of the HUD. A $1000ms$ timer acts as an anti-flicker debounce. When the danger clears (light turns green, or car stops braking), the system transitions to a `GOING` state, flashing **GO** three times before returning to normal `IDLE` operation.

---

## 🎨 User Interface & AR Aesthetics

The dashboard is built entirely with **Vanilla CSS** utilizing modern **Glassmorphism** techniques (translucency, background blurring, glowing borders).
The AR overlays map semantic colors to different classes (e.g., Cars = Cyan, Pedestrians = Amber, Bicycles = Purple) to provide instant visual clarity to the user.

---

## 🚀 How to Run Locally

Since this project requires webcam/screen-share access, modern browsers require it to be served over a secure origin (`localhost` or `https`).

1. Clone the repository.
2. Use a local development server to serve the files in the project root:
   - **Python 3:** `python -m http.server 8000`
   - **Node.js:** `npx http-server`
3. Open your browser and navigate to `http://localhost:8000`.
4. Click **Initialize Camera** to use your webcam, or **Share Screen** to pipe in a dashcam video from YouTube or `driveandlisten.com`.
