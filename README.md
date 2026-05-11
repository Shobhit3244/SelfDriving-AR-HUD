# Autonomous Perception Engine: Self-Driving AR HUD

[![Test Me](https://img.shields.io/badge/Test_Me-Live_Demo-0ea5e9?style=for-the-badge)](https://shobhit3244.github.io/SelfDriving-AR-HUD/)

## 1. Problem Description
The project involves designing and developing a real-time computer vision web application that simulates the "Perception Engine" of a self-driving car. The system processes live video (via webcam or screen-sharing tools) to autonomously detect drivable lanes, identify traffic light states, and track braking vehicles to provide Forward Collision Warnings (FCW) within a dynamic Augmented Reality (AR) Heads-Up Display (HUD).

**System Overview:**
The objective is to create a performant, browser-based autonomous driving dashboard using a hybrid approach. It integrates **TensorFlow.js** for object detection and **OpenCV.js** for pure Digital Image Processing (DIP). To maintain a smooth user interface, the system decouples computationally expensive tasks from the main rendering thread into asynchronous loops.

**Key Requirements:**
* Process live webcam or screen-share video feeds entirely client-side.
* Maintain a smooth ~60 FPS UI rendering despite heavy ML inference loads.
* Detect and visualize drivable path lanes using OpenCV line detection.
* Detect vehicles, pedestrians, and traffic signs using the COCO-SSD neural network.
* Implement a Forward Collision Warning system based on heuristic brake light analysis.
* Provide an interactive AR HUD using Glassmorphism design principles.

## 2. Sequence of Operation and Approach to Problem
We identified the following critical operational sequence, managed through three decoupled, asynchronous execution loops:

**Step-by-Step System Operation:**

**Phase 1: Video Input & Initialization**
1. User grants webcam access or shares a screen containing dashcam footage.
2. The `video` element is piped into hidden canvases for processing.

**Phase 2: UI Rendering (`drawLoop`)**
1. Bound to `requestAnimationFrame` running at ~60 FPS.
2. Continuously draws the raw video frame to the main canvas.
3. Renders the latest ML bounding boxes and OpenCV lane polygons over the video.

**Phase 3: Object Detection (`mlLoop`)**
1. Runs asynchronously, throttled to ~42ms (~24 FPS).
2. The COCO-SSD model processes the current frame to identify `car`, `truck`, `person`, `stop sign`, and `traffic light`.
3. Bounding boxes are scaled and stored in global memory for the `drawLoop` to render.

**Phase 4: Digital Image Processing (`dipLoop`)**
1. Downscales the video feed to 640px to optimize performance.
2. Converts to Grayscale and applies a 5x5 Gaussian Blur.
3. Applies Canny Edge Detection (thresholds 30, 100).
4. Masks the upper half of the screen using a Dynamic ROI polygon that adapts to the calculated vanishing point.
5. Uses Hough Line Transform to extract line segments, grouping them into 'Near' and 'Far' depth bands.
6. Calculates the Drivable Lane polygon as a Dynamic Hexagon to visualize road curvature, enforcing strict non-overlapping geometry.

## 3. Creativity Aspects & Algorithmic Decisions

**1. Decoupled Asynchronous Architecture**
To achieve real-time performance inside a web browser, we decoupled the rendering thread from the processing threads. Even if ML inference spikes to 150ms on slower hardware, the HUD UI and camera feed remain silky smooth and responsive.

**2. Hybrid ML + DIP Approach**
Instead of relying entirely on heavy ML models (which struggle in browsers), we used classical OpenCV techniques for lane detection. This drastically reduced computational overhead while maintaining high accuracy for structural road features.

**3. Strict Chromaticity Heuristics for Braking**
Instead of training a specific ML model for brake lights, we developed a highly robust mathematical heuristic. When a car is detected, the system isolates the bottom 30% of its bounding box and evaluates pixels: `if (R > 180 && R > G * 2.0 && R > B * 2.0)`. This accurately identifies braking vehicles with minimal processing power.

**4. Dynamic Hexagons for Curved Lanes**
To visualize curves without the severe computational cost of polynomial curve fitting, the algorithm groups lane lines into "Near" and "Far" depth bands. Calculating the intersections between these bands yields a 6-point polygon (Hexagon) that smoothly bends with the road.

**5. Dynamic Vanishing Point & Horizon Adaptation**
The system mathematically calculates the true vanishing point by finding the intersection of the far lane lines. This coordinate is smoothed via an Exponential Moving Average (EMA) and used to dynamically adjust the ROI mask, allowing the HUD to seamlessly adapt to hills, bumps, and varying camera tilt angles.

## 4. Project Deliverables: Virtual System with HUD Integration

**Input Variables (Sensors/Feeds):**
| Component | Function |
| :--- | :--- |
| `MediaDevices.getUserMedia` | Webcam video feed |
| `MediaDevices.getDisplayMedia` | Screen-share video feed (Dashcam footage) |
| TF.js COCO-SSD | Machine Learning object detection arrays |

**Output Variables (AR Visualizations):**
| Visual Component | Function |
| :--- | :--- |
| Drivable Path Polygon | Highlights the safe driving area (Dynamic Hexagon) |
| Red/Amber/Cyan Bounding Boxes | Visualizes semantic classes (e.g. Threat, Pedestrian, Vehicle) |
| Status Overlay Text | Displays STOP/GO and FCW alerts |
| Debug View Toggles | Visualizes real-time OpenCV DIP transformations (Grayscale, Canny, ROI Mask) |

### 4.1 DIP Algorithm Implementation Details

**Lane Detection Logic Explanation:**
The system uses mathematical intersections of the "Near" and "Far" line segments to construct a 6-point lane polygon. The top boundary dynamically tapers to 5% below the continuously tracked vanishing point. Strict anti-crossover logic mathematically prevents the left and right lane boundaries from ever overlapping. If lane data is lost, the system relies on an EMA cache, clearing it if no lines are detected for >15 frames to prevent screen freezing.

### 4.2 Process Flow & State Machine Behaviour
**Flow Logic for Forward Collision Warning:**
COCO-SSD detects Vehicle → Proximity Filter ($width_{bbox} > 20\%$) → Trajectory Filter (Vehicle is in center 40% of screen) → Extract bottom 30% of Bounding Box → Apply Chromaticity Threshold → If High Red Pixel Count → Trigger `STOPPING` state.

**State Machine Behaviour:**
The system implements a finite state machine for traffic signal and FCW handling:
* **State: `IDLE`**
  * Conditions: No red lights, no braking cars ahead.
  * Action: Standard AR rendering.
* **State: `STOPPING`**
  * Conditions: Red light pixel density > 5% OR Braking car detected in trajectory path.
  * Action: Flash red "STOP" warning in HUD. Start 1000ms debounce timer.
* **State: `GOING`**
  * Conditions: Danger clears (light turns green or car accelerates).
  * Action: Flash "GO" three times before transitioning back to `IDLE`.

## 5. Results and Observations
**System Performance Metrics:**
* **Rendering Speed:** `drawLoop` maintains a solid 60 FPS regardless of background processing.
* **Inference Speed:** COCO-SSD and OpenCV processes run effectively at ~24 FPS (42ms delay) on standard consumer hardware.
* **Lane Accuracy:** Robust lane detection in clear daylight conditions; relies heavily on Canny thresholds.

**Key Findings:**
* Browsers are capable of complex, real-time autonomous perception tasks if the architecture is heavily decoupled.
* A hybrid DIP and ML approach provides the best balance of speed and structural awareness.
* Mathematical heuristics applied *after* ML object detection are a highly efficient way to build complex logic (like FCW) without needing custom-trained models.

## 6. Technical Learning Outcomes
**Computer Vision (OpenCV):**
1. **Matrix Operations:** Learned to manipulate image tensors (`cv.Mat`) inside the browser.
2. **Edge Detection & Hough Transforms:** Mastered extracting geometric line equations from raw pixel data.
3. **Region Masking:** Implemented complex polygon masking to isolate Regions of Interest.

**Web Technologies:**
1. **Asynchronous Architecture:** Bridged the gap between heavy synchronous operations and smooth UI rendering using `requestAnimationFrame` and `setTimeout`.
2. **HTML5 Canvas:** Handled high-performance, layered pixel manipulation directly on the DOM.
3. **TensorFlow.js:** Successfully implemented pre-trained neural networks running entirely client-side.

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
