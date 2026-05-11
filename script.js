// Core Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const shareBtn = document.getElementById('share-screen-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

// UI Elements
const sysStatusDot = document.getElementById('system-status-dot');
const sysStatusText = document.getElementById('system-status-text');
const fpsValue = document.getElementById('fps-value');
const inferenceValue = document.getElementById('inference-value');
const resolutionValue = document.getElementById('resolution-value');
const detectionCount = document.getElementById('detection-count');
const detectionLog = document.getElementById('detection-log');
const toggleObjects = document.getElementById('toggle-objects');
const toggleLanes = document.getElementById('toggle-lanes');

// State
let model = null;
let isDetecting = false;
let currentPredictions = [];
let currentLanePolygon = null;
let currentNoGoZones = [];
let mlInferencing = false;
let dipInferencing = false;
let frameCount = 0;
let lastFpsTime = performance.now();
let trafficState = 'IDLE';
let lastGoFlashTime = 0;
let stopDebounceTime = 0;
let missedLaneFrames = 0;
let currentVP = null;
let currentDebugView = 'none';
let dipViewCanvas = document.createElement('canvas');

// Targets to detect
const TARGET_CLASSES = ['car', 'person', 'bicycle', 'motorcycle', 'truck', 'bus', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bear', 'zebra', 'giraffe', 'stop sign', 'traffic light'];
const MAX_LOG_ITEMS = 15;

// Initialization
async function init() {
    try {
        startBtn.disabled = true;
        startBtn.textContent = "Loading Model...";
        loadingOverlay.classList.remove('hidden');
        
        // Load the COCO-SSD model
        console.log("Loading COCO-SSD Model...");
        model = await cocoSsd.load();
        console.log("Model loaded successfully.");
        
        startBtn.disabled = false;
        shareBtn.disabled = false;
        startBtn.textContent = "Initialize Camera";
        loadingOverlay.classList.add('hidden');
        
        startBtn.addEventListener('click', toggleCamera);
        shareBtn.addEventListener('click', toggleScreenShare);
        
        document.querySelectorAll('input[name="dip-view"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                currentDebugView = e.target.value;
            });
        });
    } catch (error) {
        console.error("Error loading model:", error);
        loadingText.textContent = "Error Loading Perception Model.";
        sysStatusDot.style.backgroundColor = 'var(--danger)';
        sysStatusText.textContent = "Model Error";
    }
}

// Media Control
function stopStream() {
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    isDetecting = false;
    
    startBtn.textContent = "Initialize Camera";
    shareBtn.textContent = "Share Screen";
    startBtn.style.background = "";
    shareBtn.style.background = "";
    shareBtn.style.borderColor = "";
    startBtn.disabled = false;
    shareBtn.disabled = false;
    
    sysStatusDot.classList.remove('active');
    sysStatusText.textContent = "System Offline";
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    fpsValue.textContent = "0";
    inferenceValue.textContent = "0ms";
    detectionCount.textContent = "0";
    resolutionValue.textContent = "--";
}

async function startMediaStream(useScreenShare = false) {
    if (isDetecting) {
        stopStream();
        return;
    }
    try {
        let stream;
        if (useScreenShare) {
            stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
            shareBtn.textContent = "Stop Screen Share";
            shareBtn.style.background = "rgba(239, 68, 68, 0.2)";
            shareBtn.style.borderColor = "var(--danger)";
            startBtn.disabled = true;
        } else {
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
            });
            startBtn.textContent = "Stop Camera";
            startBtn.style.background = "linear-gradient(135deg, var(--danger), #b91c1c)";
            shareBtn.disabled = true;
        }
        
        video.srcObject = stream;
        
        video.addEventListener('loadeddata', () => {
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
            resolutionValue.textContent = `${video.videoWidth}x${video.videoHeight}`;
            
            isDetecting = true;
            sysStatusDot.classList.add('active');
            sysStatusText.textContent = "System Online";
            
            detectionLog.innerHTML = '';
            startDetectionLoops();
        }, { once: true });
        
        stream.getVideoTracks()[0].addEventListener('ended', stopStream);
        
    } catch (error) {
        console.error("Error accessing media:", error);
        alert("Could not access the media. Please check permissions.");
    }
}

function toggleCamera() { startMediaStream(false); }
function toggleScreenShare() { startMediaStream(true); }

// Main Detection Loops
function startDetectionLoops() {
    requestAnimationFrame(drawLoop);
    mlLoop();
    dipLoop();
}

async function mlLoop() {
    if (!isDetecting || !model) return;
    
    if (!mlInferencing) {
        mlInferencing = true;
        const startInferenceTime = performance.now();
        try {
            const predictions = await model.detect(video);
            const infTime = Math.round(performance.now() - startInferenceTime);
            currentPredictions = predictions.filter(p => TARGET_CLASSES.includes(p.class));
            
            let stopDetected = false;
            for (let p of currentPredictions) {
                if (p.class === 'stop sign') {
                    stopDetected = true;
                    break;
                }
                if (p.class === 'traffic light') {
                    if (isTrafficLightRed(p.bbox)) {
                        stopDetected = true;
                        break;
                    }
                }
                if (p.class === 'car' || p.class === 'truck') {
                    if (isCarBrakingAndClose(p.bbox)) {
                        stopDetected = true;
                        break;
                    }
                }
            }
            
            if (stopDetected) {
                trafficState = 'STOPPING';
                stopDebounceTime = performance.now();
            } else if (trafficState === 'STOPPING') {
                if (performance.now() - stopDebounceTime > 1000) { // 1 second debounce
                    trafficState = 'GOING';
                    lastGoFlashTime = performance.now();
                }
            }
            
            updateMetrics(infTime, currentPredictions.length);
            updateLogs(currentPredictions);
        } catch (e) {
            console.error("ML Error:", e);
        }
        mlInferencing = false;
    }
    setTimeout(mlLoop, 42); // Throttle to ~24 fps
}

function dipLoop() {
    if (!isDetecting) return;
    
    if (!dipInferencing) {
        dipInferencing = true;
        processDIP();
        dipInferencing = false;
    }
    setTimeout(dipLoop, 42); // Throttle to ~24 fps
}

function drawLoop() {
    if (!isDetecting) return;

    try {
        if (!video || video.videoWidth === 0 || video.videoHeight === 0 || video.clientWidth === 0) {
            requestAnimationFrame(drawLoop);
            return;
        }

        if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (currentDebugView !== 'none' && dipViewCanvas.width > 0) {
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(dipViewCanvas, 0, 0, canvas.width, canvas.height);
        }
        
        if (!toggleObjects || toggleObjects.checked) {
            renderPredictions(currentPredictions);
        }
        
        if (!toggleLanes || toggleLanes.checked) {
            // Draw DIP Lane Polygon
            if (currentLanePolygon && (currentLanePolygon.length === 4 || currentLanePolygon.length === 6 || currentLanePolygon.length === 8)) {
                ctx.fillStyle = 'rgba(16, 185, 129, 0.2)'; // Translucent emerald
                ctx.beginPath();
                ctx.moveTo(currentLanePolygon[0].x, currentLanePolygon[0].y);
                for (let i = 1; i < currentLanePolygon.length; i++) {
                    ctx.lineTo(currentLanePolygon[i].x, currentLanePolygon[i].y);
                }
                ctx.closePath();
                ctx.fill();
                
                // Draw lane boundary lines
                ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
                ctx.lineWidth = 3;
                
                let midPoint = currentLanePolygon.length / 2;
                
                ctx.beginPath();
                ctx.moveTo(currentLanePolygon[0].x, currentLanePolygon[0].y);
                for (let i = 1; i < midPoint; i++) {
                    ctx.lineTo(currentLanePolygon[i].x, currentLanePolygon[i].y);
                }
                ctx.stroke();
                
                ctx.beginPath();
                ctx.moveTo(currentLanePolygon[midPoint].x, currentLanePolygon[midPoint].y);
                for (let i = midPoint + 1; i < currentLanePolygon.length; i++) {
                    ctx.lineTo(currentLanePolygon[i].x, currentLanePolygon[i].y);
                }
                ctx.stroke();
            }
        
        // Draw No-Go Zones (Non-drivable areas outside the main lane)
        if (currentNoGoZones && currentNoGoZones.length > 0) {
            for (let poly of currentNoGoZones) {
                if (!poly || poly.length < 3) continue;
                
                ctx.fillStyle = 'rgba(239, 68, 68, 0.15)'; // Translucent Red
                ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
                ctx.lineWidth = 2;
                
                ctx.beginPath();
                ctx.moveTo(poly[0].x, poly[0].y);
                for (let i = 1; i < poly.length; i++) {
                    ctx.lineTo(poly[i].x, poly[i].y);
                }
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                
                // Add text label roughly in the center-top of the zone
                let minX = Math.min(...poly.map(p => p.x));
                let maxX = Math.max(...poly.map(p => p.x));
                let minY = Math.min(...poly.map(p => p.y));
                let avgX = (minX + maxX) / 2;
                
                ctx.fillStyle = '#ffffff';
                ctx.font = '14px "JetBrains Mono", monospace';
                ctx.fillText("NO-GO ZONE", avgX - 40, minY + 30);
            }
        }
        } // End of toggleLanes check

        // Draw Traffic Sign Flashes
        const now = performance.now();
        if (trafficState === 'STOPPING') {
            if (Math.floor(now / 500) % 2 === 0) {
                drawCenterFlash("STOP", "rgba(239, 68, 68, 0.9)");
            }
        } else if (trafficState === 'GOING') {
            let timeInGo = now - lastGoFlashTime;
            let cycle = Math.floor(timeInGo / 500);
            if (cycle < 6) { // 3 full flashes (on/off)
                if (cycle % 2 === 0) {
                    drawCenterFlash("GO", "rgba(16, 185, 129, 0.9)");
                }
            } else {
                trafficState = 'IDLE';
            }
        }

        calculateFPS();
    } catch (err) {
        console.error("Rendering Error: ", err);
        const fpsElement = document.getElementById('fps-value');
        if (fpsElement) {
            // Display the actual error message in the UI so it can be debugged
            fpsElement.textContent = String(err.message || err).substring(0, 15);
            fpsElement.style.fontSize = "12px";
            fpsElement.style.color = "#ff4444";
        }
    } finally {
        requestAnimationFrame(drawLoop);
    }
}

// Render bounding boxes and AR HUD
function renderPredictions(predictions) {
    
    // Calculate scale factors because video resolution might differ from canvas size
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    predictions.forEach(prediction => {
        // Extract original coordinates
        const [origX, origY, origWidth, origHeight] = prediction.bbox;
        
        // Scale to canvas size
        const scaledWidth = origWidth * scaleX;
        const scaledHeight = origHeight * scaleY;
        const scaledX = origX * scaleX;
        const scaledY = origY * scaleY;

        const x = scaledX;
        const y = scaledY;
        const width = scaledWidth;
        const height = scaledHeight;

        const CLASS_COLORS = {
            'car': { color: '#0ea5e9', bgOpacity: 'rgba(14, 165, 233, 0.15)' }, // Blue
            'person': { color: '#f59e0b', bgOpacity: 'rgba(245, 158, 11, 0.15)' }, // Amber
            'bicycle': { color: '#8b5cf6', bgOpacity: 'rgba(139, 92, 246, 0.15)' }, // Purple
            'motorcycle': { color: '#ec4899', bgOpacity: 'rgba(236, 72, 153, 0.15)' }, // Pink
            'truck': { color: '#14b8a6', bgOpacity: 'rgba(20, 184, 166, 0.15)' }, // Teal
            'bus': { color: '#eab308', bgOpacity: 'rgba(234, 179, 8, 0.15)' }, // Yellow
            'stop sign': { color: '#ef4444', bgOpacity: 'rgba(239, 68, 68, 0.15)' }, // Red
            'traffic light': { color: '#f59e0b', bgOpacity: 'rgba(245, 158, 11, 0.15)' }, // Amber
            'default': { color: '#10b981', bgOpacity: 'rgba(16, 185, 129, 0.15)' } // Emerald
        };

        const theme = CLASS_COLORS[prediction.class] || CLASS_COLORS['default'];
        const color = theme.color;
        const bgOpacity = theme.bgOpacity;
        
        // Draw Fill
        ctx.fillStyle = bgOpacity;
        ctx.fillRect(x, y, width, height);

        // Draw Bounding Box HUD Style (Corners instead of full box)
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        const cornerLength = Math.min(width, height) * 0.15;
        
        ctx.beginPath();
        // Top Left
        ctx.moveTo(x, y + cornerLength);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLength, y);
        // Top Right
        ctx.moveTo(x + width - cornerLength, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerLength);
        // Bottom Right
        ctx.moveTo(x + width, y + height - cornerLength);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width - cornerLength, y + height);
        // Bottom Left
        ctx.moveTo(x + cornerLength, y + height);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x, y + height - cornerLength);
        ctx.stroke();

        // Draw Label Background
        const confText = Math.round(prediction.score * 100) + '%';
        const labelText = `${prediction.class.toUpperCase()} [${confText}]`;
        ctx.font = '14px "JetBrains Mono", monospace';
        const textWidth = ctx.measureText(labelText).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 24, textWidth + 10, 24);
        
        // Draw Label Text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x + 5, y - 7);
        
        // Draw crosshair center
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        
        ctx.beginPath();
        ctx.moveTo(centerX - 10, centerY);
        ctx.lineTo(centerX + 10, centerY);
        ctx.moveTo(centerX, centerY - 10);
        ctx.lineTo(centerX, centerY + 10);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}

function updateMetrics(infTime, count) {
    inferenceValue.textContent = `${infTime}ms`;
    detectionCount.textContent = count;
}

function calculateFPS() {
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        fpsValue.textContent = frameCount;
        frameCount = 0;
        lastFpsTime = now;
    }
}

// Throttle log updates to avoid flooding
let lastLogTime = 0;
function updateLogs(predictions) {
    const now = performance.now();
    if (now - lastLogTime < 500) return; // Update logs max twice a second
    lastLogTime = now;

    if (predictions.length === 0) return;

    // Take the most confident prediction for logging if there are multiple
    const topPred = predictions.reduce((prev, current) => (prev.score > current.score) ? prev : current);
    
    const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false, hour: "numeric", minute: "numeric", second: "numeric", fractionalSecondDigits: 2 });
    
    const logItem = document.createElement('div');
    const className = topPred.class.replace(' ', '-');
    logItem.className = `log-item ${className}`;
    
    const conf = Math.round(topPred.score * 100);
    
    logItem.innerHTML = `
        <span class="log-time">[${timeStr}]</span>
        <span class="log-class">${topPred.class.toUpperCase()}</span>
        <span class="log-conf font-mono">CONF: ${conf}%</span>
    `;
    
    detectionLog.prepend(logItem);
    
    // Keep max items
    while (detectionLog.children.length > MAX_LOG_ITEMS) {
        detectionLog.removeChild(detectionLog.lastChild);
    }
}

// Start app
window.addEventListener('load', init);

// OpenCV DIP Processing
let srcMat, dstMat, edgesMat, linesMat, hsvMat, mask1, mask2, redMask, contours, hierarchy;
let hiddenCanvas, hiddenCtx;

function processDIP() {
    if (typeof cv === 'undefined' || !cv.Mat) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;
    
    try {
        if (!hiddenCanvas) {
            hiddenCanvas = document.createElement('canvas');
            hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });
        }
        
        const MAX_DIP_WIDTH = 640;
        let procWidth = video.videoWidth;
        let procHeight = video.videoHeight;
        
        if (procWidth > MAX_DIP_WIDTH) {
            procHeight = Math.floor(procHeight * (MAX_DIP_WIDTH / procWidth));
            procWidth = MAX_DIP_WIDTH;
        }

        // Ensure size match if resolution changes
        if (hiddenCanvas.width !== procWidth || hiddenCanvas.height !== procHeight) {
            hiddenCanvas.width = procWidth;
            hiddenCanvas.height = procHeight;
            if (srcMat) {
                srcMat.delete(); dstMat.delete(); edgesMat.delete(); linesMat.delete();
                hsvMat.delete(); mask1.delete(); mask2.delete(); redMask.delete();
                contours.delete(); hierarchy.delete();
                srcMat = null;
            }
        }

        // Reliable way to capture video frame into OpenCV
        hiddenCtx.drawImage(video, 0, 0, procWidth, procHeight);
        let imageData = hiddenCtx.getImageData(0, 0, procWidth, procHeight);

        if (!srcMat) {
            srcMat = cv.matFromImageData(imageData);
            dstMat = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
            edgesMat = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
            linesMat = new cv.Mat();
            hsvMat = new cv.Mat(procHeight, procWidth, cv.CV_8UC3);
            mask1 = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
            mask2 = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
            redMask = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
            contours = new cv.MatVector();
            hierarchy = new cv.Mat();
        } else {
            srcMat.data.set(imageData.data);
        }
        
        const scaleX = canvas.width / procWidth;
        const scaleY = canvas.height / procHeight;

        // ==========================================
        // 1. Lane Detection (Green Lines)
        // ==========================================
        cv.cvtColor(srcMat, dstMat, cv.COLOR_RGBA2GRAY);
        let ksize = new cv.Size(5, 5);
        cv.GaussianBlur(dstMat, dstMat, ksize, 0, 0, cv.BORDER_DEFAULT);
        cv.Canny(dstMat, edgesMat, 30, 100, 3);
        
        let mask = cv.Mat.zeros(edgesMat.rows, edgesMat.cols, cv.CV_8UC1);
        let horizonY = currentVP ? currentVP.y : procHeight * 0.45;
        horizonY = Math.max(procHeight * 0.2, Math.min(horizonY, procHeight * 0.8));
        
        let pts = new cv.Mat(4, 1, cv.CV_32SC2);
        pts.data32S.set([
            0, procHeight,
            0, Math.floor(horizonY),
            procWidth, Math.floor(horizonY),
            procWidth, procHeight
        ]);
        let ptsVec = new cv.MatVector();
        ptsVec.push_back(pts);
        cv.fillPoly(mask, ptsVec, new cv.Scalar(255));
        
        let maskedEdges = new cv.Mat();
        cv.bitwise_and(edgesMat, mask, maskedEdges);
        
        cv.HoughLinesP(maskedEdges, linesMat, 1, Math.PI / 180, 15, 15, 20);
        
        if (currentDebugView === 'grayscale') cv.imshow(dipViewCanvas, dstMat);
        else if (currentDebugView === 'edges') cv.imshow(dipViewCanvas, edgesMat);
        else if (currentDebugView === 'mask') cv.imshow(dipViewCanvas, mask);
        else if (currentDebugView === 'maskedEdges') cv.imshow(dipViewCanvas, maskedEdges);
        
        mask.delete(); pts.delete(); ptsVec.delete(); maskedEdges.delete();
        
        let yBottom = procHeight;
        let vpY = currentVP ? currentVP.y : procHeight * 0.45;
        let yTop = vpY + (procHeight * 0.05);
        
        let yMidTop = yBottom - (yBottom - yTop) * 0.5;

        let leftLinesAll = [], rightLinesAll = [];
        let leftLinesNear = [], leftLinesFar = [];
        let rightLinesNear = [], rightLinesFar = [];
        
        for (let i = 0; i < linesMat.rows; ++i) {
            let x1 = linesMat.data32S[i * 4];
            let y1 = linesMat.data32S[i * 4 + 1];
            let x2 = linesMat.data32S[i * 4 + 2];
            let y2 = linesMat.data32S[i * 4 + 3];
            
            let dX = x2 - x1;
            if (dX === 0) continue; // Ignore vertical lines
            
            let slope = (y2 - y1) / dX;
            if (Math.abs(slope) < 0.1 || Math.abs(slope) > 30) continue;
            
            let midY = (y1 + y2) / 2;
            
            if (slope < -0.1 && x1 < procWidth * 0.9 && x2 < procWidth * 0.9) {
                leftLinesAll.push([x1, y1, x2, y2]);
                if (midY > yMidTop) leftLinesNear.push([x1, y1, x2, y2]);
                else leftLinesFar.push([x1, y1, x2, y2]);
            } else if (slope > 0.1 && x1 > procWidth * 0.1 && x2 > procWidth * 0.1) {
                rightLinesAll.push([x1, y1, x2, y2]);
                if (midY > yMidTop) rightLinesNear.push([x1, y1, x2, y2]);
                else rightLinesFar.push([x1, y1, x2, y2]);
            }
        }

        function averageLines(lines) {
            if (lines.length === 0) return null;
            let sumX = 0, sumY = 0, sumSlope = 0;
            for (let line of lines) {
                sumX += line[0] + line[2];
                sumY += line[1] + line[3];
                let lineDx = line[2] - line[0];
                if (lineDx === 0) lineDx = 0.001; 
                sumSlope += (line[3] - line[1]) / lineDx;
            }
            let avgX = sumX / (lines.length * 2);
            let avgY = sumY / (lines.length * 2);
            let avgSlope = sumSlope / lines.length;
            if (Math.abs(avgSlope) < 0.001) avgSlope = avgSlope < 0 ? -0.001 : 0.001; 
            return { slope: avgSlope, b: avgY - avgSlope * avgX };
        }

        let leftLane = averageLines(leftLinesAll);
        let rightLane = averageLines(rightLinesAll);
        
        let leftNear = averageLines(leftLinesNear);
        let leftFar = averageLines(leftLinesFar);
        
        let rightNear = averageLines(rightLinesNear);
        let rightFar = averageLines(rightLinesFar);
        
        if (!leftNear) leftNear = leftFar || leftLane;
        if (!leftFar) leftFar = leftNear || leftLane;
        
        if (!rightNear) rightNear = rightFar || rightLane;
        if (!rightFar) rightFar = rightNear || rightLane;
        
        if (leftLane && rightLane) {
            let intersectX = (rightLane.b - leftLane.b) / (leftLane.slope - rightLane.slope);
            let intersectY = leftLane.slope * intersectX + leftLane.b;
            
            if (intersectX > -procWidth && intersectX < procWidth * 2 && intersectY > 0 && intersectY < procHeight * 0.9) {
                if (!currentVP) {
                    currentVP = {x: intersectX, y: intersectY};
                } else {
                    currentVP.x = currentVP.x * 0.9 + intersectX * 0.1;
                    currentVP.y = currentVP.y * 0.9 + intersectY * 0.1;
                }
            }
            
            // Calculate X intersections for left lane bounds
            let xL_bottom = (yBottom - leftNear.b) / leftNear.slope;
            let xL_midTop = (yMidTop - leftNear.b) / leftNear.slope;
            let xL_top = (yTop - leftFar.b) / leftFar.slope;
            
            // Calculate X intersections for right lane bounds
            let xR_bottom = (yBottom - rightNear.b) / rightNear.slope;
            let xR_midTop = (yMidTop - rightNear.b) / rightNear.slope;
            let xR_top = (yTop - rightFar.b) / rightFar.slope;
            
            // Strict Anti-crossover logic (lanes must never overlap or touch)
            const minWidth = procWidth * 0.05; // 5% minimum logical width
            
            if ((xL_bottom + minWidth < xR_bottom) && 
                (xL_midTop + minWidth < xR_midTop) && 
                (xL_top + minWidth < xR_top)) {
                
                let newPolygon = [
                    {x: xL_bottom * scaleX, y: yBottom * scaleY},
                    {x: xL_midTop * scaleX, y: yMidTop * scaleY},
                    {x: xL_top * scaleX, y: yTop * scaleY},
                    {x: xR_top * scaleX, y: yTop * scaleY},
                    {x: xR_midTop * scaleX, y: yMidTop * scaleY},
                    {x: xR_bottom * scaleX, y: yBottom * scaleY}
                ];
                
                if (currentLanePolygon && currentLanePolygon.length === 6 && missedLaneFrames === 0) {
                    for (let i = 0; i < 6; i++) {
                        currentLanePolygon[i].x = currentLanePolygon[i].x * 0.7 + newPolygon[i].x * 0.3;
                        currentLanePolygon[i].y = currentLanePolygon[i].y * 0.7 + newPolygon[i].y * 0.3;
                    }
                } else {
                    currentLanePolygon = newPolygon;
                }
                
                // Dynamically assign areas outside the lane as No-Go Zones
                currentNoGoZones = [
                    [ // Left No-Go Zone
                        {x: 0, y: currentLanePolygon[0].y},
                        {x: 0, y: currentLanePolygon[2].y},
                        {x: currentLanePolygon[2].x, y: currentLanePolygon[2].y},
                        {x: currentLanePolygon[1].x, y: currentLanePolygon[1].y},
                        {x: currentLanePolygon[0].x, y: currentLanePolygon[0].y}
                    ],
                    [ // Right No-Go Zone
                        {x: currentLanePolygon[5].x, y: currentLanePolygon[5].y},
                        {x: currentLanePolygon[4].x, y: currentLanePolygon[4].y},
                        {x: currentLanePolygon[3].x, y: currentLanePolygon[3].y},
                        {x: procWidth * scaleX, y: currentLanePolygon[3].y},
                        {x: procWidth * scaleX, y: currentLanePolygon[5].y}
                    ]
                ];
                missedLaneFrames = 0;
            } else {
                missedLaneFrames++;
            }
        } else {
            missedLaneFrames++;
        }

        if (missedLaneFrames > 15) {
            currentLanePolygon = null;
            currentNoGoZones = [];
        }
    } catch (e) {
        console.error("OpenCV processing error:", e);
    }
}

// Helpers for Traffic Signs
function isTrafficLightRed(bbox) {
    if (typeof hiddenCtx === 'undefined' || !hiddenCtx || !hiddenCanvas) return false;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return false;
    
    const [x, y, w, h] = bbox;
    const scaleX = hiddenCanvas.width / video.videoWidth;
    const scaleY = hiddenCanvas.height / video.videoHeight;
    
    const sx = Math.max(0, Math.floor(x * scaleX));
    const sy = Math.max(0, Math.floor(y * scaleY));
    const sw = Math.min(hiddenCanvas.width - sx, Math.floor(w * scaleX));
    const sh = Math.min(hiddenCanvas.height - sy, Math.floor(h * scaleY));
    
    if (sw <= 0 || sh <= 0) return false;
    
    const imgData = hiddenCtx.getImageData(sx, sy, sw, sh);
    const data = imgData.data;
    let redPixels = 0;
    let totalPixels = sw * sh;
    
    // Strict check for dominant red to avoid amber/glare
    for (let i = 0; i < data.length; i += 4) {
        let r = data[i];
        let g = data[i+1];
        let b = data[i+2];
        if (r > 180 && r > g * 2.0 && r > b * 2.0) {
            redPixels++;
        }
    }
    
    return (redPixels / totalPixels) > 0.05; // 5% of pixels are red
}

function isCarBrakingAndClose(bbox) {
    if (typeof hiddenCtx === 'undefined' || !hiddenCtx || !hiddenCanvas) return false;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return false;
    
    const [x, y, w, h] = bbox;
    
    // 1. Proximity Check (car must take up >20% of screen width)
    if (w < video.videoWidth * 0.20) return false;
    
    // 2. Lane Position Check (car center must be in center 40% of screen)
    const centerX = x + (w / 2);
    if (centerX < video.videoWidth * 0.3 || centerX > video.videoWidth * 0.7) return false;
    
    // 3. Brake Light DIP Check (sample the bottom 30% of the bounding box)
    const scaleX = hiddenCanvas.width / video.videoWidth;
    const scaleY = hiddenCanvas.height / video.videoHeight;
    
    const bottomY = y + (h * 0.7);
    const bottomH = h * 0.3;
    
    const sx = Math.max(0, Math.floor(x * scaleX));
    const sy = Math.max(0, Math.floor(bottomY * scaleY));
    const sw = Math.min(hiddenCanvas.width - sx, Math.floor(w * scaleX));
    const sh = Math.min(hiddenCanvas.height - sy, Math.floor(bottomH * scaleY));
    
    if (sw <= 0 || sh <= 0) return false;
    
    const imgData = hiddenCtx.getImageData(sx, sy, sw, sh);
    const data = imgData.data;
    let redPixels = 0;
    let totalPixels = sw * sh;
    
    // Strict red threshold for brake lights
    for (let i = 0; i < data.length; i += 4) {
        let r = data[i];
        let g = data[i+1];
        let b = data[i+2];
        if (r > 180 && r > g * 2.0 && r > b * 2.0) {
            redPixels++;
        }
    }
    
    // If >2% of the bottom area is intensely red, it is braking
    return (redPixels / totalPixels) > 0.02;
}

function drawCenterFlash(text, color) {
    ctx.save();
    let boxWidth = canvas.width * 0.2; // 20% of the page
    let boxHeight = boxWidth * 0.4;
    let x = (canvas.width - boxWidth) / 2;
    let y = (canvas.height - boxHeight) / 2;
    
    ctx.fillStyle = color;
    ctx.fillRect(x, y, boxWidth, boxHeight);
    
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, boxWidth, boxHeight);
    
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `bold ${Math.floor(boxHeight * 0.5)}px "JetBrains Mono", monospace`;
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    ctx.restore();
}
