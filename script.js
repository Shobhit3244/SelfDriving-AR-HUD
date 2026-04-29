// Core Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
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

// State
let model = null;
let isDetecting = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let lastFpsTime = performance.now();

// Targets to detect
const TARGET_CLASSES = ['car', 'stop sign'];
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
        startBtn.textContent = "Initialize Camera";
        loadingOverlay.classList.add('hidden');
        
        startBtn.addEventListener('click', toggleCamera);
    } catch (error) {
        console.error("Error loading model:", error);
        loadingText.textContent = "Error Loading Perception Model.";
        sysStatusDot.style.backgroundColor = 'var(--danger)';
        sysStatusText.textContent = "Model Error";
    }
}

// Camera Control
async function toggleCamera() {
    if (!isDetecting) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
            });
            video.srcObject = stream;
            
            video.addEventListener('loadeddata', () => {
                // Set canvas dimensions to match video
                canvas.width = video.clientWidth;
                canvas.height = video.clientHeight;
                resolutionValue.textContent = `${video.videoWidth}x${video.videoHeight}`;
                
                isDetecting = true;
                startBtn.textContent = "Stop Camera";
                startBtn.style.background = "linear-gradient(135deg, var(--danger), #b91c1c)";
                
                sysStatusDot.classList.add('active');
                sysStatusText.textContent = "System Online";
                
                // Clear wait log
                detectionLog.innerHTML = '';
                
                // Start detection loop
                detectFrame();
            });
            
        } catch (error) {
            console.error("Error accessing webcam:", error);
            alert("Could not access the webcam. Ensure you are on localhost or HTTPS.");
        }
    } else {
        // Stop Camera
        const stream = video.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        video.srcObject = null;
        isDetecting = false;
        
        startBtn.textContent = "Initialize Camera";
        startBtn.style.background = "";
        sysStatusDot.classList.remove('active');
        sysStatusText.textContent = "System Offline";
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        fpsValue.textContent = "0";
        inferenceValue.textContent = "0ms";
        detectionCount.textContent = "0";
        resolutionValue.textContent = "--";
    }
}

// Main Detection Loop
async function detectFrame() {
    if (!isDetecting || !model) return;

    // Ensure canvas dimensions match video display size in case of resize
    if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
    }

    const startInferenceTime = performance.now();
    
    // Perform detection
    const predictions = await model.detect(video);
    
    const endInferenceTime = performance.now();
    const infTime = Math.round(endInferenceTime - startInferenceTime);
    
    // Filter predictions for our targets
    const filteredPredictions = predictions.filter(p => TARGET_CLASSES.includes(p.class));
    
    // Render AR elements
    renderPredictions(filteredPredictions);
    
    // Update Metrics
    updateMetrics(infTime, filteredPredictions.length);
    updateLogs(filteredPredictions);
    
    // Calculate FPS
    calculateFPS();

    // Loop
    requestAnimationFrame(detectFrame);
}

// Render bounding boxes and AR HUD
function renderPredictions(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scale factors because video resolution might differ from canvas size
    // Note: CSS scaleX(-1) mirrors the video visually, so we must mirror the X coordinates
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

        // Apply mirroring calculation for X axis
        const x = canvas.width - scaledX - scaledWidth;
        const y = scaledY;
        const width = scaledWidth;
        const height = scaledHeight;

        const isCar = prediction.class === 'car';
        const color = isCar ? '#0ea5e9' : '#ef4444'; // CSS Variables equivalent
        const bgOpacity = isCar ? 'rgba(14, 165, 233, 0.15)' : 'rgba(239, 68, 68, 0.15)';
        
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
