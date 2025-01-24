const video = document.getElementById('video');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdValue = document.getElementById('threshold-value');

let threshold = 0.5;

// Update threshold dynamically
thresholdSlider.addEventListener('input', (event) => {
    threshold = parseFloat(event.target.value);
    thresholdValue.textContent = threshold.toFixed(1);
});

// Set TensorFlow.js backend to WebGL
async function setupBackend() {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('Using backend:', tf.getBackend());
}

// Setup camera
async function setupCamera() {
    video.width = 360;
    video.height = 270;

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

// Load MoveNet model
async function loadMoveNet() {
    return await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    });
}

// Draw keypoints with dynamic threshold
function drawKeypoints(keypoints) {
    keypoints.forEach((keypoint) => {
        if (keypoint.score > threshold) {
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'yellow';
            ctx.fill();
        }
    });
}

// Draw skeleton with dynamic threshold
function drawSkeleton(keypoints) {
    const adjacentKeyPoints = poseDetection.util.getAdjacentPairs(
        poseDetection.SupportedModels.MoveNet
    );

    adjacentKeyPoints.forEach(([i, j]) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];

        if (kp1.score > threshold && kp2.score > threshold) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'blue';
            ctx.stroke();
        }
    });
}

// Detect and draw poses
async function detectPose(detector) {
    const poses = await detector.estimatePoses(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (poses.length > 0) {
        const keypoints = poses[0].keypoints;
        drawKeypoints(keypoints);
        drawSkeleton(keypoints);
    }

    requestAnimationFrame(() => detectPose(detector));
}

// Main function
async function main() {
    await setupBackend();
    await setupCamera();
    video.play();

    canvas.width = video.width;
    canvas.height = video.height;

    const detector = await loadMoveNet();
    detectPose(detector);
}

main();
