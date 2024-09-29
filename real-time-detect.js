// Define labels of the TF Lite model.
const classLabels = {};

// Define colours for the labels.
const classColors = {};

let detectedProducts = [];


// Define max no. of detections per image, threshold and model path.
const MAX_DETECTIONS = 5;
const THRESHOLD = 0.4;
const MODEL_PATH = "./model/lush_naked.tflite";

const FRAME_COUNT_THRESHOLD = 10;

const SECONDS_THRESHOLD = 1000;

let objectDetector;

async function loadProductNames() {
    try {
        // Fetch the product names file from the model directory
        const response = await fetch('./model/dict.txt'); // Adjust path as needed
        if (!response.ok) throw new Error('Network response was not ok');

        const contents = await response.text(); // Read the contents of the file

        // Create the classLabels object
      
        // Process the contents into the desired format
        const lines = contents.split('\n').map(line => line.trim()).filter(Boolean); // Remove empty lines

        lines.forEach((value, index) => {
			if (value != "background"){
				classLabels[index -1] = value; // Add each product with an index	
				classColors[value] = '#'+(Math.random() * 0xFFFFFF << 0).toString(16).padStart(6, '0');	
			}

        });
		

        // Display the classLabels object
        return classLabels;
    } catch (error) {
        console.error('Error loading file:', error);
        return null; // Return null if there's an error
    }
}


async function loadWebGPUBackend() {
    try {
        await tf.ready(); // Ensure TensorFlow.js is fully loaded
        await tf.setBackend('webgpu'); // Set the WebGPU backend
        console.log('WebGPU backend loaded successfully');
    } catch (error) {
        console.error('Error loading WebGPU backend:', error);
    }
}

// Access the webcam and start object detection.
async function startCamera() {
    const videoElement = document.querySelector('.video');
	
    //await loadWebGPUBackend(); // Ensure backend is loaded before starting

	removeAllChildren();
	
	await loadProductNames();
	
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: {facingMode: 'environment'} });
        videoElement.srcObject = stream;
        videoElement.play();
        detect(videoElement);
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

async function removeAllChildren(){
	const buttonStack = document.querySelector('.button-stack');
	if (buttonStack != undefined){
		while (buttonStack.firstChild) {
		    buttonStack.removeChild(buttonStack.firstChild);
		}		
	}
}

let lastFrameTime = Date.now(); // Last frame timestamp for FPS calculation
let inferenceTime = 0; // Inference time for each frame
let fps = 0; // Calculated FPS

// Detect objects in the video feed.
async function detect(videoElement) {
    if (!objectDetector) {
        objectDetector = await tflite.loadTFLiteModel(MODEL_PATH);
    }

    async function processFrame() {
		
        const startFrameTime = Date.now(); // Start time of the frame
    
        const frame = await grabFrame(videoElement);
    
        if (!frame) {
            requestAnimationFrame(processFrame);
            return;
        }
    
        let input = tf.image.resizeBilinear(tf.browser.fromPixels(frame), [192, 192]);
        input = tf.cast(tf.expandDims(input), 'int32');
    
        const startInferenceTime = performance.now(); // Start time of inference
        
        // Run the inference and get the output tensors.
        let result = await objectDetector.predict(input);
    
        const endInferenceTime = performance.now(); // End time of inference
        inferenceTime = endInferenceTime - startInferenceTime; // Calculate inference time
        
        let boxes = Array.from(await result[Object.keys(result)[0]].data());
        let classes = Array.from(await result[Object.keys(result)[1]].data());
        let scores = Array.from(await result[Object.keys(result)[2]].data());
        let n = Array.from(await result[Object.keys(result)[3]].data());
    
        inferenceResults(boxes, classes, scores, n, frame);
    
        // Calculate FPS
        const now = Date.now();
        fps = Math.round(1000 / (now - lastFrameTime));
        lastFrameTime = now;
    
        // Update the FPS and Inference Time in the HTML
        const fpsCounter = document.getElementById('fps-counter');
        fpsCounter.textContent = `FPS: ${fps} | Inference Time: ${inferenceTime.toFixed(2)} ms`;
    
        // Continue to process the next frame
        requestAnimationFrame(processFrame);
    }

    processFrame();
}


/*
async function grabFrame(videoElement) {
    const imageCapture = new ImageCapture(videoElement.srcObject.getVideoTracks()[0]);
    try {
        return await imageCapture.grabFrame();
    } catch (error) {
        console.error("Error grabbing frame:", error);
        return null;
    }
}

*/

async function grabFrame(videoElement) {
    // Check if ImageCapture is supported
  /*  if ('ImageCapture' in window) {
        try {
            const imageCapture = new ImageCapture(videoElement.srcObject.getVideoTracks()[0]);
            return await imageCapture.grabFrame(); // Use ImageCapture API if available
        } catch (error) {
            console.error("Error grabbing frame with ImageCapture:", error);
        }
    }*/
    
    // Fallback method using canvas if ImageCapture is not available or fails
    return captureFromCanvas(videoElement);
}

function captureFromCanvas(videoElement) {
    const canvas = document.createElement('canvas');

    // Ensure video is ready and has valid dimensions
    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) {
        console.error("Video dimensions are not valid.");
        return null; // Return null to indicate failure
    }

    canvas.width = videoWidth;
    canvas.height = videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Get the image data (similar to grabFrame)
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// Render and inference the detection results.
function inferenceResults(boxes, classes, scores, n, frame) {
    const boxesContainer = document.querySelector(".boxes-container");
    boxesContainer.innerHTML = "";

    const detections = [];

    for (let i = 0; i < n; i++) {
        const boundingBox = boxes.slice(i*4, (i+1)*4);
        const classIndex = classes[i];
        const className = classLabels[classIndex];
        const score = scores[i];
        detections.push({ boundingBox, className, score, index: i });
    }
    
    // Sort the results in the order of confidence to get top results.
    detections.sort((a, b) => b.score - a.score);
    const numDetectionsToShow = Math.min(MAX_DETECTIONS, detections.length);

    for (let i = 0; i < numDetectionsToShow; i++) {
        const detection = detections[i];
        const { boundingBox, className, score, index } = detection;
        const y_min = Math.floor(boundingBox[0] * frame.height);
        const y_max = Math.floor(boundingBox[2] * frame.height);
        const x_min = Math.floor(boundingBox[1] * frame.width);
        const x_max = Math.floor(boundingBox[3] * frame.width);

        if (score > THRESHOLD) {
            const color = classColors[className]
            const boxContainer = drawBoundingBoxes(
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min,
                className,
                score,
                color
            );
			
			let found = false
			for (let j = 0; j < detectedProducts.length; j++){
				if (detectedProducts[j].className == className){
					let count = detectedProducts[j].count + 1;
					detectedProducts[j] = {
						className: className,
						count: count,
						lastTime: Date.now()
					};
					found = true
				}
			}
			if (!found){
				detectedProducts.push({
					className: className,
					count: 1,
					lastTime: Date.now()
				})	
			}
			
			
            boxesContainer.appendChild(boxContainer);
        }
    }
}


// Draw bounding boxes for top 'N' detections.
function drawBoundingBoxes(left, top, width, height, className, score, color) {
		
    const container = document.createElement("div");
    container.classList.add("box-container");

    const box = document.createElement("div");
    box.classList.add("box");
    box.style.borderColor = color;
    box.style.borderWidth = "4px";
	
    container.appendChild(box);

    const label = document.createElement("div");
    label.classList.add("label");
    label.style.backgroundColor = color;
    label.textContent = `${className} (${score.toFixed(2)})`;
    container.appendChild(label);

    const inputVideoElement = document.getElementById("input-video");
    const vidRect = inputVideoElement.getBoundingClientRect();
    const offsetX = vidRect.left;
    const offsetY = vidRect.top;

    container.style.left = `${left + offsetX - 1}px`;
    container.style.top = `${top + offsetY}px`;
    box.style.width = `${width + 1}px`;
    box.style.height = `${height + 1}px`;

    return container;
}


function triggerAndroidCallback() {
    if (typeof Android !== 'undefined') {
        Android.onCustomEvent("This is a custom event from the web page!");
    } else {
        console.log("Android interface not available.");
    }
}


function checkDetectedProduct() {

	const buttonStack = document.querySelector('.button-stack');
	
	detectedProducts = detectedProducts.filter(product => {
	    const currentTime = Date.now();
	    return ((currentTime - product.lastTime) <= SECONDS_THRESHOLD); // Keep only if less than 2 seconds old
	});
	
	for (let i = 0; i < detectedProducts.length; i++){
		if (detectedProducts[i].count > FRAME_COUNT_THRESHOLD){
			
			let found = false
			
			buttonStack.childNodes.forEach((node) => {
			    if (node.nodeType === Node.ELEMENT_NODE) { // Check if it's an element
			        if (node.textContent == detectedProducts[i].className){
						found = true;
					}
			    }
			});
			

			if (!found){
	            const newButton = document.createElement('button');

                
	            newButton.textContent = detectedProducts[i].className;
	            buttonStack.appendChild(newButton);				
                newButton.addEventListener("click", function() {
                    triggerAndroidCallback();
                });

			}
		}
	}
	
	buttonStack.childNodes.forEach((node) => {
	    if (node.nodeType === Node.ELEMENT_NODE) { // Check if it's an element

			let found = false;
			for (let j = 0; j < detectedProducts.length; j++){
				if (node.textContent == detectedProducts[j].className){
					found = true;
				}
			}
			if (!found){
				buttonStack.removeChild(node);
			}

	    }
	});
	
	
    // Schedule the next frame
    requestAnimationFrame(checkDetectedProduct);
}

// Start the frame processing loop
requestAnimationFrame(checkDetectedProduct);

startCamera();

