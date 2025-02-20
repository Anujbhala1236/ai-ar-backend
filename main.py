from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

# ✅ Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
@app.get("/")
def read_root():
    return {"message": "Welcome to AI-AR Backend!"}
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        print(f"📸 Received file: {file.filename}")

        # ✅ Read image data
        image_data = await file.read()
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image format. Try another image."}

        # ✅ Convert BGR to RGB (required by MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ✅ Detect pose
        results = pose.process(image_rgb)
        pose_detected = "Pose detected ✅" if results.pose_landmarks else "No pose detected ❌"

        return {"poses": pose_detected, "lighting": analyze_lighting(image)}

    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return {"error": f"Failed to process image: {str(e)}"}

def analyze_lighting(image):
    """ 
    Analyze lighting direction based on brightness in different regions of the image 
    """
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Divide image into 3 vertical sections: Left, Center, Right
        height, width = gray.shape
        left_region = gray[:, :width // 3]
        center_region = gray[:, width // 3: 2 * width // 3]
        right_region = gray[:, 2 * width // 3:]

        # Calculate the average brightness for each region
        left_brightness = np.mean(left_region)
        center_brightness = np.mean(center_region)
        right_brightness = np.mean(right_region)

        # Determine the main light source based on brightness
        if left_brightness > right_brightness and left_brightness > center_brightness:
            return "Light source from the LEFT 🔆"
        elif right_brightness > left_brightness and right_brightness > center_brightness:
            return "Light source from the RIGHT 🔆"
        elif center_brightness > left_brightness and center_brightness > right_brightness:
            return "Light source from the CENTER 🔆"
        else:
            return "Lighting is evenly distributed"

    except Exception as e:
        print(f"❌ Error analyzing lighting: {str(e)}")
        return "Error detecting lighting"