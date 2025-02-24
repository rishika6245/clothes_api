# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk
# from inference_sdk import InferenceHTTPClient

# # Initialize Roboflow Inference Client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="5tlAx5MicHhsHE2XtNXO"
# )

# class ObjectDetectionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Object Detection GUI")

#         self.image_path = None
#         self.processed_image = None

#         # Buttons
#         self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, width=20, height=2)
#         self.upload_btn.pack(pady=10)

#         self.detect_btn = tk.Button(root, text="Get Details", command=self.detect_objects, width=20, height=2, state=tk.DISABLED)
#         self.detect_btn.pack(pady=10)

#         # Image Display
#         self.image_label = tk.Label(root)
#         self.image_label.pack()

#     def upload_image(self):
#         """ Open file dialog to upload an image """
#         file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
#         if file_path:
#             self.image_path = file_path
#             self.display_image(file_path)
#             self.detect_btn.config(state=tk.NORMAL)  # Enable detect button

#     def detect_objects(self):
#         """ Perform object detection and update the displayed image """
#         if not self.image_path:
#             return
        
#         # Perform Inference
#         result = CLIENT.infer(self.image_path, model_id="clothes-classfy/1")

#         # Read Image
#         image = cv2.imread(self.image_path)

#         # Process Detection Results
#         for prediction in result.get('predictions', []):
#             x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
#             class_name = prediction['class']

#             x1, y1 = int(x - width / 2), int(y - height / 2)
#             x2, y2 = int(x + width / 2), int(y + height / 2)

#             # Draw Bounding Box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Convert image for display
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         self.processed_image = Image.fromarray(image)

#         # Update the displayed image
#         self.display_image(self.processed_image)

#     def display_image(self, img):
#         """ Display image in the GUI """
#         if isinstance(img, str):  # If it's a file path
#             img = Image.open(img)

#         img = img.resize((400, 400), Image.Resampling.LANCZOS)
#         img_tk = ImageTk.PhotoImage(img)

#         self.image_label.config(image=img_tk)
#         self.image_label.image = img_tk  # Keep reference to prevent garbage collection

# # Run the Application
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ObjectDetectionApp(root)
#     root.mainloop()






# API DEPLOY CODE
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io

app = FastAPI()

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="5tlAx5MicHhsHE2XtNXO"
)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    """Receive an image, perform detection, and return the processed image."""
    
    # Read Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Save image temporarily
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    # Perform Inference
    result = CLIENT.infer(image_path, model_id="clothes-classfy/1")

    # Convert to OpenCV format
    image = cv2.imread(image_path)

    # Draw Bounding Boxes
    for prediction in result.get('predictions', []):
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        class_name = prediction['class']

        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert Image to Bytes
    _, encoded_image = cv2.imencode(".jpg", image)
    return {"image": encoded_image.tobytes()}

