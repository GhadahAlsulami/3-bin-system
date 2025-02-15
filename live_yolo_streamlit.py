import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLO model
model = YOLO(r"C:\Users\user\Desktop\capstone\runs\classify\train2\weights\last.pt")  # Load your model

# Path to your logo image
logo_path = r"C:\Users\user\Desktop\capstone\WhatsApp_Image_2024-11-27_at_9.44.16_PM-removebg-preview.jpg"

# Display the logo, title, and subheader
st.image(logo_path, width=150, use_column_width=False)

# Button to capture an image
capture_button = st.button("Capture Photo")
camera = cv2.VideoCapture(1)  # Open webcam (change to 1 if the default doesn't work)

if capture_button:
    ret, frame = camera.read()
    if ret:
        # Convert to RGB and display the captured image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption="Captured Image", use_column_width=True)

        # Save the captured frame temporarily
        temp_image_path = "temp.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Predict on the captured image using the YOLO model
        results = model(temp_image_path)  # Use model directly on the image

        if results and len(results) > 0:
            # Extract class names and probabilities
            names_dict = results[0].names  # Get class names
            probs = results[0].probs.data.tolist()  # Get probabilities

            # Get the index of the top prediction
            top_index = np.argmax(probs)
            top_class_name = names_dict[top_index]

            # Define background color based on the predicted class
            background_color = {
                "Black Bin": "black",
                "Blue Bin": "blue",
                "Green Bin": "green",
                "Tuwaiq Cup": "purple"
            }.get(top_class_name, "white")  # Default to white if class is not found

            # Inject background color change with a smooth transition using CSS
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-color: {background_color} !important;
                    transition: background-color 1s ease-in-out; /* Smooth transition */
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the top prediction result with a white background
            st.markdown(
                f"""
                <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <h3 style="margin: 0; color: #333333;"> {top_class_name}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("No results found. Ensure the model is correctly configured.")
    else:
        st.error("Failed to capture an image. Please check your webcam connection.")

# Release the camera after use
camera.release()
