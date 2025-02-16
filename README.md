# ZIMA - Waste Classification App
ZIMA is an AI-powered garbage classification model designed to help users identify the appropriate bin (Blue, Green, or Black) for waste disposal. Leveraging a YOLO model, the app classifies garbage with high accuracy by analyzing photos captured via webcam.
# Features
Real-time Image Capture: Uses a webcam to capture garbage images.
## Accurate Classification: Classifies waste into:
- Blue Box
- Green Box
- Black Box
Confidence Scores: Displays the model's prediction confidence for each classification.
User-Friendly Interface: Interactive and intuitive design built with Streamlit.
# 🚀 Installation
Ensure you have the following installed:
- Python 3.9 or later
- Streamlit
- OpenCV
- PyTorch
- ultralytics library (for YOLO)
  # Steps
  1. Clone this repository:
```
  git clone https://github.com/AhoodNaif/Capstone-Project
```
  2. Download the trained YOLO model weights:
Place the best.pt file in the project directory.
# 🖥️ Usage
Run the Streamlit app:
```
streamlit run live_yolo_streamlit.py
```
Use your webcam to capture an image of garbage.
The app will classify the image into one of the three categories:
Blue Box: Recyclable materials
Green Box: Organic waste
Black Box: General waste
View the top prediction with confidence score on the interface.
