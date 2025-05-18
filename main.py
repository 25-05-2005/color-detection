import cv2
import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image

# Load the predefined color dataset
COLOR_CSV = "colors.csv"

# Create color dataset if not available
data = {
    "color_name": ["Red", "Green", "Blue", "Black", "White", "Yellow", "Purple", "Cyan", "Magenta", "Orange"],
    "R": [255, 0, 0, 0, 255, 255, 128, 0, 255, 255],
    "G": [0, 255, 0, 0, 255, 255, 0, 255, 0, 165],
    "B": [0, 0, 255, 0, 255, 0, 128, 255, 255, 0]
}
colors_df = pd.DataFrame(data)
colors_df.to_csv(COLOR_CSV, index=False)  # Save CSV file

# Function to find the closest color match
def get_color_name(R, G, B):
    min_diff = float("inf")
    color_name = ""
    for index, row in colors_df.iterrows():
        diff = abs(R - row["R"]) + abs(G - row["G"]) + abs(B - row["B"])
        if diff < min_diff:
            min_diff = diff
            color_name = row["color_name"]
    return color_name

# Streamlit UI
st.title("Color Detection App 🎨")

# Upload Image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    img_np = np.array(image)

    # Display Image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Select Pixel
    x = st.slider("Select X-coordinate:", 0, img_np.shape[1]-1, 50)
    y = st.slider("Select Y-coordinate:", 0, img_np.shape[0]-1, 50)

    # Extract Pixel Color
    pixel_color = img_np[y, x]
    detected_color = get_color_name(pixel_color[0], pixel_color[1], pixel_color[2])

    # Show Results
    st.write(f"**Detected Color:** {detected_color}")
    st.write(f"**RGB Value:** ({pixel_color[0]}, {pixel_color[1]}, {pixel_color[2]})")

    # Display Detected Color
    st.markdown(f'<div style="width:50px;height:50px;background-color:rgb({pixel_color[0]},{pixel_color[1]},{pixel_color[2]})"></div>', unsafe_allow_html=True)
