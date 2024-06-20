import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import cv2
import tempfile

# Load the trained model
model = load_model('best_model.keras')

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image

# Function to generate Grad-CAM heatmap
def get_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to overlay heatmap on image
def overlay_heatmap(heatmap, image_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    image = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlayed_image = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    return overlayed_image

# Streamlit app
st.title("Cell Image Classification and Heatmap Visualization")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    image_array = preprocess_image(image, target_size=(150, 150))
    preds = model.predict(image_array)
    pred_class = 'Parasitized' if preds[0] > 0.5 else 'Uninfected'
    confidence = preds[0][0]  # Extract the scalar value from the array
    st.write(f"Prediction: {pred_class} (Confidence: {confidence:.2f})")

    heatmap = get_gradcam_heatmap(model, image_array, 'conv2d_2')
    
    # Save uploaded image temporarily to apply heatmap overlay
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        image.save(tmpfile.name)
        overlayed_image = overlay_heatmap(heatmap, tmpfile.name)
        overlayed_image_path = tmpfile.name

    st.image(overlayed_image_path, caption='Grad-CAM Heatmap', use_column_width=True)
