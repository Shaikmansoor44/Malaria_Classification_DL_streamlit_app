# Malaria_Classification_DL_streamlit_app
 Build an explainable deep learning model to classify malaria-infected cells and deploy the model in a simple dashboard.
# Malaria Cell Classification with Explainable Deep Learning Model

## Overview

This project aims to build and deploy a deep learning model to classify malaria-infected cells from uninfected cells using a dataset of cell images. The project involves creating an explainable model that not only predicts the class of the cell images but also highlights the regions in the images that contribute to the prediction. The final model is deployed as a web application using Streamlit.

## Project Structure

- **models/**: This directory contains all the code and scripts used to train the deep learning model and generate the results.
- **app.py**: Streamlit app script for deploying the model and visualizing the predictions along with the heatmaps.
- **demo.mp4**: A video demonstration of the working Streamlit app.
- **report.pdf**: A detailed technical report documenting the approach, methodology, and findings of the project.
- **readme.txt**: Additional instructions and comments about the project.

## Requirements

### Libraries and Packages

To run the project, you need the following libraries and packages installed:

- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- tensorflow
- keras
- cv2 (OpenCV)
- streamlit

You can install these packages using pip:

```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras opencv-python-headless streamlit
```

## Models Directory Structure

The `models` directory includes the following:

- Data preprocessing and visualization scripts.
- Model training script with data augmentation and model checkpointing.
- Evaluation scripts for generating confusion matrices and classification reports.

## Instructions to Run the Code

### Model Training

1. **Data Preparation**: Ensure that the dataset is organized into two directories: `Parasitized` and `Uninfected` under `cell_images/cell_images`.

2. **Training the Model**: Navigate to the `models` directory and run the training script. This will preprocess the images, split the data into training and validation sets, and train the deep learning model with early stopping and model checkpointing.

3. **Saving the Model**: The best model during training will be saved as `best_model.keras`.

### Streamlit Application

1. **Run the Streamlit App**: In the main project directory, execute the following command to start the Streamlit app:

    ```
    streamlit run app.py
    ```

2. **Upload and Classify Images**: Use the app to upload cell images. The app will display the uploaded image, classify it as either 'Parasitized' or 'Uninfected', and show the confidence score. Additionally, the app will generate a Grad-CAM heatmap overlay to highlight the regions in the image that influenced the prediction.

### Demonstration Video

- A video demonstration of the app's functionality is included as `demo.mp4`.

### Technical Report

- The `report.pdf` contains a detailed description of the project, including the data exploration, model architecture, training process, evaluation metrics, and conclusions. It provides insights into the model's performance and the rationale behind the chosen methods.

## Conclusion

This project successfully demonstrates the development and deployment of an explainable deep learning model for classifying malaria-infected cells. The use of Grad-CAM heatmaps provides interpretability, making it easier to understand the model's decision-making process. The Streamlit app offers a user-friendly interface for uploading and classifying cell images, enhancing the accessibility and usability of the model.
