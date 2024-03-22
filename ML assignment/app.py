import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Function to load the model
def load_models():
    try:
        ann_model = load_model("iris_ann_model.h5")
        logreg_model = joblib.load("iris_logreg_model.pkl")
        knn_model = joblib.load("iris_knn_model.pkl")
        scaler = joblib.load("iris_scaler.pkl")
        le = joblib.load("iris_label_encoder.pkl")
        return ann_model, logreg_model, knn_model, scaler, le
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Function to preprocess input data for prediction
def preprocess_input(sepal_length, sepal_width, petal_length, petal_width, scaler):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input = scaler.transform(input_data)
    return scaled_input

# Streamlit app
st.title("Iris Species Classification App")
with st.expander('App Description'):
    st.write('The Iris Species Classification App is a web application built using Streamlit, a Python library for creating interactive web applications, along with machine learning models trained on the famous Iris dataset. The app allows users to input the features of an Iris flower (sepal length, sepal width, petal length, and petal width) and predicts the species of the Iris flower based on these features using various machine learning models.')


# Background color for the entire app
st.markdown(
    """
    <style>
    body {
        background-color: #FFFF00:;
        color: #00008B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input
with st.sidebar:
    st.markdown('<span style="font-size:30px; font-weight:bold;">Enter Flower Details</span>', unsafe_allow_html=True)
    sepal_length = st.text_input("Sepal Length", "5.2")
    sepal_width = st.text_input("Sepal Width", "3.2")
    petal_length = st.text_input("Petal Length", "1.5")
    petal_width = st.text_input("Petal Width", "0.2")

# Model selection
model_selected = st.sidebar.radio("Select Model", ("Artificial Neural Network", "Logistic Regression", "K-Nearest Neighbors"))

# Load models
ann_model, logreg_model, knn_model, scaler, le = load_models()

# Dictionary to map classes to images
class_to_image = {
    0: "Iris-Setosa.png",
    1: "Iris-Versicolor.png",
    2: "Iris-Virginica.png"
}

# Make predictions based on the selected model
if ann_model and logreg_model and knn_model and scaler and le:
    if st.sidebar.button("Predict"):
        try:
            # Convert input to float
            sepal_length = float(sepal_length)
            sepal_width = float(sepal_width)
            petal_length = float(petal_length)
            petal_width = float(petal_width)

            input_data = preprocess_input(sepal_length, sepal_width, petal_length, petal_width, scaler)

            if model_selected == "Artificial Neural Network":
                prediction = ann_model.predict(input_data)
                probabilities = prediction[0]
            elif model_selected == "Logistic Regression":
                prediction = logreg_model.predict_proba(input_data)
                probabilities = prediction[0]
            else:  # K-Nearest Neighbors
                prediction = knn_model.predict_proba(input_data)
                probabilities = prediction[0]

            # Extracting the predicted class and species
            if model_selected == "Logistic Regression" or model_selected == "K-Nearest Neighbors":
                predicted_class = np.argmax(prediction)
                species = le.inverse_transform([predicted_class])[0]
            else:
                predicted_class = np.argmax(prediction)
                species = le.classes_[predicted_class]

            # Display the predicted species
            st.write("## Predicted Species")
            st.write(species)
            # Display the corresponding image
            image_path = class_to_image.get(predicted_class)
            if image_path:
                image = Image.open(image_path)
                st.image(image, caption=f"Image for {species}", width=400)
            # Display the probability distribution
            st.write("## Probability Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(le.classes_, probabilities, color='skyblue', edgecolor='black')
            plt.xlabel('Species')
            plt.ylabel('Probability')
            plt.title('Probability Distribution')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)


        except Exception as e:
            st.error(f"Error predicting: {str(e)}")
else:
    st.error("Models could not be loaded. Please check the model files.")

# Show details on the left side
st.write("## Details")
st.write("### User Input:")
st.write(f"- Sepal Length: {sepal_length}")
st.write(f"- Sepal Width: {sepal_width}")
st.write(f"- Petal Length: {petal_length}")
st.write(f"- Petal Width: {petal_width}")

# Explanation about the selected model
st.write("### Model Used:")
if model_selected == "Artificial Neural Network":
    st.write("An artificial neural network model trained using TensorFlow/Keras.")
elif model_selected == "Logistic Regression":
    st.write("A logistic regression model trained using scikit-learn.")
else:
    st.write("A K-Nearest Neighbors model trained using scikit-learn.")
