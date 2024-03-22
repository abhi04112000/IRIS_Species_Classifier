import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Iris Species Classification App",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"  # Expand the sidebar by default
)

# Load the saved artificial neural network model, scaler, and label encoder
ann_model = load_model("iris_ann_model.h5")
scaler = joblib.load("iris_scaler.pkl")
le = joblib.load("iris_label_encoder.pkl")

# Load the saved logistic regression and KNN models
logreg_model = joblib.load("iris_logreg_model.pkl")
knn_model = joblib.load("iris_knn_model.pkl")

# Dictionary to map classes to images
class_to_image = {
    0: "Iris-Setosa.png",
    1: "Iris-Versicolor.png",
    2: "Iris-Virginica.png"
}

# Function to preprocess input data for prediction
def preprocess_input(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input = scaler.transform(input_data)
    return scaled_input

# Streamlit app
st.title("Iris Species Classification App")
with st.expander('APP Description'):
    st.write('The Iris Species Classification App is a web application built using Streamlit, a Python library for creating interactive web applications, along with machine learning models trained on the famous Iris dataset. The app allows users to input the features of an Iris flower (sepal length, sepal width, petal length, and petal width) and predicts the species of the Iris flower based on these features using various machine learning models.')

# Background color for the entire app
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input
st.sidebar.header("User Input")
sepal_length = st.sidebar.text_input("Enter Sepal Length", "5.0")
sepal_width = st.sidebar.text_input("Enter Sepal Width", "3.0")
petal_length = st.sidebar.text_input("Enter Petal Length", "4.0")
petal_width = st.sidebar.text_input("Enter Petal Width", "1.3")

# Model selection
model_selected = st.sidebar.radio("Select Model", ("Artificial Neural Network", "Logistic Regression", "K-Nearest Neighbors"))

# Make predictions based on the selected model
if st.sidebar.button("Predict"):
    # Convert input to float
    sepal_length = float(sepal_length)
    sepal_width = float(sepal_width)
    petal_length = float(petal_length)
    petal_width = float(petal_width)

    input_data = preprocess_input(sepal_length, sepal_width, petal_length, petal_width)
    
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
    fig, ax = plt.subplots(figsize=(6/2.54, 6/2.54))  # 6 cm x 6 cm size
    ax.bar(le.classes_, probabilities, color='skyblue', edgecolor='black')
    plt.xlabel('Species', fontsize=8)  # Adjusted font size for x-label
    plt.ylabel('Probability', fontsize=8)  # Adjusted font size for y-label
    plt.title('Probability Distribution', fontsize=10)  # Adjusted font size for title
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=6)  # Adjusted font size for x-ticks
    plt.yticks(fontsize=6)  # Adjusted font size for y-ticks
    plt.tight_layout()  # Ensures tight layout to minimize padding
    st.pyplot(fig)

else:
    st.sidebar.error("Please select a model and click Predict.")

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
