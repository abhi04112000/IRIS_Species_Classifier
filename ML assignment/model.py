import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_data = pd.read_csv(url, header=None, names=column_names)

# Preprocess the data
X = iris_data.iloc[:, :-1].values
y = iris_data.iloc[:, -1].values

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)

# Logistic Regression Model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)
logreg_predictions = logreg_model.predict(X_test_scaled)

# Artificial Neural Network Model
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Build the neural network model
ann_model = Sequential()
ann_model.add(Dense(8, input_dim=4, activation='relu'))
ann_model.add(Dense(3, activation='softmax'))

# Compile the model
ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
ann_model.fit(X_train_scaled, y_train_categorical, epochs=100, batch_size=10, verbose=1)

# Save the models, scaler, and label encoder
joblib.dump(knn_model, "iris_knn_model.pkl")
joblib.dump(logreg_model, "iris_logreg_model.pkl")
ann_model.save("iris_ann_model.h5")
joblib.dump(scaler, "iris_scaler.pkl")
joblib.dump(le, "iris_label_encoder.pkl")

# Evaluate models
knn_accuracy = accuracy_score(y_test, knn_predictions)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
_, ann_accuracy = ann_model.evaluate(X_test_scaled, y_test_categorical, verbose=0)

print("K-Nearest Neighbors Accuracy:", knn_accuracy)
print("Logistic Regression Accuracy:", logreg_accuracy)
print("Artificial Neural Network Accuracy:", ann_accuracy)
