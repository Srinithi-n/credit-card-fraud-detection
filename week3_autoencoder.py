import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================
# Scale Features
# ==========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train only on normal transactions

X_train_normal = X_train[y_train == 0]

# ==========================
# Build Autoencoder
# ==========================

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

# ==========================
# Train Model
# ==========================

history = autoencoder.fit(
    X_train_normal,
    X_train_normal,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# ==========================
# Reconstruction Error
# ==========================

reconstructions = autoencoder.predict(X_test)

mse = np.mean(np.square(X_test - reconstructions), axis=1)

threshold = np.percentile(mse, 95)

predictions = (mse > threshold).astype(int)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report")
print(classification_report(y_test, predictions))