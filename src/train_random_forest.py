import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    "features/flavia_features.csv"
)

# ==========================
# Features and Labels
# ==========================

X = df.drop(
    columns=["image", "class"]
)

y = df["class"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Random Forest Model
# ==========================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# ==========================
# Training
# ==========================

print("\nTraining Started...")

model.fit(
    X_train,
    y_train
)

print("Training Completed!")

# ==========================
# Prediction
# ==========================

y_pred = model.predict(
    X_test
)

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nAccuracy : {accuracy*100:.2f}%"
)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================
# Save Model
# ==========================

pickle.dump(
    model,
    open(
        "models/leaf_classifier.pkl",
        "wb"
    )
)

print(
    "\nRandom Forest Model Saved Successfully!"
)