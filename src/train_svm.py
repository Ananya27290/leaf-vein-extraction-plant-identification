import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# ==========================
# SVM Model
# ==========================

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)

model.fit(
    X_train,
    y_train
)

# ==========================
# Prediction
# ==========================

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
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
        "models/svm_model.pkl",
        "wb"
    )
)

pickle.dump(
    scaler,
    open(
        "models/scaler.pkl",
        "wb"
    )
)

print("\nSVM Model Saved Successfully")