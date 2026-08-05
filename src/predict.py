import cv2
import pickle
import numpy as np

from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from class_names import CLASS_NAMES

print("HELLO PREDICT FILE RUNNING")

# ==========================
# Load Model
# ==========================

model = pickle.load(
    open(
        "models/leaf_classifier.pkl",
        "rb"
    )
)

# ==========================
# Input Image
# ==========================

image_path = r"dataset/archive (3)/Leaves/1300.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# ==========================
# Preprocessing
# ==========================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

clahe_img = clahe.apply(gray)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (21,21)
)

tophat = cv2.morphologyEx(
    clahe_img,
    cv2.MORPH_TOPHAT,
    kernel
)

blur = cv2.GaussianBlur(
    tophat,
    (7,7),
    0
)

adaptive = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    8
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5,5)
)

opening = cv2.morphologyEx(
    adaptive,
    cv2.MORPH_OPEN,
    kernel
)

# ==========================
# Feature Extraction
# ==========================

vein_area = cv2.countNonZero(
    opening
)

total_pixels = (
    opening.shape[0]
    * opening.shape[1]
)

vein_density = (
    vein_area / total_pixels
)

num_labels, labels = cv2.connectedComponents(
    opening
)

components = num_labels - 1

contours, _ = cv2.findContours(
    opening,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

largest_contour = max(
    contours,
    key=cv2.contourArea
)

leaf_area = cv2.contourArea(
    largest_contour
)

perimeter = cv2.arcLength(
    largest_contour,
    True
)

x, y, w, h = cv2.boundingRect(
    largest_contour
)

aspect_ratio = w / h

extent = leaf_area / (w * h)

hull = cv2.convexHull(
    largest_contour
)

hull_area = cv2.contourArea(
    hull
)

solidity = leaf_area / hull_area

# ==========================
# Additional Shape Features
# ==========================

ellipse = cv2.fitEllipse(
    largest_contour
)

major_axis = max(
    ellipse[1]
)

minor_axis = min(
    ellipse[1]
)

eccentricity = np.sqrt(
    1 - (minor_axis**2 / major_axis**2)
)

equivalent_diameter = np.sqrt(
    4 * leaf_area / np.pi
)

# ==========================
# GLCM Features
# ==========================

glcm = graycomatrix(
    gray,
    distances=[1],
    angles=[0],
    levels=256,
    symmetric=True,
    normed=True
)

contrast = graycoprops(
    glcm,
    'contrast'
)[0,0]

correlation = graycoprops(
    glcm,
    'correlation'
)[0,0]

energy = graycoprops(
    glcm,
    'energy'
)[0,0]

homogeneity = graycoprops(
    glcm,
    'homogeneity'
)[0,0]

# ==========================
# Prediction
# ==========================

features = np.array([[
    vein_area,
    vein_density,
    components,
    leaf_area,
    perimeter,
    aspect_ratio,
    extent,
    solidity,
    major_axis,
    minor_axis,
    eccentricity,
    equivalent_diameter,
    contrast,
    correlation,
    energy,
    homogeneity
]])

print(features.shape)
print(features)
print("vein_area =", vein_area)
print("vein_density =", vein_density)
print("components =", components)
print("leaf_area =", leaf_area)

prediction = model.predict(
    features
)

probabilities = model.predict_proba(
    features
)

confidence = np.max(
    probabilities
) * 100
predicted_class = int(
    prediction[0]
)

if confidence < 60:

    print(
        "\nPlant not found in Flavia dataset"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )

else:

    plant_name = CLASS_NAMES.get(
        predicted_class,
        "Unknown Plant"
    )

    print(
        "\nPredicted Class :",
        predicted_class
    )

    print(
        "Plant Name      :",
        plant_name
    )

    print(
        f"Confidence      : {confidence:.2f}%"
    )