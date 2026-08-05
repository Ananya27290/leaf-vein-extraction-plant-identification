import cv2
import numpy as np

# ==========================
# Load Image
# ==========================

image_path = r"dataset\archive (3)\Leaves\1087.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# ==========================
# Grayscale
# ==========================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    "output/grayscale/1087_gray.jpg",
    gray
)

# ==========================
# CLAHE
# ==========================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_img = clahe.apply(gray)

cv2.imwrite(
    "output/clahe/1087_clahe.jpg",
    clahe_img
)

# ==========================
# Top-Hat Transform
# ==========================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (21, 21)
)

tophat = cv2.morphologyEx(
    clahe_img,
    cv2.MORPH_TOPHAT,
    kernel
)

cv2.imwrite(
    "output/tophat/1087_tophat.jpg",
    tophat
)

# ==========================
# Gaussian Blur
# ==========================

blur = cv2.GaussianBlur(
    tophat,
    (7, 7),
    0
)

cv2.imwrite(
    "output/blur/1087_blur.jpg",
    blur
)

# ==========================
# Adaptive Threshold
# ==========================

adaptive = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    8
)

cv2.imwrite(
    "output/threshold/1087_adaptive.jpg",
    adaptive
)

print("Threshold image saved successfully")

# ==========================
# Morphological Opening
# ==========================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

opening = cv2.morphologyEx(
    adaptive,
    cv2.MORPH_OPEN,
    kernel
)

cv2.imwrite(
    "output/morphology/1087_opening.jpg",
    opening
)

print("Opening completed")

# ==========================
# Final Vein Image
# ==========================

cv2.imwrite(
    "output/final_vein/1087_final_vein.jpg",
    opening
)

print("Final vein image generated successfully")