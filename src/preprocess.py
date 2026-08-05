import cv2
import os


def extract_vein(image_path, save_steps=False):

    # ==========================
    # Load Image
    # ==========================

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Image not found")

    # ==========================
    # Grayscale
    # ==========================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ==========================
    # CLAHE
    # ==========================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(gray)

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

    # ==========================
    # Gaussian Blur
    # ==========================

    blur = cv2.GaussianBlur(
        tophat,
        (7, 7),
        0
    )

    # ==========================
    # Adaptive Thresholding
    # ==========================

    threshold = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    # ==========================
    # Morphological Opening
    # ==========================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    opening = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )
    closing = cv2.morphologyEx(
       opening,
       cv2.MORPH_CLOSE,
       kernel
    )

    # ==========================
    # Save Processing Steps
    # ==========================

    if save_steps:

        os.makedirs(
            "static/outputs",
            exist_ok=True
        )

        cv2.imwrite(
            "static/outputs/grayscale.jpg",
            gray
        )

        cv2.imwrite(
            "static/outputs/clahe.jpg",
            clahe_img
        )

        cv2.imwrite(
            "static/outputs/tophat.jpg",
            tophat
        )

        cv2.imwrite(
            "static/outputs/threshold.jpg",
            threshold
        )

        cv2.imwrite(
            "static/outputs/opening.jpg",
            closing
        )

        cv2.imwrite(
            "static/outputs/final_vein.jpg",
            closing
        )

    return closing
