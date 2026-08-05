import cv2
import numpy as np

from skimage.feature import graycomatrix
from skimage.feature import graycoprops




def extract_features(original_image,vein_image):

    gray_leaf = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2GRAY
    )

    _, leaf_mask = cv2.threshold(
        gray_leaf,
        240,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    leaf_contour = max(
        contours,
        key=cv2.contourArea
    )
    # ==========================
    # Feature 1
    # Vein Area
    # ==========================

    vein_area = cv2.countNonZero(
        vein_image
    )

    total_pixels = (
        vein_image.shape[0]
        * vein_image.shape[1]
    )

    vein_density = (
        vein_area / total_pixels
    )

    # ==========================
    # Feature 2
    # Components
    # ==========================

    num_labels, labels = cv2.connectedComponents(
        vein_image
    )

    components = num_labels - 1

    
    # ==========================
    # Shape Features
    # ==========================

    leaf_area = cv2.contourArea(
        leaf_contour
    )

    perimeter = cv2.arcLength(
        leaf_contour,
        True
    )

    x, y, w, h = cv2.boundingRect(
        leaf_contour
    )

    aspect_ratio = w / h

    extent = leaf_area / (
        w * h
    )

    hull = cv2.convexHull(
        leaf_contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    solidity = (
        leaf_area / hull_area
        if hull_area > 0
        else 0
    )

    # ==========================
    # Ellipse Features
    # ==========================

    if len(leaf_contour) >= 5:

        ellipse = cv2.fitEllipse(
            leaf_contour
        )

        major_axis = max(
            ellipse[1]
        )

        minor_axis = min(
            ellipse[1]
        )

    else:

        major_axis = 0
        minor_axis = 0

    eccentricity = np.sqrt(
        1 -
        (
            minor_axis ** 2 /
            (major_axis ** 2 + 1e-10)
        )
    )

    equivalent_diameter = np.sqrt(
        4 * leaf_area / np.pi
    )

    # ==========================
    # GLCM Features
    # ==========================

    glcm = graycomatrix(
        vein_image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        "contrast"
    )[0, 0]

    correlation = graycoprops(
        glcm,
        "correlation"
    )[0, 0]

    energy = graycoprops(
        glcm,
        "energy"
    )[0, 0]

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    )[0, 0]

    # ==========================
    # Statistical Features
    # ==========================

    mean_intensity = np.mean(
        vein_image
    )

    std_intensity = np.std(
        vein_image
    )

    hist = cv2.calcHist(
        [vein_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist = hist / hist.sum()

    entropy = -np.sum(
        hist * np.log2(
            hist + 1e-10
        )
    )

    # ==========================
    # Vein Length
    # ==========================

    vein_length = vein_area

    # ==========================
    # Final Feature Vector
    # ==========================

    features = [

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
        homogeneity,

       
    ]
    
    return features
  