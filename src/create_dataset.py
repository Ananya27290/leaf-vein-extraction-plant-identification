import pandas as pd
from pathlib import Path

from preprocess import extract_vein
from feature_extraction import extract_features

# ==========================
# Dataset Paths
# ==========================

csv_file = r"dataset/archive (3)/Leaves/all.csv"

image_folder = r"dataset/archive (3)/Leaves"

# ==========================
# Read Labels
# ==========================

df = pd.read_csv(
    csv_file
)

dataset = []

# ==========================
# Process Images
# ==========================

for _, row in df.iterrows():

    image_name = row["id"]

    class_label = row["y"]

    image_path = str(
        Path(image_folder) /
        image_name
    )

    try:

        import cv2

        original_image = cv2.imread(
            image_path
        )

        vein_image = extract_vein(
            image_path
        )

        features = extract_features(
            original_image,
            vein_image
        )

        if features is None:
            continue

        dataset.append(
            [image_name, class_label]
            + features
        )

        print(
            f"Processed {image_name}"
        )

    except Exception as e:

        print(
            f"Error: {image_name}"
        )

# ==========================
# Column Names
# ==========================

columns = [

    "image",
    "class",

    "vein_area",
    "vein_density",
    "components",

    "leaf_area",
    "perimeter",
    "aspect_ratio",
    "extent",
    "solidity",

    "major_axis",
    "minor_axis",
    "eccentricity",
    "equivalent_diameter",

    "contrast",
    "correlation",
    "energy",
    "homogeneity",

    
]

# ==========================
# Create DataFrame
# ==========================

feature_df = pd.DataFrame(
    dataset,
    columns=columns
)

# ==========================
# Save CSV
# ==========================

feature_df.to_csv(
    "features/flavia_features.csv",
    index=False
)

print("\nDataset Created Successfully")
print(feature_df.head())