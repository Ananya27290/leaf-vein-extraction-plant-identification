from src.preprocess import extract_vein
import cv2

vein = extract_vein(
    r"dataset/archive (3)/Leaves/1087.jpg"
)

cv2.imwrite(
    "output/final_vein/test.jpg",
    vein
)

print("Success")