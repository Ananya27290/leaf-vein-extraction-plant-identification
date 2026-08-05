import pandas as pd

df = pd.read_csv(
    "dataset/archive (3)/Leaves/all.csv"
)

for cls in range(32):

    sample = df[df["y"] == cls].iloc[0]

    print(
        f"Class {cls}:",
        sample["id"]
    )