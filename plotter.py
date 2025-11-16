import pandas as pd
import matplotlib.pyplot as plt

# --- Specify CSV files manually ---
csv_files = [
    "keypoint_counts_indoor.csv"
    # add or remove as needed
]

# --- Load and merge ---
dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)
merged_df["frame"] = range(len(merged_df))  # continuous frame numbering

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(merged_df["frame"], merged_df["yolo_keypoints"], label="YOLO keypoints", color="blue")
plt.plot(merged_df["frame"], merged_df["zed_keypoints"], label="ZED keypoints", color="red")
plt.xlabel("Frame index")
plt.ylabel("Valid keypoints")
plt.title("YOLO vs ZED Keypoints per Frame")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Totals ---
print(f"Files merged: {len(csv_files)}")
print(f"Total frames: {len(merged_df)}")
print(f"Total YOLO keypoints: {merged_df['yolo_keypoints'].sum()}")
print(f"Total ZED keypoints: {merged_df['zed_keypoints'].sum()}")
