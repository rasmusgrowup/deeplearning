import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd

# Load IoU results from the CSV file
iou_df = pd.read_csv("data/results/iou_results.csv")
iou_scores = iou_df["IoU Score"][:-1].astype(float)  # Exclude the "Average" row
average_iou = float(iou_df["IoU Score"].iloc[-1])

# Plot IoU scores
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(iou_scores) + 1), iou_scores, label="IoU Scores")
plt.axhline(y=average_iou, color='r', linestyle='--', label=f"Average IoU: {average_iou:.4f}")
plt.xlabel("Sample Index")
plt.ylabel("IoU Score")
plt.title("IoU Scores for Test Set")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

# Save the plot
plt.savefig("data/results/iou_scores_plot.png", dpi=300)
plt.show()