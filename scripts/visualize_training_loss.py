import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "data/results/training_results.csv"  # Adjust the path if necessary
df = pd.read_csv(file_path)

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Training Loss"], label="Training Loss", marker="o")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("data/results/loss_trends.png")  # Save the plot as a PNG
plt.show()