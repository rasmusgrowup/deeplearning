import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from scripts.unet import UNet
from scripts.preprocess import create_data_loaders

# Load the test dataset
_, test_loader = create_data_loaders(batch_size=1, image_size=128)

# Load the trained model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("data/models/unet_model.pth"))  # Update path as necessary
model.eval()  # Set the model to evaluation mode

# Visualization Function
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()  # Ensure the model is in evaluation mode
    count = 0
    for images, masks in test_loader:
        images, masks = images.to("cpu"), masks.to("cpu")
        with torch.no_grad():
            preds = model(images)
            preds = (preds > 0.5).float()

        # Plot input, ground truth, and predictions
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(images[0].permute(1, 2, 0).numpy())
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(masks[0][0].numpy(), cmap="gray")
        plt.title("Ground Truth Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(preds[0][0].numpy(), cmap="gray")
        plt.title("Predicted Mask")
        plt.show()

        count += 1
        if count >= num_samples:
            break

# Call the visualization function
visualize_predictions(model, test_loader)