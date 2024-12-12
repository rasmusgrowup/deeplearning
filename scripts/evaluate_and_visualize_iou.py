import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import os

# IoU computation function
def compute_iou(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()  # Apply threshold for binary predictions
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Function to visualize and save a sample
def visualize_and_save_sample(image, ground_truth, prediction, sample_index, save_folder):
    plt.figure(figsize=(15, 5))
    
    # Input Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(f"Input Image (Sample {sample_index})")
    
    # Ground Truth Mask
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth[0].numpy(), cmap="gray")
    plt.title("Ground Truth Mask")
    
    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(prediction[0].numpy(), cmap="gray")
    plt.title("Predicted Mask")
    
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_folder, f"low_iuo_score_sample_{sample_index}_visualization.png")
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")
    plt.close()

# Evaluate IoU and visualize/save low scores
def evaluate_and_save_low_iou_samples(model, test_loader, save_folder):
    model.eval()  # Set the model to evaluation mode
    low_iou_threshold = 0.5  # Threshold for low IoU
    sample_index = 0  # Counter to track sample indices

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to("cpu"), masks.to("cpu")
            preds = model(images)
            iou_scores = compute_iou(preds, masks)

            # Check if any IoU score is below the threshold
            for i, iou in enumerate(iou_scores):
                if iou < low_iou_threshold:
                    print(f"Low IoU Detected (Sample {sample_index}, IoU: {iou.item():.4f})")
                    visualize_and_save_sample(images[i], masks[i], preds[i], sample_index, save_folder)
            
            sample_index += 1

# Load the test data loader
from scripts.preprocess import create_data_loaders
from scripts.unet import UNet

_, test_loader = create_data_loaders(batch_size=1, image_size=128)

# Load the trained model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("data/models/unet_model.pth"))  # Update path as needed

# Define the folder to save the results
results_folder = "data/results/low_iou_samples"

# Run the evaluation and save visualizations
evaluate_and_save_low_iou_samples(model, test_loader, results_folder)