import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scripts.unet import UNet
from scripts.preprocess import create_data_loaders
from tqdm import tqdm

# Load the trained model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("data/models/unet_model.pth"))
model.eval()

# Load the test dataset
_, test_loader = create_data_loaders(batch_size=1, image_size=128)

# Define metrics
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()  # Threshold predictions
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

# Evaluate on test set
dice_scores = []
for images, masks in tqdm(test_loader, desc="Evaluating on Test Set"):
    images, masks = images.to("cpu"), masks.to("cpu")
    with torch.no_grad():
        preds = model(images)
        dice_scores.append(dice_coefficient(preds, masks))

# Calculate and print average Dice Coefficient
average_dice = sum(dice_scores) / len(dice_scores)
print(f"Average Dice Coefficient on Test Set: {average_dice:.4f}")

# Results
# Average Dice Coefficient on Test Set: 0.8755