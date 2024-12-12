import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
from scripts.preprocess import create_data_loaders
from scripts.unet import UNet

def visualize_specific_sample(loader, model, sample_index, save_path=None):
    """
    Visualizes the input image, ground truth mask, and predicted mask for a specific sample index.
    
    Args:
        loader (DataLoader): DataLoader for the dataset (e.g., test_loader).
        model (torch.nn.Module): Trained segmentation model.
        sample_index (int): Index of the sample to visualize (0-based index).
        save_path (str): Optional. Path to save the visualization as an image.
    """
    model.eval()  # Ensure model is in evaluation mode
    
    # Iterate through the DataLoader to find the sample
    for i, (images, masks) in enumerate(loader):
        if i == sample_index:
            images, masks = images.to("cpu"), masks.to("cpu")
            
            with torch.no_grad():
                preds = model(images)
                preds = (preds > 0.5).float()  # Apply threshold for binary mask
            
            # Visualize the input image, ground truth, and predicted mask
            plt.figure(figsize=(15, 5))
            
            # Input Image
            plt.subplot(1, 3, 1)
            plt.imshow(images[0].permute(1, 2, 0).numpy())
            plt.title("Input Image")
            
            # Ground Truth
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0][0].numpy(), cmap="gray")
            plt.title("Ground Truth Mask")
            
            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(preds[0][0].numpy(), cmap="gray")
            plt.title("Predicted Mask")
            
            plt.tight_layout()
            
            # Save the plot if save_path is provided
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"Visualization saved to {save_path}")
            
            plt.show()
            return

    print(f"Sample index {sample_index} not found in the dataset.")

if __name__ == "__main__":
    # Load the test data loader
    _, test_loader = create_data_loaders(batch_size=1, image_size=128)
    
    # Load the trained model
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load("data/models/unet_model.pth"))  # Update path as needed
    model.eval()  # Set to evaluation mode
    
    # Visualize the specific sample
    sample_index = 92  # Set the sample index
    save_path = f"data/results/sample_{sample_index}_visualization.png"  # Set save path
    
    visualize_specific_sample(test_loader, model, sample_index, save_path)