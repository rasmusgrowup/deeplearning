# train.py
# Main script for training the U-Net model
import sys
import os

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from scripts.unet import UNet                 # U-Net model
from scripts.loss import BCEDiceLoss          # Loss function
from scripts.preprocess import create_data_loaders  # Data loaders
import torch.optim as optim                   # Optimizer
import torch                                  # PyTorch core
from tqdm import tqdm
import time

device = "cpu"
def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10, device=device):
    """
    Trains the U-Net model on the training dataset and evaluates on the validation dataset.

    Args:
        model (torch.nn.Module): The U-Net model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_fn (torch.nn.Module): Loss function (e.g., BCEDiceLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ("cpu").

    Returns:
        torch.nn.Module: The trained model.
    """
    model = model.to(device)

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0

        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_start_time = time.time()

        # Use tqdm for progress bar
        for images, masks in tqdm(train_loader, desc="Training", unit="batch"):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            preds = model(images)

            # Calculate loss
            loss = loss_fn(preds, masks)
            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")

        # Validation Phase
        if val_loader:
            val_loss = evaluate_model(model, val_loader, loss_fn, device)
            print(f"Validation Loss: {val_loss:.4f}")

        print(f"Epoch completed in {time.time() - epoch_start_time:.2f} seconds\n")

    print("Training Complete!")
    return model

def evaluate_model(model, val_loader, loss_fn, device=device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The U-Net model.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_fn (torch.nn.Module): Loss function (e.g., BCEDiceLoss).
        device (str): Device to evaluate on ("cpu").

    Returns:
        float: Validation loss.
    """
    model.eval()
    val_loss = 0

    # Use tqdm for progress bar
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating", unit="batch"):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            preds = model(images)
            loss = loss_fn(preds, masks)
            val_loss += loss.item()

    return val_loss / len(val_loader)

# Main Training Script
if __name__ == "__main__":
    # Initialize model, loss function, optimizer, and data loaders
    model = UNet(in_channels=3, out_channels=1)
    loss_fn = BCEDiceLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load data
    train_loader, val_loader = create_data_loaders(batch_size=16, image_size=128)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=20)

    # Save the trained model
    torch.save(trained_model.state_dict(), "unet_model.pth")