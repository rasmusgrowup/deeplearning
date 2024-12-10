import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")

        # Load corresponding mask
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.jpg', '_segmentation.png'))
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define transformations
def get_transforms(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images and masks
        transforms.ToTensor(),                       # Convert to PyTorch tensors
    ])

# Create data loaders
def create_data_loaders(batch_size=16, image_size=128):
    transform = get_transforms(image_size)

    # Define dataset paths
    train_image_dir = "data/images/"
    train_mask_dir = "data/masks/"
    test_image_dir = "data/test_images/"
    test_mask_dir = "data/test_masks/"

    # Create datasets
    train_dataset = SkinLesionDataset(train_image_dir, train_mask_dir, transform)
    test_dataset = SkinLesionDataset(test_image_dir, test_mask_dir, transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Test the data loaders
if __name__ == "__main__":
    train_loader, test_loader = create_data_loaders()

    # Visualize a sample batch
    images, masks = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")

    import matplotlib.pyplot as plt

# Visualize images and masks from the DataLoader
def visualize_batch(data_loader, n_samples=4):
    images, masks = next(iter(data_loader))
    plt.figure(figsize=(15, 10))

    for i in range(n_samples):
        # Plot image
        plt.subplot(2, n_samples, i+1)
        plt.imshow(images[i].permute(1, 2, 0))  # Convert [C, H, W] to [H, W, C]
        plt.title("Image")
        plt.axis("off")

        # Plot mask
        plt.subplot(2, n_samples, i+1+n_samples)
        plt.imshow(masks[i][0], cmap="gray")  # Show mask as grayscale
        plt.title("Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Test visualization
train_loader, _ = create_data_loaders()
visualize_batch(train_loader, n_samples=4)