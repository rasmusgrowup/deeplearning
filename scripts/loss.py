import torch
import torch.nn as nn

# Dice Loss
def dice_loss(preds, targets, smooth=1e-6):
    """
    Computes Dice Loss for binary segmentation.
    Args:
        preds (torch.Tensor): Predicted segmentation masks (output of the model).
        targets (torch.Tensor): Ground truth segmentation masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice loss value.
    """
    preds = preds.contiguous()
    targets = targets.contiguous()

    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()

# Combined BCE + Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Combines Binary Cross-Entropy (BCE) Loss and Dice Loss.
        Args:
            alpha (float): Weight for BCE Loss in the combined loss.
        """
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        """
        Computes the combined BCE + Dice Loss.
        Args:
            preds (torch.Tensor): Predicted segmentation masks.
            targets (torch.Tensor): Ground truth segmentation masks.

        Returns:
            float: Combined loss value.
        """
        bce = self.bce(preds, targets)
        dice = dice_loss(preds, targets)
        return self.alpha * bce + (1 - self.alpha) * dice