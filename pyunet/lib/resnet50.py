import torch.nn as nn
import torchvision.models as models

class ResNetSegmentation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetSegmentation, self).__init__()

        # Load the pre-trained ResNet50 model
        self.backbone = models.resnet50(weights=None)

        # Replace the first layer with a convolutional layer with the desired input channels
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the fully connected layer and average pooling layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Define the segmentation head with the desired output channels
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, ch_out, kernel_size=1)  # Output: ch_out channels
        )
