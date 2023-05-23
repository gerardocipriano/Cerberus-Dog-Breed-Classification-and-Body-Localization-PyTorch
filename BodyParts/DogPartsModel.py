import torchvision
import torch.nn as nn

class DogPartsModel(nn.Module):
    def __init__(self, num_classes):
        super(DogPartsModel, self).__init__()

        # Define the backbone CNN (e.g., ResNet, VGG, or MobileNet)
        self.backbone = torchvision.models.resnet50(pretrained=True)
        # Modify the last layer to output the desired number of features for each body part
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes * 5)  # 5 values per body part (object type, x, y, w, h)

    def forward(self, x):
        # Pass the input through the backbone CNN
        features = self.backbone(x)

        # Reshape the features to separate object type and bounding box information
        features = features.view(features.size(0), -1, 5)  # Reshape to (batch_size, num_classes, 5)

        return features

# Example usage:
num_classes = 3  # Number of body parts (nose, eyes, tail)
model = DogPartsModel(num_classes)
print(model)
