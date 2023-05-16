# Load the dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from StanfordDogsDataset import StanfordDogsDataset

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = StanfordDogsDataset(root_dir='StanfordDogs', transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Loaded {len(dataset)} samples from {len(dataset.classes)} classes.")

# Split the dataset into training, validation and test sets
from torch.utils.data import random_split

print("Splitting dataset...")
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print(f"Split dataset into {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples.")

# Create data loaders for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
