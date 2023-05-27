import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DogBodyPartsDataset import DogBodyPartsDataset
from DogBodyPartsDetector import DogBodyPartsDetector
from net_runner import NetRunner

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 16

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define dataset
    image_dir = 'BodyParts/images'
    annotation_dir = 'BodyParts/annotations'
    dataset = DogBodyPartsDataset(image_dir, annotation_dir, transform=transform)

    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    num_classes = 3  # Number of body parts to detect
    model = DogBodyPartsDetector(num_classes).to(device)

    # Define loss function
    criterion = torch.nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create an instance of NetRunner
    net_runner = NetRunner(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    # Train the model
    net_runner.train()
    validation_loss = net_runner.evaluate()
    print("Validation Loss:", validation_loss)
    
    # Save the trained model
    torch.save(model.state_dict(), 'dog_body_parts_detector.pth')