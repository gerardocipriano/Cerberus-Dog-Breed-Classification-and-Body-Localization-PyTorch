import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from net_runner import NetRunner
from DogBodyPartsDataset import DogBodyPartsDataset

if __name__ == "__main__":


    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to a fixed size
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    image_dir = 'BodyParts\images'
    annotation_dir = r'BodyParts\annotations'

    # Create an instance of the DogBodyPartsDataset
    dataset = DogBodyPartsDataset(image_dir, annotation_dir, transform=transform)

    shuffle = True
    batch_size = 16

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for images, annotations in data_loader:
        print("Batch Size:", images.shape[0])  # Number of images in the batch
        print("Image Shape:", images.shape)  # Shape of a single image in the batch
        print("Annotations Shape:", annotations.shape)  # Shape of the annotations tensor

    # Perform further operations with the images and annotations
    # ...


    """
    custom_trainset = CustomDataset(root='../generatore_forme/dst/training', transform=transform)
    custom_testset = CustomDataset(root='../generatore_forme/dst/test', transform=transform)
    pt_trainset = torchvision.datasets.ImageFolder(root='../generatore_forme/dst/training', transform=transform)
    pt_testset = torchvision.datasets.ImageFolder(root='../generatore_forme/dst/test', transform=transform)

    trainset = custom_trainset if custom else pt_trainset
    testset = custom_testset if custom else pt_testset
    classes = custom_trainset.classes

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    runner = NetRunner(classes, batch_size)

    if train:
        runner.train(trainloader, preview)
    else:
        runner.test(testloader, True, preview)

    for images, boxes in dataloader:
        # Perform your model training or evaluation here
        # images: tensor of shape (batch_size, channels, height, width)
        # boxes: tensor of shape (batch_size, num_boxes, 5), where 5 represents (xmin, ymin, xmax, ymax, class_id)
        pass
    """