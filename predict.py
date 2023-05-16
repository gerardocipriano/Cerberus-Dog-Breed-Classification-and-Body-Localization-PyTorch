from PIL import Image
from torchvision import transforms
from NetRunner import *

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image
image = Image.open('dog.jpg')

# Preprocess the image
input_tensor = preprocess(image)

# Add a batch dimension to the tensor
input_batch = input_tensor.unsqueeze(0)

# Create an instance of the NetRunner class
net_runner = NetRunner(train_loader, val_loader, test_loader)

# Load the trained model weights
net_runner.model.load_state_dict(torch.load('model.pth'))

# Make a prediction using the trained model
predicted = net_runner.predict(input_batch)

# Print the predicted breed
breed_names = ['Beagle', 'Husky', 'Toy Poodle']
print(f"Predicted breed: {breed_names[predicted.item()]}")
