import torch
from torchvision import transforms
from PIL import Image

class Prediction:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.breeds = ['Beagle', 'Siberian Husky', 'Toy Poodle']

    def predict(self, img_path):
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
        output = self.model(image)
        _, predicted = torch.max(output.data, 1)
        return self.breeds[predicted]
