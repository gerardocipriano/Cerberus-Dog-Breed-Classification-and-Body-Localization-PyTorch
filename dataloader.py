import os
from torch.utils.data import Dataset
from PIL import Image

class DogBreedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.breeds = ['n02088364-beagle', 'n02110185-Siberian_husky', 'n02113624-toy_poodle']
        self.data = []
        for breed in self.breeds:
            breed_dir = os.path.join(self.root_dir, 'Images', breed)
            for file in os.listdir(breed_dir):
                if file.endswith('.jpg'):
                    img_path = os.path.join(breed_dir, file)
                    annotation_path = os.path.join(self.root_dir, 'Annotation', breed, file.replace('.jpg', '.xml'))
                    self.data.append((img_path, annotation_path, breed))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, annotation_path, breed = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, breed