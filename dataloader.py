import os
from torch.utils.data import Dataset
from PIL import Image

class DogDataset(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.classes = ['Beagle', 'Siberian Husky', 'Toy Poodle']
        self.class_to_folder = {
            'Beagle': 'Beagle',
            'Siberian Husky': 'Siberian Husky',
            'Toy Poodle': 'Toy Poodle'
        }
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.data = []
        for c in self.classes:
            class_dir = os.path.join(self.root_dir, 'StanfordDogs', 'Images', self.dataset_type, self.class_to_folder[c])
            for fname in os.listdir(class_dir):
                if fname.endswith('.jpg'):
                    path = os.path.join(class_dir, fname)
                    item = (path, self.class_to_idx[c])
                    self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
