import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize

class DogPartsDataset(Dataset):
    def __init__(self, data_dir, target_size=(224, 224)):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.annotation_dir = os.path.join(data_dir, 'annotations')
        self.image_files = sorted(os.listdir(self.image_dir))
        self.target_size = target_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.jpg', '.txt'))

        image = Image.open(image_path).convert('RGB')
        image = Resize(self.target_size)(image)
        annotation = self._parse_annotation(annotation_path, image.size)

        # Apply any desired data augmentation transformations here
        # e.g., random cropping, flipping, rotation, etc.

        # Convert image and annotation to tensors
        image = ToTensor()(image)
        annotation = torch.tensor(annotation)

        return image, annotation

    def _parse_annotation(self, annotation_path, image_size):
        image_width, image_height = image_size

        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        annotation = []
        for line in lines:
            values = line.strip().split()

            # Parse the object type and bounding box coordinates
            object_type = int(values[0])
            x, y, w, h = map(float, values[1:])

            # Convert relative coordinates to absolute coordinates
            x = x * image_width
            y = y * image_height
            w = w * image_width
            h = h * image_height

            # Append the bounding box coordinates and object type to the annotation list
            annotation.append([object_type, x, y, w, h])

        return annotation


# Example usage:
data_dir = r'BodyParts'
dataset = DogPartsDataset(data_dir)
image, annotation = dataset[0]
print('Image shape:', image.shape)
print('Annotation:', annotation)
