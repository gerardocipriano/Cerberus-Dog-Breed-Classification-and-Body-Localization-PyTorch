import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class DogBodyPartsDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, num_body_parts=3, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.num_body_parts = num_body_parts

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, image_file.replace(".jpg", ".txt"))

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Load the bounding box annotations
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()

        processed_annotations = []
    
        for annotation in annotations:
            # Split the annotation values (class_label, x_center, y_center, width, height)
            class_label, x_center, y_center, width, height = map(float, annotation.strip().split())

            # Convert the YOLO format to absolute pixel values or normalized values based on your model's requirements
            # Example: convert normalized values to absolute pixel values
            image_width, image_height = image.size
            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height

            # Convert the coordinates to the format expected by the model (e.g., [x_min, y_min, x_max, y_max])
            processed_annotation = [x_min, y_min, x_max, y_max, class_label]

            processed_annotations.append(processed_annotation)

        # Pad or truncate the annotations to match self.num_body_parts
        num_annotations = len(processed_annotations)
        if num_annotations < self.num_body_parts:
            # Pad the annotations with dummy values
            pad_size = self.num_body_parts - num_annotations
            dummy_annotation = [0.0, 0.0, 0.0, 0.0, -1.0]  # Dummy values for padding
            processed_annotations.extend([dummy_annotation] * pad_size)
        elif num_annotations > self.num_body_parts:
            # Truncate the extra annotations
            processed_annotations = processed_annotations[:self.num_body_parts]

        if self.transform:
            image = self.transform(image)

        annotations_tensor = torch.tensor(processed_annotations, dtype=torch.float32)
        return image, annotations_tensor