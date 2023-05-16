from torch.utils.data import Dataset
from PIL import Image
import os
import xml.etree.ElementTree as ET
import torch

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_objects=10):
        self.root_dir = root_dir
        self.transform = transform
        self.max_objects = max_objects
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'Images')))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            img_dir = os.path.join(root_dir, 'Images', cls)
            ann_dir = os.path.join(root_dir, 'Annotation', cls)
            for img_file in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img_file)
                ann_file = os.path.splitext(img_file)[0]
                ann_path = os.path.join(ann_dir, ann_file)
                item = (img_path, ann_path, self.class_to_idx[cls])
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        tree = ET.parse(ann_path)
        objects = tree.findall('object')
        bboxes = []
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            bboxes.append([label, x_center, y_center, width, height])
        
        # Pad the list of bounding boxes with zeros if necessary
        while len(bboxes) < self.max_objects:
            bboxes.append([0] * 5)
        
        bboxes = torch.tensor(bboxes)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label