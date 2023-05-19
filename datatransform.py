from torchvision import transforms

class DataTransform:
    def __init__(self, resize=256, center_crop=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], augment=True):
        transform_list = [
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop)
        ]
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                transforms.RandomAffine(degrees=10)
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, image):
        return self.transform(image)

