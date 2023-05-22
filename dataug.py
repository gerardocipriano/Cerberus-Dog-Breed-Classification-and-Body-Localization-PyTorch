import os
import random
import shutil
import imgaug as ia
import imageio.v2 as imageio
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def augment_images_with_bboxes(image_dir, bbox_dir, output_image_dir, output_bbox_dir, num_images):
    # Create output directories if they don't exist
    #os.makedirs(output_image_dir, exist_ok=True)
    #os.makedirs(output_bbox_dir, exist_ok=True)
    
    # List the images and their corresponding bounding boxes
    images = os.listdir(image_dir)
    bboxes = os.listdir(bbox_dir)

    # Define the augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-40, 40)),
        iaa.Multiply((0.2, 1.8)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.MultiplyHueAndSaturation(mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)),
        iaa.AverageBlur(k=(3, 7)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 1.5))
    ])
    
    # Initialize a counter for generated images
    generated_images = 0
    
    # Generate augmented images
    while generated_images < num_images:
        # Select a random image and its corresponding bounding box
        image = random.choice(images)
        bbox = image.replace('.jpg', '.txt')
        
        image_path = os.path.join(image_dir, image)
        bbox_path = os.path.join(bbox_dir, bbox)
        
        # Read the image and its bounding box
        image_data = imageio.imread(image_path)
        bbox_data = read_bboxes(bbox_path)
        
        # Create the BoundingBoxesOnImage object
        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbox[1], y1=bbox[2], x2=bbox[3], y2=bbox[4]) for bbox in bbox_data
        ], shape=image_data.shape)
        
        # Apply the augmentation to the image and bounding box
        augmented_image, augmented_bbs = seq(images=[image_data], bounding_boxes=[bbs])
        
        # Save the augmented image and bounding box in the output directories
        output_image_path = os.path.join(output_image_dir, f"augmented_{generated_images}.jpg")
        output_bbox_path = os.path.join(output_bbox_dir, f"augmented_{generated_images}.txt")
        
        imageio.imsave(output_image_path, augmented_image[0])
        save_bboxes(output_bbox_path, [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in augmented_bbs[0].bounding_boxes])
        
        generated_images += 1
    
    print("Augmentation completed!")

def read_bboxes(bbox_path):
    with open(bbox_path, 'r') as f:
        lines = f.readlines()
        bboxes = [[float(val) for val in line.strip().split()] for line in lines]
    return bboxes


def save_bboxes(bbox_path, bboxes):
    with open(bbox_path, 'w') as f:
        for bbox in bboxes:
            f.write(' '.join(str(val) for val in bbox) + '\n')

# Example usage
image_dir = r'Images\02088364-beagle'  # Percorso delle immagini originali
bbox_dir = r'Annotation\02088364-beagle'  # Percorso delle bounding box originali

output_image_dir = r'Output\Images\02088364-beagle'  # Percorso delle immagini modificate di output
output_bbox_dir = r'Output\Annotation\02088364-beagle' 

num_images = 5  # Number of images/bounding boxes to generate


augment_images_with_bboxes(image_dir, bbox_dir, output_image_dir, output_bbox_dir, num_images)



