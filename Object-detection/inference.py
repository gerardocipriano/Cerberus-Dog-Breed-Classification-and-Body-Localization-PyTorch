"""
import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/Cerberus/training_test_13/weights/best.pt')

# Preprocess the image
image_path = '1.jpg'
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0)

# Run inference
output = model(input_image)

# Access the predicted bounding boxes, labels, and scores
# Modify this section based on the structure of the output tensor
# Example structure: output.pred[0].xyxy, output.pred[0].labels, output.pred[0].scores
boxes = output.pred[0].xyxy  # Bounding box coordinates (xmin, ymin, xmax, ymax)
labels = output.pred[0].labels  # Class labels
scores = output.pred[0].scores  # Confidence scores

# Visualize the predictions
output.show()  # Show the annotated image with predicted bounding boxes

# Iterate over the predictions
for box, label, score in zip(boxes, labels, scores):
    print(f"Label: {label}, Score: {score:.2f}, Box: {box}")
"""
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp7/weights/best.pt', force_reload=True)
# path='./yolov5/Cerberus/training_test_13/weights/best.pt'
# Image
im = '1.jpg'
# Inference
results = model(im)

print(results.pandas().xyxy[0])

# Access the bounding box predictions
boxes = results.pandas().xyxy[0]

# Load the image
image = cv2.imread(im)

# Iterate over the bounding boxes
for _, box in boxes.iterrows():
    # Extract the coordinates
    xmin, ymin, xmax, ymax = box[['xmin', 'ymin', 'xmax', 'ymax']]

    # Draw the bounding box on the image
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

# Save the image with bounding boxes
output_path = 'output.jpg'
cv2.imwrite(output_path, image)

print(f"Image with bounding boxes saved to: {output_path}")

"""import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp7/weights/best.pt')
# path='./yolov5/Cerberus/training_test_13/weights/best.pt'
# Image
im = './1.jpg'

# Inference
results = model(im)

print(results.pandas().xyxy[0])

# Access the bounding box predictions
boxes = results.pandas().xyxy[0]

# Load the image
image = plt.imread(im)

# Create a figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(image)

# Iterate over the bounding boxes
for _, box in boxes.iterrows():
    # Extract the coordinates
    xmin, ymin, xmax, ymax = box[['xmin', 'ymin', 'xmax', 'ymax']]

    # Calculate the width and height of the bounding box
    width = xmax - xmin
    height = ymax - ymin

    # Create a rectangle patch
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the rectangle patch to the axes
    ax.add_patch(rect)

# Show the plot
plt.show()
"""