
Cerberus - v5 2023-05-23 12:24pm
==============================

This dataset was exported via roboflow.com on May 23, 2023 at 10:25 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 24 images.
Body-parts are annotated in YOLOv8 format.

The following pre-processing was applied to each image:

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random Gaussian blur of between 0 and 1 pixels

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -15 and +15 degrees
* Random exposure adjustment of between -25 and +25 percent


