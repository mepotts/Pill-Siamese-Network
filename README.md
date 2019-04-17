# Pill Similarity with Siamese Networks in Pytorch

## Demo
git pull and run Pill Demo notebook.

pill object detection can be found [here](https://github.com/mepotts/Pill-Detection/blob/master/pill-detect-demo.mp4).

## Business Case
**Problem Statement:** Incorrect drug ingestion causes both harm to individual and is costly to the healthcare system.

**Solution Objective:** Accurately and efficiently identify pills by way of image capture. The solution uses a siamese network, which is similar to Apple's FaceID system. Users input their pills into the app, then when they need to identify a pill without a bottle they simply take a picture of the pill to identify it. 

**Focus Use Case:**  Help people accurately and automatically identify pills based on images taken through a mobile device before. 

## Data

The data for this project comes from the [NIH Pill Image Recognition Challenge](https://pir.nlm.nih.gov/challenge/).

## Pill Image Detection

For a detailed description of the training process see the [Pill-Detection Repo](https://github.com/mepotts/Pill-Detection).

**Training Data set:** 200 annotated images based on the reference image and consumer quality image

**Model:** TensorflowLite model (ssd_mobilenet_v2_coco) trained on annotated pill images. 

**Inference Speed (standard):**  31ms

**Model Size:** 70 MB

**Results:** 5,000 consumer taken image, 4,882 identified a pill. 117 were not identified. 97.64% accuracy. The quality of the auto cropping can also be improved.

## Siamese Network

The siamese network is two CNNs that share weights, learning the same features. 

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/images/siamese-network.png)

The input is pairs of pills. 

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/images/pill-pairings.png)

The output is a dissimilarity score where the higher the more dissimilar the pills are. 

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/images/network-output.png)

Accuracy was tested on multiple inventory levels of pills. 

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/images/accuracy-inventory.png)

For the inventory of 50 pills, accuracy looks like the image below.

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/images/accuracy-50.png)

## Siamese Results

**Training Data set:** 4,832 cropped pill images based on the consumer quality images

**Model:** Custom Siamese Network. 

**Inference Speed :**  24ms

**Model Size:** 20 MB

**Results:** Depends on pill inventory

### Combined Pipeline

**Inference Speed:** 31 ms + (24 ms * Inventory Size(5-50)) = 151 ms - 1231ms

**Model Size:**  70 MB + 20 MB = 90 MB

**Est. App Size:** <250 MB (Airbnb 180 MB, Uber 205 MB)
