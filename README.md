# Pill Similarity with Siamese Networks in Pytorch

## Business Case
**Problem Statement:** Incorrect drug ingestion causes both harm to individual and is costly to the healthcare system.

**Solution Objective:** Accurately and efficiently identify pills by way of image capture.

**Focus Use Case:**  Help people accurately and automatically identify pills based on images taken through a mobile device before. 

## Data

The data for this project comes from the [NIH Pill Image Recognition Challenge](https://pir.nlm.nih.gov/challenge/).

## Pill Image Detection

For a detailed description of the training process see the [Pill-Detection Repo]().

## Siamese Network

The siamese network is two CNNs that share weights, learning the same features. 

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/siamese-network.png)

The input is pairs of pills. 

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/pill-pairings.png)

The output is a dissimilarity score where the higher the more dissimilar the pills are. 

![](https://raw.githubusercontent.com/mepotts/Pill-Siamese-Network/master/network-output.png)

