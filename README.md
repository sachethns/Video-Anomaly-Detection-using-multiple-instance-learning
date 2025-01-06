# Video Anomaly Detection using surveillance videos

This repository contains an implementation for detecting anomalies in surveillance videos using the Multiple Instance Learning (MIL) algorithm. The model is trained on the UCF Crime Dataset, which features various instances of criminal activities captured by surveillance cameras.

## Overview

Anomaly detection in surveillance videos is essential for identifying unusual or suspicious behavior. This project utilizes the Multiple Instance Learning (MIL) algorithm, which is well-suited for weakly labeled data. By training on the UCF Crime Dataset, the model learns to detect anomalous activities in unseen videos.

## UCF Crime Dataset

The UCF Crime Dataset is a popular benchmark for anomaly detection, containing labeled surveillance videos that depict various criminal activities. It is widely used in the research community to evaluate the performance of anomaly detection models.

**Performance Metric:**  
The model achieves an AUC score of **82.64** on the UCF Crime Dataset.

## Example Output

The following graph shows anomaly scores plotted against the number of frames for a surveillance video:

![Anomaly Scores vs Number of Frames](https://github.com/sachethns/Video-Anomaly-Detection-using-multiple-instance-learning/blob/main/output_graph.png?raw=true)
