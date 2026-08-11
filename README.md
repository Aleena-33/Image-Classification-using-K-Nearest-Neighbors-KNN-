# Fashion MNIST KNN

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify images from the Fashion-MNIST dataset.

## Requirements

Install the required libraries:

```bash
pip install tensorflow scikit-learn matplotlib
```

## Description

* Loads the Fashion-MNIST dataset.
* Uses 10,000 training images and 2,000 testing images.
* Tests KNN with different K values: `1, 3, 5, 7, 9`.
* Calculates the accuracy for each K value.
* Measures the prediction time.
* Displays graphs for accuracy and time.

## Run

```bash
python filename.py
```

The program will print the accuracy and execution time for each K value and display the results in graphs.

## Dataset

**Fashion-MNIST** contains images of clothing items such as shirts, trousers, shoes, bags, etc.

## Output

The program shows:

* Accuracy for each K value
* Prediction time for each K value
* Accuracy vs K graph
* Time vs K graph

## Purpose

This project is created to understand how KNN performs on the Fashion-MNIST image classification dataset.
