# Handwritten Digit Recognition with a Low-Level Neural Network

An MLP for MNIST digit classification implemented entirely from scratch
using low-level TensorFlow primitives — custom dense layers,and a hand-coded Adam optimizer — with model export and
confusion-matrix evaluation.

## Overview

- **Dataset:** MNIST, loaded via `tensorflow_datasets`
  - Train: 50,000 images (`train[10000:]`)
  - Validation: 10,000 images (`train[0:10000]`)
  - Test: 10,000 images
  - Batch size: 128
- **Task:** 10-class handwritten digit classification (0–9)
- **Goal:** demonstrate the mechanics of a neural network (forward pass,
  loss, backprop, optimizer) without relying on `tf.keras.layers` or
  `model.fit`

## Pipeline

1. Load MNIST train/val/test splits via `tensorflow_datasets`
2. Visualize a 3×3 grid of sample digits with true labels
3. Preprocess: reshape images to `(784,)`, normalize to `[0, 1]`
4. Visualize activation functions used (ReLU, Softmax) as sanity checks
5. Define `DenseLayer`, `MLP`, custom `Adam` optimizer, loss and accuracy
   functions
6. Manual training loop (`train_step` / `val_step` with `tf.GradientTape`)
   for 10 epochs, tracking per-epoch train/val loss and accuracy
7. Plot loss and accuracy curves (train vs. validation)
8. Wrap the trained model in an `ExportModule` (bundles preprocessing +
   inference + class prediction into a single `tf.function` with a fixed
   input signature) and save via `tf.saved_model.save`
9. Reload the saved model and evaluate on the test set
10. Report overall test accuracy and **per-digit accuracy breakdown**
11. Plot a normalized confusion matrix (Seaborn heatmap)

## Setup

```bash
pip install tensorflow tensorflow-datasets pandas matplotlib seaborn scikit-learn
```

