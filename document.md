# Using MPI to Implement Kernel Ridge Regression in Parallel

## Overview
This document outlines a project that leverages the power of Message Passing Interface (MPI) to parallelize the computation of Kernel Ridge Regression (KRR). KRR is a sophisticated machine learning technique that extends ridge regression by incorporating kernel methods to map data into a higher-dimensional space. This project focuses on the efficient computation of the Gram matrix, the parallel computation of the regression coefficients using the conjugate gradient method, and the prediction phase. The project also includes a systematic approach to hyperparameter tuning to optimize model performance.

## 1. Introduction to Kernel Ridge Regression

Kernel Ridge Regression (KRR) is a machine learning model that integrates ridge regression with kernel methods. It elevates the original data into a higher-dimensional space using a kernel function, which gauges the similarity between two data points in this feature space without the need for explicit data transformation. In this project, we implement L2 regularization to mitigate overfitting and adopt the Radial Basis Function (RBF) kernel to capture intricate data patterns.

## 2. Computing the Gram Matrix $K$ in Parallel

$K_{ij} = K(X_i, X_j) = \exp\left(-\frac{\|X_i - X_j\|^2}{2\sigma^2}\right)$

In this phase, each process is exposed to a subset of the training data. We define a function `compute_k` that efficiently computes a kernel matrix in a distributed environment using MPI (broadcast and send-receive). It begins by broadcasting the number of local training data points among all processes. Each process then initializes a local kernel matrix and computes kernel values based on its own training data and the data received from other processes. Through $N$ rounds of send-receive operations (where $N$ is the number of processes used), it gathers the necessary data, calculates local kernels, and finally concatenates these results, resulting in each process having a segment of the Gram matrix.

## 3. Computing $\alpha$ in Parallel Using the Conjugate Gradient Method

Given the equation $A\alpha = y$ where $A = K + \lambda I$, each process now has access to a segment of the matrix $A$ and vector $y$. We aim to compute $\alpha$ using the conjugate gradient method.

We define the function `mv_mul` to perform a parallel matrix-vector multiplication, gathering vector $\alpha$ from all processes and computing $A \alpha$. The `inner_product` function computes the local inner product of two vectors and utilizes MPI.Allreduce to aggregate the results across all processes. The `solve_linear` function iteratively updates $\alpha$ until the squared error falls below a specified threshold.

## 4. Predicting $y$ and Computing the Loss

In this section, we predict the values of $y$ for the test dataset using the `predict_y` function. Each process calculates its local summation ( $y_{\text{pred}} = \sum_{i=1}^{N} \alpha_{i} k(x, x_{i})$ , using local $k$ and $\alpha$ ), which are then aggregated using MPI.Reduce to combine results into the global prediction vector $y$. After all predictions are computed, the loss is evaluated using the `compute_loss` function, which calculates the mean squared error between the predicted and actual values.

## 5. Hyperparameter Tuning and Test Results

We perform a grid search through the hyperparameters:

$\lambda = [0.01, 0.1, 1, 2] \quad \text{and} \quad \sigma = [0.1, 0.5, 1, 2]$

By comparing the root mean squared error, we ultimately select $\lambda = 0.1$ and $\sigma = 1$, with the corresponding RMSE being 0.47.
