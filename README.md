# MLOps PyTorch Pipeline

A complete MLOps pipeline for training and serving a CIFAR-10 image classifier using PyTorch, Docker, and Kubernetes.

## Project Overview

This project demonstrates an end-to-end machine learning workflow:

1. Train a PyTorch image classification model.
2. Save the trained model as a checkpoint.
3. Containerize the training workload with Docker.
4. Run model training as a Kubernetes Job.
5. Containerize the inference API separately.
6. Deploy the model-serving application to Kubernetes.
7. Expose the inference API through a Kubernetes Service.
8. Provide health checks and autoscaling configuration.
9. Test the prediction endpoint.

The model classifies CIFAR-10 images into 10 classes.

## Project Structure

```text
mlops-pytorch-pipeline/
│
├── .github/
│
├── checkpoints/
│   └── classifier_v1.pt
│
├── configs/
│   └── training_config.yaml
│
├── data/
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── storage.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
│
├── tests/
│
├── .dockerignore
│
└── README.md