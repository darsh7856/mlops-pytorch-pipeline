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
9. Validate the deployed components using Kubernetes and HTTP checks.
10. Document validation evidence using captured terminal screenshots.

The model classifies CIFAR-10 images into 10 classes.

---

## Architecture

```text
                    ┌─────────────────────┐
                    │   CIFAR-10 Dataset  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PyTorch Training  │
                    │      train.py       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Checkpoint    │
                    │ classifier_v1.pt    │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                     ▼                   ▼
              Docker Training      Docker Serving
                     │                   │
                     ▼                   ▼
             Kubernetes Job      Kubernetes Deployment
                                         │
                                         ▼
                                Kubernetes Service
                                         │
                                         ▼
                                  Inference API
                               /health /predict
                                         │
                                         ▼
                                  CIFAR-10 Result


mlops-pytorch-pipeline/
│
├── .github/
│   └── workflows/
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
├── docs/
│   └── evidence/
│       ├── docker-serving-image-local-verification.png
│       ├── docker-local-serving-health-check.png
│       ├── docker-container-run-verification.png
│       ├── docker-predict-endpoint-test.png
│       ├── kubernetes-namespace-pvc-verification.png
│       ├── kubernetes-training-job-running.png
│       ├── kubernetes-training-job-logs.png
│       ├── kubernetes-serving-deployment-status.png
│       └── kubernetes-service-hpa-status.png
│
├── .dockerignore
│
└── README.md