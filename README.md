# torch-to
🚀 Implementing and Optimizing Variational Autoencoders (VAEs) for Anomaly Detection
📌 Project Overview

This project implements a Variational Autoencoder (VAE) using PyTorch and applies it to an anomaly detection task on the Fashion-MNIST dataset.
The goal is to rigorously analyze how latent space dimensionality and β-VAE regularization affect reconstruction quality and anomaly detection performance.

Unlike standard autoencoders, VAEs model data probabilistically, making them highly suitable for outlier detection, density estimation, and uncertainty-aware learning, which are critical in domains such as fraud detection and industrial monitoring.

🎯 Objectives

Implement a VAE from scratch with correct reparameterization

Train the model only on normal data

Introduce anomalies during evaluation

Tune latent dimension and β (beta) hyperparameter

Evaluate anomaly detection performance using AUC-ROC

Analyze the trade-off between reconstruction accuracy and latent regularization

🧠 Key Concepts Used

Variational Autoencoders (VAEs)

Reparameterization Trick

KL Divergence Regularization

β-VAE

Reconstruction Error–based Anomaly Detection

AUC-ROC Evaluation Metric

📂 Dataset

Fashion-MNIST

70,000 grayscale images (28×28)

10 clothing categories

One class is treated as anomalous during testing

Model is trained only on normal classes

🏗 Model Architecture
Encoder

Input: 28×28 image (flattened)

Dense layer → ReLU

Outputs:

Mean (μ)

Log-variance (log σ²)

Latent Space

Dimension: configurable (2, 8, 16, 32)

Sampling via reparameterization trick

Decoder

Dense layers

Sigmoid output for image reconstruction

🔢 Loss Function

The VAE loss is defined as:

𝐿
=
Reconstruction Loss
+
𝛽
⋅
KL Divergence
L=Reconstruction Loss+β⋅KL Divergence

Reconstruction Loss: Binary Cross Entropy

KL Divergence: Regularizes latent space towards a unit Gaussian

β (Beta): Controls the strength of regularization

⚙ Hyperparameter Tuning

The following parameters were systematically tuned:

Parameter	Values Explored
Latent Dimension	2, 8, 16, 32
β (Beta)	0.1, 1, 5, 10
Optimizer	Adam
Learning Rate	0.001
Epochs	20

Each configuration was evaluated using AUC-ROC to determine the optimal balance between reconstruction quality and anomaly separation.

🚨 Anomaly Detection Strategy

The VAE is trained only on normal data

During testing, anomalies are introduced

Reconstruction error (MSE) is used as the anomaly score

Higher reconstruction error ⇒ higher likelihood of anomaly

📊 Evaluation Metric

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) is used because:

It is threshold-independent

It handles class imbalance effectively

It is standard for anomaly detection tasks

📈 Results & Observations

Low β values lead to excellent reconstruction but poor anomaly separation

High β values enforce stronger latent regularization but reduce reconstruction quality

An optimal β achieves the highest AUC-ROC, balancing both objectives

Moderate latent dimensions (e.g., 16) perform better than extremely small or large ones

🧩 Trade-off Analysis
Aspect	Low β	High β
Reconstruction	Excellent	Poor
Latent Structure	Weak	Strong
Anomaly Detection	Weak	Strong
Overall Balance	❌	✅ (Optimal range)
🛠 How to Run the Project
pip install torch torchvision scikit-learn numpy

python vae_anomaly_detection.py

📁 Project Structure
├── vae_anomaly_detection.py
├── README.md
├── data/
└── results/

✅ Conclusion

This project demonstrates that Variational Autoencoders, when properly regularized using β-VAE, are powerful tools for anomaly detection.
Through systematic hyperparameter tuning and principled evaluation, the model effectively distinguishes anomalous samples using reconstruction-based metrics.

The analysis highlights the importance of balancing reconstruction fidelity and latent space regularization to achieve optimal detection performance.

