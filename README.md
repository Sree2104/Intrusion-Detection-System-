# Design and Implementation of Intrusion Detection System using Bi-LSTM

## 📌 Project Overview
This project presents the design and implementation of a Distributional Denial of Service (DDoS) Intrusion Detection System (IDS) using a Bidirectional Long Short-Term Memory (Bi-LSTM) model. The system is designed to identify and classify malicious traffic patterns in network data with high accuracy and low false alarm rates.

## 🚀 Motivation
DDoS attacks are one of the most severe threats to network security, leading to network congestion and service disruption. Existing detection methods often fail to handle high-volume, complex attack patterns. This project leverages the power of Bi-LSTM models to capture sequential dependencies in network traffic and improve detection performance.

---

## 📂 Datasets
We used two main datasets for model training and evaluation:
1. **InSDN Dataset** – Used for initial training.
2. **MCAD-SDN Dataset** – Used for transfer learning and fine-tuning.

### 🔹 Preprocessing Steps:
- **Data Cleaning** – Removed missing values and handled anomalies.
- **Standardization** – Used `StandardScaler` to normalize feature distributions.
- **Dimensionality Reduction** – Applied PCA with 25 components to reduce feature dimensions.
- **Class Balancing** – Applied SMOTE to handle class imbalance.

---

## 🧠 Model Architecture
The Bi-LSTM model was carefully designed using TensorFlow/Keras:

1. **Input Layer** – Dense layer with ReLU activation.
2. **Batch Normalization** – To stabilize learning.
3. **Dropout Layer** – Regularization to prevent overfitting.
4. **First Bidirectional LSTM Layer** – 128 units with return sequences enabled.
5. **Layer Normalization** – For stable gradient flow.
6. **Dropout Layer** – Additional regularization.
7. **Second Bidirectional LSTM Layer** – 64 units with return sequences enabled.
8. **Dropout + Batch Normalization** – For enhanced stability and performance.
9. **Third Bidirectional LSTM Layer** – 32 units.
10. **Output Layer** – Dense layer with Softmax activation for multi-class classification (6 attack classes).

---

## ⚙️ Training and Optimization
### **Training Setup**
- Learning rate: `0.001`
- Loss function: `SparseCategoricalCrossentropy`
- Optimizer: `Adam`
- Batch size: `64`
- Epochs: `50`
- Early stopping: Enabled (patience = 5)
- Learning rate reduction: `ReduceLROnPlateau` (patience = 3, factor = 0.2)
- Model checkpointing: Enabled

### **Fine-Tuning (Transfer Learning)**
- Pretrained model was loaded using `load_model`.
- Existing layers were frozen to retain learned weights.
- New output layer added for adapting to new attack classes.
- Learning rate adjusted to `0.0001` for fine-tuning.

---

## 📊 Results
| Metric | InSDN Dataset | MCAD-SDN Dataset |
|--------|---------------|------------------|
| **Accuracy** | 99.98% | 95.50% |
| **Precision** | 0.9998 | 0.955 |
| **Recall** | 0.9998 | 0.955 |
| **F1 Score** | 0.9998 | 0.955 |
| **False Alarm Rate** | 0.0000 | 0.0000 |

---

## 📜 Code Structure
