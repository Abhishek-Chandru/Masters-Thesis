An Approach to Medical Image Processing Based on Generative Adversarial Networks (VQGAN+D)

This repository contains the implementation of my Master's Thesis at SRH University Heidelberg, focusing on anomaly detection in brain MRI scans using Vector Quantized Generative Adversarial Networks (VQGAN+D).

The project explores how discrete latent representations can improve image reconstruction fidelity and early anomaly detection, addressing challenges in medical imaging such as limited annotated data, subtle pathological features, and model interpretability.

📂 Repository Structure
master-thesis/
│── training_code.ipynb       # Model training implementation
│── testing_code.ipynb        # Model evaluation and anomaly detection
│── requirements.txt          # Dependencies
│── results/                  # Example outputs (generated images, confusion matrix, ROC curve)
│── thesis.pdf                # Full Master's Thesis (optional)
│── README.md                 # Project documentation

⚙️ Requirements

Python 3.9+
PyTorch
Torchvision
NumPy
OpenCV
Matplotlib & Seaborn
TensorBoard

Install dependencies:
pip install -r requirements.txt

CPU-only installation
If you don’t have a GPU, the default install works fine:
pip install torch torchvision

GPU installation (with CUDA support)
If you have an NVIDIA GPU, install the PyTorch version matching your CUDA.


📊 Dataset
This project uses publicly available brain MRI datasets:

Figshare MRI Dataset
Sartaj Brain MRI Dataset
Br35H Dataset

⚠️ Due to privacy and licensing restrictions, datasets are not included in this repository. Please download them separately and update the dataset paths in the code.

🚀 How to Run

Clone the repository
git clone https://github.com/Abhishek-Chandru/Masters-Thesis.git
cd master-thesis

Install all dependencies with:
pip install -r requirements.txt

CPU-only installation
If you don’t have a GPU, the default install works fine:

pip install torch torchvision
GPU installation (with CUDA support)

If you have an NVIDIA GPU, install the PyTorch version matching your CUDA.
For example, for CUDA 12.1:
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

Prepare dataset
Organize data into train/normal, train/anomaly, test/normal, test/anomaly
Preprocessing is handled inside the notebooks.

Train the model
jupyter notebook "Thesis Training Code.ipynb"


Test and evaluate
jupyter notebook "Thesis Testing Code.ipynb"

📈 Results
Key findings from the thesis:
VQGAN+D produced superior reconstructions compared to DCGAN.
Improved anomaly detection performance with ROC-AUC > 0.9.
Stable training due to discrete latent representations.
Clearer interpretability of anomaly localization through reconstruction errors.


📄 Read Full Thesis (PDF) (./Abhishek Chandru Master's Thesis Report.pdf)

🔮 Future Work

Extension to multi-modal MRI scans.
Integration with federated learning for privacy-preserving training.
High-resolution anomaly detection.
Improving explainability for clinical use.
