# Adversarial Attacks 

This project explores a range of **adversarial attack techniques** in both **computer vision** and **natural language processing**. It provides hands-on implementations, visual analysis, and step-by-step explanations for each method.

🎯 The goal is to demonstrate the vulnerabilities of modern deep learning models and help users understand how small, often imperceptible, perturbations can cause misclassification.

---

## 📚 Project Overview

The repository includes the following Jupyter notebooks, each implementing a specific adversarial method:

| 📁 Notebook | 🧪 Description & Justification |
|-------------|-------------------------------|
| `deepfool.ipynb` | **DeepFool** computes minimal perturbations that cross decision boundaries. Included to demonstrate how imperceptibly small changes can fool even robust models. |
| `projected_gradient_descent_method.ipynb` | **PGD (Projected Gradient Descent)** is a widely used, iterative attack and benchmark for adversarial robustness. This notebook demonstrates a strong untargeted image attack. |
| `one_multi_pixel_attacks.ipynb` | **One-Pixel and Multi-Pixel Attacks** show that even single-pixel changes can drastically affect model output. These are useful for understanding boundary sensitivity. |
| `latent_masking.ipynb` | Explores internal **latent representation manipulation** rather than raw inputs. Offers insights into vulnerabilities in feature space. |
| `CLIP.ipynb` | Attacks **CLIP models** by manipulating text-image embeddings. Included due to the growing use of multimodal models in real-world systems. |
| `TextBugger.ipynb` | **TextBugger** is a powerful NLP attack that alters text minimally while fooling sentiment or classification models. Highlights risks in language applications. |
| `fast_gradient_sign_method.ipynb` | **FGSM (Fast Gradient Sign Method)** is a foundational one-step attack based on gradient sign. Included as a baseline and educational reference. |

---

## 📊 Attack Comparison Table

This table provides a high-level comparison of all the adversarial attacks implemented in this project.

|  **Attack** | 🎯 **Type** | 📂**Domain** |⚙️**Method** |🔐**Access** | 🚨**Class** | 🔁**Steps** |👁️**Perceptibility** | 📝**Notes** |
|---------------|-------------|---------------|--------------------------|----------------|----------------|----------------|----------------------|---------------------------|
| **FGSM** | Untargeted | Vision | Gradient-based | White-box | Evasion | 1 | Low | Fast & simple baseline attack |
| **PGD** | Untargeted | Vision | Gradient-based (iterative) | White-box | Evasion | Many | Low | Stronger than FGSM; widely used benchmark |
| **DeepFool** | Untargeted | Vision | Optimization-based | White-box | Evasion | Iterative | Very Low | Computes minimal boundary-crossing perturbations |
| **One Pixel Attack** | Untargeted | Vision | Score-based | Black-box | Evasion | Few | Low | Alters only 1–5 pixels using optimization |
| **Latent Masking** | Untargeted | Vision | Optimization-based | White-box | Evasion | Iterative | Medium | Perturbs internal (latent) representations |
| **CLIP Embedding Attack** | Untargeted | Multimodal | Optimization-based | White-box | Evasion | Iterative | Low–Medium | Disrupts CLIP’s text–image embedding alignment |
| **TextBugger** | Untargeted | NLP | Heuristic & Score-based | Black-box | Evasion | Varies | Low | Changes characters/words while preserving grammar and semantics |
| **Fast Gradient Sign Method (FGSM)** | Untargeted | Vision | Gradient-based | White-box | Evasion | 1 | Low | Classic one-step attack using gradient sign |
| **Clean-Label Feature Collision** | Targeted | Vision | Optimization-based | White-box | Poisoning | Iterative | Very Low | Inserts clean-looking samples into training to misclassify a specific target at test time |


## ⚙️ Installation Instructions

1. 🧬 Clone this repository:

```bash
git clone https://github.com/your-username/adversarial-attacks.git
cd adversarial-attacks
```

## 📦 Install dependencies:

```bash
pip install -r requirements.txt
```

# 🚀 Usage Instructions
Open a Jupyter Notebook environment:

``` bash
jupyter notebook
```

Run any of the following notebooks:

* `deepfool.ipynb`
* `projected_gradient_descent_method.ipynb`
* `one_multi_pixel_attacks.ipynb`
* `latent_masking.ipynb`
* `CLIP.ipynb`
* `TextBugger.ipynb`
* `fast_gradient_sign_method.ipynb`
* `clean_label_feature_collision.ipynb`


Each notebook includes:

* ` Model loading`
* `Attack generation`
* `Saliency and perturbation visualizations`
* `Interpretation of results`
     

## 💻 Cross-Platform Compatibility

This project works on:

    🐧 Linux – Fully supported. Recommended for GPU acceleration and large model training.

    🍎 macOS – Fully supported for CPU-based execution. GPU support is limited due to PyTorch constraints.

    🪟 Windows – Fully supported. Make sure to properly activate venv and use pip.

  Make sure to use a GPU-enabled environment (e.g., CUDA-compatible machine or Google Colab) for optimal performance. 

## 📦 Requirements Snapshot

Key libraries (see requirements.txt for full list):

    torch==2.6.0+cu124
    torchvision==0.15.2
    numpy==1.26.2
    matplotlib==3.7.1
    Pillow==9.5.0
    requests==2.31.0
    urllib3==2.0.7
    transformers==4.30.0
    diffusers==0.18.0
    torchattacks==3.5.1
    nltk==3.8.1




