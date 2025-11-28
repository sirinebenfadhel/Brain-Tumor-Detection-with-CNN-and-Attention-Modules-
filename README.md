# Brain-Tumor-Detection-with-CNN-and-Attention-Modules-
This repository contains the code and documentation for a deep learning project focused on detecting brain tumors from MRI scans. The project compares the performance of different models, including a custom CNN, LeNet-5, and enhanced versions incorporating attention mechanisms like Channel Attention, Spatial Attention, and CBAM.

**The notebook contais all the comments and interpretations needed under each section**

## Project Overview

- **Objective:** To build and compare Convolutional Neural Network (CNN) models capable of classifying the presence or absence of a brain tumor from MRI images.
- **Approach:** A comparative study of SVM, a custom CNN, LeNet-5, and the integration of attention modules to improve model performance.
- **Key Result:** The custom CNN model achieved the best overall performance, demonstrating the effectiveness of tailored architectures for specific medical imaging data.

## Dataset

The project uses a dataset of brain MRI images, categorized into two classes:
- **Tumor: Yes** (Positive Cases)
- **Tumor: No** (Negative Cases)
- The data: Brain MRI Images for Brain Tumor Detection (LINK: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection )

## Installation & Requirements

To run this project, ensure you have the following libraries installed:

```bash
pip install numpy matplotlib scikit-learn pandas seaborn tensorflow pillow
```
## Models Implemented

This project implements and compares the following models:

1.  **Support Vector Machine (SVM)**
    - Used as a baseline model.
    - Shows good precision but suffers from variability in recall.

2.  **Custom CNN**
    - A deeper convolutional network with Batch Normalization and Dropout.
    - **Architecture:**
        - Input: MRI Image
        - 3x [Conv2D → BatchNorm → ReLU → MaxPooling2D]
        - Flatten → Dense(128) → BatchNorm → ReLU → Dropout(30%)
        - Output: Dense(1) with Sigmoid activation

3.  **LeNet-5**
    - A classic CNN architecture adapted for this binary classification task.

4.  **Attention Modules**
    - **Channel Attention Module:** Focuses on 'what' is meaningful in an input image.
    - **Spatial Attention Module:** Focuses on 'where' the informative parts are located.
    - **Convolutional Block Attention Module (CBAM):** A sequential application of both channel and spatial attention.

## Results & Performance

### Model Comparison Summary

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Custom CNN** | **0.88** | **0.84** | **1.00** | **0.91** |
| LeNet-5 | 0.83 | 0.80 | 0.96 | 0.87 |
| SVM (Median) | ~0.65 | ~0.85 | ~0.60 | ~0.70 |

### Key Findings

- The **Custom CNN model outperformed** both LeNet-5 and SVM, achieving a higher overall accuracy and a better balance between precision and recall.
- It demonstrated a perfect recall (1.00) for the "Tumor" class, meaning it successfully identified all positive cases, which is crucial in a medical context.
- The integration of **Spatial Attention** provided a noticeable improvement in accuracy (0.8553) over the base LeNet-5 model.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sirinebenfadhel/Brain-Tumor-Detection-with-CNN-and-Attention-Modules-.git
    cd Brain-Tumor-Detection-with-CNN-and-Attention-Modules-
    ```

2.  **Prepare your data:**
    - Organize your MRI dataset into appropriate directories for training and validation.

3.  **Run the models:**
    - Execute the Jupyter notebooks or Python scripts for the respective models (SVM, Custom CNN, LeNet-5).
    - The scripts for attention modules (`channel_attention.py`, `spatial_attention.py`, `cbam.py`) can be run to apply these mechanisms to the base models.

## Metrics

The performance of all models was evaluated using the following metrics:
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of true positives among all predicted positives.
- **Recall (Sensitivity):** Ability of the model to identify all relevant positive cases.
- **F1-Score:** Harmonic mean of Precision and Recall.

## Conclusion

The custom CNN model proved to be the most effective for brain tumor detection from MRI scans in this study. The project also highlights the potential of attention mechanisms, particularly Spatial Attention, to further enhance model performance. An interactive dashboard was developed to facilitate the use of the best-performing model.

## Contributors

- **Sirine Ben Fadhel** - [GitHub](https://github.com/sirinebenfadhel)
- **Supervised by:** Rym Sessi

---

*This project is for academic and research purposes.*

(the Accuracy was 0.88, but now it has dropped to 0.8 and that is normal because of the consistent changes that i'm making you can improve the code yourself if you want :)
