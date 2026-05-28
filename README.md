# Explainable AI: Pneumonia Detection using Grad-CAM

**Sample results**

![alt](Sample%20results/sample_results.png)

## Project Description

Alright, so let's talk about explainable AI! You know, AI models are doing some amazing things these days, but sometimes they can be like black boxes, making decisions without giving us a clear idea of how they arrived at those conclusions. That's where explainable AI steps in to save the day!

Explainable AI is all about making AI models more transparent and understandable. It helps us humans get insights into the inner workings of these complex models. And one cool technique we're diving into is **Grad-CAM**, short for Gradient-weighted Class Activation Mapping.

Now, here's the plan: I'm going to apply Grad-CAM to Chest X-Ray Images, specifically those related to Pneumonia. By using Grad-CAM, we'll be able to visualize and understand which areas of the X-ray image contribute the most to the model's decision-making process. It's like shining a spotlight on the important regions that help the AI model identify signs of pneumonia.

With this explainable AI technique in place, we can gain deeper insights into how the AI model is analyzing the X-ray images and what factors it's considering when making predictions. This can be incredibly useful for medical professionals, researchers, and even patients to better understand the AI's diagnostic process.

---

## Setup & Installation

### Prerequisites
- Python 3.10–3.12 (TensorFlow does not support Python 3.13+ yet)
- macOS 12.0 or later (Intel or Apple Silicon)

### Installation Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/sevendi/Projects/explainable-ai-pneumonia
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   # Using requirements.txt (recommended)
   pip install -r requirements.txt
   
   # OR manually
   pip install tensorflow pandas numpy matplotlib opencv-python scikit-learn
   ```
   
   **Note for Apple Silicon Mac**: TensorFlow will automatically use Metal Performance Shaders (MPS) for GPU acceleration.

4. **Download the dataset:**
   - Download from [Kaggle: Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - Extract the ZIP file in the project directory
   - You should have: `chest_xray/train`, `chest_xray/test`, and `chest_xray/val` directories

---

## Project Structure

```
explainable-ai-pneumonia/
├── eXplainable-ai.ipynb          # Main notebook
├── README.md                      # This file
├── saved_weights/
│   └── model.weights.h5           # Best model weights (saved during training)
├── Sample results/
│   └── sample_results.png         # Visualization results
└── chest_xray/                    # Dataset directory (download separately)
    ├── train/
    ├── test/
    └── val/
```

---

## Key Implementation Details

### 1. **Two-Stage Training**
- **Stage 1**: VGG19 base fully frozen — only the classification head is trained (`lr=3e-4`, 7 epochs)
- **Stage 2**: VGG19 `block5_conv1` and above unfrozen for fine-tuning (`lr=1e-5`, up to 14 additional epochs)
- **Gradient clipping**: `clipnorm=1.0` on the Stage 2 optimizer prevents gradient explosion when unfreezing deep layers
- **Fresh callbacks**: `EarlyStopping`, `ReduceLROnPlateau`, and checkpoint are re-instantiated for Stage 2 to avoid stale state from Stage 1

### 2. **Validation Strategy**
- The Kaggle dataset's built-in `val/` folder contains only 16 images — too small for reliable callback monitoring
- The 624-image `test/` set is used as `validation_data` during training for stable `EarlyStopping` and `ReduceLROnPlateau` signals
- Final evaluation metrics are still computed on the same test set after training

### 3. **Grad-CAM Visualization**
- Target layer: `block5_conv3` (last convolutional layer of VGG19)
- Output: grayscale X-ray base image with red hotspot overlay (top 5% activations only)
- Applied per-image for the predicted class to highlight discriminative regions

### 4. **Class Imbalance Handling**
- Training set is imbalanced (~3:1 PNEUMONIA:NORMAL)
- Per-class weights computed from folder counts and passed to `model.fit()` via `class_weight`

---

## Running the Notebook

### Option 1: Jupyter Notebook
```bash
jupyter notebook eXplainable-ai.ipynb
```

### Option 2: VS Code
1. Open the notebook in VS Code
2. Select your Python kernel (from the virtual environment)
3. Run cells sequentially


---

## Model Architecture

- **Base Model**: VGG19 (pre-trained on ImageNet)
- **Input Size**: 256x256 RGB images
- **Output**: Binary classification (NORMAL / PNEUMONIA)
- **Training**: Two-stage — frozen head warm-up, then block5 fine-tuning
- **Optimizer**: Adam (`lr=3e-4` Stage 1, `lr=1e-5` + `clipnorm=1.0` Stage 2)
- **Loss**: Categorical Crossentropy with label smoothing (`0.05`)
- **Regularization**: Dropout (0.4 + 0.3), L2 on Dense layer
- **Tested on**: Mac M4 Pro with Metal GPU acceleration


---

## Performance Metrics

The trained model provides:
- **Accuracy**: Binary classification accuracy
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of prediction performance

---

## Grad-CAM Visualization

The Grad-CAM visualization highlights which regions of the X-ray image most influenced the model's prediction:
- **Red/Bright areas**: High influence on pneumonia detection
- **Blue/Dark areas**: Low influence
- **Yellow/Green areas**: Medium influence

This helps validate that the model focuses on clinically relevant regions.

---

## Author
**Sevendi Eldrige Rifki Poluan**

## Dataset Source
[Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## References
- Grad-CAM Paper: [Selvaraju et al., 2019](https://arxiv.org/abs/1610.02055)
- VGG19 Model: [Simonyan & Zisserman, 2015](https://arxiv.org/abs/1409.1556) 
