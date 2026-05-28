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
│   ├── model.weights.h5           # Trained model weights
│   └── model.weights.retrain.h5   # Retrained model weights
├── Sample results/
│   └── sample_results.png         # Visualization results
└── chest_xray/                    # Dataset directory (download separately)
    ├── train/
    ├── test/
    └── val/
```

---

## Key Components & Fixes Applied

### 1. **Data Loading (FIXED)**
- **Issue**: All three datasets (train, test, val) were loading from the same "train" directory
- **Fix**: Updated to load from correct paths:
  - Training: `chest_xray/train/`
  - Testing: `chest_xray/test/`
  - Validation: `chest_xray/val/`
- **Impact**: Prevents data leakage and ensures proper model evaluation

### 2. **API Updates (FIXED)**
- **Issue**: Using deprecated `tf.keras.preprocessing.image_dataset_from_directory`
- **Fix**: Updated to `tf.keras.utils.image_dataset_from_directory`
- **Impact**: Ensures compatibility with TensorFlow 2.11+

### 3. **Variable Naming (FIXED)**
- **Issue**: Variable named `efficient_net` but using `VGG19`
- **Fix**: Renamed to `base_model` for clarity
- **Impact**: Better code readability and maintenance

### 4. **Early Stopping (FIXED)**
- **Issue**: Monitoring training loss instead of validation loss
- **Fix**: Changed to monitor `val_loss` with `restore_best_weights=True`
- **Impact**: Better generalization and model performance

### 5. **GradCAM Configuration (FIXED)**
- **Issue**: Default layer name `'black_max_pool_2'` doesn't exist in VGG19
- **Fix**: Updated to correct layer `'block5_pool'` (VGG19's final pooling layer)
- **Impact**: Proper heatmap generation for explainability

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

### Expected Execution Flow
1. **Cell 1-3**: Setup and imports
2. **Cell 4-8**: Dataset download and preparation
3. **Cell 9**: Create and compile the VGG19 model
4. **Cell 10-11**: Training with callbacks and early stopping
5. **Cell 12**: Model evaluation
6. **Cell 13-17**: Generate Grad-CAM visualizations
7. **Cell 18-19**: Compute confusion matrix and metrics

---

## Model Architecture

- **Base Model**: VGG19 (pre-trained on ImageNet)
- **Input Size**: 256x256 RGB images
- **Output**: Binary classification (NORMAL / PNEUMONIA)
- **Fine-tuning**: Base layers frozen, only top layers trained
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Tested on**: Mac M4 Pro with Metal GPU acceleration

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution**: Ensure virtual environment is activated and packages installed:
```bash
source venv/bin/activate
pip install tensorflow
```

### Issue: Slow performance
**Solution**: For Apple Silicon Mac, verify Metal GPU is being used:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # Should show GPU device
```

### Issue: Dataset not found
**Solution**: Verify directory structure:
```bash
ls -la chest_xray/
# Should show: train  test  val
```

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
