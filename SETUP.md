# Mac Setup Guide - Explainable AI Pneumonia Detection

*Note: This setup was tested and optimized on Mac M4 Pro, but works on any Mac with Python 3.9+*

## System Requirements
- **OS**: macOS 12.0 or later
- **Python**: 3.10–3.12 (TensorFlow does not support Python 3.13+ yet)
- **RAM**: 16GB minimum (recommended for training)
- **Disk Space**: 30GB minimum (for dataset + models)

---

## Step 1: Set Up Project Directory

```bash
# Navigate to project
cd /Users/sevendi/Projects/explainable-ai-pneumonia

# Create virtual environment using Python 3.10 or 3.12 (NOT 3.13+)
python3.10 -m venv venv  # or: python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your terminal prompt should now show (venv) prefix
```

## Step 2: Install Dependencies

### Install TensorFlow
```bash
# Upgrade pip first
pip install --upgrade pip

# Install TensorFlow (Apple Silicon Macs will automatically use Metal Performance Shaders)
pip install tensorflow

# Verify TensorFlow Metal backend
python -c "import tensorflow as tf; print('Num GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

### Install Additional Libraries
**Option A: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Install individually**
```bash
pip install pandas numpy matplotlib opencv-python scikit-learn jupyter
```

### Verify All Installations
```bash
python -c "
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import cv2
import sklearn
print('✓ TensorFlow:', tf.__version__)
print('✓ Metal GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)
print('✓ All packages installed successfully!')
"
```

---

## Step 3: Download the Dataset

### Option A: Using Kaggle CLI (Recommended if you have API token)
```bash
# Install Kaggle CLI
pip install kaggle

# Create ~/.kaggle/kaggle.json with your credentials (if not already done)
# Then run:
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract the dataset
unzip chest-xray-pneumonia.zip

# Verify structure
ls -la chest_xray/
# Should output: test  train  val
```

### Option B: Manual Download
1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Click "Download" button
3. Extract ZIP file to project directory

### Verify Dataset Structure
```bash
# Check directory structure find chest_xray -type d | head -5

# Count images per class
echo "Training set:"
ls -la chest_xray/train/NORMAL | wc -l
ls -la chest_xray/train/PNEUMONIA | wc -l

echo "Test set:"
ls -la chest_xray/test/NORMAL | wc -l
ls -la chest_xray/test/PNEUMONIA | wc -l
```

---

## Step 4: Run the Notebook

### Option A: Jupyter Notebook (Recommended)
```bash
# Start Jupyter server
jupyter notebook eXplainable-ai.ipynb

# Browser will open automatically. Click on the notebook to run cells
```

### Option B: VS Code
1. Open VS Code
2. Install "Jupyter" extension (if not already)
3. Open `eXplainable-ai.ipynb`
4. Select kernel: `./venv/bin/python`
5. Run cells sequentially with ▶ buttons

### Option C: Command Line
```bash
# Convert notebook to Python script and run (not recommended for interactive visualization)
jupyter nbconvert --to script eXplainable-ai.ipynb
python eXplainable-ai.py
```

---

## Performance Optimization

### GPU Acceleration (Apple Silicon)
Metal Performance Shaders are automatically enabled in TensorFlow 2.11+. Verify with:
```python
import tensorflow as tf
gpu_devices = tf.config.list_physical_devices('GPU')
print("GPU Devices:", gpu_devices)
print("Is GPU available:", tf.config.list_physical_devices('GPU') != [])
```

### Optional: Configure Memory Growth
To prevent out-of-memory errors:
```python
# Add this at the beginning of your notebook
import tensorflow as tf
gpu_devices = tf.config.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Recommended Training Parameters
```python
BATCH_SIZE = 32  # Reduce to 16 if out-of-memory errors occur
IMG_SIZE = [256, 256]  # Can reduce to [224, 224] for faster training
EPOCHS = 10  # Adjust based on your system
```

---

## After Training

### Save Model Weights
```bash
# Already done in notebook, saved to:
ls -lah saved_weights/
```

### Run on New Images
```python
# In notebook, modify the path:
test_image_path = "your_xray_image.png"
image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(256, 256))
```

---

## Deactivate Virtual Environment
```bash
deactivate
```

---

## Additional Resources

- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [VGG19 Architecture](https://arxiv.org/abs/1409.1556)
- [Jupyter Notebook Documentation](https://jupyter.org/)

---

Good luck! 🚀
