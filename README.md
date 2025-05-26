# 🐶 Dog Breed Classification Project

Welcome to the **Dog Breed Classification** project! This repository contains a full pipeline for identifying the breed of a dog from an image using deep learning and transfer learning with TensorFlow and TensorFlow Hub.

The goal of this project is to classify images of dogs into one of 120 different breeds. We make use of **MobileNetV2**, a lightweight and efficient CNN model, for feature extraction and fine-tuning.

This project is structured to:

- Load and explore the dataset  
- Preprocess image data and labels  
- Build and train a deep learning model using transfer learning  
- Evaluate the model's performance  
- Visualize predictions and confidence levels  
- Save and reload the trained model  
- Generate predictions on test and custom images  
- Prepare the final submission file in Kaggle format  

---

## 📂 Dataset

The dataset consists of:

- **Training images**: 10,222 dog images across 120 breeds  
- **Test images**: Unlabeled dog images for prediction  
- **Labels CSV**: Contains image IDs and corresponding breeds  

📥 Download the dataset from Kaggle: [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

---

## 🧱 Model Architecture

- **Base Model**: [MobileNetV2](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) via TensorFlow Hub  
- **Input Shape**: 224 × 224 × 3  
- **Output Layer**: Dense layer with softmax activation (`units = 120`)  

The MobileNetV2 base is used with frozen weights initially, and later fine-tuned for improved accuracy.

---

## 🚀 Training Process

- Images are preprocessed, resized to 224x224, and batched.  
- The dataset is split into training and validation sets.  
- Augmentations like flipping and rotation are applied.  
- Training uses:
  - `EarlyStopping` to prevent overfitting
  - `ModelCheckpoint` to save the best model
  - `TensorBoard` to track performance

Training is done on a small subset (1,000 images) and then on the full set (10,222 images).

---

## 📊 Evaluation & Visualization

Model performance is visualized using:

- Accuracy and loss curves  
- Bar charts of top predicted breeds  
- Side-by-side display of:
  - Actual breed  
  - Predicted breed  
  - Prediction confidence  
- Confusion matrix (optional)

This helps in understanding the strengths and weaknesses of the model.

---

## 🧪 Inference

### ✅ On Validation Data

Evaluate using:

```python
model.evaluate(validation_data)
```

Visualize predictions using `matplotlib`.

### ✅ On Test Data

- Load and preprocess test images  
- Predict using `model.predict()`  
- Save results in `final_predictions.csv` in required Kaggle format

### ✅ On Custom Images

- Upload your own dog photos to the `Dog Images/` folder  
- Run prediction on uploaded image and display the top 5 predicted breeds with probabilities

---

## 💾 Saving & Loading Models

Save trained model:

```python
model.save("best_model.h5")
```

Load model:

```python
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

loaded_model = load_model("best_model.h5", custom_objects={"KerasLayer": hub.KerasLayer})
```

---

## 📈 TensorBoard Logs

Track training progress with TensorBoard:

```python
%tensorboard --logdir="logs/"
```

Log directory can be customized, and logs are saved automatically via callbacks.

---

## 📤 Submission File Format

The final submission CSV includes:

- `id`: Image file name (without extension)  
- One column for each of the 120 dog breeds with prediction probabilities  

Sample row:

```
id,golden_retriever,pug,...,malamute
abc123,0.0012,0.0009,...,0.8754
```

Save as `final_predictions.csv` and upload to Kaggle.

---

## 📸 Example Predictions on Custom Images

| Custom Image | Predicted Breed   | Confidence |
|--------------|-------------------|------------|
| `dog1.jpg`   | golden_retriever  | 87.5%      |
| `dog2.jpg`   | pug               | 92.3%      |
| `dog3.jpg`   | malamute          | 81.2%      |

---

## 📚 References

- 🐶 [Kaggle: Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)  
- 🧠 [TensorFlow Hub](https://www.tensorflow.org/hub)  
- 📄 [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)  

---

## 🛠️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Denistanb/Dog-Breed-Classification.git
cd Dog-Breed-Classification
```

### Step 2: Install Dependencies

Install required Python packages:

```bash
pip install tensorflow tensorflow_hub pandas numpy matplotlib scikit-learn
```

### Step 3: Use Google Colab (Recommended)

- Mount Google Drive to access datasets and save checkpoints.
- Use GPU runtime for faster training:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
---
