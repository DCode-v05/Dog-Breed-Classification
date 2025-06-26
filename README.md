# Dog Breed Classification Project

## Project Description
This project provides a complete pipeline for classifying dog breeds from images using deep learning and transfer learning. The goal is to accurately identify one of 120 dog breeds from a given image, leveraging modern convolutional neural networks and efficient data handling practices.

## Project Details

### Dataset
- **Training images:** 10,222 labeled dog images across 120 breeds (in `train/`)
- **Test images:** Unlabeled images for prediction (in `test/`)
- **Labels:** `labels.csv` contains image IDs and corresponding breeds
- **Sample Submission:** `sample_submission.csv` for Kaggle format reference
- **Custom Images:** Place your own images in `Dog Images/` for inference

Dataset source: [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

### Model Architecture
- **Base Model:** MobileNetV2 (TensorFlow Hub)
- **Input Shape:** 224 × 224 × 3
- **Output Layer:** Dense layer with softmax activation (120 units)
- **Transfer Learning:** The base model is first used with frozen weights, then fine-tuned for improved accuracy.

### Training Process
- Images are preprocessed, resized, and batched
- Dataset split into training and validation sets
- Data augmentation (flipping, rotation, etc.)
- Training callbacks: EarlyStopping, ModelCheckpoint, TensorBoard
- Training is performed on a subset and then the full dataset

### Evaluation & Visualization
- Accuracy and loss curves
- Bar charts of top predicted breeds
- Side-by-side display of actual vs. predicted breed and confidence
- Optional confusion matrix

### Inference
- **Validation:** Evaluate with `model.evaluate(validation_data)`
- **Test:** Predict on test images and save results in `final_predictions.csv`
- **Custom Images:** Predict on images in `Dog Images/` and display top 5 breeds with probabilities

### Saving & Loading Models
- Save: `model.save("best_model.h5")`
- Load: 
```python
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
loaded_model = load_model("best_model.h5", custom_objects={"KerasLayer": hub.KerasLayer})
```

### Submission File Format
- `id`: Image file name (no extension)
- One column per breed with prediction probabilities
- Example row:
  id,golden_retriever,pug,...,malamute
  abc123,0.0012,0.0009,...,0.8754
- Save as `final_predictions.csv` for Kaggle submission

### Example Predictions on Custom Images
| Custom Image | Predicted Breed   | Confidence |
|--------------|-------------------|------------|
| dog1.jpg     | golden_retriever  | 87.5%      |
| dog2.jpg     | pug               | 92.3%      |
| dog3.jpg     | malamute          | 81.2%      |

## Tech Stack
- Python 3.x
- TensorFlow & TensorFlow Hub
- Pandas, NumPy
- Matplotlib
- scikit-learn
- Jupyter/Colab (recommended for GPU support)

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/TensoRag/Dog-Breed-Classification.git
cd Dog-Breed-Classification
```

### 2. Install Dependencies
```bash
pip install tensorflow tensorflow_hub pandas numpy matplotlib scikit-learn
```

### 3. (Recommended) Use Google Colab
- Mount Google Drive to access datasets and save checkpoints
- Use GPU runtime for faster training
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage
- Run the main notebook or scripts to train, evaluate, and generate predictions
- Place custom images in `Dog Images/` and use the provided code to predict their breeds
- Use TensorBoard (`logs/`) to monitor training progress

## Project Structure
```
Dog-Breed-Classification/
├── train/                  # Training images (labeled)
├── test/                   # Test images (unlabeled)
├── Dog Images/             # Custom images for inference
├── model/                  # Saved model files
├── logs/                   # TensorBoard logs
├── labels.csv              # Image IDs and breed labels
├── sample_submission.csv   # Kaggle sample submission
├── predictions.csv         # Model predictions
├── final_predictions.csv   # Final formatted predictions
├── predictions.xlsx        # Model predictions (Excel)
├── Dog Breed Classification.ipynb # Main notebook
├── README.md               # Project documentation
```

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

## Contact
- **GitHub:** [TensoRag](https://github.com/TensoRag)
- **Email:** denistanb05@gmail.com

---
