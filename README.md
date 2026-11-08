# Dog Breed Classification

Transfer-learning image classifier for the Kaggle Dog Breed Identification competition: given a photo of a dog, predict one of 120 breeds. It fine-tunes a MobileNetV2 model from TensorFlow Hub (`mobilenet_v2_130_224`) with a `Dense(120, softmax)` head.

## Data

- `train/`: 10,222 labelled images, with breeds listed in `labels.csv`
- `test/`: 10,357 unlabelled images
- `sample_submission.csv`: the Kaggle submission format (an `id` column plus one probability column per breed)
- `Dog Images/`: three photos (basset, Bedlington terrier, pug) for trying the model on your own images

## Pipeline

Everything is in `Dog Breed Classification.ipynb`:

1. Read `labels.csv` and one-hot encode the 120 breeds.
2. Decode each JPEG, scale to 0-1, resize to 224x224 and batch with `tf.data` (batch size 32).
3. Train first on a 1,000-image subset to check the pipeline, then on all 10,222 images. Callbacks are EarlyStopping on `val_accuracy` and TensorBoard.
4. Save each run as `.h5` in `model/` (one for the 1,000-image run, one for the full run).
5. Predict the test set, write `predictions.csv` and `predictions.xlsx`, and format `final_predictions.csv` as the Kaggle submission.
6. Predict the top 5 breeds for the images in `Dog Images/`.

Reloading a saved model needs the TF-Hub layer registered:

```python
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
model = load_model("model/<file>.h5", custom_objects={"KerasLayer": hub.KerasLayer})
```

## Results

The full-data run reached about 87% accuracy, but that is training accuracy: it was trained on all labelled images with no validation split. The Kaggle metric is multi-class log loss on the hidden test set, which is not computed in the notebook. Treat the figure as a sign that training worked, not as a test score.

## Running it

```bash
git lfs install
git clone https://github.com/DCode-v05/Dog-Breed-Classification.git
pip install tensorflow tensorflow-hub tf-keras pandas numpy matplotlib scikit-learn
jupyter notebook "Dog Breed Classification.ipynb"
```

The notebook, model files and logs are stored with Git LFS. The notebook was written for Google Colab with a GPU and Google Drive for the data, so the paths need adjusting to run elsewhere. TensorBoard event files from seven training runs are in `logs/`.
