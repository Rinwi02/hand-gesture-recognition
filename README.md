# Hand Gesture Recognition System

Real-time hand tracking and gesture classification from a webcam, built with deep learning.

A MobileNetV3 classifier, fine-tuned on a sample of the HaGRID dataset, recognises ten hand gestures. MediaPipe finds and follows the hand in every frame, so the gesture is recognised wherever the hand appears, with no stationary box.

> Summer Internship Project, by Rinwi Prosper Jibuin

## Features

- Hand tracking with 21 landmarks (MediaPipe HandLandmarker)
- Gesture classification with a fine-tuned MobileNetV3-Large (PyTorch)
- Smoothed box and predictions, so the display is steady
- Browser frontend (Gradio) and a desktop window (OpenCV)
- A fixed-box version for comparison with the tracking version

**Gestures:** call, dislike, fist, like, ok, one, palm, peace, rock, stop

## How it works

```
Webcam frame -> MediaPipe finds the hand -> padded, smoothed box -> crop
             -> MobileNetV3 classifier -> average of last 5 predictions -> label
```

## Technologies

Python 3.12, PyTorch, torchvision (MobileNetV3), MediaPipe, OpenCV, Pillow, NumPy, Gradio, scikit-learn, Matplotlib, Google Colab, Kaggle.

## Project structure

```
.
├── hand_gesture_training.ipynb   Colab notebook: data prep, training, evaluation
├── gesture_model.pt              Trained model (weights, class names, crop margin)
├── tracker.py                    Hand tracking + classification + smoothing
├── app.py                        Gradio app with hand tracking
├── live_cam.py                   OpenCV window with hand tracking
├── app_fixed_box.py              Gradio app with a fixed box (baseline)
├── requirements.txt
└── figures/                      Images used in the report
```

## Setup (Windows)

Run each command on its own line from the project folder.

```
py -3.12 -m venv gesture-env
gesture-env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```
python app.py             # browser app with hand tracking
python live_cam.py        # desktop window with hand tracking (press q to quit)
python app_fixed_box.py   # browser app with a fixed box
```

The browser apps print a local address (usually http://127.0.0.1:7860). Open it and allow camera access. On the first run, the MediaPipe hand model (`hand_landmarker.task`) is downloaded automatically, so an internet connection is needed once.

## Training your own model

1. Open `hand_gesture_training.ipynb` in Google Colab and choose a GPU runtime.
2. Upload your Kaggle API token (`kaggle.json`) when asked. Never commit this file.
3. Run the cells in order. The last cell downloads `gesture_model.pt`; place it in the project folder.

## Dataset

Trained on a 384p sample of [HaGRID](https://github.com/hukenovs/hagrid) (HAnd Gesture Recognition Image Dataset), released under CC BY-SA 4.0. The dataset is not included in this repository.

Kapitanov, A., Makhlyarchuk, A., and Kvanchiani, K. (2022). *HaGRID - HAnd Gesture Recognition Image Dataset*. arXiv:2206.08219.

## Results

Add your results here: test accuracy, confusion matrix, and screenshots (place the images in `figures/`).

## Author

Rinwi Prosper Jibuin
