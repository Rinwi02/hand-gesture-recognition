# tracker.py: follows your hand and classifies its gesture.

import os
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "gesture_model.pt"
LANDMARKER_PATH = "hand_landmarker.task"
# Official MediaPipe hand model. 
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Pairs of landmark indices to draw the hand skeleton.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]


def ensure_landmarker():
    if not os.path.exists(LANDMARKER_PATH):
        print("Downloading the MediaPipe hand model (one time)...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)


class HandGestureTracker:
    def __init__(self, smooth_frames=5, box_alpha=0.5, conf_threshold=0.6):
        """
        smooth_frames : how many recent predictions to average (higher = steadier, slower to react)
        box_alpha     : box responsiveness 0-1 (higher = follows faster, lower = smoother)
        conf_threshold: below this confidence the label shows "unsure"
        """
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        self.classes = ckpt["classes"]
        self.pad = ckpt.get("pad", 0.25)       # same margin used when cropping the training images

        self.model = models.mobilenet_v3_large(weights=None)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        ensure_landmarker()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=vision.RunningMode.VIDEO,   # VIDEO mode = tracks the hand across frames
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.lock = threading.Lock()
        self.last_ts = 0
        self.box = None                            # smoothed (cx, cy, side)
        self.missed = 0
        self.history = deque(maxlen=smooth_frames)
        self.box_alpha = box_alpha
        self.conf_threshold = conf_threshold

    def process(self, frame_rgb):
        frame_rgb = np.ascontiguousarray(frame_rgb[..., :3])
        H, W = frame_rgb.shape[:2]
        shown = frame_rgb.copy()

        with self.lock:
            ts = int(time.monotonic() * 1000)
            ts = max(ts, self.last_ts + 1)         # timestamps must keep increasing
            self.last_ts = ts
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.landmarker.detect_for_video(mp_image, ts)

        if not result.hand_landmarks:
            self.missed += 1
            if self.missed > 5:                    # lost the hand for a few frames: reset
                self.box = None
                self.history.clear()
            cv2.putText(shown, "No hand found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 60, 60), 2)
            return shown, {}, None
        self.missed = 0

        pts = result.hand_landmarks[0]
        xs, ys = [p.x * W for p in pts], [p.y * H for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * self.pad)

        new = np.array([cx, cy, side])
        self.box = new if self.box is None else self.box_alpha * new + (1 - self.box_alpha) * self.box
        cx, cy, side = self.box
        x0, y0, x1, y1 = int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)

        crop = Image.fromarray(frame_rgb).crop((x0, y0, x1, y1))   # PIL pads with black outside the frame
        with torch.no_grad():
            probs = torch.softmax(self.model(self.tfm(crop).unsqueeze(0)), dim=1)[0].numpy()
        self.history.append(probs)
        avg = np.mean(self.history, axis=0)
        top = int(avg.argmax())
        label = self.classes[top] if avg[top] >= self.conf_threshold else "unsure"

        # Draw the skeleton, the following box, and the label
        px = [(int(p.x * W), int(p.y * H)) for p in pts]
        for a, b in CONNECTIONS:
            cv2.line(shown, px[a], px[b], (255, 255, 255), 2)
        for p in px:
            cv2.circle(shown, p, 4, (255, 0, 80), -1)
        cv2.rectangle(shown, (x0, y0), (x1, y1), (0, 200, 0), 3)
        cv2.putText(shown, f"{label} {avg[top]:.0%}", (max(x0, 5), max(y0 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        return shown, {c: float(p) for c, p in zip(self.classes, avg)}, label
