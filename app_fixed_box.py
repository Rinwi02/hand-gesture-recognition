"""
Live hand-gesture recognition with a Gradio frontend.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
from torchvision import models, transforms

# ---------- 1. Load the trained model ----------
ckpt = torch.load("gesture_model.pt", map_location="cpu")
classes = ckpt["classes"]
PAD = ckpt.get("pad", 0.25)          # same margin used when cropping the training images

model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- 2. Hand finder (MediaPipe), with a fallback box ----------
try:
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
except Exception:
    hands = None
    print("MediaPipe Hands not available: using a fixed center box. Put your hand inside it.")


def find_hand(frame):
    """Return (x0, y0, x1, y1) of a square crop around the hand, or None."""
    H, W = frame.shape[:2]
    if hands is not None:
        res = hands.process(frame)                      # frame is RGB
        if not res.multi_hand_landmarks:
            return None
        pts = res.multi_hand_landmarks[0].landmark
        xs, ys = [p.x * W for p in pts], [p.y * H for p in pts]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    else:
        side = min(H, W) * 0.5
        x, y, w, h = (W - side) / 2, (H - side) / 2, side, side
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * PAD)
    return int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)


# ---------- 3. Prediction for each webcam frame ----------
def predict(frame):
    if frame is None:
        return None, {}
    frame = np.ascontiguousarray(frame[..., :3])
    box = find_hand(frame)
    shown = frame.copy()
    if box is None:
        cv2.putText(shown, "No hand found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return shown, {}

    x0, y0, x1, y1 = box
    crop = Image.fromarray(frame).crop(box)             # PIL pads with black outside the frame
    with torch.no_grad():
        probs = torch.softmax(model(tfm(crop).unsqueeze(0)), dim=1)[0]
    top = int(probs.argmax())
    text = classes[top] if probs[top] > 0.6 else "unsure"

    cv2.rectangle(shown, (x0, y0), (x1, y1), (0, 200, 0), 3)
    cv2.putText(shown, f"{text} {probs[top]:.0%}", (max(x0, 5), max(y0 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    return shown, {c: float(p) for c, p in zip(classes, probs)}


# ---------- 4. The frontend ----------
with gr.Blocks(title="Hand Gesture Recognition") as demo:
    gr.Markdown("# Hand Gesture Recognition\nAllow camera access and show a gesture.")
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        out_img = gr.Image(label="Detection")
    out_label = gr.Label(num_top_classes=3, label="Prediction")
    cam.stream(predict, inputs=cam, outputs=[out_img, out_label], stream_every=0.3, time_limit=600)

if __name__ == "__main__":
    demo.launch()
