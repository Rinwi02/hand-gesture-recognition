"""
Gradio frontend for live hand-gesture recognition (uses tracker.py).

"""
import gradio as gr
from tracker import HandGestureTracker

tracker = HandGestureTracker()


def predict(frame):
    if frame is None:
        return None, {}
    shown, probs, _ = tracker.process(frame)
    return shown, probs


with gr.Blocks(title="Hand Gesture Recognition") as demo:
    gr.Markdown("# Hand Gesture Recognition\nAllow camera access and show a gesture. The box follows your hand.")
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        out_img = gr.Image(label="Tracking")
    out_label = gr.Label(num_top_classes=3, label="Prediction")
    cam.stream(predict, inputs=cam, outputs=[out_img, out_label], stream_every=0.2, time_limit=600)

if __name__ == "__main__":
    demo.launch()
