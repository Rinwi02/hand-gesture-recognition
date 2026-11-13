"""
Smoothest option: a normal OpenCV window (no browser), uses tracker.py.

Run:  python live_cam.py      Press q to quit.
"""
import cv2
from tracker import HandGestureTracker

tracker = HandGestureTracker()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)                       # mirror view, feels natural
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    shown, probs, label = tracker.process(rgb)
    cv2.imshow("Hand Gesture Tracker (q to quit)", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
