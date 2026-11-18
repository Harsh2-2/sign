import time
import pickle
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# 1. Load the Kaggle model
MODEL_PATH = "asl_mediapipe_rf.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
        else:
            model = model_data
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# 29-Class Alphabetical Mapping matching your dataset folders
labels_dict = {
    0: "A", 1: "B", 2: "C", 3: "D",
    4: "del",
    5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N",
    15: "nothing",
    16: "O", 17: "P", 18: "Q", 19: "R", 20: "S",
    21: "space",
    22: "T", 23: "U", 24: "V", 25: "W", 26: "X", 27: "Y", 28: "Z"
}

WINDOW_NAME = "ASL Gesture Sentence Builder"
cv2.namedWindow(WINDOW_NAME)

current_word = ""
full_sentence = ""
last_prediction = ""
sign_start_time = None
HOLD_THRESHOLD = 1.0

print("=" * 60)
print("ASL Recognition Running with Bounding-Box Normalization...")
print("=" * 60)

while True:
    success, img = cap.read()
    if not success or img is None:
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True)

    detected_sign = "nothing"

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # 21 points (x, y, z)
        x_bbox, y_bbox, w_bbox, h_bbox = hand["bbox"]

        # Prevent division by zero if hand is too small
        w_bbox = max(w_bbox, 1)
        h_bbox = max(h_bbox, 1)

        data_aux = []

        # Minimum coordinates for bounding box shift
        min_x = min([lm[0] for lm in lmList])
        min_y = min([lm[1] for lm in lmList])

        # Bounding-Box Normalization (scales coordinates to roughly 0.0 - 1.0 range)
        for lm in lmList:
            norm_x = (lm[0] - min_x) / w_bbox
            norm_y = (lm[1] - min_y) / h_bbox
            norm_z = lm[2] / w_bbox  # Scale depth proportionally to hand width

            data_aux.append(norm_x)
            data_aux.append(norm_y)
            data_aux.append(norm_z)

        input_data = np.asarray(data_aux).reshape(1, -1)

        try:
            prediction = model.predict(input_data)
            predicted_class = prediction[0]
            
            # Map predicted index or label directly
            if isinstance(predicted_class, (int, np.integer)):
                detected_sign = labels_dict.get(int(predicted_class), "nothing")
            else:
                detected_sign = str(predicted_class)

            # Draw prediction over hand
            cv2.rectangle(img, (x_bbox - 20, y_bbox - 20), (x_bbox + w_bbox + 20, y_bbox + h_bbox + 20), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"Sign: {detected_sign}",
                (x_bbox, y_bbox - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        except Exception as e:
            detected_sign = "nothing"

    # --- Hold Timer Logic ---
    if detected_sign != "nothing":
        if detected_sign == last_prediction:
            elapsed_time = time.time() - sign_start_time
            progress = min(elapsed_time / HOLD_THRESHOLD, 1.0)
            
            cv2.rectangle(img, (20, 130), (20 + int(progress * 200), 140), (0, 255, 0), -1)
            cv2.rectangle(img, (20, 130), (220, 140), (255, 255, 255), 1)

            if elapsed_time >= HOLD_THRESHOLD:
                if detected_sign == "space":
                    if current_word:
                        full_sentence += current_word + " "
                        current_word = ""
                elif detected_sign == "del":
                    if current_word:
                        current_word = current_word[:-1]
                    elif full_sentence:
                        words = full_sentence.strip().split(" ")
                        full_sentence = " ".join(words[:-1]) + (" " if len(words) > 1 else "")
                else:
                    current_word += detected_sign

                sign_start_time = time.time()
        else:
            last_prediction = detected_sign
            sign_start_time = time.time()
    else:
        last_prediction = "nothing"
        sign_start_time = None

    # --- UI Banner ---
    cv2.rectangle(img, (0, 0), (img.shape[1], 115), (20, 20, 20), -1)

    cv2.putText(img, f"Word: {current_word}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(img, f"Sentence: {full_sentence}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Sign 'space' = Next Word | Sign 'del' = Backspace | Sign 'nothing' = Idle", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow(WINDOW_NAME, img)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("App closed cleanly.")