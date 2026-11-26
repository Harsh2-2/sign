"""
app.py – ASL Sign Language Translator
=======================================
Features added / improved:
  1. CONFIDENCE SCORE overlay on prediction box (colour-coded bar + %)
  2. TOP-3 ALTERNATIVES shown in a mini panel
  3. HOLD PROGRESS BAR with animated fill and countdown ring
  4. SENTENCE HISTORY panel (last 3 spoken sentences)
  5. TTS (Text-to-Speech) via spd-say / speech-dispatcher (non-blocking thread)
  6. KEYBOARD SHORTCUTS: Q/Esc = quit, S = speak now, C = clear all, Enter = confirm word
  7. FPS counter in corner
  8. STATUS INDICATOR (Ready / Detecting / Locked)

TTS Engine:
  Uses the system's `spd-say` (Speech Dispatcher) which works with
  espeak-ng out of the box on Ubuntu/Debian.  Falls back to a silent
  no-op if unavailable so the app still runs without audio.
"""

import time
import pickle
import threading
import subprocess
import warnings
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH      = "asl_mediapipe_rf.pkl"
HOLD_THRESHOLD  = 1.0        # seconds to hold a sign before it registers
MIN_CONFIDENCE  = 0.30       # ignore predictions below this confidence
HISTORY_MAX     = 3          # number of past sentences to show
WINDOW_NAME     = "ASL – Sign Language Translator"

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C_BG        = (18,  18,  18)
C_BANNER    = (28,  28,  28)
C_GREEN     = (80, 200, 80)
C_YELLOW    = (40, 210, 210)
C_RED       = (60,  60, 220)
C_WHITE     = (240, 240, 240)
C_GREY      = (140, 140, 140)
C_CYAN      = (200, 200,  40)
C_ORANGE    = (40, 140, 240)
C_BOX_HIGH  = (50, 220,  50)   # high confidence bbox
C_BOX_MID   = (40, 190, 220)   # medium confidence
C_BOX_LOW   = (40,  40, 220)   # low confidence


# ─────────────────────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH  (non-blocking)
# ─────────────────────────────────────────────────────────────────────────────
_tts_lock    = threading.Lock()
_tts_process = None   # track current spd-say process

def speak(text: str):
    """Speak *text* asynchronously using spd-say (speech-dispatcher)."""
    if not text.strip():
        return
    def _run():
        global _tts_process
        with _tts_lock:
            if _tts_process and _tts_process.poll() is None:
                _tts_process.terminate()   # stop any in-progress speech
            try:
                _tts_process = subprocess.Popen(
                    ["spd-say", "--wait", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _tts_process.wait()
            except FileNotFoundError:
                pass   # spd-say not installed – silent fallback
    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  ASL Sign Language Translator  –  Loading …")
print("=" * 60)

try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"] if isinstance(model_data, dict) and "model" in model_data else model_data
    class_names = [str(c) for c in model.classes_]
    print(f"✓ Model loaded  –  {len(class_names)} classes  |  {model.n_features_in_} features")
except Exception as exc:
    print(f"✗ Could not load model: {exc}")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA + DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
cap      = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
current_word    = ""
full_sentence   = ""
last_prediction = ""
sign_start_time = None
sentence_history: list[str] = []

# FPS tracking
fps_time = time.time()
fps      = 0.0
frame_count = 0

print(f"✓ Camera opened")
print("  Keyboard: S=Speak | C=Clear | Enter=Confirm word | Q/Esc=Quit")
print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=10, filled=False):
    """Draw a rectangle with rounded corners (approximated with circles at corners)."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    if filled:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        cv2.circle(img, (x1+r, y1+r), r, color, -1)
        cv2.circle(img, (x2-r, y1+r), r, color, -1)
        cv2.circle(img, (x1+r, y2-r), r, color, -1)
        cv2.circle(img, (x2-r, y2-r), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0,  90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0,  90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0,  90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0,  90, color, thickness)


def confidence_color(conf: float):
    """Return a BGR colour interpolating red→yellow→green."""
    if conf >= 0.75:
        return C_BOX_HIGH
    elif conf >= 0.45:
        return C_BOX_MID
    else:
        return C_BOX_LOW


def draw_top_banner(img, current_word, full_sentence, fps, status):
    h, w = img.shape[:2]
    banner_h = 120
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), C_BANNER, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Sentence
    sentence_display = full_sentence if len(full_sentence) <= 55 else "…" + full_sentence[-54:]
    cv2.putText(img, f"Sentence: {sentence_display}",
                (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_WHITE, 2)
    # Current word
    cv2.putText(img, f"Building: {current_word}",
                (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_CYAN, 2)
    # Hints
    cv2.putText(img, "S=Speak  C=Clear  Enter=Word  Q=Quit",
                (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)
    # FPS
    cv2.putText(img, f"FPS:{fps:.0f}", (w - 80, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREY, 1)
    # Status pill
    pill_color = C_GREEN if status == "Ready" else C_YELLOW if status == "Detecting" else C_RED
    cv2.circle(img, (w - 15, 50), 7, pill_color, -1)
    cv2.putText(img, status, (w - 95, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, pill_color, 1)


def draw_prediction_overlay(img, x, y, w, h, sign, confidence, top3):
    """Draw bounding box + confidence bar + top-3 alternatives."""
    col = confidence_color(confidence)
    # Main bounding box
    draw_rounded_rect(img, (x-20, y-20), (x+w+20, y+h+20), col, 2, radius=12)

    # ── Prediction chip (above box) ──────────────────────────────────────────
    chip_y = max(y - 70, 5)
    draw_rounded_rect(img, (x-20, chip_y), (x+w+20, chip_y+52), (30,30,30), -1,
                      radius=8, filled=True)
    cv2.putText(img, sign, (x - 10, chip_y + 28),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, col, 2)
    conf_pct = f"{confidence*100:.0f}%"
    cv2.putText(img, conf_pct, (x + w - 20, chip_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    # ── Confidence bar ───────────────────────────────────────────────────────
    bar_x, bar_y = x - 20, chip_y + 38
    bar_w = w + 40
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50,50,50), -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + 8), col, -1)

    # ── Top-3 alternatives panel ─────────────────────────────────────────────
    if top3:
        panel_x = x + w + 28
        panel_y = y - 20
        panel_w = 140
        panel_h = 10 + len(top3) * 28
        # Clamp to frame
        img_h, img_w = img.shape[:2]
        if panel_x + panel_w > img_w:
            panel_x = x - 20 - panel_w - 4
        draw_rounded_rect(img, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + panel_h),
                          (40,40,40), -1, radius=6, filled=True)
        cv2.putText(img, "Top picks", (panel_x + 6, panel_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1)
        for i, (lbl, prob) in enumerate(top3):
            ty = panel_y + 14 + (i + 1) * 26
            bar_len = int(100 * prob)
            cv2.rectangle(img, (panel_x + 6, ty - 10),
                          (panel_x + 6 + bar_len, ty - 2), (70,70,180), -1)
            cv2.putText(img, f"{lbl}: {prob*100:.0f}%",
                        (panel_x + 6, ty + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)


def draw_hold_progress(img, progress, x_start=16, y_start=130):
    """Draw the hold-progress bar with animated fill."""
    bar_w = 220
    bar_h = 12
    # Background
    draw_rounded_rect(img, (x_start, y_start),
                      (x_start + bar_w, y_start + bar_h),
                      (60, 60, 60), -1, radius=6, filled=True)
    # Fill
    fill_w = int(bar_w * progress)
    fill_col = C_GREEN if progress < 0.8 else C_YELLOW
    if fill_w > 0:
        draw_rounded_rect(img, (x_start, y_start),
                          (x_start + fill_w, y_start + bar_h),
                          fill_col, -1, radius=6, filled=True)
    # Label
    cv2.putText(img, f"Hold: {progress*100:.0f}%",
                (x_start + bar_w + 8, y_start + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREY, 1)


def draw_history(img, history):
    """Draw the last N spoken sentences in a small panel."""
    if not history:
        return
    img_h, img_w = img.shape[:2]
    panel_h = 20 + len(history) * 22
    panel_x = img_w - 310
    panel_y = img_h - panel_h - 10
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (img_w - 10, img_h - 10), (28, 28, 28), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, "History", (panel_x + 6, panel_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GREY, 1)
    for i, sent in enumerate(reversed(history[-HISTORY_MAX:])):
        display = sent if len(sent) <= 36 else sent[:33] + "…"
        cv2.putText(img, f"  {display}",
                    (panel_x + 6, panel_y + 14 + (i + 1) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
cv2.namedWindow(WINDOW_NAME)

while True:
    success, img = cap.read()
    if not success or img is None:
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True)

    # FPS
    frame_count += 1
    now = time.time()
    if now - fps_time >= 0.5:
        fps = frame_count / (now - fps_time)
        frame_count = 0
        fps_time = now

    detected_sign  = "nothing"
    confidence     = 0.0
    top3: list     = []
    status         = "Ready"

    if hands:
        hand   = hands[0]
        lmList = hand["lmList"]
        x_bbox, y_bbox, w_bbox, h_bbox = hand["bbox"]
        w_bbox = max(w_bbox, 1)
        h_bbox = max(h_bbox, 1)

        min_x = min(lm[0] for lm in lmList)
        min_y = min(lm[1] for lm in lmList)

        data_aux = []
        for lm in lmList:
            data_aux.append((lm[0] - min_x) / w_bbox)
            data_aux.append((lm[1] - min_y) / h_bbox)
            data_aux.append(lm[2] / w_bbox)

        input_data = np.asarray(data_aux).reshape(1, -1)

        try:
            proba      = model.predict_proba(input_data)[0]
            pred_idx   = int(np.argmax(proba))
            confidence = float(proba[pred_idx])

            if confidence >= MIN_CONFIDENCE:
                detected_sign = class_names[pred_idx]
                status = "Detecting"

            # Top-3 (excluding the top prediction)
            sorted_idx = np.argsort(proba)[::-1]
            top3 = [(class_names[i], float(proba[i]))
                    for i in sorted_idx[1:4] if float(proba[i]) > 0.05]

            draw_prediction_overlay(img, x_bbox, y_bbox, w_bbox, h_bbox,
                                     detected_sign, confidence, top3)

        except Exception:
            detected_sign = "nothing"

    # ── Hold-timer logic ──────────────────────────────────────────────────────
    committed = False
    if detected_sign not in ("nothing",):
        if detected_sign == last_prediction and sign_start_time is not None:
            elapsed  = time.time() - sign_start_time
            progress = min(elapsed / HOLD_THRESHOLD, 1.0)
            draw_hold_progress(img, progress)
            status = "Locking…" if progress < 1.0 else "Locked"

            if elapsed >= HOLD_THRESHOLD:
                committed = True
                if detected_sign == "space":
                    if current_word:
                        full_sentence += current_word + " "
                        current_word   = ""
                        speak(full_sentence.strip())
                elif detected_sign == "del":
                    if current_word:
                        current_word = current_word[:-1]
                    elif full_sentence:
                        words = full_sentence.strip().split()
                        full_sentence = " ".join(words[:-1]) + (" " if len(words) > 1 else "")
                else:
                    current_word += detected_sign

                sign_start_time = time.time()   # reset timer
        else:
            last_prediction = detected_sign
            sign_start_time = time.time()
    else:
        last_prediction = "nothing"
        sign_start_time = None

    # ── Top banner ────────────────────────────────────────────────────────────
    draw_top_banner(img, current_word, full_sentence, fps, status)

    # ── Sentence history panel ────────────────────────────────────────────────
    draw_history(img, sentence_history)

    cv2.imshow(WINDOW_NAME, img)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), ord('Q'), 27):       # Q / Esc – quit
        break

    elif key == ord('s') or key == ord('S'):  # S – speak current sentence
        text = (full_sentence + current_word).strip()
        if text:
            speak(text)

    elif key == ord('c') or key == ord('C'):  # C – clear all
        if full_sentence or current_word:
            combined = (full_sentence + current_word).strip()
            if combined:
                sentence_history.append(combined)
        current_word  = ""
        full_sentence = ""

    elif key == 13:                            # Enter – confirm word
        if current_word:
            full_sentence += current_word + " "
            current_word   = ""
            speak(full_sentence.strip())

    # Window closed by user
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("\nApp closed cleanly.")
