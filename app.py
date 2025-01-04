import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import easyocr
import re

def gocr(img, d):
    x, y, w, h = int(d[0]), int(d[1]), int(d[2]), int(d[3])

    cropped_img = img[y:h, x:w]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    res = reader.readtext(gray)

    max_area = 0
    best_text = ""

    for r in res:
        bbox, text, confidence = r
        if confidence > 0.2:
            (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
            area = (x_max - x_min) * (y_max - y_min)
            if area > max_area:
                max_area = area
                best_text = text

    best_text = "".join(best_text.split()).upper()
    best_text = re.sub(r"[^A-Z0-9]", "", best_text)

    if best_text.startswith("I"):
        best_text = best_text[1:]

    match = re.match(r"^([A-Z]{1,2}\d{1,4}[A-Z]{1,3})", best_text)
    if match:
        best_text = match.group(1)

    return best_text

# Model and OCR Reader
car_model = YOLO("best.pt")
reader = easyocr.Reader(["en"])

# Predefined data
bank_accounts = {
    "E2101PAD": 100000,
    "B2540BFA": 50000,
    "T1192EV": 25000,
    "BG1632RA": 4000,
    "B1716SDC": 80000,
    "AD8693AS": 9000,
    "B6703WJF": 780000,
    "B1770SCY": 82000,
    "D1006QZZ": 20000,
    "B7716SDC": 0        
}
toll_fee = 16000

# Streamlit app
st.title("Automatic Toll Payment System")

# Disclaimer section
if not st.session_state.get("stream_active", False):
    st.markdown(
        """
        <style>
            .disclaimer { font-size: 16px; color: red; margin-bottom: 20px; }
            .disclaimer-title { font-weight: bold; }
            .disclaimer-text, ul { color: white; }
        </style>
        <div class="disclaimer">
            <span class="disclaimer-title">Disclaimer:</span> 
            <span class="disclaimer-text">This model will only successfully read a license plate if it is valid. A valid license plate must contain at least one letter in the front, numbers in the middle, and letters at the end. For example, a valid license plate could be 'B 1234 AEK'. If these conditions are not met, the license plate will not be successfully detected.</span>
            <ul>
                <li>This model only works for license plates that fit the specified pattern, containing a combination of letters and numbers.</li>
            </ul>
        </div>
        <div class="disclaimer">
            <span class="disclaimer-title">Additional Note:</span> 
            <ul>
                <li>After conducting extensive tests, we realized that the webcam quality has a significant impact on the OCR process for license plates. A better-quality webcam will result in more accurate OCR results.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Sidebar
st.sidebar.title("Control Panel")
st.sidebar.markdown(
    """
    **To start plate detection and recognition:**
    - Click on **Start Stream** to begin the process. You will be asked to grant access to your webcam.
    - Once a license plate that meets the conditions is detected, click **Stop Stream** to view the results.
    - If the detected plate matches one in the database, the toll fee will be deducted from the associated balance.
    """
)

start_stream = st.sidebar.button("Start Stream")
stop_stream = st.sidebar.button("Stop Stream")

plate_placeholder = st.sidebar.empty()

if "stream_active" not in st.session_state:
    st.session_state["stream_active"] = False
if "detected_texts" not in st.session_state:
    st.session_state["detected_texts"] = {}

if start_stream:
    st.session_state["stream_active"] = True
if stop_stream:
    st.session_state["stream_active"] = False

if st.session_state["stream_active"]:
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("Error: Unable to open video source.")
    else:
        st.info("Streaming... Click 'Stop Stream' to end.")

    while cap.isOpened() and st.session_state["stream_active"]:
        ret, frame = cap.read()
        if not ret:
            st.warning("End of video stream or cannot fetch the frame.")
            break

        detections = car_model(frame)[0] if car_model else []
        for d in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = d
            if score > 0.5:
                detected_text = gocr(frame, [x1, y1, x2, y2])

                if re.match(r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$", detected_text):
                    if detected_text in st.session_state["detected_texts"]:
                        st.session_state["detected_texts"][detected_text] += 1
                    else:
                        st.session_state["detected_texts"][detected_text] = 1

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, detected_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

if not st.session_state["stream_active"] and stop_stream:
    detected_texts = st.session_state["detected_texts"]
    if detected_texts:
        most_frequent_plate = max(detected_texts, key=detected_texts.get)
        st.sidebar.subheader("Most Frequent Detected Plate:")
        st.sidebar.write(most_frequent_plate)

        if most_frequent_plate in bank_accounts:
            balance = bank_accounts[most_frequent_plate]
            if balance >= toll_fee:
                bank_accounts[most_frequent_plate] -= toll_fee
                st.sidebar.success(f"Toll fee of {toll_fee} deducted.")
                st.sidebar.write(f"Remaining Balance: {bank_accounts[most_frequent_plate]}")
            else:
                st.sidebar.error(f"Insufficient funds. Current Balance: {balance}")
        else:
            st.sidebar.error(f"Plate {most_frequent_plate} not found in bank accounts.")
    else:
        st.sidebar.write("No plates detected.")
