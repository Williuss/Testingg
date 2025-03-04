import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import easyocr
import re
import tempfile

def gocr(img, d):
    x = int(d[0])
    y = int(d[1])
    w = int(d[2])
    h = int(d[3])

    img = img[y:h, x:w]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

car_model = YOLO("best.pt")
reader = easyocr.Reader(["en"])

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

st.title("Automatic Toll Payment System")

if not st.session_state.get("stream_active", False):
    st.markdown(""" 
        <style>
            .disclaimer {
                font-size: 16px;
                color: red;  
                margin-bottom: 20px;
            }
            .disclaimer-title {
                font-weight: bold;
            }
            .disclaimer-text {
                color: white;
            }
            ul {
                color: white;
            }
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
                <li>Our model was originally designed to perform plate detection and recognition in real-time using a webcam input. 
            However, due to the limitations of Streamlit, which currently does not support real-time webcam input for processing, 
            we have opted to use video uploads from the user instead. We apologize for any inconvenience this may cause and 
            appreciate your understanding.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.title("Control Panel")

video_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

st.sidebar.markdown(""" 
    **To start plate detection and recognition:**
    - Upload a video file above and click **Start Stream** to begin processing.
    - Once a license plate that meets the conditions is detected, click **Stop Stream** to view the results.
    - If the detected plate matches one in the database, the toll fee will be deducted from the associated balance.
""")

start_stream = st.sidebar.button("Start Stream", key="start_stream")
stop_stream = st.sidebar.button("Stop Stream", key="stop_stream")

plate_placeholder = st.sidebar.empty()

if "stream_active" not in st.session_state:
    st.session_state.stream_active = False
if "detected_texts" not in st.session_state:
    st.session_state.detected_texts = {}

if start_stream and video_file is not None:
    st.session_state.stream_active = True
    st.session_state.video_file = video_file

if stop_stream:
    st.session_state.stream_active = False

if st.session_state.stream_active and video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video_file:
        tmp_video_file.write(video_file.getbuffer())  
        tmp_video_file_path = tmp_video_file.name  
    
    cap = cv2.VideoCapture(tmp_video_file_path)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("Error: Unable to open video source.")
    else:
        st.info("Processing video... Click 'Stop Stream' to end.")

    while cap.isOpened() and st.session_state.stream_active:
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
                    if detected_text in st.session_state.detected_texts:
                        st.session_state.detected_texts[detected_text] += 1
                    else:
                        st.session_state.detected_texts[detected_text] = 1

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, detected_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

if not st.session_state.stream_active and stop_stream:
    detected_texts = st.session_state.detected_texts
    if detected_texts:
        most_frequent_plate = max(detected_texts, key=detected_texts.get)
        st.sidebar.subheader("Most Frequent Detected Plate:")
        st.sidebar.write(most_frequent_plate)
        st.subheader("Most Frequent Detected Plate:")
        st.write(most_frequent_plate)

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
        st.sidebar.write("Result : No plates detected.")
        st.subheader("Result : No plates detected.")
