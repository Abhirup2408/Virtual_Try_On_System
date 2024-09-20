import os
import cv2
import cvzone
import streamlit as st
from cvzone.PoseModule import PoseDetector
from PIL import Image
import numpy as np

st.title("Virtual Shirt Try-On")

# Initialize session state variables
if "webcam_started" not in st.session_state:
    st.session_state["webcam_started"] = False

if "shirt_mode" not in st.session_state:
    st.session_state["shirt_mode"] = None  # Possible values: "database", "upload"

if "uploaded_shirt" not in st.session_state:
    st.session_state["uploaded_shirt"] = None

def start_webcam():
    st.session_state["webcam_started"] = True

def stop_webcam():
    st.session_state["webcam_started"] = False
    st.session_state["shirt_mode"] = None

def set_mode_database():
    st.session_state["shirt_mode"] = "database"
    start_webcam()

def set_mode_upload():
    st.session_state["shirt_mode"] = "upload"

# Create buttons for shirt mode selection and webcam control
normal_database_button = st.button("Try on Normal Database", on_click=set_mode_database)
upload_shirt_button = st.button("Upload the Shirt You Want to Try On", on_click=set_mode_upload)
stop_button = st.button("Stop Webcam", on_click=stop_webcam)

# Upload shirt functionality
if st.session_state["shirt_mode"] == "upload":
    uploaded_file = st.file_uploader("Choose a PNG file", type="png")
    if uploaded_file is not None:
        st.session_state["uploaded_shirt"] = Image.open(uploaded_file)
        st.button("Start Webcam", on_click=start_webcam)

if st.session_state["webcam_started"]:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1420)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    detector = PoseDetector()

    shirtFolderPath = "Resources/Shirts"
    listShirts = os.listdir(shirtFolderPath)
    fixedRatio = 262 / 190
    shirtRatioHeightWidth = 581 / 440
    imageNumber = 0
    imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
    imgButtonLeft = cv2.flip(imgButtonRight, 1)
    counterRight = 0
    counterLeft = 0
    selectionSpeed = 10

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            lm11 = lmList[11][:3]
            lm12 = lmList[12][:3]

            if st.session_state["shirt_mode"] == "database":
                imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
            else:
                imgShirt = cv2.cvtColor(np.array(st.session_state["uploaded_shirt"]), cv2.COLOR_RGBA2BGRA)
            
            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)

            try:
                img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except Exception as e:
                print(f"Error overlaying PNG: {e}")

            if st.session_state["shirt_mode"] == "database":
                img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
                img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

                cv2.rectangle(img, (1074, 293), (1074 + imgButtonRight.shape[1], 293 + imgButtonRight.shape[0]), (0, 255, 0), 2)
                cv2.rectangle(img, (72, 293), (72 + imgButtonLeft.shape[1], 293 + imgButtonLeft.shape[0]), (0, 255, 0), 2)

                if lmList[16][0] < 300:
                    counterRight += 1
                    cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
                    if counterRight * selectionSpeed > 360:
                        counterRight = 0
                        if imageNumber < len(listShirts) - 1:
                            imageNumber += 1
                elif lmList[15][0] > 900:
                    counterLeft += 1
                    cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
                    if counterLeft * selectionSpeed > 360:
                        counterLeft = 0
                        if imageNumber > 0:
                            imageNumber -= 1
                else:
                    counterRight = 0
                    counterLeft = 0

        # Display the frame
        stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")
        #if cv2.waitKey(1) & 0xFF == ord('q'):
           # break

    cap.release()
    cv2.destroyAllWindows()
