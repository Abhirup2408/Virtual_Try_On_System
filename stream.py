import os
import cv2
import cvzone
import streamlit as st
from cvzone.PoseModule import PoseDetector
from screeninfo import get_monitors

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if multiple webcams are connected
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)  # Set width of the frame to screen width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)  # Set height of the frame to screen height
detector = PoseDetector()

# Set paths and parameters
shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440
imageNumber = 0
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

# Streamlit setup
st.title("Real-Time Webcam Pose Detection with Virtual Shirt Overlay")
st.subheader("Press 'Start' to access the webcam and see the output in real-time.")

start_button = st.button("Start")
stop_button = st.button("Stop")
frame_window = st.image([])

running = False

if start_button:
    running = True

if stop_button:
    running = False

while running:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip the image horizontally
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        lm11 = lmList[11][:3]
        lm12 = lmList[12][:3]
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
        except Exception as e:
            print(f"Error overlaying PNG: {e}")

        img = cvzone.overlayPNG(img, imgButtonRight, (screen_width - 100, int(screen_height / 2) - 100))
        img = cvzone.overlayPNG(img, imgButtonLeft, (20, int(screen_height / 2) - 100))

        # Draw rectangles around buttons for debugging
        cv2.rectangle(img, (screen_width - 100, int(screen_height / 2) - 100), 
                      (screen_width - 100 + imgButtonRight.shape[1], int(screen_height / 2) - 100 + imgButtonRight.shape[0]), 
                      (0, 255, 0), 2)
        cv2.rectangle(img, (20, int(screen_height / 2) - 100), 
                      (20 + imgButtonLeft.shape[1], int(screen_height / 2) - 100 + imgButtonLeft.shape[0]), 
                      (0, 255, 0), 2)  

        if lmList[16][0] < 300:
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0,
                        counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts) - 1:
                    imageNumber += 1
        elif lmList[15][0] > screen_width - 300:
            counterLeft += 1
            cv2.ellipse(img, (screen_width - 139, 360), (66, 66), 0, 0,
                        counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1
        else:
            counterRight = 0
            counterLeft = 0

    # Display the frame in Streamlit
    frame_window.image(img, channels='BGR')

cap.release()
cv2.destroyAllWindows()
