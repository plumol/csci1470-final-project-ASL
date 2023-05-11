import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf


def run_hd_real_time(model, labels):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1) # want just one hand
    offset = 40 # add a bit of padding around the hand
    imgSize = 28

    while True:
        ret, frame = cap.read()

        if ret:
            hands = detector.findHands(frame, draw=False)

            # Cropped frame that will contain only the hand
            imgCrop = np.ones((imgSize, imgSize, 3), np.uint8)

            # Final image that will be passed to the model
            imgFinal = np.ones((imgSize, imgSize, 1), np.uint8)

            if hands:
                hand = hands[0]
                # Get bounding box info
                x, y, w, h = hand['bbox']

                # Crop img and leave some padding
                imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]

            # Sometimes the bounding box info will briefly return a 0, which will cause errors down the line
            if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
                pass
            else:
                # Similar to preprocessing steps
                imgResized = cv2.resize(imgCrop, (imgSize, imgSize))
                imgGrayscale = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)
                imgFinal = imgGrayscale / 255.

            # Make a prediction based on the processed picture
            prediction = model.predict(np.array([imgFinal]))
            num_list = [np.argmax(i) for i in prediction]
            pred_list = [labels[i] for i in num_list]
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Prediction: {}".format(pred_list[0])
        cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ASL Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 1 ms delay
            break

    cap.release()
    cv2.destroyAllWindows()