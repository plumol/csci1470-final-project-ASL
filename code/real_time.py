import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf

# category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
# cap.release()

def run_rt(model, labels):

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1) # want just one hand
    offset = 40
    imgSize = 28 # might need to change

    while True:
        ret, frame = cap.read()
        hands = detector.findHands(frame, draw=False)

        if ret:
            imgCrop = np.ones((imgSize, imgSize, 3), np.uint8)
            imgFinal = np.ones((imgSize, imgSize, 1), np.uint8)

            if hands:
                hand = hands[0]
                # get bounding box info
                x, y, w, h = hand['bbox']

                # crop img and leave some padding: might not need this if we 
                # are just resizing all the imgs
                imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]

            # cv2.imshow("Cropped ASL Classificaton", imgCrop)
            # cv2.imshow("White ASL Classification", imgWhite)

            if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
                pass
            else:
                # lower_white = np.array([0, 48, 80], dtype=np.uint8)
                # upper_white = np.array([20, 255, 255], dtype=np.uint8)
                # mask = cv2.inRange(imgCrop, lower_white, upper_white) # could also use threshold
                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
                # mask = cv2.bitwise_not(mask)  # invert mask
                # bg = np.full(imgCrop.shape, 0, dtype=np.uint8)  # black bg
                # imgMasked = cv2.bitwise_and(imgCrop, imgCrop, mask=mask)
                # # get masked background, mask must be inverted 
                # mask = cv2.bitwise_not(mask)
                # bk_masked = cv2.bitwise_and(bg, bg, mask=mask)
                # # combine masked foreground and masked background 
                # final = cv2.bitwise_or(imgMasked, bk_masked)
                # mask = cv2.bitwise_not(mask)  # revert mask to original

                imgResized = cv2.resize(imgCrop, (imgSize, imgSize))
                imgGrayscale = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)
                imgFinal = imgGrayscale / 255.

            prediction = model.predict(np.array([imgFinal]))
            num_list = [np.argmax(i) for i in prediction]
            pred_list = [labels[i] for i in num_list]
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Prediction: {}".format(pred_list[0])
        cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ASL Classification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break # 1 ms delay

    cap.release()
    cv2.destroyAllWindows()