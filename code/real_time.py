import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
# cap.release()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) # want just one hand
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
offset = 20
imgSize = 300 # might need to change

while True:
    ret, frame = cap.read()
    hands, img = detector.findHands(frame)
    if hands:
        hand = hands[0]
        # get bounding box info
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        # crop img and leave some padding: might not need this if we 
        # are just resizing all the imgs
        imgCrop = img[y-offset:y+h+offset][x-offset:x+w+offset]

        imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop
        cv2.imshow("Cropped ASL Classificaton", imgCrop)
        cv2.imshow("White ASL Classification", imgWhite)

    cv2.imshow("ASL Classification", frame)
    cv2.waitKey(1) # 1 ms delay