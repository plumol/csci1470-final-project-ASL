import cv2
import numpy as np
from model import ASLClassifier as model

def extract(frame):
    # Convert the image to the YCrCb color space
    img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    # Threshold the image in the YCrCb color space to get a binary mask
    mask = cv2.inRange(img_ycrcb, (0, 135, 85), (255, 180, 135))

    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours to keep only those that are likely to be hands
    hands_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000 and area < 100000:
            hull = cv2.convexHull(contour)
            hands_contours.append(hull)

    # Draw the convex hulls on the original image
    cv2.drawContours(frame, hands_contours, -1, (0, 255, 0), 3)

def reduce_noise(frame):
    # Apply multiple dilation steps with different kernel sizes to reduce noise
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        frame = cv2.dilate(frame, kernel, iterations=1)

    # Apply erosion to remove small details
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frame = cv2.erode(frame, kernel, iterations=1)

    # Apply median blurring to further reduce noise
    frame = cv2.medianBlur(frame, 5)

    # Apply thresholding to create a binary image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return ret, binary

def contour_and_bounding(binary, frame):
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_contour = max(contours, key=cv2.contourArea)

    # Extract the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Draw the bounding rectangle on the original frame in blue
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Crop the region of interest from the original image
    # roi = binary[y:y+h, x:x+w]
    roi = frame[y:y+h, x:x+w]

    # Then, resize the cropped image to 28x28 using OpenCV's resize function
    resized_roi = cv2.resize(roi, (28, 28))

    gray = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY) 
    # _, thresholded_img = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO) 

    return gray / 255.0, max_contour



def cnn(img):
    float_img = np.float32(img)
    return float_img



def run_contour_real_time(model, labels):
    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()
        prediction = None

        if ret:
            extract(frame)
            nr_ret, nr_binary = reduce_noise(frame)
            resized_img, max_contour = contour_and_bounding(nr_binary, frame)

            img = cnn(resized_img)
            prediction = model.predict(np.array([img]))
            num_list = [np.argmax(i) for i in prediction]
            pred_list = [labels[i] for i in num_list]

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Prediction: {}".format(pred_list[0])
        cv2.putText(frame, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()