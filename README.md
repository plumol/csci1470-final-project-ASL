# Real-time ASL Fingerspelling Recognition

**Team members:** Kyle Lam, Jennifer Li, Grace Wan, Claire Yang

**Project structure:**

- code: contains all code related to the implementation of the model
  - assignment.py: compiles, trains, and tests the model, then starts the real-time classifier
  - contour_real_time.py: our real-time implementation using the contour-based hand detection approach
  - hand_detector_real_time.py: our real-time implementation using cvzone's HandDetector package
  - model.py: defines the architecture of our model, as well as its train, test, and call functions. Loss and accuracy metrics are also defined here
  - preprocessing.py: preprocesses images in the dataset by resizing them to 28x28, converting them to grayscale, and normalizing them
- data: contains all the images in our dataset. More information about the dataset itself can be found in our final writeup

**Running the project**

- To run the model, open the `assignment.py` file in VSCode and clicking the play button at the top. This will train and test the model, as well as start the real-time classifier.
