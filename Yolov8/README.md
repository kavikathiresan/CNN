# YOLOv8
YOLOv8 is a new state-of-the-art computer vision model built by Ultralytics, the creators of YOLOv5. The YOLOv8 model contains out-of-the-box support for object detection, classification, and segmentation tasks, accessible through a Python package as well as a command line interface.
# Capturing the live video
We create a yolov8.py file where we import the initial requirements, such as Opencv and get the base ready for detecting through the webcam.
Install all the necessary dependencies, such as ultralytics, opencv-python and other dependencies.
we’ll capture frames from the webcam using OpenCV. This can be done using the VideoCapture function in OpenCV.
In the above code, we create a VideoCapture object and set it to capture frames from the default camera (0). It sets the resolution of the webcam to 3280x2000. 
We install the ultralytics library that makes working with YOLO very easy and hassle-free.The YOLO model is loaded using the ultralytics library and specifies the location of the YOLO weights file in the yolo-Weights/yolov8n.pt.We instantiate a classNames variable containing a list of object classes that the YOLO model is trained to detect.
The while loop starts and it reads each frame from the webcam using cap.read(). Then it passes the frame to the YOLO model for object detection. The results of object detection are stored in the ‘results’ variable.
For each result, the code extracts the bounding box coordinates of the detected object and draws a rectangle around it using cv2.rectangle(). It also prints the confidence score and class name of the detected object on the console.
