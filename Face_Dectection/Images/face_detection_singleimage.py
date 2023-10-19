# Import the Libraries

import matplotlib.pyplot as plt
import cv2

# Read the file
images=cv2.imread('T:\Advance AI\cnn\images\single.jpeg',1)
print(f'shape of original image:',images.shape)# since image size is large lets resize it
images=cv2.resize(images,(400,600))
print(f'shape of resize image:',images.shape)

# take the math file
math_file=cv2.CascadeClassifier('T:\Advance AI\cnn\harcascade_code\haarcascade_frontalface_default.xml')
cv2.imshow('image',images)
cv2.waitKey()
cv2.destroyAllWindows()

# math_file given to cordinates,num of faces
cordinates,num_of_faces=math_file.detectMultiScale2(images) # detect multi scale means detecting the faces
print(f'cordinates:{cordinates}')

# Take the cordinates of image
x1= cordinates[0][0]
y1= cordinates[0][1]
x2= cordinates[0][2] + x1
y2= cordinates[0][3] + y1

# we can create rectangle on top of the images
cv2.rectangle(images,(x1,y1),(x2,y2),(255,0,0),2)
cv2.imshow('image',images)
cv2.waitKey()
cv2.destroyAllWindows()

# put the text like faces:0
text='Faces :0'
cv2.rectangle(images,(x1,y1),(x2,y2),(255,0,0),2)
cv2.putText(images,text,(x1,y1-4),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),1)
cv2.imshow('image',images)
cv2.waitKey()
cv2.destroyAllWindows()