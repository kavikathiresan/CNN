# import the libraries
import cv2
import matplotlib.pyplot as plt
# Read the image
images = cv2.imread('T:\Advance AI\cnn\images\multiple (1).jpg')
cv2.imshow('mult_image',images)
cv2.waitKey()
cv2.destroyAllWindows()

# take the math file
math_file=cv2.CascadeClassifier('T:\Advance AI\cnn\harcascade_code\haarcascade_frontalface_default.xml')
cordinates,num_of_faces=math_file.detectMultiScale2(images)
print(f'cordinates:{cordinates}')
print(f'length of cordinates:{len(cordinates)}')
for i in range(len(cordinates)):
    cordinates_1=cordinates[i]
    x1 = cordinates_1[0]
    y1 = cordinates_1[1]
    x2 = cordinates_1[2] + x1
    y2 = cordinates_1[3] + y1
    # lets create rectangle on image
    cv2.rectangle(images, (x1, y1), (x2, y2), (0, 255, 0), 1)
    text=f'Face:{i}'
    cv2.putText(images,text,(x1,y1-3),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
    cv2.imshow('mult_image', images)
    cv2.waitKey()
    cv2.destroyAllWindows()





