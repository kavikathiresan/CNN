#Import the libraries
import cv2
import matplotlib.pyplot as plt
def click(events,x,y,flags,pattern):
    if events==cv2.EVENT_LBUTTONDOWN:
        text=f'x:{x} y:{y}'
        convert_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        cv2.imshow('im', convert_image)
        cv2.imwrite("T:\Advance AI\cnn\images\black_color_mouse.png", convert_image)

image=cv2.imread('T:\Advance AI\cnn\images\lena_color_256.tif')
cv2.imshow('im',image)

cv2.setMouseCallback('im',click)
cv2.waitKey()
cv2.destroyAllWindows()
