#Using mouse we can detect x and y cordinates
#Import thr Libraries
import cv2

def click(events,x,y,flags,pattern):
    if events==cv2.EVENT_LBUTTONDOWN:
        text=f'x:{x} y:{y}'
        cv2.putText(image,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        cv2.imshow('show',image)
        cv2.imwrite("T:\Advance AI\cnn\images\MOUSE_HANDLING_OUTPUT.png", image)

image=cv2.imread('T:\Advance AI\cnn\images\lena_color_256.tif')
cv2.imshow('show',image) # it wil show the image
cv2.setMouseCallback('show',click)
cv2.waitKey()
cv2.destroyAllWindows()
