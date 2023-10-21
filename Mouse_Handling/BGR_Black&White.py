# Import the libraries
import cv2

def click(events,x,y,flags,pattern):
    # Taking color of images
    blue=images[:,:,0]
    green=images[:,:,1]
    red=images[:,:,2]
    # assigning cordinates to color
    b=blue[x,y]
    g=green[x,y]
    r=red[x,y]
    if events==cv2.EVENT_LBUTTONDOWN:
        Text=f'B:{b},G:{g},R:{r}'
        convert_images=cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
        cv2.putText(images,Text,(x,y),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),2)
        cv2.imshow('B_G_R_BlackWhite',convert_images)
        cv2.imwrite('T:/Advance AI/cnn/images/BGR_WHITEBLACK.png',images)

#Read the image
images=cv2.imread('T:/Advance AI/cnn/images/lena_color_512.tif')
images=cv2.resize(images,(300,300))
cv2.imshow('image_color',images)
cv2.setMouseCallback('image_color',click)

cv2.waitKey()
cv2.destroyAllWindows()












