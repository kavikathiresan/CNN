#Import the Libraries
import cv2

def click(events,x,y,flags,pattern):
    if events == cv2.EVENT_LBUTTONDOWN:
        blue=image[:,:,0] # Taking image of BGR color
        green=image[:,:,1]
        red=image[:,:,2]

        b=blue[x,y] # assgining color to cordinates
        g=green[x,y]
        r=red[x,y]

        Text=f'B:{b},G:{g},R:{r}'
        cv2.putText(image,Text,(x,y),cv2.FONT_ITALIC,0.5,(0,255,0),2)
        cv2.imshow('color_image',image)
        cv2.imwrite('T:/Advance AI/cnn/images/B_G_R_images.png',image)

image=cv2.imread('T:/Advance AI/cnn/images/lena_color_256.tif')
image=cv2.resize(image,(250,250))
cv2.imshow('color_image',image)
cv2.setMouseCallback('color_image',click)

cv2.waitKey()
cv2.destroyAllWindows()