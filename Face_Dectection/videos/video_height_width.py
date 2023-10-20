# Import The Libraries
import cv2

# Capture the video
cap=cv2.VideoCapture('T:/Advance AI/cnn/videos/obama.mp4')
math_file=cv2.CascadeClassifier('T:\Advance AI\cnn\harcascade_code\haarcascade_frontalface_default.xml')# Take math file
print(cap.read())

#2WAYS TO GET HEIGHT AND WIDTH
print(f'video of width:{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
print(f'video of height:{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
print(f'width:{cap.get(3)}')
print(f'height:{cap.get(4)}')
while True:
    res,frame=cap.read()
    if res==True:
        #print(f'shape of video:{frame.shape}')
        cv2.imshow('frame',frame)
        Text = f'width:{cap.get(3)} and height:{cap.get(4)}'
        cv2.putText(frame,Text,(10,38,),cv2.FONT_HERSHEY_PLAIN,4,(255,0,0),2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(3) & 0xFF==ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
