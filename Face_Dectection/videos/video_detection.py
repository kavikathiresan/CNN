# import the libraries
import matplotlib.pyplot as plt
import cv2
# Capture the video
cap = cv2.VideoCapture('T:/Advance AI/cnn/videos/obama.mp4')
math_file=cv2.CascadeClassifier('T:\Advance AI\cnn\harcascade_code\haarcascade_frontalface_default.xml')

while cap.read():
    res,frame=cap.read() # read of video will go to frame and res is condition
    if res==True:
        print(f'orginal shape:{frame.shape}')
        frame=cv2.resize(frame,(500,500))
        cordinates,num_of_faces=math_file.detectMultiScale2(frame)
        print(f'cordinates:{cordinates}')
        if len(cordinates)>0: # Taking the cordinates for single image in video
            x1=cordinates[0][0]
            y1=cordinates[0][1]
            x2=cordinates[0][2]+x1
            y2=cordinates[0][3]+y1
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
            text='face:Obama'
            cv2.putText(frame,text,(x1,y1-3),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            cv2.imshow('video',frame)
        else:
            pass
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()



