from keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
cap=cv2.VideoCapture(1)
detector=MTCNN()
model1=load_model('mask_trained.h5')
labels_dict={0:'NO MASK',1:'WITH MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}
while True:
    ret,frame=cap.read()  
    faces=detector.detect_faces(frame)
    for face in faces:
        x,y,w,h=face['box']
        roi_head=frame[y:y+h,x:x+h]
        roi_head1=cv2.resize(roi_head,(150,150))
        img=roi_head1/255.0
        img_pred=np.reshape(img,(1,150,150,3))
        result=model1.predict(img_pred)
        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(frame,(x,y+h),(x+w,y+h+40),color_dict[label],-1)
        cv2.putText(frame, labels_dict[label], (x, y+h+20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow("Live CAM",frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()   